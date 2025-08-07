import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple

from torch import Tensor

from .byte_segment_compressor import ByteSegmentCompressor, CompressorOutput
from .expander import CodeExpander, DecoderOnlyExpander
from .code_sequence_transformer import CodeSequenceTransformer

logger = logging.getLogger(__name__)

def _fix_len(seq: torch.Tensor,           # (B, S, D) *or* (B, S)
             kpm: Optional[torch.Tensor], # (B, Sₖ) or None
             tgt_len: int,
             pad_val: float | int,
             ):
    B, S, *tail = seq.shape

    # ---- clip both when too long ----
    if S > tgt_len:
        seq = seq[:, -tgt_len:, ...]
        if kpm is not None:
            kpm = kpm[:, -tgt_len:]
        return seq, kpm                           # done

    # ---- pad sequence when too short ----
    if S < tgt_len:
        pad = (0, 0) * len(tail) + (0, tgt_len - S)
        seq = F.pad(seq, pad, value=pad_val)

    # ---- pad kpm independently (if present) ----
    if kpm is not None:
        S_kpm = kpm.shape[1]
        if S_kpm > tgt_len:                       # rare, but mirror clip-logic
            kpm = kpm[:, -tgt_len:]
        elif S_kpm < tgt_len:
            kpm = F.pad(kpm, (0, tgt_len - S_kpm), value=True)

    return seq, kpm


def _trim_trailing_pad(seq, kpm):
    """Remove all-pad suffix so we append inside the fixed window."""
    if kpm is None:
        return seq, kpm
    # every batch item has identical length because we pad uniformly
    eff_len = int((~kpm).sum(dim=1).max().item())
    return seq[:, :eff_len, ...], kpm[:, :eff_len]


class HierarchicalAutoencoder(nn.Module):
    """Multi-level autoencoder built from compressors and expanders.

    The model compresses a byte sequence through a stack of
    :class:`ByteSegmentCompressor` modules, producing progressively shorter
    sequences of discrete codes.  During decoding the codes are fed through the
    corresponding :class:`CodeExpander` modules in reverse order to reconstruct
    the original bytes.  Optionally a :class:`CodeSequenceTransformer` can model
    the highest-level codes.

    Args:
        num_levels: Number of compression/expansion levels.
        compressor_level_configs: Configuration dictionary for each
            ``ByteSegmentCompressor``.
        initial_vocab_size: Vocabulary size of the raw byte tokens.
        expander_level_configs: Configuration dictionary for each
            ``CodeExpander`` or ``DecoderOnlyExpander``. The list should be in
            order from bottom (closest to the bytes) to top.
        propagate_key_padding_mask: Whether to propagate padding masks between
            levels.
        aux_lm_loss_weight: Weight of auxiliary language-modeling losses on
            compressor inputs.
        top_transformer_config: Optional configuration for a
            ``CodeSequenceTransformer`` over the top-level codes.
        top_lm_loss_weight: Weight for the top-level language-modeling loss.
        top_lm_mse_weight: Weight for the MSE component of the top LM loss.
        top_lm_ce_weight: Weight for the cross-entropy component of the top LM loss.
    """

    def __init__(self,
                 num_levels: int,
                 compressor_level_configs: List[Dict[str, Any]],
                 expander_level_configs: List[Dict[str, Any]],
                 initial_vocab_size: int = 259,
                 propagate_key_padding_mask: bool = True,
                 aux_lm_loss_weight: float = 0.1,
                 top_transformer_config: Optional[Dict[str, Any]] = None,
                 top_lm_loss_weight: float = 1.0,
                 top_lm_mse_weight: float = 1.0,
                 top_lm_ce_weight: float = 1.0,
                 use_flex_attention: bool = True
                 ):
        super().__init__()

        if len(compressor_level_configs) != num_levels:
            raise ValueError("Length of compressor_level_configs must match num_levels.")
        if num_levels <= 0:
            raise ValueError("num_levels must be positive.")

        self.num_levels = num_levels
        self.initial_vocab_size = initial_vocab_size
        self.propagate_key_padding_mask = propagate_key_padding_mask
        self.aux_lm_loss_weight = aux_lm_loss_weight  # Weight for auxiliary LM loss

        self.top_transformer_config = top_transformer_config
        self.top_lm_loss_weight = top_lm_loss_weight
        self.top_lm_mse_weight = top_lm_mse_weight
        self.top_lm_ce_weight = top_lm_ce_weight
        self.use_flex_attention = use_flex_attention

        self.target_compression_ratios: List[Optional[float]] = []
        self.compression_loss_weights: List[float] = []

        # ---- Configure Compressor Stack ----
        self.compressors = nn.ModuleList()
        self.actual_codebook_sizes: List[int] = []
        current_input_vocab_size = initial_vocab_size

        for i in range(num_levels):
            config = compressor_level_configs[i]
            compressor = ByteSegmentCompressor(
                vocab_size=current_input_vocab_size,
                dim=config['dim'],
                heads=config['heads'],
                window=config['window'],
                lm_window=config.get('lm_window'),

                compression_window=config.get('compression_window'),
                num_encoder_layers=config.get('num_encoder_layers', 3),
                encoder_ffn_dim_multiplier=config.get('encoder_ffn_dim_multiplier', 4),
                num_shared_encoder_layers=config.get('num_shared_encoder_layers', 0),
                num_lm_encoder_layers=config.get('num_lm_encoder_layers'),
                num_compression_encoder_layers=config.get('num_compression_encoder_layers'),
                num_queries=config['num_queries'],
                codebook_size=config['codebook_size'],
                beta=config['beta'],
                vq_reset_interval=config.get('vq_reset_interval', 250),
                entropy_delta=config.get('entropy_delta', 0.2),
                entropy_abs_threshold=config.get('entropy_abs_threshold'),
                use_flex_attention=use_flex_attention,
                output_length=config.get('output_length'),
            )
            self.compressors.append(compressor)
            self.actual_codebook_sizes.append(config['codebook_size'])
            current_input_vocab_size = config['codebook_size']

            target_ratio = config.get('target_compression_ratio')
            if isinstance(target_ratio, list):
                target_ratio = target_ratio[0] if target_ratio else None
            self.target_compression_ratios.append(target_ratio)
            self.compression_loss_weights.append(config.get('compression_loss_weight', 1.0))

        if self.top_transformer_config and self.num_levels > 0:
            embed_dim = compressor_level_configs[-1]["dim"]
            cfg = self.top_transformer_config
            transformer_dim = cfg.get("dim", embed_dim)

            self.top_transformer_continuous = cfg.get("continuous", True)
            self.top_lm_mse_weight = cfg.get("mse_weight", self.top_lm_mse_weight)
            self.top_lm_ce_weight = cfg.get("ce_weight", self.top_lm_ce_weight)

            self.code_sequence_transformer = CodeSequenceTransformer(
                embed_dim=embed_dim,
                dim=transformer_dim,
                num_layers=cfg.get("num_layers", 4),
                num_heads=cfg.get("num_heads", 8),
                ffn_dim_multiplier=cfg.get("ffn_dim_multiplier", 4),
                kv_comp_dim =cfg.get("kv_comp_dim", 64),  # d_c
                q_comp_dim = cfg.get("q_comp_dim", 96),  # d_c`
                retr_dim = cfg.get("retr_dim", 32),  # r
                vq=self.compressors[-1].vq,
            )
            mode = "continuous" if self.top_transformer_continuous else "discrete"
            logger.info(
                "Initialized CodeSequenceTransformer for %s codes (embed_dim: %s, dim: %s)",
                mode,
                embed_dim,
                transformer_dim,
            )

        else:
            self.code_sequence_transformer = None
            self.top_transformer_continuous = True

        # ---- Configure Expander Stack ----
        if len(expander_level_configs) != num_levels:
            raise ValueError("Length of expander_level_configs must match num_levels.")

        self.use_continuous_expander_inputs = any(
            cfg.get("use_continuous_inputs", False) for cfg in expander_level_configs
        )
        self.expander_eos_id = expander_level_configs[-1].get("eos_id", 1)

        self.expanders = nn.ModuleList()
        for i in range(num_levels - 1, -1, -1):
            comp_cfg = compressor_level_configs[i]
            exp_cfg = expander_level_configs[i]

            k_hi = self.actual_codebook_sizes[i]
            k_lo = self.initial_vocab_size if i == 0 else self.actual_codebook_sizes[i - 1]

            exp_dim = int(comp_cfg["dim"] * exp_cfg.get("dim_scale", 1.0))
            exp_heads = max(1, int(comp_cfg["heads"] * exp_cfg.get("heads_scale", 1.0)))
            n_enc = exp_cfg.get("num_enc_layers", 1)
            n_dec = exp_cfg.get("num_dec_layers", 1)
            eos_id = exp_cfg.get("eos_id", 1)
            max_len = exp_cfg.get("max_len", 2048)

            if exp_cfg.get("use_decoder_only", True):
                expander = DecoderOnlyExpander(
                    K_hi=k_hi,
                    K_lo=k_lo,
                    D=exp_dim,
                    N_dec=n_dec,
                    H=exp_heads,
                    # cross_window=comp_cfg.get("compression_window", comp_cfg.get("window", 128)),
                    cross_window=1,
                    eos_id=eos_id,
                    max_len=max_len,
                    hi_dim=exp_cfg.get("hi_dim", 256),
                    lo_dim=exp_cfg.get("lo_dim", 128),
                )
            else:
                expander = CodeExpander(
                    K_hi=k_hi,
                    K_lo=k_lo,
                    D=exp_dim,
                    N_enc=n_enc,
                    N_dec=n_dec,
                    H=exp_heads,
                    eos_id=eos_id,
                    max_len=max_len,
                )
            self.expanders.append(expander)

    @torch.no_grad()
    def _gen_segment_at_level(
            self,
            e_idx: int,  # 0 = top expander, L-1 = bottom (bytes)
            hi_seq: torch.Tensor,  # (1, S_hi) or (1, S_hi, D) if continuous
            *,
            comp_for_masks: Optional[Dict[str, Any]],  # latest compression dict (for seg_ids/kpm reuse)
            decode_max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate ONE *segment* at expander `e_idx`, i.e., a variable-length sequence
        of child tokens terminated by EOP (or EOS/EOP at bottom). Returns the
        *materialized* child level output:
          - if not bottom: returns the concatenated child codes (1, S_child_total)
          - if bottom:     returns the emitted bytes             (1, B_new)

        Strategy:
          • Step the expander by exactly one token at a time (max_len = prefix+1).
          • On each new child token:
              – If it's EOP (or EOS/EOP at bottom), stop.
              – Else, if not bottom, recursively expand that child token into its
                own (grandchild) sequence and move on.
        """
        assert hi_seq.size(0) == 1, "Generation currently supports B=1."

        exp = self.expanders[e_idx]
        is_bottom = (e_idx == self.num_levels - 1)
        child_accum = hi_seq.new_zeros((1, 0), dtype=torch.long) if not is_bottom else hi_seq.new_zeros((1, 0),
                                                                                                        dtype=torch.long)

        # Prepare src KPM and seg_ids for this expander call
        if comp_for_masks is not None:
            # seg_ids index mapping: expander 0 -> seg_ids[L-1], ..., expander L-1 -> seg_ids[0]
            seg_idx = self.num_levels - 1 - e_idx
            comp_seg_ids = comp_for_masks["all_seg_ids"][seg_idx] if seg_idx < len(
                comp_for_masks["all_seg_ids"]) else None
            if e_idx == 0:
                src_kpm = comp_for_masks.get("final_key_padding_mask", None)
            else:
                src_kpm = None
        else:
            comp_seg_ids, src_kpm = None, None

        seg_ids = self._maybe_coerce_seg_ids(hi_seq if hi_seq.dim() == 2 else hi_seq[..., 0], comp_seg_ids)
        src_kpm = self._maybe_coerce_kpm(hi_seq if hi_seq.dim() == 2 else hi_seq[..., 0], src_kpm)

        # Target (child) prefix buffers
        lo = child_accum  # (1, 0) initially
        tgt_kpm = self._all_false_kpm_like(torch.zeros((1, 0), device=hi_seq.device, dtype=torch.long))

        # Stop ids for this level
        if is_bottom:
            eos_id, eop_id = self._bottom_ids()
            stop_set = {eos_id, eop_id}
        else:
            eop_id = self._eop_id_for_expander(e_idx)
            stop_set = {eop_id}

        # Generate until boundary
        steps = 0
        hard_cap = decode_max_len if decode_max_len is not None else 10_000_000  # practically unbounded
        while steps < hard_cap:
            steps += 1
            # One-step extend: run expander.generate to lo_len+1
            out = exp.generate(
                codes_hi=hi_seq,
                codes_lo=lo,
                src_key_padding_mask=src_kpm,
                tgt_key_padding_mask=tgt_kpm if tgt_kpm.numel() > 0 else None,
                max_len=lo.size(1) + 1,
                seg_ids=seg_ids,
            )
            # `out` is the whole prefix including the new token
            assert out.size(1) == lo.size(1) + 1, "Expander.generate must extend by exactly one."

            new_tok = out[:, -1] if out.dim() == 2 else out[:, -1, 0]  # (1,)
            lo = out  # keep the prefix
            tgt_kpm = self._all_false_kpm_like(lo)

            # Stop?
            if int(new_tok.item()) in stop_set:
                break

            # Not stopping: if not bottom, recursively expand this child token
            if not is_bottom:
                # Child token becomes *hi_seq* for the next expander level
                # Shape (1,1), dtype long (the expander will embed internally)
                child_hi = new_tok.unsqueeze(1)  # (1,1)
                _ = self._gen_segment_at_level(
                    e_idx=e_idx + 1,
                    hi_seq=child_hi,
                    comp_for_masks=None,  # no valid comp for generated child; we'll fabricate masks
                    decode_max_len=decode_max_len,
                )
                # We don't need the returned bytes/codes here; the recursion materializes
                # at the bottom only. Parent just continues emitting children.

        # Return what this level produced
        return lo if not is_bottom else lo

    @torch.no_grad()
    def generate_bytes_recursive(
            self,
            prompt_tokens: torch.Tensor,  # (1, S₀)
            key_padding_mask: Optional[torch.Tensor] = None,
            *,
            max_top_codes: int = 256,
            decode_max_len: Optional[int] = None,  # per-patch cap
    ) -> torch.Tensor:
        """
        New recursive AR generator.

        Flow:
          1) Compress the running byte buffer to get masks/seg_ids.
          2) Use the top CodeSequenceTransformer to predict ONE next top symbol
             (continuous by default).
          3) Decode *one segment* at expander 0 (variable length, ends on EOP),
             recursively expanding each produced child down to the bottom, where
             bytes are emitted until EOS/EOP.
          4) Append the NEW bytes to the global buffer and loop.

        Notes:
          • Requires B=1 (asserted).
          • Keeps seg_ids/KPM valid via fresh compression at the start of each top step
            (for top expander) and safe fabrication for deeper levels.
        """
        self.eval()
        assert prompt_tokens.size(0) == 1, "Generation currently supports B=1."

        device = prompt_tokens.device
        use_cont = self.use_continuous_expander_inputs

        # Running byte buffer
        byte_buf = prompt_tokens.clone()  # (1, S_bytes)
        byte_kpm = key_padding_mask

        # Helper: run top CST for one step
        def _predict_one_top_symbol(comp) -> torch.Tensor:
            # Prepare top input (continuous or discrete)
            if use_cont:
                top_mem = comp["all_pre_vq_embeddings"][-1]  # (1, L_top, D)
            else:
                top_codes = comp["all_vq_indices"][-1]  # (1, L_top)
                top_mem = F.embedding(top_codes, self.compressors[-1].vq.codebook)

            top_kpm = comp.get("final_key_padding_mask", None)

            # Optional fixed LM length from config
            lm_len = 0
            if self.top_transformer_config and self.top_transformer_config.get("lm_fixed_length"):
                lm_len = int(self.top_transformer_config["lm_fixed_length"])
            if lm_len:
                pad_val = 0.0 if use_cont else self.top_transformer_config.get("lm_pad_id", 258)
                top_mem, top_kpm = _fix_len(top_mem, top_kpm, lm_len, pad_val)

            cst_out = self.code_sequence_transformer(
                input_embeddings=top_mem,
                key_padding_mask=top_kpm,
            )
            next_vec = cst_out["predictions_pre_vq"][:, -1, :]  # (1, D)
            # You can switch to discrete indices if your config says so
            return next_vec  # continuous by default

        for _ in range(max_top_codes):
            # 1) Compress the current bytes to build masks for the top expander
            comp = self.compress(byte_buf, key_padding_mask=byte_kpm)

            # 2) Predict ONE next top symbol
            if self.code_sequence_transformer is None:
                raise RuntimeError("Top-level transformer not initialized.")
            next_top_symbol = _predict_one_top_symbol(comp)  # (1, D)

            # Build the top-level hi sequence = existing top (continuous) + new vector
            if self.use_continuous_expander_inputs:
                top_hi = torch.cat([comp["all_pre_vq_embeddings"][-1], next_top_symbol.unsqueeze(1)],
                                   dim=1)  # (1, L+1, D)
            else:
                # If you switch to discrete: use cst_out["indices"][:, -1] and cat as ints
                raise NotImplementedError("Discrete top-symbol path not wired in here.")

            # 3) Generate one full segment through expander 0, recursively
            child_codes = self._gen_segment_at_level(
                e_idx=0,
                hi_seq=top_hi,
                comp_for_masks=comp,  # reuse seg_ids/kpm for top expander
                decode_max_len=decode_max_len,
            )

            # 4) Materialize bytes for THIS top symbol by expanding every child down to bottom.
            #    The recursion already did that; we just need to *reconstruct* bottom bytes
            #    given the new top_hi (which includes the just-added symbol).
            #    Easiest (and consistent with your current code): call `decompress(...)`.
            decomp = self.decompress(
                top_codes=top_hi,  # continuous
                top_codes_key_padding_mask=self._maybe_coerce_kpm(top_hi[..., 0] if top_hi.dim() == 3 else top_hi,
                                                                  comp.get("final_key_padding_mask")),
                targets_for_teacher_forcing=None,
                max_len_override=decode_max_len,
                teacher_forcing_embeddings=None,
                all_seg_ids=comp["all_seg_ids"],
                compression_results=comp,
            )
            full_bytes = decomp["final_reconstructed_tokens"]  # (1, S_total)
            new_bytes = full_bytes[:, byte_buf.size(1):]  # take the delta

            # If nothing new, stop (shouldn't really happen if expanders behave)
            if new_bytes.numel() == 0:
                break

            # Append
            byte_buf = torch.cat([byte_buf, new_bytes], dim=1)

            # Stop when last byte is EOS or EOP
            eos_id, eop_id = self._bottom_ids()
            last = int(byte_buf[0, -1].item())
            if last == eos_id or last == eop_id:
                break

            # Keep KPM length in sync (no pads during gen)
            byte_kpm = self._all_false_kpm_like(byte_buf)

        return byte_buf

    def _bottom_ids(self) -> Tuple[int, int]:
        """(eos_id, eop_id) for the byte level."""
        # Prefer explicit config if present; fall back to common defaults.
        eos = getattr(self, "bottom_eos_id", getattr(self, "expander_eos_id", 257))
        eop = getattr(self, "bottom_eop_id", 259)
        return int(eos), int(eop)

    def _eop_id_for_expander(self, e_idx: int) -> int:
        """
        EOP for the *output* space of expander e_idx:
          e_idx == L-1 -> bytes (bottom)
          else -> codebook of compressor[e_idx-1]
        """
        if e_idx == self.num_levels - 1:
            _, eop = self._bottom_ids()
            return eop
        # intermediate: child space is compressor[e_idx-1]
        return int(self.compressors[e_idx - 1].vq.eop_token_id)

    def _all_false_kpm_like(self, x: torch.Tensor) -> torch.Tensor:
        """Return a KPM (True=pad) of zeros matching x's sequence length."""
        B, S = x.shape[:2]
        return torch.zeros(B, S, dtype=torch.bool, device=x.device)

    def _seg_ids_like(self, x: torch.Tensor) -> torch.Tensor:
        """Fabricate a per-position segment id (each token is its own segment)."""
        B, S = x.shape[:2]
        base = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        return base

    def _maybe_coerce_seg_ids(self, want: torch.Tensor, have: Optional[torch.Tensor]) -> torch.Tensor:
        """
        If provided `have` matches length, use it; else fabricate a safe one.
        """
        if have is not None and have.size(1) == want.size(1) and have.size(0) == want.size(0):
            return have
        return self._seg_ids_like(want)

    def _maybe_coerce_kpm(self, want: torch.Tensor, have: Optional[torch.Tensor]) -> torch.Tensor:
        """
        If provided `have` matches length, use it; else return all-False (no pad).
        """
        if have is not None and have.size(1) == want.size(1) and have.size(0) == want.size(0):
            return have
        return self._all_false_kpm_like(want)

    def _bottom_ids(self) -> tuple[int, int]:
        eos = getattr(self, "bottom_eos_id", getattr(self, "expander_eos_id", 257))
        eop = getattr(self, "bottom_eop_id", 259)
        return int(eos), int(eop)

    def _all_false_kpm_like(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)

    def _seg_ids_like(self, x: torch.Tensor) -> torch.Tensor:
        B, S = x.shape[:2]
        return torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)

    def _maybe_kpm(self, want_len: int, have: Optional[torch.Tensor], device) -> torch.Tensor:
        if have is not None and have.size(1) == want_len and have.size(0) == 1:
            return have
        return torch.zeros(1, want_len, dtype=torch.bool, device=device)  # all real, no PAD

    def _compress_one_level(self,
                            tokens: torch.Tensor,
                            key_padding_mask: Optional[torch.Tensor] = None,
                            level: int = 0) -> Dict[str, Any]:
        """Compress a sequence at a single level.

        Args:
            tokens: Tensor of raw byte tokens ``(B, S)``.
            key_padding_mask: Optional boolean mask ``(B, S)`` where ``True``
                marks padded positions.
            level: Index of the compressor level to use.

        Returns:
            Dictionary with keys:
                ``'codes'`` – output codes for this level;
                ``'continuous'`` – quantized segment embeddings;
                ``'pre_vq_embeddings'`` – embeddings prior to vector quantization;
                ``'vq_loss'`` – vector‑quantization loss for this level;
                ``'encoder_logits'`` – logits from the encoder (if applicable);
                ``'seg_id'`` – segment IDs for the input sequence;
                plus auxiliary statistics used for language‑modeling losses.
        """

        compressor = self.compressors[level]
        return compressor(tokens, key_padding_mask=key_padding_mask)


    def compress(self, tokens: torch.Tensor,
                 key_padding_mask: Optional[torch.Tensor] = None
                 ) -> Dict[str, Any]:
        """Run all compressors on an input sequence.

        Args:
            tokens: Tensor of raw byte tokens ``(B, S)``.
            key_padding_mask: Optional boolean mask ``(B, S)`` where ``True``
                marks padded positions.

        Returns:
            Dictionary with keys:
                ``'top_codes'`` – output of the last compressor;
                ``'all_codes'`` – list of code tensors from each level;
                ``'all_continuous'`` – list of quantized segment embeddings;
                ``'all_pre_vq'`` – list of embeddings prior to vector quantization;
                ``'vq_loss'`` – accumulated vector‑quantization loss;
                ``'final_key_padding_mask'`` – padding mask for ``top_codes``;
                ``'compression_ratios'`` – ratio of output length to input length
                for each level;
                ``'all_compressor_level_valid_masks'`` – masks identifying valid
                segments at each level;
                plus auxiliary statistics used for language‑modeling losses.
        """
        all_vq_indices_list: List[torch.Tensor] = []
        all_vq_embeddings: List[torch.Tensor] = []
        all_pre_vq_embeddings_list: List[torch.Tensor] = []
        all_input_seq_lengths: List[float] = []
        all_output_seq_lengths: List[float] = []
        all_seg_ids: List[torch.Tensor] = []
        all_vq_loss_list: List[torch.Tensor] = []
        # Stores valid_masks (per segment) from each compressor to reconstruct KPMs later.
        # all_compressor_level_valid_masks: List[Optional[torch.Tensor]] = []
        total_vq_loss = torch.tensor(0., device=tokens.device, dtype=torch.float32)

        # For auxiliary LM loss
        all_entropy_model_logits_list: List[torch.Tensor] = []
        all_compressor_input_tokens_list: List[torch.Tensor] = []
        all_compressor_input_kpms_list: List[Optional[torch.Tensor]] = []

        all_perplexities_list: List[torch.Tensor] = []

        all_patch_end_masks: List[Optional[torch.Tensor]] = []
        all_first_byte_idx: List[Optional[torch.Tensor]] = []

        current_input_tokens = tokens
        current_kpm = key_padding_mask
        steps: List[CompressorOutput] = []

        for i, compressor in enumerate(self.compressors):
            all_compressor_input_tokens_list.append(current_input_tokens) # Store input for aux LM loss
            all_compressor_input_kpms_list.append(current_kpm) # Store KPM for aux LM loss
            if current_kpm is not None:
                input_seq_len = (~current_kpm).sum(dim=1).float().mean().item()
            else:
                input_seq_len = float(current_input_tokens.size(1))
            all_input_seq_lengths.append(input_seq_len)

            comp_out = compressor(current_input_tokens, key_padding_mask=current_kpm)
            steps.append(comp_out)

            all_seg_ids.append(comp_out.seg_id)
            all_patch_end_masks.append(comp_out.patch_end_mask)
            all_perplexities_list.append(comp_out.vq_perplexity)
            all_first_byte_idx.append(comp_out.first_byte_idx)
            all_vq_loss_list.append(comp_out.vq_loss)

            all_vq_indices_list.append(comp_out.vq_indices)
            all_vq_embeddings.append(comp_out.vq_embeddings)
            all_pre_vq_embeddings_list.append(comp_out.pre_vq_embeddings)
            all_entropy_model_logits_list.append(comp_out.entropy_model_logits)
            total_vq_loss += comp_out.vq_loss

            # all_compressor_level_valid_masks.append(comp_out.valid_mask)

            num_q_this_level = compressor.num_queries_per_segment  # L value

            if comp_out.valid_mask is not None:
                # Effective output length based on valid segments * num_queries
                output_seq_len = (comp_out.valid_mask.sum(
                    dim=1) * num_q_this_level).float().mean().item()
            else:
                output_seq_len = float(comp_out.vq_indices.size(1))
            all_output_seq_lengths.append(output_seq_len)

            current_input_tokens = comp_out.vq_indices
            if self.propagate_key_padding_mask:
                # This is just inverting the valid mask to get the KPM for the next level


                if comp_out.valid_mask is not None:
                    # KPM for next level codes: True where segment was NOT valid, repeated for L queries.
                    # Shape: (B, S_hat_segments * L)
                    current_kpm = ~comp_out.valid_mask.repeat_interleave(
                        num_q_this_level, dim=1)
                else:
                    if i < self.num_levels - 1:
                        logger.warning(
                            "'valid_mask' not returned by compressor %s. KPM propagation to next level disabled.",
                            i,
                        )
                    current_kpm = None
            else:
                current_kpm = None

        compression_ratios = [
            out_len / in_len if in_len > 0 else 0.0
            for in_len, out_len in zip(all_input_seq_lengths, all_output_seq_lengths)
        ]
        compression_ratios = [
            tokens.new_tensor(r, dtype=torch.float32) for r in compression_ratios
        ]

        return {
            'all_vq_indices': all_vq_indices_list,
            'all_vq_embeddings': all_vq_embeddings,
            'all_pre_vq_embeddings': all_pre_vq_embeddings_list,
            'all_vq_loss': all_vq_loss_list,
            'vq_loss': total_vq_loss,
            'final_key_padding_mask': current_kpm,  # KPM for top_codes
            'compression_ratios': compression_ratios,
            # 'input_seq_lengths': all_input_seq_lengths,
            # 'output_seq_lengths': all_output_seq_lengths,
            # 'all_compressor_level_valid_masks': all_compressor_level_valid_masks,
            'all_encoder_logits': all_entropy_model_logits_list,
            'all_compressor_input_tokens': all_compressor_input_tokens_list,
            'all_compressor_input_kpms': all_compressor_input_kpms_list,
            'all_codebook_perplexities': all_perplexities_list,
            'all_patch_end_masks': all_patch_end_masks,
            'all_seg_ids': all_seg_ids,
            'all_first_byte_idx': all_first_byte_idx,
            'steps': steps,
        }


    def decompress(
        self,
        top_codes: torch.Tensor,
        top_codes_key_padding_mask: Optional[torch.Tensor] = None,
        targets_for_teacher_forcing: Optional[List[torch.Tensor]] = None,
        target_key_padding_masks: Optional[List[Optional[torch.Tensor]]] = None,
        max_len_override: Optional[int] = None,
        teacher_forcing_embeddings: Optional[List[torch.Tensor]] = None,
        all_seg_ids: Optional[List[torch.Tensor]] = None,
        decomp_codes_lo: Optional[torch.Tensor] = None,
        tf_inputs: Optional[Dict[str, torch.Tensor]] = None,
        compression_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Decode a sequence of top-level codes back to bytes.

        Args:
            top_codes: Codes produced by the highest compressor ``(B, S_top)``.
            top_codes_key_padding_mask: Optional mask for ``top_codes``.
            targets_for_teacher_forcing: When provided, a list of target
                sequences for each level used during teacher forcing.
            target_key_padding_masks: Padding masks corresponding to
                ``targets_for_teacher_forcing``.
            max_len_override: Optional length limit used when generating
                autoregressively.
            teacher_forcing_embeddings: Optional list of continuous embeddings
                corresponding to the targets. When provided and teacher forcing
                is active, these are fed as the high-level inputs for the next
                expander.

        Returns:
            If ``targets_for_teacher_forcing`` is given, returns a dictionary
            ``{'all_logits': [...], 'final_reconstructed_logits': tensor}``.
            Otherwise returns ``{'generated_sequences': [...],
            'final_reconstructed_tokens': tensor}``.
        """
        current_input_codes_hi = top_codes
        # This is the KPM for the input to the current expander (codes_hi)
        current_src_kpm = top_codes_key_padding_mask

        all_output_logits_list: List[torch.Tensor] = []
        all_generated_tokens_list: List[torch.Tensor] = []

        is_teacher_forcing = targets_for_teacher_forcing is not None
        if is_teacher_forcing:
            if targets_for_teacher_forcing is None or len(
                    targets_for_teacher_forcing) != self.num_levels:
                raise ValueError("Length of targets_for_teacher_forcing must match num_levels.")
            if target_key_padding_masks and len(target_key_padding_masks) != self.num_levels:
                raise ValueError(
                    "Length of target_key_padding_masks must match num_levels if provided.")
            if teacher_forcing_embeddings is not None and len(teacher_forcing_embeddings) != self.num_levels - 1:
                raise ValueError(
                    "Length of teacher_forcing_embeddings must be num_levels - 1 when provided.")

        for i in range(self.num_levels):  # From TopExpander to BottomExpander
            expander = self.expanders[i]

            tgt_kpm_for_this_level = None
            if is_teacher_forcing and target_key_padding_masks:
                tgt_kpm_for_this_level = target_key_padding_masks[i]

            if is_teacher_forcing:
                target_sequence_for_this_level = targets_for_teacher_forcing[i]  # Must not be None

                exp_out_dict = expander(
                    codes_hi=current_input_codes_hi,
                    codes_lo=target_sequence_for_this_level,
                    src_key_padding_mask=current_src_kpm,  # KPM for codes_hi
                    # src_key_padding_mask=None,
                    tgt_key_padding_mask=tgt_kpm_for_this_level,  # KPM for codes_lo (targets)
                    # attn_mask=attn_mask,
                    seg_ids = compression_results["all_seg_ids"][self.num_levels - 1 - i],
                )
                all_output_logits_list.append(exp_out_dict['logits'])

                if i < self.num_levels - 1:
                    if teacher_forcing_embeddings is not None:
                        current_input_codes_hi = teacher_forcing_embeddings[i]
                    else:
                        current_input_codes_hi = target_sequence_for_this_level
                    # The KPM for the next expander's input is the KPM of the current target
                    current_src_kpm = tgt_kpm_for_this_level
            else:  # Autoregressive generation
                # comp_results_reversed = compression_results["steps"][self.num_levels - 1 - i]
                generated_tokens = expander.generate(
                    codes_hi=current_input_codes_hi,
                    # codes_lo=compression_results["final_reconstructed_tokens"],
                    codes_lo=compression_results["steps"][self.num_levels - 1 - i].input_sequence,
                    src_key_padding_mask=current_src_kpm,  # KPM for codes_hi
                    tgt_key_padding_mask=compression_results["steps"][self.num_levels - 1 - i].input_padding_mask,
                    max_len=max_len_override,
                    seg_ids=compression_results["all_seg_ids"][self.num_levels - 1 - i]
                )
                all_generated_tokens_list.append(generated_tokens)
                current_input_codes_hi = generated_tokens
                current_src_kpm = None  # KPM for purely generated sequences is typically None for next stage

        if is_teacher_forcing:
            return {'all_logits': all_output_logits_list,
                    'final_reconstructed_logits': all_output_logits_list[-1]}
        else:
            return {'generated_sequences': all_generated_tokens_list,
                    'final_reconstructed_tokens': all_generated_tokens_list[-1]}

    # ------------------------------------------------------------------
    # Helper utilities for the training forward pass
    # ------------------------------------------------------------------

    @staticmethod
    def _insert_eop(
            seq: torch.Tensor,
            end_mask: torch.Tensor,
            kpm: Optional[torch.Tensor],
            eop_id: int,
            pad_id: int = 0  # ← pass your actual pad value here
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Same signature as before but returns tensors of the *same* length S.

        * seq ............ (B, S) original codes
        * end_mask ....... (B, S) True at last query of each patch
        * kpm ............ (B, S) True = padding
        """
        B, S = seq.shape

        # ----- step a: normalise end_mask length -----
        if end_mask.size(1) != S:
            if end_mask.size(1) > S:  # clip
                end_mask = end_mask[:, :S]
            else:  # pad
                end_mask = F.pad(end_mask, (0, S - end_mask.size(1)), value=False)

        # no tokens at all -> nothing to do
        if S == 0:
            return seq, kpm

        if kpm is None:
            raise ValueError("`kpm` is required to keep the length constant.")

        # ----- step b: budget check -----
        inserts = end_mask.sum(dim=1)  # (B,)
        pad_slots = kpm.sum(dim=1)  # (B,)

        # Vectorised clip: keep only the first `pad_slots[b]` True values per row
        mask_i = end_mask.to(torch.long)  # 0/1
        keep_msk = torch.cumsum(mask_i, dim=1) <= pad_slots.unsqueeze(1)
        end_mask &= keep_msk  # in-place prune

        # ----- step c: shift map (unchanged) -----
        mask_i = end_mask.to(torch.long)
        cumsum_end = torch.cumsum(mask_i, dim=1)
        shift = cumsum_end - mask_i  # (B, S)

        # ----- step d: restrict to real tokens -----
        data_mask = ~kpm  # True == real token
        shift_data = shift[data_mask]  # 1-D flattened view

        # ----- allocate outputs (all-pad by default) -----
        out = seq.new_full((B, S), pad_id)
        out_kpm = None if kpm is None else torch.ones_like(kpm, dtype=torch.bool)

        # conveniences for scatter
        token_idx = torch.arange(S, device=seq.device).unsqueeze(0).expand(B, -1)
        batch_idx = torch.arange(B, device=seq.device).unsqueeze(1).expand_as(seq)

        # ----- step e-1: scatter original tokens -----
        token_pos = token_idx + shift  # (B,S)
        token_pos_flat = token_pos[data_mask]  # flattened view aligned w/ shift_data
        out[batch_idx[data_mask], token_pos_flat] = seq[data_mask]

        # ----- step e-2: scatter EOP tokens -----
        eop_mask = data_mask & end_mask  # positions that need an EOP
        eop_src_pos = token_idx[eop_mask] + shift[eop_mask] + 1
        out[batch_idx[eop_mask], eop_src_pos] = eop_id

        # ----- step f: update kpm -----
        if out_kpm is not None:
            out_kpm[batch_idx[data_mask], token_pos_flat] = False  # real tokens
            out_kpm[batch_idx[eop_mask], eop_src_pos] = False  # EOP tokens

        return out, out_kpm


    def _compute_compression_loss(
        self,
        compression_ratios: List[torch.Tensor],
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        total_unweighted = torch.tensor(0., device=device, dtype=torch.float32)
        total_weighted = torch.tensor(0., device=device, dtype=torch.float32)
        details: Dict[str, torch.Tensor] = {}
        for i, ratio in enumerate(compression_ratios):
            tgt = self.target_compression_ratios[i] if i < len(self.target_compression_ratios) else None
            loss = torch.tensor(0., device=device, dtype=torch.float32)
            if tgt is not None:
                loss = (ratio.mean() - tgt) ** 2
                total_weighted += self.compression_loss_weights[i] * loss
            total_unweighted += loss
            details[f"compression_loss_L{i}"] = loss
        avg = total_unweighted / self.num_levels if self.num_levels > 0 else torch.tensor(0., device=device, dtype=torch.float32)
        return avg, total_weighted, details

    def _compute_top_code_lm_loss(
        self,
        compression_results: Dict[str, Any],
        kpm_for_top_codes: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = compression_results['all_pre_vq_embeddings'][-1].device
        avg_loss = torch.tensor(0., device=device, dtype=torch.float32)
        details: Dict[str, torch.Tensor] = {}

        if (self.code_sequence_transformer is not None and
                self.top_lm_loss_weight > 0 and
                compression_results['all_pre_vq_embeddings'][-1].numel() > 0):
            # cont = compression_results['all_pre_vq'][-1]
            cont = compression_results['all_vq_embeddings'][-1]
            codes = compression_results['all_vq_indices'][-1]
            kpm = kpm_for_top_codes  # (B, S?) or None

            transformer_input = cont if self.top_transformer_continuous else F.embedding(
                codes, self.compressors[-1].vq.codebook
            )
            cst_out = self.code_sequence_transformer(
                input_embeddings=transformer_input,
                key_padding_mask=kpm,
            )
            preds_pre_vq = cst_out['predictions_pre_vq']
            vq_loss_top = cst_out['vq_loss']

            if cont.size(1) <= 1:
                mse_loss = torch.tensor(0., device=device, dtype=torch.float32)
                ce_loss = torch.tensor(0., device=device, dtype=torch.float32)
            else:
                pred = preds_pre_vq[:, :-1, :]
                mask = kpm[:, 1:] if kpm is not None else None

                target_cont = cont[:, 1:, :]
                mse_per_tok = F.mse_loss(pred, target_cont, reduction='none').mean(dim=-1)

                target_codes = codes[:, 1:]
                codebook = self.compressors[-1].vq.codebook
                logits = -torch.cdist(pred, codebook)
                ce_per_tok = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_codes.reshape(-1),
                    reduction='none'
                ).view_as(target_codes)

                if mask is not None:
                    loss_mask = ~mask
                    valid_targets = loss_mask.sum()
                    if valid_targets > 0:
                        mse_loss = (mse_per_tok * loss_mask).sum() / valid_targets.clamp(min=1e-9)
                        ce_loss = (ce_per_tok * loss_mask).sum() / valid_targets.clamp(min=1e-9)
                    else:
                        mse_loss = torch.tensor(0., device=device, dtype=torch.float32)
                        ce_loss = torch.tensor(0., device=device, dtype=torch.float32)
                else:
                    mse_loss = mse_per_tok.mean()
                    ce_loss = ce_per_tok.mean()

            avg_loss = (
                self.top_lm_mse_weight * mse_loss
                + self.top_lm_ce_weight * ce_loss
                + vq_loss_top
            )
            details['top_code_mse'] = mse_loss
            details['top_code_ce'] = ce_loss
            details['top_code_vq_loss'] = vq_loss_top

        return avg_loss, details


    def _prepare_teacher_forcing_inputs(
            self,
            compression_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        After this call we guarantee, for every level j (0 = bottom bytes):
            • targets[j]             : (B, S_j)
            • target_kpms[j] or None : same shape
            • seg_ids_matching_j     : same shape  (# used later for attn_mask)

          so downstream code never has to pad/clip again.
        """
        all_compressor_input_tokens = compression_results["all_compressor_input_tokens"]
        all_compressor_input_kpms = compression_results["all_compressor_input_kpms"]
        all_codes = compression_results["all_vq_indices"]
        all_pre_vq = compression_results["all_pre_vq_embeddings"]
        # all_valid = compression_results["all_compressor_level_valid_masks"]
        all_patch_mask = compression_results["all_patch_end_masks"]
        # seg_ids = compression_results["all_seg_ids"]
        parent_ids_aligned: list[Tensor] = []
        start_bytes_aligned: list[Tensor] = []

        targets, kpms, seg_ids_aligned = [], [], []

        # ---------- walk from top to bottom ----------
        for lvl in range(self.num_levels-1, -1, -1):
            tgt = all_compressor_input_tokens[lvl]
            kpm = all_compressor_input_kpms[lvl]

            # ---- insert EOP, if we have a patch mask for this level ----
            if lvl < len(all_patch_mask) and all_patch_mask[lvl] is not None:
                eop_id = self.compressors[lvl - 1].vq.eop_token_id
                pad_id = self.compressors[lvl - 1].vq.padding_token_id
                tgt, kpm = self._insert_eop(tgt, all_patch_mask[lvl], kpm, eop_id, pad_id)

            targets.append(tgt)
            kpms.append(kpm)

        top_codes_for_decompress = (all_pre_vq[-1] if self.use_continuous_expander_inputs
                                    else all_codes[-1])


        tf_continuous = ([all_pre_vq[-2 - i] for i in range(self.num_levels - 1)]
                         if self.use_continuous_expander_inputs else None)

        return {
            'targets': targets,
            'target_kpms': kpms,
            'top_codes': top_codes_for_decompress,
            'teacher_forcing_embeddings': tf_continuous,
            'parent_ids_aligned': parent_ids_aligned,  # NEW
            'start_bytes_aligned': start_bytes_aligned,  # NEW
            # 'all_seg_ids': seg_ids_aligned,  # NEW: already padded
        }

    def _compute_reconstruction_loss(
        self,
        all_logits: List[torch.Tensor],
        targets: List[torch.Tensor],
        target_kpms: List[Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = all_logits[0].device if all_logits else torch.device('cpu')
        total_loss = torch.tensor(0., device=device, dtype=torch.float32)
        details: Dict[str, torch.Tensor] = {}

        for i in range(self.num_levels):
            logits_i = all_logits[i]
            target_i = targets[i]
            target_kpm_i = target_kpms[i]

            loss_per_token = F.cross_entropy(
                logits_i.reshape(-1, logits_i.size(-1)),
                target_i.reshape(-1),
                reduction='none',
            ).view_as(target_i)

            if target_kpm_i is not None:
                loss_mask = ~target_kpm_i
                current_loss = (loss_per_token * loss_mask).sum() / loss_mask.sum().clamp(min=1e-9)
            else:
                current_loss = loss_per_token.mean()

            expander_k_lo = self.expanders[i].K_lo
            if expander_k_lo == self.initial_vocab_size:
                target_level_desc = '_bytes'
            else:
                reconstructed_code_level_idx = self.num_levels - 2 - i
                if reconstructed_code_level_idx >= 0:
                    target_level_desc = f"_codesL{reconstructed_code_level_idx}"
                else:
                    target_level_desc = '_intermediate_codes'
            level_name = f"reconstruction_to_K{expander_k_lo}{target_level_desc}"

            details[level_name] = current_loss
            total_loss += current_loss

        avg_loss = total_loss / self.num_levels if self.num_levels > 0 else torch.tensor(0., device=device, dtype=torch.float32)
        return avg_loss, details

    def _compute_auxiliary_losses(
        self,
        tokens: torch.Tensor,
        all_encoder_logits: List[torch.Tensor],
        all_compressor_input_tokens: List[torch.Tensor],
        all_compressor_input_kpms: List[Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = tokens.device
        total_unweighted = torch.tensor(0., device=device, dtype=torch.float32)
        details: Dict[str, torch.Tensor] = {}

        if self.aux_lm_loss_weight > 0 and all_encoder_logits:
            num_valid_levels = 0
            for i in range(self.num_levels):
                encoder_logits_i = all_encoder_logits[i]
                input_tokens_i = all_compressor_input_tokens[i]
                input_kpm_i = all_compressor_input_kpms[i]

                if input_tokens_i.size(1) <= 1:
                    current_aux_loss = torch.tensor(0., device=device, dtype=torch.float32)
                else:
                    logits_for_next = encoder_logits_i[:, :-1, :]
                    target_tokens_for_next = input_tokens_i[:, 1:]
                    kpm_for_shifted_targets = input_kpm_i[:, 1:] if input_kpm_i is not None else None

                    if logits_for_next.size(1) == 0:
                        current_aux_loss = torch.tensor(0., device=device, dtype=torch.float32)
                    else:
                        loss_per_tok = F.cross_entropy(
                            logits_for_next.reshape(-1, logits_for_next.size(-1)),
                            target_tokens_for_next.reshape(-1),
                            reduction='none'
                        ).view_as(target_tokens_for_next)

                        if kpm_for_shifted_targets is not None:
                            loss_mask = ~kpm_for_shifted_targets
                            valid_targets_count = loss_mask.sum()
                            if valid_targets_count > 0:
                                current_aux_loss = (loss_per_tok * loss_mask).sum() / valid_targets_count.clamp(min=1e-9)
                            else:
                                current_aux_loss = torch.tensor(0., device=device, dtype=torch.float32)
                        else:
                            current_aux_loss = loss_per_tok.mean()

                    num_valid_levels += 1

                details[f"aux_lm_loss_L{i}"] = current_aux_loss
                total_unweighted += current_aux_loss

            avg_loss = total_unweighted / num_valid_levels if num_valid_levels > 0 else torch.tensor(0., device=device, dtype=torch.float32)
        else:
            avg_loss = torch.tensor(0., device=device, dtype=torch.float32)

        return avg_loss, details

    @torch.no_grad()
    def _gen_segment_at_level(
            self,
            e_idx: int,  # 0 = top expander, L-1 = bottom (bytes)
            hi_seq: torch.Tensor,  # (1, S_hi) or (1, S_hi, D)
            *,
            comp_for_masks: Optional[Dict[str, Any]],  # fresh compress(...) for top, else None
            decode_max_len: Optional[int],
    ) -> torch.Tensor:
        """
        Returns: bottom bytes (1, B_new) generated for this parent segment.
        Uses DecoderOnlyExpander.generate which fills PADs up to EOP/EOS or budget.
        """
        assert hi_seq.size(0) == 1, "Gen supports B=1 for now."
        exp = self.expanders[e_idx]
        is_bottom = (e_idx == self.num_levels - 1)
        device = hi_seq.device

        # ---- build per-call buffers for child's sequence ----
        budget = int(decode_max_len) if decode_max_len is not None else int(exp.max_len)
        # Seed with a single valid token (EOS-as-BOS), rest PAD.
        lo = torch.zeros(1, budget, dtype=torch.long, device=device)
        tgt_kpm = torch.ones_like(lo, dtype=torch.bool)  # True == PAD
        lo[:, 0] = exp.eos_id
        tgt_kpm[:, 0] = False
        prev_valid = 1

        # src KPM for memory (top expander gets real mask; deeper levels fabricate)
        if comp_for_masks is not None and e_idx == 0:
            src_kpm = comp_for_masks.get("final_key_padding_mask", None)
            src_kpm = self._maybe_kpm(hi_seq.size(1) if hi_seq.dim() == 2 else hi_seq.size(1), src_kpm, device)
            seg_ids = self._seg_ids_like(lo)
        else:
            src_kpm = self._maybe_kpm(hi_seq.size(1) if hi_seq.dim() == 2 else hi_seq.size(1), None, device)
            seg_ids = self._seg_ids_like(lo)

        # stopping set for this level
        if is_bottom:
            eos_id, eop_id = self._bottom_ids()
            stop_set = {eos_id, eop_id}
        else:
            # child space is compressor[e_idx-1]
            stop_set = {int(self.compressors[e_idx - 1].vq.eop_token_id)}

        # ---- let the expander fill until EOP/EOS/budget/full ----
        out = exp.generate(
            codes_hi=hi_seq,
            codes_lo=lo,
            src_key_padding_mask=src_kpm,
            tgt_key_padding_mask=tgt_kpm,  # will be mutated in-place
            seg_ids=seg_ids,
            max_len=budget,
        )
        # After the call, tgt_kpm has been updated in-place
        new_valid = int((~tgt_kpm).sum().item())
        if new_valid <= prev_valid:
            # no progress; return empty
            return lo.new_zeros(1, 0, dtype=torch.long)

        # consume newly produced tokens sequentially
        bottom_bytes = lo.new_zeros(1, 0, dtype=torch.long)
        for pos in range(prev_valid, new_valid):
            tok = int(out[0, pos].item())
            if tok in stop_set:
                break
            if is_bottom:
                # direct byte emission
                bottom_bytes = torch.cat([bottom_bytes, out[:, pos:pos + 1]], dim=1)
            else:
                # recurse: this child token becomes `hi_seq` for the next level
                child_hi = out[:, pos:pos + 1]  # (1,1) long
                child_bytes = self._gen_segment_at_level(
                    e_idx=e_idx + 1,
                    hi_seq=child_hi,
                    comp_for_masks=None,
                    decode_max_len=decode_max_len,
                )
                bottom_bytes = torch.cat([bottom_bytes, child_bytes], dim=1)

        # include terminal at bottom if you want it in the stream (keep it: caller stops on it)
        if is_bottom and new_valid > prev_valid:
            last_tok = int(out[0, new_valid - 1].item())
            if last_tok in stop_set:
                bottom_bytes = torch.cat([bottom_bytes, out[:, new_valid - 1:new_valid]], dim=1)

        return bottom_bytes

    def forward(self, tokens: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None
                ) -> Dict[str, Any]:
        """Training forward pass combining compression and reconstruction.

        The input tokens are compressed level by level and then decoded back
        using teacher forcing.  Losses from vector quantization, reconstruction
        and optional language-modeling objectives are aggregated and returned.

        Args:
            tokens: Batch of byte tokens ``(B, S)``.
            key_padding_mask: Optional padding mask for ``tokens``.

        Returns:
            Dictionary containing the total loss and detailed statistics for
            each loss term, along with compression metrics.
        """
        compression_results = self.compress(tokens, key_padding_mask=key_padding_mask)
        vq_loss = compression_results['vq_loss']
        kpm_for_top_codes = compression_results['final_key_padding_mask']
        compression_ratios = compression_results['compression_ratios']
        # input_seq_lengths_c = compression_results['input_seq_lengths']
        # output_seq_lengths_c = compression_results['output_seq_lengths']
        all_encoder_logits = compression_results['all_encoder_logits']
        all_compressor_input_tokens = compression_results['all_compressor_input_tokens']
        all_compressor_input_kpms = compression_results['all_compressor_input_kpms']
        all_codebook_perplexities = compression_results['all_codebook_perplexities']

        avg_compression_loss, total_compression_loss_weighted, compression_loss_details = (
            self._compute_compression_loss(compression_ratios, tokens.device)
        )

        avg_top_code_lm_loss, top_code_lm_loss_details = self._compute_top_code_lm_loss(
            compression_results, kpm_for_top_codes
        )

        tf_inputs = self._prepare_teacher_forcing_inputs(compression_results)
        targets_for_expander_stack = tf_inputs['targets']
        target_kpms_for_expander_stack = tf_inputs['target_kpms']
        top_codes_for_decompress = tf_inputs['top_codes']
        tf_continuous = tf_inputs['teacher_forcing_embeddings']

        decompression_results = self.decompress(
            top_codes=top_codes_for_decompress,
            top_codes_key_padding_mask=kpm_for_top_codes,
            targets_for_teacher_forcing=targets_for_expander_stack,
            target_key_padding_masks=target_kpms_for_expander_stack,
            teacher_forcing_embeddings=tf_continuous,
            # all_seg_ids=tf_inputs['all_seg_ids'],
            tf_inputs=tf_inputs,  # for attn_mask
            compression_results=compression_results,
        )

        avg_reconstruction_loss, reconstruction_loss_details = self._compute_reconstruction_loss(
            decompression_results['all_logits'],
            targets_for_expander_stack,
            target_kpms_for_expander_stack,
        )

        avg_aux_lm_loss, aux_lm_loss_details = self._compute_auxiliary_losses(
            tokens,
            all_encoder_logits,
            all_compressor_input_tokens,
            all_compressor_input_kpms,
        )

        final_total_loss = (
            avg_reconstruction_loss
            + vq_loss
            + (self.aux_lm_loss_weight * avg_aux_lm_loss)
            + (self.top_lm_loss_weight * avg_top_code_lm_loss)
            + total_compression_loss_weighted
        )

        return {
            'total_loss': final_total_loss,
            'vq_loss': vq_loss,
            'avg_reconstruction_loss': avg_reconstruction_loss,
            'reconstruction_loss_details': reconstruction_loss_details,
            'avg_aux_lm_loss': avg_aux_lm_loss,
            'aux_lm_loss_details': aux_lm_loss_details,
            'avg_top_code_lm_loss': avg_top_code_lm_loss,
            'top_code_lm_loss_details': top_code_lm_loss_details,
            'avg_compression_loss': avg_compression_loss,
            'compression_loss_details': compression_loss_details,
            'final_reconstructed_logits': decompression_results['final_reconstructed_logits'],
            'compression_ratios': compression_ratios,
            # 'input_seq_lengths_compressors': input_seq_lengths_c,
            # 'output_seq_lengths_compressors': output_seq_lengths_c,
            'all_codebook_perplexities': all_codebook_perplexities,
            'compression_results': compression_results,
            'decompression_results': decompression_results,
        }

    @torch.no_grad()
    def generate_bytes(
            self,
            prompt_tokens: torch.Tensor,  # (1, S0)
            key_padding_mask: Optional[torch.Tensor] = None,
            *,
            max_top_codes: int = 256,
            decode_max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Recursive AR: one top symbol → one variable-length segment per level → bytes."""
        self.eval()
        assert prompt_tokens.size(0) == 1, "Gen supports B=1 for now."
        device = prompt_tokens.device
        use_cont = self.use_continuous_expander_inputs

        byte_buf = prompt_tokens.clone()  # running bytes
        byte_kpm = key_padding_mask

        def _predict_one_top_symbol(comp) -> torch.Tensor:
            if use_cont:
                top_mem = comp["all_pre_vq_embeddings"][-1]
            else:
                top_codes = comp["all_vq_indices"][-1]
                top_mem = F.embedding(top_codes, self.compressors[-1].vq.codebook)

            top_kpm = comp.get("final_key_padding_mask", None)
            lm_len = 0
            if self.top_transformer_config and self.top_transformer_config.get("lm_fixed_length"):
                lm_len = int(self.top_transformer_config["lm_fixed_length"])
            if lm_len:
                pad_val = 0.0
                top_mem, top_kpm = _fix_len(top_mem, top_kpm, lm_len, pad_val)

            out = self.code_sequence_transformer(input_embeddings=top_mem, key_padding_mask=top_kpm)
            return out["predictions_pre_vq"][:, -1, :]  # (1, D) continuous

        for _ in range(max_top_codes):
            # compress current bytes to get masks/seg_ids for top expander
            comp = self.compress(byte_buf, key_padding_mask=byte_kpm)

            # one more top symbol (continuous)
            next_top_vec = _predict_one_top_symbol(comp)  # (1, D)
            top_hi = torch.cat([comp["all_pre_vq_embeddings"][-1], next_top_vec.unsqueeze(1)], dim=1)  # (1, L+1, D)

            # generate the entire descendant bottom bytes for this top symbol
            new_bytes = self._gen_segment_at_level(
                e_idx=0,
                hi_seq=top_hi,
                comp_for_masks=comp,  # provides top KPM
                decode_max_len=decode_max_len,
            )

            if new_bytes.numel() == 0:
                break

            byte_buf = torch.cat([byte_buf, new_bytes], dim=1)
            # stop on EOS/EOP at bottom
            eos_id, eop_id = self._bottom_ids()
            last = int(byte_buf[0, -1].item())
            if last in (eos_id, eop_id):
                break

            byte_kpm = self._all_false_kpm_like(byte_buf)

        return byte_buf
    #
    # @torch.no_grad()
    # def generate_bytes(
    #     self,
    #     prompt_tokens: torch.Tensor,                       # (B,S₀)
    #     key_padding_mask: Optional[torch.Tensor] = None,
    #     *,
    #     max_top_codes: int = 256,                          # hard stop
    #     decode_max_len: Optional[int] = None,              # per-segment
    # ) -> torch.Tensor:
    #     """
    #     Autoregressive decoding loop:
    #
    #       1. Compress the prompt → top-level code sequence C.
    #       2. repeat until EOS or `max_top_codes`:
    #            a. CodeSequenceTransformer predicts **one** next top symbol cᵢ.
    #            b. Append cᵢ to C.
    #            c. Decompress *entire* C through all expanders → bytes B.
    #            d. Emit only the new byte span B[prev_len:].
    #       3. Return `prompt_tokens ⊕ generated_bytes`.
    #
    #     Notes
    #     -----
    #     • Works for both continuous and discrete expander inputs.
    #     • Assumes `self.expander_eos_id` marks the *end of sequence* at the
    #       bottom level (byte stream).
    #     """
    #     self.eval()
    #
    #     B = prompt_tokens.size(0)
    #     device = prompt_tokens.device
    #     use_cont = self.use_continuous_expander_inputs
    #
    #     # hard coded 1024 to prevent recompilations during testing
    #     _fix_len(prompt_tokens, key_padding_mask, 1024, self.compressors[0].vq.padding_token_id)
    #
    #     # ------------------------------------------------------------------
    #     # 1. Compress the prompt
    #     # ------------------------------------------------------------------
    #
    #     # Keep a running byte buffer
    #     generated_bytes = prompt_tokens                   # list for torch.cat
    #     prev_decoded_len = prompt_tokens.size(1)
    #
    #
    #     # ------------------------------------------------------------------
    #     # 2. Main loop
    #     # ------------------------------------------------------------------
    #     for _ in range(max_top_codes):
    #
    #         comp = self.compress(generated_bytes, key_padding_mask=key_padding_mask)
    #         # for idx, mask in enumerate(comp["all_compressor_level_valid_masks"]):
    #         #     print(f'mask {idx} valid: {mask.sum().item()}')
    #         if use_cont:
    #             top_mem = comp["all_pre_vq_embeddings"][-1].clone()  # (B, L_top, D)
    #             top_codes_discrete = None
    #         else:
    #             top_mem = comp["all_vq_indices"][-1].clone()  # (B, L_top)
    #             top_codes_discrete = top_mem.clone()
    #
    #         # kpm for transformer (None = no padding)
    #         top_kpm = comp["final_key_padding_mask"].clone() if comp["final_key_padding_mask"] is not None else None
    #
    #         # ---- 2a. predict ONE next top symbol -------------------------
    #         if self.code_sequence_transformer is None:
    #             raise RuntimeError("top-level transformer not initialised")
    #
    #         top_lm_len = self.top_transformer_config.get("lm_fixed_length", False)
    #         if top_lm_len:
    #             top_lm_pad_id = self.top_transformer_config.get("lm_pad_id", 258)
    #             top_mem, top_kpm = _fix_len(top_mem, top_kpm, top_lm_len, 0.0)
    #             if not use_cont:
    #                 top_codes_discrete, _ = _fix_len(top_codes_discrete, None, top_lm_len, top_lm_pad_id)
    #
    #         # Transformer input depends on `continuous` flag
    #         tr_input = (
    #             top_mem if (use_cont or top_codes_discrete is None)
    #             else F.embedding(top_codes_discrete,
    #                              self.compressors[-1].vq.codebook)
    #         )
    #
    #         tr_out = self.code_sequence_transformer(
    #             input_embeddings=tr_input,
    #             key_padding_mask=top_kpm,
    #         )
    #
    #         next_vec  = tr_out["predictions_pre_vq"][:, -1, :]          # (B,D)
    #         next_idx  = tr_out["indices"][:, -1] if tr_out["indices"] is not None else None
    #
    #         # drop the pad tail so we overwrite it, not grow past it
    #         top_mem, top_kpm = _trim_trailing_pad(top_mem, top_kpm)
    #         if not use_cont:
    #             top_codes_discrete, _ = _trim_trailing_pad(top_codes_discrete, top_kpm)
    #
    #         # ---- 2b. append to top sequence -----------------------------
    #         if use_cont:
    #             top_mem = torch.cat([top_mem, next_vec.unsqueeze(1)], dim=1)
    #         else:
    #             if next_idx is None:
    #                 raise RuntimeError("transformer did not return indices in discrete mode")
    #             top_mem = torch.cat([top_mem, next_idx.unsqueeze(1)], dim=1)  # codes_hi
    #             top_codes_discrete = top_mem
    #
    #         if top_kpm is not None:                                       # extend KPM with 0-pad
    #             pad = torch.zeros(B, 1, dtype=top_kpm.dtype, device=device)
    #             top_kpm = torch.cat([top_kpm, pad], dim=1)


            # for i in range(self.num_levels - 1):
            #     expander = self.expanders[i]
            #     generated = expander.generate(
            #         codes_hi=top_mem,
            #         codes_lo=None,  # no codes_lo in generation mode
            #         src_key_padding_mask=top_kpm,  # KPM for codes_hi
            #         max_len=decode_max_len,  # per-segment
            #         all_seg_ids=seg_ids[i],
            #     )
            #     seg


        #     # ---- 2c. full downward decode -------------------------------
        #     decomp_inp = top_mem if use_cont else top_codes_discrete
        #     decomp = self.decompress(
        #         top_codes=decomp_inp,
        #         decomp_codes_lo=generated_bytes,
        #         top_codes_key_padding_mask=top_kpm,
        #         targets_for_teacher_forcing=None,             # generation mode
        #         max_len_override=decode_max_len,
        #         teacher_forcing_embeddings=None,
        #         all_seg_ids=comp["all_seg_ids"],
        #         compression_results=comp
        #     )
        #     reconstructed = decomp["final_reconstructed_tokens"]        # (B, S_total)
        #
        #     # ---- 2d. take only the *new* bytes --------------------------
        #     new_bytes = reconstructed[:, prev_decoded_len:]              # (B, Δ)
        #
        #     # prev_decoded_len = reconstructed.size(1)
        #     if new_bytes.size(1) == 0:
        #         # No new bytes generated, stop early
        #         print("No new bytes generated, stopping early.")
        #         break
        #
        #     # now cat the new bytes to the tensor in generated bytes
        #     generated_bytes = torch.cat((generated_bytes, new_bytes), dim=1)
        #
        #     # if the last byte is EOS, we stop here
        #     if generated_bytes[:, -1] == self.expander_eos_id:
        #         print("Last byte is EOS, stopping generation.")
        #         break
        #
        #     _fix_len(generated_bytes, key_padding_mask, 1024, top_codes_discrete)
        #
        #
        #     # ---- 2e. stopping condition --------------------------------
        #     if (next_idx is not None) and (next_idx == self.expander_eos_id).all():
        #         break
        #
        #     prev_decoded_len = generated_bytes.size(1)  # update for the next iteration
        #
        #
        # # ------------------------------------------------------------------
        # # 3. return prompt + continuation
        # # ------------------------------------------------------------------
        # return generated_bytes
