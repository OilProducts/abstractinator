import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple

from .byte_segment_compressor import ByteSegmentCompressor
from .expander import CodeExpander, DecoderOnlyExpander
from .code_sequence_transformer import CodeSequenceTransformer

logger = logging.getLogger(__name__)

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
        expander_dim_scale: Multiplier applied to compressor dims when creating
            expanders.
        expander_num_enc_layers: Number of layers in each expander encoder.
        expander_num_dec_layers: Number of layers in each expander decoder.
        expander_heads_scale: Multiplier for attention heads in expanders.
        expander_eos_id: Token used as EOS/BOS for expanders.
        expander_max_len: Maximum length an expander may generate.
        use_decoder_only_expander: If ``True`` use :class:`DecoderOnlyExpander`
            instead of the default :class:`CodeExpander`.
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
                 initial_vocab_size: int = 259,
                 expander_dim_scale: float = 1.0,
                 expander_num_enc_layers: int = 4,
                 expander_num_dec_layers: int = 4,
                 expander_heads_scale: float = 1.0,
                 expander_eos_id: int = 1,
                 expander_max_len: int = 2048,
                 use_decoder_only_expander: bool = False,
                 propagate_key_padding_mask: bool = True,
                 aux_lm_loss_weight: float = 0.1,
                 top_transformer_config: Optional[Dict[str, Any]] = None,
                 top_lm_loss_weight: float = 1.0,
                 use_continuous_expander_inputs: bool = False,
                 top_lm_mse_weight: float = 1.0,
                 top_lm_ce_weight: float = 1.0,):
        super().__init__()

        if len(compressor_level_configs) != num_levels:
            raise ValueError("Length of compressor_level_configs must match num_levels.")
        if num_levels <= 0:
            raise ValueError("num_levels must be positive.")

        self.num_levels = num_levels
        self.initial_vocab_size = initial_vocab_size
        self.expander_eos_id = expander_eos_id  # Used as BOS in CodeExpander
        self.propagate_key_padding_mask = propagate_key_padding_mask
        self.aux_lm_loss_weight = aux_lm_loss_weight  # Weight for auxiliary LM loss
        self.use_decoder_only_expander = use_decoder_only_expander

        self.top_transformer_config = top_transformer_config
        self.top_lm_loss_weight = top_lm_loss_weight
        self.top_lm_mse_weight = top_lm_mse_weight
        self.top_lm_ce_weight = top_lm_ce_weight
        self.use_continuous_expander_inputs = use_continuous_expander_inputs

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
        self.expanders = nn.ModuleList()
        for i in range(num_levels - 1, -1, -1):
            k_hi = self.actual_codebook_sizes[i]
            base_comp_config = compressor_level_configs[i]
            k_lo = self.initial_vocab_size if i == 0 else self.actual_codebook_sizes[i - 1]
            exp_dim = int(base_comp_config['dim'] * expander_dim_scale)
            exp_heads = max(1, int(base_comp_config['heads'] * expander_heads_scale))

            if self.use_decoder_only_expander:
                expander = DecoderOnlyExpander(
                    K_hi=k_hi,
                    K_lo=k_lo,
                    D=exp_dim,
                    N_dec=expander_num_dec_layers,
                    H=exp_heads,
                    cross_window=base_comp_config.get('compression_window', base_comp_config.get('window', 128)),
                    eos_id=expander_eos_id,
                    max_len=expander_max_len,
                )
            else:
                expander = CodeExpander(
                    K_hi=k_hi,
                    K_lo=k_lo,
                    D=exp_dim,
                    N_enc=expander_num_enc_layers,
                    N_dec=expander_num_dec_layers,
                    H=exp_heads,
                    eos_id=expander_eos_id,
                    max_len=expander_max_len,
                )
            self.expanders.append(expander)

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
        all_codes_list: List[torch.Tensor] = []
        all_continuous_list: List[torch.Tensor] = []
        all_pre_vq_list: List[torch.Tensor] = []
        all_input_seq_lengths: List[float] = []
        all_output_seq_lengths: List[float] = []
        # Stores valid_masks (per segment) from each compressor to reconstruct KPMs later.
        all_compressor_level_valid_masks: List[Optional[torch.Tensor]] = []
        total_vq_loss = torch.tensor(0., device=tokens.device, dtype=torch.float32)

        # For auxiliary LM loss
        all_encoder_logits_list: List[torch.Tensor] = []
        all_compressor_input_tokens_list: List[torch.Tensor] = []
        all_compressor_input_kpms_list: List[Optional[torch.Tensor]] = []

        all_perplexities_list: List[torch.Tensor] = []
        all_smoothed_perplexities_list: List[Optional[torch.Tensor]] = []

        all_patch_end_masks: List[Optional[torch.Tensor]] = []

        current_input_tokens = tokens
        current_kpm = key_padding_mask

        for i, compressor in enumerate(self.compressors):
            all_compressor_input_tokens_list.append(current_input_tokens) # Store input for aux LM loss
            all_compressor_input_kpms_list.append(current_kpm) # Store KPM for aux LM loss

            if current_kpm is not None:
                input_seq_len = (~current_kpm).sum(dim=1).float().mean().item()
            else:
                input_seq_len = float(current_input_tokens.size(1))
            all_input_seq_lengths.append(input_seq_len)

            comp_out = compressor(current_input_tokens, key_padding_mask=current_kpm)
            output_codes = comp_out['codes']
            all_patch_end_masks.append(comp_out.get('patch_end_mask'))
            encoder_logits_level_i = comp_out['encoder_logits']
            all_perplexities_list.append(comp_out['current_codebook_perplexity'])
            all_smoothed_perplexities_list.append(comp_out.get('smoothed_codebook_perplexity'))

            all_codes_list.append(output_codes)
            all_continuous_list.append(comp_out['continuous'])
            all_pre_vq_list.append(comp_out['pre_vq_embeddings'])
            all_encoder_logits_list.append(encoder_logits_level_i)
            total_vq_loss += comp_out['vq_loss']

            # valid_mask_for_output from ByteSegmentCompressor is (B, S_hat_segments)
            valid_mask_for_output_segments = comp_out.get('valid_mask')
            all_compressor_level_valid_masks.append(valid_mask_for_output_segments)

            num_q_this_level = compressor.num_queries_per_segment  # L value

            if valid_mask_for_output_segments is not None:
                # Effective output length based on valid segments * num_queries
                output_seq_len = (valid_mask_for_output_segments.sum(
                    dim=1) * num_q_this_level).float().mean().item()
            else:
                output_seq_len = float(output_codes.size(1))
            all_output_seq_lengths.append(output_seq_len)

            current_input_tokens = output_codes
            if self.propagate_key_padding_mask:
                if valid_mask_for_output_segments is not None:
                    # KPM for next level codes: True where segment was NOT valid, repeated for L queries.
                    # Shape: (B, S_hat_segments * L)
                    current_kpm = ~valid_mask_for_output_segments.repeat_interleave(
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
            'top_codes': all_codes_list[-1] if all_codes_list else torch.empty(0,
                                                                              device=tokens.device),
            'all_codes': all_codes_list,
            'all_continuous': all_continuous_list,
            'all_pre_vq': all_pre_vq_list,
            'vq_loss': total_vq_loss,
            'final_key_padding_mask': current_kpm,  # KPM for top_codes
            'compression_ratios': compression_ratios,
            'input_seq_lengths': all_input_seq_lengths,
            'output_seq_lengths': all_output_seq_lengths,
            'all_compressor_level_valid_masks': all_compressor_level_valid_masks,
            'all_encoder_logits': all_encoder_logits_list,
            'all_compressor_input_tokens': all_compressor_input_tokens_list,
            'all_compressor_input_kpms': all_compressor_input_kpms_list,
            'all_codebook_perplexities': all_perplexities_list,
            'all_smoothed_perplexities': all_smoothed_perplexities_list,
            'all_patch_end_masks': all_patch_end_masks
        }


    def decompress(
        self,
        top_codes: torch.Tensor,
        top_codes_key_padding_mask: Optional[torch.Tensor] = None,
        targets_for_teacher_forcing: Optional[List[torch.Tensor]] = None,
        target_key_padding_masks: Optional[List[Optional[torch.Tensor]]] = None,
        max_len_override: Optional[int] = None,
        teacher_forcing_embeddings: Optional[List[torch.Tensor]] = None,
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
                    tgt_key_padding_mask=tgt_kpm_for_this_level  # KPM for codes_lo (targets)
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
                generated_tokens = expander.generate(
                    codes_hi=current_input_codes_hi,
                    src_key_padding_mask=current_src_kpm,  # KPM for codes_hi
                    max_len=max_len_override
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
    def _insert_eop(seq: torch.Tensor, end_mask: torch.Tensor,
                    kpm: Optional[torch.Tensor], eop_id: int
                    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Insert ``eop_id`` after positions indicated by ``end_mask``."""

        B, S = seq.shape
        if S == 0:
            lengths = end_mask.sum(dim=1)
            max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
            out = seq.new_full((B, max_len), eop_id)
            out_kpm = None if kpm is None else torch.ones((B, max_len),
                                                          dtype=torch.bool,
                                                          device=seq.device)
            return out, out_kpm

        end_cumsum = torch.cumsum(end_mask, dim=1)
        lengths = S + end_cumsum[:, -1]
        max_len = int(lengths.max().item())

        out = seq.new_full((B, max_len), eop_id)
        out_kpm = None
        if kpm is not None:
            out_kpm = torch.ones((B, max_len), dtype=torch.bool, device=seq.device)

        shift = end_cumsum - end_mask
        token_pos = torch.arange(S, device=seq.device).unsqueeze(0) + shift
        batch_idx = torch.arange(B, device=seq.device).unsqueeze(1)

        out[batch_idx, token_pos] = seq
        if out_kpm is not None:
            out_kpm[batch_idx, token_pos] = kpm

        eop_pos = token_pos + 1
        mask = end_mask.bool()
        if out_kpm is not None:
            out_kpm[batch_idx.expand_as(end_mask)[mask], eop_pos[mask]] = False

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
        device = compression_results['all_pre_vq'][-1].device
        avg_loss = torch.tensor(0., device=device, dtype=torch.float32)
        details: Dict[str, torch.Tensor] = {}

        if (self.code_sequence_transformer is not None and
                self.top_lm_loss_weight > 0 and
                compression_results['all_pre_vq'][-1].numel() > 0):
            cont = compression_results['all_pre_vq'][-1]
            codes = compression_results['all_codes'][-1]
            transformer_input = cont if self.top_transformer_continuous else F.embedding(
                codes, self.compressors[-1].vq.codebook
            )
            cst_out = self.code_sequence_transformer(
                input_embeddings=transformer_input,
                key_padding_mask=kpm_for_top_codes,
            )
            preds_pre_vq = cst_out['predictions_pre_vq']
            vq_loss_top = cst_out['vq_loss']

            if cont.size(1) <= 1:
                mse_loss = torch.tensor(0., device=device, dtype=torch.float32)
                ce_loss = torch.tensor(0., device=device, dtype=torch.float32)
            else:
                pred = preds_pre_vq[:, :-1, :]
                mask = kpm_for_top_codes[:, 1:] if kpm_for_top_codes is not None else None

                target_cont = cont[:, 1:, :]
                mse_per_tok = F.mse_loss(pred, target_cont, reduction='none').sum(dim=-1)

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
        tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        compression_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        all_compressed_codes = compression_results['all_codes']
        all_pre_vq = compression_results['all_pre_vq']
        all_valid_masks = compression_results['all_compressor_level_valid_masks']
        all_patch_end_masks = compression_results.get('all_patch_end_masks', [])

        targets: List[torch.Tensor] = []
        kpms: List[Optional[torch.Tensor]] = []

        for j in range(self.num_levels):
            target_lvl_idx = self.num_levels - 1 - j
            target_kpm = None
            if target_lvl_idx == 0:
                target = tokens
                target_kpm = key_padding_mask
            else:
                target_code_idx = target_lvl_idx - 1
                target = all_compressed_codes[target_code_idx]
                if self.propagate_key_padding_mask:
                    valid_mask = all_valid_masks[target_code_idx]
                    if valid_mask is not None:
                        num_q = self.compressors[target_code_idx].num_queries_per_segment
                        target_kpm = ~valid_mask.repeat_interleave(num_q, dim=1)

            patch_mask = None
            if target_lvl_idx < len(all_patch_end_masks):
                patch_mask = all_patch_end_masks[target_lvl_idx]
            if patch_mask is not None and target_lvl_idx > 0:
                eop_id = self.compressors[target_lvl_idx - 1].vq.eop_token_id
                target, target_kpm = self._insert_eop(target, patch_mask, target_kpm, eop_id)

            targets.append(target)
            kpms.append(target_kpm)

        if self.use_continuous_expander_inputs:
            top_codes_for_decompress = all_pre_vq[-1]
            tf_continuous = [
                all_pre_vq[self.num_levels - 2 - i]
                for i in range(self.num_levels - 1)
            ]
        else:
            top_codes_for_decompress = compression_results['top_codes']
            tf_continuous = None

        return {
            'targets': targets,
            'target_kpms': kpms,
            'top_codes': top_codes_for_decompress,
            'teacher_forcing_embeddings': tf_continuous,
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
        input_seq_lengths_c = compression_results['input_seq_lengths']
        output_seq_lengths_c = compression_results['output_seq_lengths']
        all_encoder_logits = compression_results['all_encoder_logits']
        all_compressor_input_tokens = compression_results['all_compressor_input_tokens']
        all_compressor_input_kpms = compression_results['all_compressor_input_kpms']
        all_codebook_perplexities = compression_results['all_codebook_perplexities']
        all_smoothed_perplexities = compression_results['all_smoothed_perplexities']

        avg_compression_loss, total_compression_loss_weighted, compression_loss_details = (
            self._compute_compression_loss(compression_ratios, tokens.device)
        )

        avg_top_code_lm_loss, top_code_lm_loss_details = self._compute_top_code_lm_loss(
            compression_results, kpm_for_top_codes
        )

        tf_inputs = self._prepare_teacher_forcing_inputs(tokens, key_padding_mask, compression_results)
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
            'input_seq_lengths_compressors': input_seq_lengths_c,
            'output_seq_lengths_compressors': output_seq_lengths_c,
            'all_codebook_perplexities': all_codebook_perplexities,
            'all_smoothed_perplexities': all_smoothed_perplexities,
        }

    @torch.no_grad()
    def generate_bytes(self,
                       tokens: torch.Tensor,
                       key_padding_mask: Optional[torch.Tensor] = None,
                       max_len_override: Optional[int] = None) -> torch.Tensor:
        """Compress ``tokens`` and decode them autoregressively.

        The top-level codes are optionally extended using the
        :class:`CodeSequenceTransformer`.  When
        ``use_continuous_expander_inputs`` is ``True`` the transformer operates
        on continuous embeddings and the resulting vectors are fed directly to
        :meth:`decompress`.  Otherwise the discrete indices predicted by the
        transformer are gathered and passed to :meth:`decompress`.

        Args:
            tokens: Input byte tokens ``(B, S)``.
            key_padding_mask: Optional mask marking padded positions.
            max_len_override: Optional maximum length for generation.

        Returns:
            Tensor containing the reconstructed byte sequences from the final
            expander.
        """
        self.eval()
        compression_results = self.compress(tokens, key_padding_mask=key_padding_mask)
        top_cont = compression_results['all_pre_vq'][-1]
        top_codes = compression_results['top_codes']
        kpm_for_top_expander_input = compression_results['final_key_padding_mask']

        generated_cont = top_cont
        generated_kpm = kpm_for_top_expander_input
        generated_codes = top_codes if not self.use_continuous_expander_inputs else None

        if self.code_sequence_transformer is not None:
            # Autoregressively extend using transformer predictions
            while True:
                if max_len_override is not None and generated_cont.size(1) >= max_len_override:
                    break

                transformer_input = (
                    generated_cont
                    if self.top_transformer_continuous or generated_codes is None
                    else F.embedding(generated_codes, self.compressors[-1].vq.codebook)
                )
                out = self.code_sequence_transformer(
                    input_embeddings=transformer_input,
                    key_padding_mask=generated_kpm,
                )
                next_vec = out["predictions_pre_vq"][:, -1, :]
                next_idx = out["indices"][:, -1] if out["indices"] is not None else None
                generated_cont = torch.cat([generated_cont, next_vec.unsqueeze(1)], dim=1)
                if generated_codes is not None and next_idx is not None:
                    generated_codes = torch.cat([generated_codes, next_idx.unsqueeze(1)], dim=1)
                if generated_kpm is not None:
                    pad = torch.zeros(generated_kpm.size(0), 1, dtype=generated_kpm.dtype, device=generated_kpm.device)
                    generated_kpm = torch.cat([generated_kpm, pad], dim=1)

                if next_idx is not None and (next_idx == self.expander_eos_id).all():
                    break

        decomp_input = generated_cont if self.use_continuous_expander_inputs else generated_codes
        decomp = self.decompress(
            top_codes=decomp_input,
            top_codes_key_padding_mask=generated_kpm,
            targets_for_teacher_forcing=None,
            max_len_override=max_len_override,
            teacher_forcing_embeddings=None,
        )

        return decomp['final_reconstructed_tokens']
