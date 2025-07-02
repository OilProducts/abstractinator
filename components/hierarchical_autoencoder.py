import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple

from .byte_segment_compressor import ByteSegmentCompressor
from .expander import CodeExpander
from .code_sequence_transformer import CodeSequenceTransformer

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
        propagate_key_padding_mask: Whether to propagate padding masks between
            levels.
        aux_lm_loss_weight: Weight of auxiliary language-modeling losses on
            compressor inputs.
        top_transformer_config: Optional configuration for a
            ``CodeSequenceTransformer`` over the top-level codes.
        top_lm_loss_weight: Weight for the top-level language-modeling loss.
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
                 propagate_key_padding_mask: bool = True,
                 aux_lm_loss_weight: float = 0.1,
                 top_transformer_config: Optional[Dict[str, Any]] = None,
                 top_lm_loss_weight: float=1.0,):
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

        self.top_transformer_config = top_transformer_config
        self.top_lm_loss_weight = top_lm_loss_weight

        # ---- Configure Compressor Stack ----
        self.compressors = nn.ModuleList()
        self.actual_codebook_sizes: List[int] = []
        current_input_vocab_size = initial_vocab_size

        for i in range(num_levels):
            config = compressor_level_configs[i]
            compressor = ByteSegmentCompressor(  # Assuming ByteSegmentCompressor is defined
                vocab_size=current_input_vocab_size,
                dim=config['dim'], heads=config['heads'], window=config['window'],
                num_encoder_layers=config.get('num_encoder_layers', 3),
                encoder_ffn_dim_multiplier=config.get('encoder_ffn_dim_multiplier', 4),
                num_queries=config['num_queries'],
                codebook_size=config['codebook_size'], beta=config['beta']
            )
            self.compressors.append(compressor)
            self.actual_codebook_sizes.append(config['codebook_size'])
            current_input_vocab_size = config['codebook_size']

        if self.top_transformer_config and self.num_levels > 0:
            if not self.actual_codebook_sizes:
                raise ValueError(
                    "Compressor stack must be configured before top_transformer (actual_codebook_sizes is empty).")

            top_code_vocab_size = self.actual_codebook_sizes[-1]  # Vocab of the highest level codes

            # The dimension for the CodeSequenceTransformer can be independent or related to
            # the last compressor's dimension. Let's assume it's specified in its config.
            # If top_transformer_config['dim'] is not set, it could default to last compressor's dim.
            # For simplicity, assume top_transformer_config contains all necessary args for CodeSequenceTransformer.

            transformer_dim = self.top_transformer_config.get('dim')
            if transformer_dim is None:
                # Default to the dimension of the codes it will process (output dim of last compressor)
                # This assumes the codes themselves are indices, and CST will embed them.
                # The 'dim' of the last compressor config is the dim of its *continuous vectors* before VQ.
                # For CodeSequenceTransformer, its 'dim' is its internal model dimension.
                # It's better if top_transformer_config explicitly defines its 'dim'.
                # Let's raise an error if not explicitly set, or define a clear default logic.
                # For this example, let's assume it's in the config:
                if 'dim' not in self.top_transformer_config:
                    raise ValueError("'dim' must be specified in top_transformer_config")

            self.code_sequence_transformer = CodeSequenceTransformer(
                code_vocab_size=top_code_vocab_size,
                dim=self.top_transformer_config['dim'],
                num_layers=self.top_transformer_config.get('num_layers', 4),
                num_heads=self.top_transformer_config.get('num_heads', 8),
                ffn_dim_multiplier=self.top_transformer_config.get('ffn_dim_multiplier', 4),
                output_lm_logits=self.top_transformer_config.get('output_lm_logits', True)
            )
            print(
                f"Initialized CodeSequenceTransformer for top codes (vocab: {top_code_vocab_size}, dim: {self.top_transformer_config['dim']})")

        else:
            self.code_sequence_transformer = None

        # ---- Configure Expander Stack ----
        self.expanders = nn.ModuleList()
        for i in range(num_levels - 1, -1, -1):
            k_hi = self.actual_codebook_sizes[i]
            base_comp_config = compressor_level_configs[i]
            k_lo = self.initial_vocab_size if i == 0 else self.actual_codebook_sizes[i - 1]
            exp_dim = int(base_comp_config['dim'] * expander_dim_scale)
            exp_heads = max(1, int(base_comp_config['heads'] * expander_heads_scale))

            expander = CodeExpander(  # Using updated CodeExpander
                K_hi=k_hi, K_lo=k_lo, D=exp_dim,
                N_enc=expander_num_enc_layers, N_dec=expander_num_dec_layers,
                H=exp_heads,
                eos_id=expander_eos_id, max_len=expander_max_len
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
                ``'all_continuous'`` – list of continuous embeddings before
                quantization;
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
                input_seq_len = (~current_kpm).sum(dim=1).float().mean()
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
            all_encoder_logits_list.append(encoder_logits_level_i)
            total_vq_loss += comp_out['vq_loss']

            # valid_mask_for_output from ByteSegmentCompressor is (B, S_hat_segments)
            valid_mask_for_output_segments = comp_out.get('valid_mask')
            all_compressor_level_valid_masks.append(valid_mask_for_output_segments)

            num_q_this_level = compressor.num_queries_per_segment  # L value

            if valid_mask_for_output_segments is not None:
                # Effective output length based on valid segments * num_queries
                output_seq_len = (valid_mask_for_output_segments.sum(
                    dim=1) * num_q_this_level).float().mean()
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
                        print(
                            f"Warning: 'valid_mask' not returned by compressor {i}. KPM propagation to next level disabled.")
                    current_kpm = None
            else:
                current_kpm = None

        compression_ratios: List[torch.Tensor] = []
        for in_len, out_len in zip(all_input_seq_lengths, all_output_seq_lengths):
            ratio = torch.where(in_len > 0, out_len / in_len, out_len.new_tensor(0.0))
            compression_ratios.append(ratio)
            # compression_ratios.append(float((out_len / in_len).cpu()) if in_len.item() > 0 else 0.0)
        # compression_ratios = [(out_len / in_len) if in_len > 0 else 0.0 for in_len, out_len in
        #                       zip(all_input_seq_lengths, all_output_seq_lengths)]

        return {
            'top_codes': all_codes_list[-1] if all_codes_list else torch.empty(0,
                                                                              device=tokens.device),
            'all_codes': all_codes_list,
            'all_continuous': all_continuous_list,
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


    def decompress(self,
                   top_codes: torch.Tensor,
                   top_codes_key_padding_mask: Optional[torch.Tensor] = None,
                   targets_for_teacher_forcing: Optional[List[torch.Tensor]] = None,
                   target_key_padding_masks: Optional[List[Optional[torch.Tensor]]] = None,
                   max_len_override: Optional[int] = None
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
        # 1. Compress
        compression_results = self.compress(tokens, key_padding_mask=key_padding_mask)
        all_compressed_codes = compression_results['all_codes']
        top_codes = compression_results['top_codes']
        vq_loss = compression_results['vq_loss']
        kpm_for_top_codes = compression_results['final_key_padding_mask']
        all_compressor_level_valid_masks = compression_results['all_compressor_level_valid_masks']
        all_codebook_perplexities = compression_results['all_codebook_perplexities']
        all_smoothed_perplexities = compression_results['all_smoothed_perplexities']
        all_patch_end_masks = compression_results.get('all_patch_end_masks', [])

        # For auxiliary LM loss
        all_encoder_logits = compression_results['all_encoder_logits']
        all_compressor_input_tokens = compression_results['all_compressor_input_tokens']
        all_compressor_input_kpms = compression_results['all_compressor_input_kpms']

        # For metrics
        compression_ratios = compression_results['compression_ratios']
        input_seq_lengths_c = compression_results['input_seq_lengths']
        output_seq_lengths_c = compression_results['output_seq_lengths']

        # --- 1.5 Process top_codes with CodeSequenceTransformer (if it exists) ---
        avg_top_code_lm_loss = torch.tensor(0., device=tokens.device, dtype=torch.float32)
        top_code_lm_loss_details: Dict[str, torch.Tensor] = {}  # For consistency, though only one level here

        if self.code_sequence_transformer is not None and self.top_lm_loss_weight > 0 and top_codes.numel() > 0:
            cst_output_dict = self.code_sequence_transformer(
                input_codes=top_codes,
                key_padding_mask=kpm_for_top_codes
            )
            top_code_lm_logits = cst_output_dict.get('logits')

            if top_code_lm_logits is not None:
                # Calculate LM loss: predict next top_code
                if top_codes.size(1) <= 1:  # Not enough length for next token prediction
                    current_top_lm_loss = torch.tensor(0., device=tokens.device, dtype=torch.float32)
                else:
                    logits_for_pred = top_code_lm_logits[:, :-1, :]  # (B, S_top-1, V_top)
                    targets_for_pred = top_codes[:, 1:]  # (B, S_top-1)

                    kpm_for_shifted_targets = None
                    if kpm_for_top_codes is not None:
                        kpm_for_shifted_targets = kpm_for_top_codes[:, 1:]

                    loss_per_token = F.cross_entropy(
                        logits_for_pred.reshape(-1, logits_for_pred.size(-1)),
                        targets_for_pred.reshape(-1),
                        reduction='none'
                    ).view_as(targets_for_pred)

                    if kpm_for_shifted_targets is not None:
                        loss_mask = ~kpm_for_shifted_targets
                        valid_targets_count = loss_mask.sum()
                        if valid_targets_count > 0:
                            current_top_lm_loss = (loss_per_token * loss_mask).sum() / valid_targets_count.clamp(
                                min=1e-9)
                        else:
                            current_top_lm_loss = torch.tensor(0., device=tokens.device, dtype=torch.float32)
                    else:
                        current_top_lm_loss = loss_per_token.mean()

                avg_top_code_lm_loss = current_top_lm_loss  # Since it's a single loss term, not averaged over levels
                top_code_lm_loss_details["top_code_lm_loss"] = avg_top_code_lm_loss

        # 2. Prepare targets and their KPMs for the expander stack (teacher forcing)
        targets_for_expander_stack: List[torch.Tensor] = []
        target_kpms_for_expander_stack: List[Optional[torch.Tensor]] = []

        def insert_eop(seq: torch.Tensor, end_mask: torch.Tensor,
                       kpm: Optional[torch.Tensor], eop_id: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            """Insert ``eop_id`` after positions indicated by ``end_mask``.

            This vectorized implementation avoids Python loops by computing
            insertion indices with cumulative sums of ``end_mask``.
            """
            B, S = seq.shape

            if S == 0:
                lengths = end_mask.sum(dim=1)
                max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
                out = seq.new_full((B, max_len), eop_id)
                out_kpm = None if kpm is None else torch.ones((B, max_len), dtype=torch.bool, device=seq.device)
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

        for j in range(self.num_levels):  # j is index for self.expanders list
            target_compressor_input_level_idx = self.num_levels - 1 - j
            target_kpm = None
            if target_compressor_input_level_idx == 0:
                target = tokens
                target_kpm = key_padding_mask
            else:
                target_code_idx = target_compressor_input_level_idx - 1
                target = all_compressed_codes[target_code_idx]

                if self.propagate_key_padding_mask:
                    valid_mask_for_target_segments = all_compressor_level_valid_masks[
                        target_code_idx]
                    if valid_mask_for_target_segments is not None:
                        num_q = self.compressors[target_code_idx].num_queries_per_segment
                        target_kpm = ~valid_mask_for_target_segments.repeat_interleave(num_q, dim=1)

            patch_mask = None
            if target_compressor_input_level_idx < len(all_patch_end_masks):
                patch_mask = all_patch_end_masks[target_compressor_input_level_idx]
            if patch_mask is not None and target_compressor_input_level_idx > 0:
                eop_id = self.compressors[target_compressor_input_level_idx - 1].vq.eop_token_id
                target, target_kpm = insert_eop(target, patch_mask, target_kpm, eop_id)

            targets_for_expander_stack.append(target)
            target_kpms_for_expander_stack.append(target_kpm)

        # 3. Decompress with teacher forcing
        decompression_results = self.decompress(
            top_codes=top_codes,
            top_codes_key_padding_mask=kpm_for_top_codes,
            targets_for_teacher_forcing=targets_for_expander_stack,
            target_key_padding_masks=target_kpms_for_expander_stack
        )
        all_reconstruction_logits = decompression_results['all_logits']

        # 4. Calculate hierarchical reconstruction losses
        total_reconstruction_loss = torch.tensor(0., device=tokens.device, dtype=torch.float32)
        reconstruction_loss_details: Dict[str, torch.Tensor] = {}

        for i in range(self.num_levels):
            logits_i = all_reconstruction_logits[i]
            target_i = targets_for_expander_stack[i]
            target_kpm_i = target_kpms_for_expander_stack[i]  # This is True where padded

            # Calculate loss per token, then mask and average
            loss_per_token = F.cross_entropy(
                logits_i.reshape(-1, logits_i.size(-1)),
                target_i.reshape(-1),
                reduction='none'
            )
            loss_per_token = loss_per_token.view_as(target_i)  # Reshape to (B, S_target_i)

            if target_kpm_i is not None:
                # loss_mask is True for valid (non-padded) tokens
                loss_mask = ~target_kpm_i
                current_level_loss = (loss_per_token * loss_mask).sum() / loss_mask.sum().clamp(
                    min=1e-9)
            else:
                current_level_loss = loss_per_token.mean()

            # --- Naming for the loss level ---
            expander_k_lo = self.expanders[i].K_lo
            target_level_desc = ""
            if expander_k_lo == self.initial_vocab_size:
                target_level_desc = "_bytes"
            else:
                # Determine which compressor's output this expander reconstructs (as input to next compressor)
                # expanders[i] reconstructs target_for_expander_stack[i]
                # if i=0 (TopExp), target is input to comp[N-1], which is output of comp[N-2]. So, codesL(N-2)
                # In general, expanders[i] reconstructs the input to compressors[num_levels - 1 - i]
                # This input is the output of compressors[num_levels - 2 - i]
                reconstructed_code_level_idx = self.num_levels - 2 - i
                if reconstructed_code_level_idx >= 0:
                    target_level_desc = f"_codesL{reconstructed_code_level_idx}"
                else:  # Should only happen if num_levels=1 and i=0, where K_lo is initial_vocab_size
                    target_level_desc = "_intermediate_codes"  # Fallback
            level_name = f"reconstruction_to_K{expander_k_lo}{target_level_desc}"
            # --- End Naming ---

            reconstruction_loss_details[level_name] = current_level_loss
            total_reconstruction_loss += current_level_loss

        avg_reconstruction_loss = total_reconstruction_loss / self.num_levels if self.num_levels > 0 else torch.tensor(
            0.)

        # --- 5. Calculate Auxiliary LM Loss for Compressors ---
        total_aux_lm_loss_unweighted = torch.tensor(0., device=tokens.device, dtype=torch.float32)
        aux_lm_loss_details: Dict[str, torch.Tensor] = {}

        if self.aux_lm_loss_weight > 0 and all_encoder_logits:  # Ensure weight is positive and logits list exists
            num_valid_aux_lm_levels = 0
            for i in range(self.num_levels):
                encoder_logits_i = all_encoder_logits[i]  # (B, S_in_i, V_in_i)
                input_tokens_i = all_compressor_input_tokens[i]  # (B, S_in_i)
                input_kpm_i = all_compressor_input_kpms[i]  # (B, S_in_i) or None

                # Ensure there's enough sequence length to predict a next token
                if input_tokens_i.size(1) <= 1:  # Cannot form shifted targets if seq_len is 0 or 1
                    current_aux_loss = torch.tensor(0., device=tokens.device, dtype=torch.float32)
                    # Optionally log that this level was skipped for aux LM loss
                    # print(f"Skipping aux LM loss for level {i} due to short sequence length: {input_tokens_i.size(1)}")
                else:
                    # Shift logits and targets for next token prediction
                    # Logits from position t are used to predict token t+1
                    logits_for_next_token_pred = encoder_logits_i[:, :-1, :]  # (B, S_in_i - 1, V_in_i)
                    # Target token at t+1 is the label for logits from position t
                    target_tokens_for_next_token_pred = input_tokens_i[:, 1:]  # (B, S_in_i - 1)

                    # Shift key_padding_mask for the targets
                    kpm_for_shifted_targets = None
                    if input_kpm_i is not None:
                        kpm_for_shifted_targets = input_kpm_i[:, 1:]  # (B, S_in_i - 1)

                    if logits_for_next_token_pred.size(1) == 0:  # Should be caught by outer if, but defensive
                        current_aux_loss = torch.tensor(0., device=tokens.device, dtype=torch.float32)
                    else:
                        loss_lm_i_per_token = F.cross_entropy(
                            logits_for_next_token_pred.reshape(-1, logits_for_next_token_pred.size(-1)),
                            target_tokens_for_next_token_pred.reshape(-1),
                            reduction='none'
                        ).view_as(target_tokens_for_next_token_pred)  # (B, S_in_i - 1)

                        if kpm_for_shifted_targets is not None:
                            loss_mask = ~kpm_for_shifted_targets  # True for valid (non-padded) target tokens
                            # Ensure loss_mask is not all False, which can happen if all shifted targets are padding
                            valid_targets_count = loss_mask.sum()
                            if valid_targets_count > 0:
                                current_aux_loss = (loss_lm_i_per_token * loss_mask).sum() / valid_targets_count.clamp(
                                    min=1e-9)
                            else:  # No valid targets to compute loss on (e.g., sequence was all padding after shift)
                                current_aux_loss = torch.tensor(0., device=tokens.device, dtype=torch.float32)
                        else:  # No padding mask for targets
                            current_aux_loss = loss_lm_i_per_token.mean()

                    num_valid_aux_lm_levels += 1

                aux_lm_loss_details[f"aux_lm_loss_L{i}"] = current_aux_loss
                total_aux_lm_loss_unweighted += current_aux_loss

            # Average only over levels where loss was actually computed meaningfully
            avg_aux_lm_loss = total_aux_lm_loss_unweighted / num_valid_aux_lm_levels if num_valid_aux_lm_levels > 0 else torch.tensor(
                0.)
        else:
            avg_aux_lm_loss = torch.tensor(0., device=tokens.device, dtype=torch.float32)  # Default if not active

        final_total_loss = (
                avg_reconstruction_loss +
                vq_loss +
                (self.aux_lm_loss_weight * avg_aux_lm_loss) +
                (self.top_lm_loss_weight * avg_top_code_lm_loss)  # <<< ADDED
        )
        return {
            'total_loss': final_total_loss, 'vq_loss': vq_loss,
            'avg_reconstruction_loss': avg_reconstruction_loss,
            'reconstruction_loss_details': reconstruction_loss_details,
            'avg_aux_lm_loss': avg_aux_lm_loss,  # For logging
            'aux_lm_loss_details': aux_lm_loss_details,  # For logging
            'avg_top_code_lm_loss': avg_top_code_lm_loss,  # <<< ADDED for logging
            'top_code_lm_loss_details': top_code_lm_loss_details,  # <<< ADDED for logging
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
        top_codes = compression_results['top_codes']
        kpm_for_top_expander_input = compression_results['final_key_padding_mask']

        generated_top_codes = top_codes
        generated_kpm = kpm_for_top_expander_input

        if self.code_sequence_transformer is not None:
            # Autoregressively extend the sequence of top codes
            while True:
                if max_len_override is not None and generated_top_codes.size(1) >= max_len_override:
                    break

                next_code = self.code_sequence_transformer.generate_codes(
                    prefix=generated_top_codes,
                    key_padding_mask=generated_kpm,
                )
                generated_top_codes = torch.cat([generated_top_codes, next_code], dim=1)
                if generated_kpm is not None:
                    pad = torch.zeros(generated_kpm.size(0), 1, dtype=generated_kpm.dtype, device=generated_kpm.device)
                    generated_kpm = torch.cat([generated_kpm, pad], dim=1)

                if (next_code == self.expander_eos_id).all():
                    break

        decomp = self.decompress(
            top_codes=generated_top_codes,
            top_codes_key_padding_mask=generated_kpm,
            targets_for_teacher_forcing=None,
            max_len_override=max_len_override,
        )

        return decomp['final_reconstructed_tokens']
