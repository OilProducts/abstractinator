import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

from .byte_segment_compressor import ByteSegmentCompressor
from .expander import CodeExpander


class HierarchicalAutoencoder(nn.Module):
    """
    Encapsulates a stack of ByteSegmentCompressors and CodeExpanders
    for hierarchical compression and decompression of byte sequences.
    (Refer to previous detailed docstring for more on attributes and general purpose)
    """

    def __init__(self,
                 num_levels: int,
                 compressor_level_configs: List[Dict[str, Any]],
                 initial_vocab_size: int = 259,
                 expander_dim_scale: float = 1.0,
                 expander_num_enc_layers: int = 4,
                 expander_num_dec_layers: int = 4,
                 expander_heads_scale: float = 1.0,
                 expander_dropout: float = 0.1,
                 expander_eos_id: int = 1,
                 expander_max_len: int = 2048,
                 propagate_key_padding_mask: bool = True):
        super().__init__()

        if len(compressor_level_configs) != num_levels:
            raise ValueError("Length of compressor_level_configs must match num_levels.")
        if num_levels <= 0:
            raise ValueError("num_levels must be positive.")

        self.num_levels = num_levels
        self.initial_vocab_size = initial_vocab_size
        self.expander_eos_id = expander_eos_id  # Used as BOS in CodeExpander
        self.propagate_key_padding_mask = propagate_key_padding_mask

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
                encoder_dropout=config.get('encoder_dropout', 0.1),
                max_seq_len_encoder=config.get('max_seq_len_encoder', 4096),
                num_queries=config['num_queries'],
                pooler_dropout=config.get('pooler_dropout', 0.1),
                codebook_size=config['codebook_size'], beta=config['beta']
            )
            self.compressors.append(compressor)
            self.actual_codebook_sizes.append(config['codebook_size'])
            current_input_vocab_size = config['codebook_size']

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
                H=exp_heads, dropout=expander_dropout,
                eos_id=expander_eos_id, max_len=expander_max_len
            )
            self.expanders.append(expander)

    def compress(self, tokens: torch.Tensor,
                 key_padding_mask: Optional[torch.Tensor] = None
                 ) -> Dict[str, Any]:
        """
        Applies the full stack of compressors.
        (Refer to previous detailed docstring for args and returns description)
        Key change: 'all_valid_masks' stores masks that can be used to reconstruct KPMs.
        """
        all_codes_list: List[torch.Tensor] = []
        all_continuous_list: List[torch.Tensor] = []
        all_input_seq_lengths: List[float] = []
        all_output_seq_lengths: List[float] = []
        # Stores valid_masks (per segment) from each compressor to reconstruct KPMs later.
        all_compressor_level_valid_masks: List[Optional[torch.Tensor]] = []
        total_vq_loss = torch.tensor(0., device=tokens.device, dtype=torch.float32)

        current_input_tokens = tokens
        current_kpm = key_padding_mask

        for i, compressor in enumerate(self.compressors):
            if current_kpm is not None:
                input_seq_len = (~current_kpm).sum(dim=1).float().mean().item()
            else:
                input_seq_len = float(current_input_tokens.size(1))
            all_input_seq_lengths.append(input_seq_len)

            comp_out = compressor(current_input_tokens, key_padding_mask=current_kpm)
            output_codes = comp_out['codes']
            all_codes_list.append(output_codes)
            all_continuous_list.append(comp_out['continuous'])
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
                        print(
                            f"Warning: 'valid_mask' not returned by compressor {i}. KPM propagation to next level disabled.")
                    current_kpm = None
            else:
                current_kpm = None

        compression_ratios = [(out_len / in_len) if in_len > 0 else 0.0 for in_len, out_len in
                              zip(all_input_seq_lengths, all_output_seq_lengths)]

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
            'all_compressor_level_valid_masks': all_compressor_level_valid_masks
            # Segment-level valid masks
        }

    def decompress(self,
                   top_codes: torch.Tensor,
                   top_codes_key_padding_mask: Optional[torch.Tensor] = None,
                   targets_for_teacher_forcing: Optional[List[torch.Tensor]] = None,
                   target_key_padding_masks: Optional[List[Optional[torch.Tensor]]] = None,
                   max_len_override: Optional[int] = None
                   ) -> Dict[str, Any]:
        """
        Applies the full stack of expanders.
        (Refer to previous detailed docstring for args and returns description)
        Key change: Passes KPMs to CodeExpander.
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
        """
        Full forward pass for training: compresses, then decompresses with teacher forcing.
        Calculates VQ loss and hierarchical reconstruction losses.
        (Refer to previous detailed docstring for args and returns description)
        Key change: Uses manual loss masking for reconstruction loss if KPMs for targets are available.
        """
        # 1. Compress
        compression_results = self.compress(tokens, key_padding_mask=key_padding_mask)
        all_compressed_codes = compression_results['all_codes']
        top_codes = compression_results['top_codes']
        vq_loss = compression_results['vq_loss']
        kpm_for_top_expander_input = compression_results['final_key_padding_mask']
        all_compressor_level_valid_masks = compression_results['all_compressor_level_valid_masks']

        compression_ratios = compression_results['compression_ratios']
        input_seq_lengths_c = compression_results['input_seq_lengths']
        output_seq_lengths_c = compression_results['output_seq_lengths']

        # 2. Prepare targets and their KPMs for the expander stack (teacher forcing)
        targets_for_expander_stack: List[torch.Tensor] = []
        target_kpms_for_expander_stack: List[Optional[torch.Tensor]] = []

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

            targets_for_expander_stack.append(target)
            target_kpms_for_expander_stack.append(target_kpm)

        # 3. Decompress with teacher forcing
        decompression_results = self.decompress(
            top_codes=top_codes,
            top_codes_key_padding_mask=kpm_for_top_expander_input,
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
        final_total_loss = avg_reconstruction_loss + vq_loss

        return {
            'total_loss': final_total_loss, 'vq_loss': vq_loss,
            'avg_reconstruction_loss': avg_reconstruction_loss,
            'reconstruction_loss_details': reconstruction_loss_details,
            'final_reconstructed_logits': decompression_results['final_reconstructed_logits'],
            'compression_ratios': compression_ratios,
            'input_seq_lengths_compressors': input_seq_lengths_c,
            'output_seq_lengths_compressors': output_seq_lengths_c,
        }

    @torch.no_grad()
    def generate_bytes(self,
                       tokens: torch.Tensor,
                       key_padding_mask: Optional[torch.Tensor] = None,
                       max_len_override: Optional[int] = None) -> torch.Tensor:
        """
        End-to-end compression and autoregressive decompression to reconstruct bytes.
        (Refer to previous detailed docstring for args and returns description)
        """
        self.eval()
        compression_results = self.compress(tokens, key_padding_mask=key_padding_mask)
        top_codes = compression_results['top_codes']
        kpm_for_top_expander_input = compression_results['final_key_padding_mask']

        decompression_results = self.decompress(
            top_codes=top_codes,
            top_codes_key_padding_mask=kpm_for_top_expander_input,
            targets_for_teacher_forcing=None,  # Inference mode
            max_len_override=max_len_override
        )
        return decompression_results['final_reconstructed_tokens']