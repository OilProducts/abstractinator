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

    This model allows for multi-stage compression, where each stage further
    compresses the representation from the previous one. Decompression mirrors
    this process in reverse.

    Attributes:
        num_levels (int): The number of compression/decompression levels.
        compressors (nn.ModuleList): A list of ByteSegmentCompressor modules.
        expanders (nn.ModuleList): A list of CodeExpander modules, ordered for
                                   decompression (top-most expander first).
        initial_vocab_size (int): Vocabulary size of the initial input (e.g., bytes).
        actual_codebook_sizes (List[int]): List storing the codebook size (K_i)
                                           for the output of each compressor_i.
        propagate_key_padding_mask (bool): If True, attempts to propagate key padding
                                           masks through compression levels.
        expander_eos_id (int): EOS token ID used by CodeExpanders, also for BOS.
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
        self.expander_eos_id = expander_eos_id
        self.propagate_key_padding_mask = propagate_key_padding_mask

        # ---- Configure Compressor Stack ----
        self.compressors = nn.ModuleList()
        # Stores K_i for compressor_i's output (which is also K_hi for expander_i)
        self.actual_codebook_sizes: List[int] = []
        current_input_vocab_size = initial_vocab_size

        for i in range(num_levels):
            config = compressor_level_configs[i]
            # Ensure all required ByteSegmentCompressor args are present or defaulted
            compressor = ByteSegmentCompressor(
                vocab_size=current_input_vocab_size,
                dim=config['dim'],
                heads=config['heads'],
                window=config['window'],
                num_encoder_layers=config.get('num_encoder_layers', 3), # Default if missing
                encoder_ffn_dim_multiplier=config.get('encoder_ffn_dim_multiplier', 4),
                encoder_dropout=config.get('encoder_dropout', 0.1),
                max_seq_len_encoder=config.get('max_seq_len_encoder', 4096),
                num_queries=config['num_queries'],
                pooler_dropout=config.get('pooler_dropout', 0.1),
                codebook_size=config['codebook_size'],
                beta=config['beta']
            )
            self.compressors.append(compressor)
            self.actual_codebook_sizes.append(config['codebook_size'])
            # Output vocabulary of this compressor is the input for the next
            current_input_vocab_size = config['codebook_size']

        # ---- Configure Expander Stack ----
        # Expanders work in reverse order of compression.
        # self.expanders[0] takes top_codes (output of self.compressors[num_levels-1])
        # self.expanders[num_levels-1] reconstructs original byte tokens (input to self.compressors[0])
        self.expanders = nn.ModuleList()

        for i in range(num_levels - 1, -1, -1):
            # This expander (say, expander_j in the list) reconstructs the input to self.compressors[i].
            # Its K_hi (input codes) are the output codes of self.compressors[i].
            # Its K_lo (output codes / target) are the input vocabulary of self.compressors[i].

            k_hi_for_this_expander = self.actual_codebook_sizes[i]
            base_compressor_config_for_this_level = compressor_level_configs[i]

            if i == 0:
                # This expander reconstructs the original input (e.g., byte tokens)
                k_lo_for_this_expander = self.initial_vocab_size
            else:
                # This expander reconstructs codes from the previous compression level's input
                # which is the output of compressor[i-1]
                k_lo_for_this_expander = self.actual_codebook_sizes[i - 1]

            exp_dim = int(base_compressor_config_for_this_level['dim'] * expander_dim_scale)
            exp_heads = int(base_compressor_config_for_this_level['heads'] * expander_heads_scale)
            if exp_heads == 0: # Ensure at least one head
                exp_heads = 1

            expander = CodeExpander(
                K_hi=k_hi_for_this_expander,
                K_lo=k_lo_for_this_expander,
                D=exp_dim,
                N_enc=expander_num_enc_layers,
                N_dec=expander_num_dec_layers,
                H=exp_heads,
                dropout=expander_dropout,
                eos_id=expander_eos_id, # Shared EOS/BOS for all expanders
                max_len=expander_max_len
            )
            self.expanders.append(expander)
        # self.expanders is now [TopExpander, ..., BottomExpander (reconstructs bytes)]

    def compress(self, tokens: torch.Tensor,
                 key_padding_mask: Optional[torch.Tensor] = None
                 ) -> Dict[str, Any]:
        """
        Applies the full stack of compressors to the input tokens.

        Args:
            tokens (torch.Tensor): Input tokens (e.g., byte tokens).
                                   Shape: (batch_size, initial_sequence_length).
            key_padding_mask (Optional[torch.Tensor]): Boolean mask for `tokens`,
                                                       `True` indicates padding.
                                                       Shape: (batch_size, initial_sequence_length).
        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'top_codes': Codes from the final (highest) compression level.
                               Shape: (batch_size, top_sequence_length).
                - 'all_codes': List of code tensors from each compression level,
                               from bottom (level 0) to top.
                - 'all_continuous': List of continuous (quantized) vector outputs
                                    from each compression level.
                - 'vq_loss': Sum of VQ losses from all compressor levels.
                - 'final_key_padding_mask': Key padding mask for 'top_codes',
                                            derived if `propagate_key_padding_mask` is True.
                - 'compression_ratios': List of effective compression ratios per level.
                - 'input_seq_lengths': List of effective input sequence lengths per level.
                - 'output_seq_lengths': List of effective output sequence lengths per level.
                - 'all_valid_masks': List of valid_masks from each compressor.
                                     `valid_mask[i]` corresponds to `all_codes[i]`.
        """
        all_codes_list: List[torch.Tensor] = []
        all_continuous_list: List[torch.Tensor] = []
        all_input_seq_lengths: List[float] = [] # Store as float due to mean
        all_output_seq_lengths: List[float] = [] # Store as float due to mean
        all_valid_masks_list: List[Optional[torch.Tensor]] = [] # To store valid_masks for KPMs in decompress
        total_vq_loss = torch.tensor(0., device=tokens.device, dtype=torch.float32)

        current_input_tokens = tokens
        current_kpm = key_padding_mask

        for i, compressor in enumerate(self.compressors):
            # Calculate effective input sequence length (average non-padded length)
            if current_kpm is not None:
                input_seq_len = (~current_kpm).sum(dim=1).float().mean().item()
            else:
                input_seq_len = float(current_input_tokens.size(1))
            all_input_seq_lengths.append(input_seq_len)

            # Run the compressor for the current level
            comp_out = compressor(current_input_tokens, key_padding_mask=current_kpm)
            output_codes = comp_out['codes'] # (B, S_level_i)
            all_codes_list.append(output_codes)
            all_continuous_list.append(comp_out['continuous'])
            total_vq_loss += comp_out['vq_loss']

            # Calculate effective output sequence length
            valid_mask_for_output = comp_out.get('valid_mask') # (B, S_level_i_segments)
            all_valid_masks_list.append(valid_mask_for_output)

            if valid_mask_for_output is not None:
                # Assumes num_queries_per_segment = 1 or valid_mask is for codes directly
                # If num_queries > 1, valid_mask (B, S_segments) needs expansion for codes (B, S_segments*L)
                # For simplicity, assume valid_mask applies to the dimension of output_codes.
                # ByteSegmentCompressor's 'valid_mask' is (B, S_hat_segments).
                # Output 'codes' is (B, S_hat_segments * L_queries).
                # A more precise output_seq_len would be (valid_mask_for_output.sum(dim=1) * L_queries).mean()
                # Let's assume for now ByteSegmentCompressor returns a valid_mask compatible with codes' seq dim
                # or this calculation needs refinement based on num_queries_per_segment.
                # For now, if valid_mask is (B, Q_max), then sum over Q_max.
                if valid_mask_for_output.shape[1] == output_codes.shape[1]:
                    output_seq_len = valid_mask_for_output.sum(dim=1).float().mean().item()
                else: # valid_mask is likely per segment, codes are per query
                    num_q = compressor.num_queries_per_segment
                    output_seq_len = (valid_mask_for_output.sum(dim=1) * num_q).float().mean().item()

            else:
                output_seq_len = float(output_codes.size(1))
            all_output_seq_lengths.append(output_seq_len)

            # Prepare for the next level
            current_input_tokens = output_codes
            if self.propagate_key_padding_mask:
                if valid_mask_for_output is not None:
                    # KPM for next level: True where segment/code slot was NOT valid.
                    # If valid_mask_for_output refers to segments (B, S_hat) and codes are (B, S_hat*L),
                    # the mask needs to be expanded.
                    num_q = compressor.num_queries_per_segment
                    if valid_mask_for_output.shape[1] * num_q == output_codes.shape[1]: # Common case
                         current_kpm = ~valid_mask_for_output.repeat_interleave(num_q, dim=1)
                    elif valid_mask_for_output.shape[1] == output_codes.shape[1]: # valid_mask is already per-code
                         current_kpm = ~valid_mask_for_output
                    else: # Fallback, shape mismatch
                         print(f"Warning: Mismatch between valid_mask shape {valid_mask_for_output.shape} and output_codes shape {output_codes.shape} for KPM propagation at compressor {i}. Disabling KPM for next level.")
                         current_kpm = None

                else:
                    if i < self.num_levels - 1: # Only warn if not the last compressor
                        print(f"Warning: 'valid_mask' not returned by compressor {i}. "
                              "Key padding mask propagation to the next level is disabled.")
                    current_kpm = None # Cannot propagate KPM without valid_mask
            else:
                current_kpm = None # KPM propagation disabled by user

        compression_ratios = [
            (out_len / in_len) if in_len > 0 else 0.0
            for in_len, out_len in zip(all_input_seq_lengths, all_output_seq_lengths)
        ]

        return {
            'top_codes': all_codes_list[-1] if all_codes_list else torch.empty(0, device=tokens.device),
            'all_codes': all_codes_list,
            'all_continuous': all_continuous_list,
            'vq_loss': total_vq_loss,
            'final_key_padding_mask': current_kpm, # KPM for top_codes
            'compression_ratios': compression_ratios,
            'input_seq_lengths': all_input_seq_lengths,
            'output_seq_lengths': all_output_seq_lengths,
            'all_valid_masks': all_valid_masks_list # For reconstructing KPMs during decompress
        }

    def decompress(self,
                   top_codes: torch.Tensor,
                   top_codes_key_padding_mask: Optional[torch.Tensor] = None,
                   targets_for_teacher_forcing: Optional[List[torch.Tensor]] = None,
                   target_key_padding_masks: Optional[List[Optional[torch.Tensor]]] = None,
                   max_len_override: Optional[int] = None
                   ) -> Dict[str, Any]:
        """
        Applies the full stack of expanders to decompress codes.

        Supports teacher forcing for training or autoregressive generation for inference.

        Args:
            top_codes (torch.Tensor): Codes from the highest compression level.
            top_codes_key_padding_mask (Optional[torch.Tensor]): KPM for `top_codes`.
                                         Crucial if CodeExpander is modified to use it.
            targets_for_teacher_forcing (Optional[List[torch.Tensor]]): List of target sequences
                for each expander, from top-most to bottom-most. If provided, enables
                teacher forcing.
            target_key_padding_masks (Optional[List[Optional[torch.Tensor]]]): List of KPMs
                for each target sequence in `targets_for_teacher_forcing`.
            max_len_override (Optional[int]): Override default `expander_max_len` for generation.

        Returns:
            Dict[str, Any]:
            - If training: {'all_logits': List[logits_level_i],
                           'final_reconstructed_logits': logits_for_original_bytes}
            - If inference: {'generated_sequences': List[tokens_level_i],
                             'final_reconstructed_tokens': generated_original_bytes}
        """
        current_input_codes_hi = top_codes # Input to the first (top-most) expander
        current_hi_kpm = top_codes_key_padding_mask

        all_output_logits_list: List[torch.Tensor] = []
        all_generated_tokens_list: List[torch.Tensor] = []

        is_teacher_forcing = targets_for_teacher_forcing is not None
        if is_teacher_forcing:
            if len(targets_for_teacher_forcing) != self.num_levels:
                raise ValueError("Length of targets_for_teacher_forcing must match num_levels.")
            if target_key_padding_masks and len(target_key_padding_masks) != self.num_levels:
                raise ValueError("Length of target_key_padding_masks must match num_levels if provided.")

        for i in range(self.num_levels):
            expander = self.expanders[i] # Iterates from TopExpander to BottomExpander

            # TODO: Modify CodeExpander.forward and .generate to accept `src_key_padding_mask`
            # (for `codes_hi`) and potentially `tgt_key_padding_mask` (for `codes_lo` in forward).
            # This `current_hi_kpm` should be passed as `src_key_padding_mask`.

            if is_teacher_forcing:
                target_sequence_for_this_level = targets_for_teacher_forcing[i]
                # `tgt_kpm_for_this_level` would be used by CodeExpander.forward for its target sequence.
                # Also, it would be used to set `ignore_index` in the loss function.
                tgt_kpm_for_this_level = target_key_padding_masks[i] if target_key_padding_masks else None

                # Call CodeExpander in teacher-forcing mode
                # Assuming CodeExpander.forward takes (codes_hi, codes_lo).
                # It should ideally also take src_key_padding_mask=current_hi_kpm and
                # tgt_key_padding_mask=tgt_kpm_for_this_level.
                exp_out_dict = expander(codes_hi=current_input_codes_hi,
                                        codes_lo=target_sequence_for_this_level)
                all_output_logits_list.append(exp_out_dict['logits'])

                # For the next expander (lower level), its input `codes_hi`
                # is the target output of the current expander.
                if i < self.num_levels - 1:
                    current_input_codes_hi = target_sequence_for_this_level
                    current_hi_kpm = tgt_kpm_for_this_level # KPM of the target becomes KPM for next input
            else: # Autoregressive generation
                # Call CodeExpander in generation mode.
                # Should ideally take src_key_padding_mask=current_hi_kpm.
                generated_tokens = expander.generate(
                    codes_hi=current_input_codes_hi,
                    max_len=max_len_override
                    # src_key_padding_mask=current_hi_kpm # If CodeExpander.generate supports it
                )
                all_generated_tokens_list.append(generated_tokens)
                # Output of this expander is input to the next (lower level) expander.
                current_input_codes_hi = generated_tokens
                # KPM for generated tokens is typically not used as input KPM for next stage,
                # unless specific EOS padding is being masked.
                current_hi_kpm = None

        if is_teacher_forcing:
            return {
                'all_logits': all_output_logits_list,
                'final_reconstructed_logits': all_output_logits_list[-1] # Logits for original bytes
            }
        else:
            return {
                'generated_sequences': all_generated_tokens_list,
                'final_reconstructed_tokens': all_generated_tokens_list[-1] # Reconstructed bytes
            }

    def forward(self, tokens: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                # Define pad_token_id if targets can be padded and need ignore_index in loss
                pad_token_id: Optional[int] = None
                ) -> Dict[str, Any]:
        """
        Full forward pass for training: compresses, then decompresses with teacher forcing.
        Calculates VQ loss and hierarchical reconstruction losses.

        Args:
            tokens (torch.Tensor): Input byte tokens. Shape: (batch_size, initial_sequence_length).
            key_padding_mask (Optional[torch.Tensor]): KPM for input tokens.
            pad_token_id (Optional[int]): The ID used for padding in target sequences.
                                          If provided, used as `ignore_index` in loss.
        Returns:
            Dict[str, Any]: Dictionary with losses, final logits, and compression metrics.
        """
        # 1. Compress the input tokens through all levels
        compression_results = self.compress(tokens, key_padding_mask=key_padding_mask)
        all_compressed_codes = compression_results['all_codes'] # List: [codes_L0, ..., codes_L(N-1)]
        top_codes = compression_results['top_codes'] # Codes from the last compressor: codes_L(N-1)
        vq_loss = compression_results['vq_loss']
        # KPM for top_codes, to be used as input KPM for the top-most expander
        kpm_for_top_expander_input = compression_results['final_key_padding_mask']
        all_compressor_valid_masks = compression_results['all_valid_masks'] # List of (B, S_hat_i) or None

        # --- Metrics for monitoring ---
        compression_ratios = compression_results['compression_ratios']
        input_seq_lengths_c = compression_results['input_seq_lengths']
        output_seq_lengths_c = compression_results['output_seq_lengths']
        # --- End Metrics ---

        # 2. Prepare targets and their KPMs for the expander stack (teacher forcing)
        # self.expanders order: [TopExpander, Exp_level_N-2, ..., BottomExpander]
        # Target for self.expanders[j] is the input to self.compressors[num_levels - 1 - j].
        # Input to self.compressors[k] is all_compressed_codes[k-1] for k > 0,
        # and `tokens` for k = 0.
        targets_for_expander_stack: List[torch.Tensor] = []
        target_kpms_for_expander_stack: List[Optional[torch.Tensor]] = []

        for j in range(self.num_levels): # j is index for expanders list
            # Expander self.expanders[j] reconstructs the input to self.compressors[num_levels - 1 - j]
            target_compressor_input_level_idx = self.num_levels - 1 - j
            if target_compressor_input_level_idx == 0:
                # This expander (bottom-most, self.expanders[num_levels-1]) reconstructs original tokens
                target = tokens
                # KPM for original tokens is the input key_padding_mask
                target_kpm = key_padding_mask
            else:
                # This expander reconstructs codes_L(target_compressor_input_level_idx - 1)
                # which is all_compressed_codes[target_compressor_input_level_idx - 1]
                target_code_idx = target_compressor_input_level_idx - 1
                target = all_compressed_codes[target_code_idx]
                # KPM for this target is derived from the valid_mask of the compressor that produced it
                # valid_mask for all_compressed_codes[k] is all_compressor_valid_masks[k]
                valid_mask_for_target = all_compressor_valid_masks[target_code_idx]
                if self.propagate_key_padding_mask and valid_mask_for_target is not None:
                    num_q = self.compressors[target_code_idx].num_queries_per_segment
                    if valid_mask_for_target.shape[1] * num_q == target.shape[1]:
                        target_kpm = ~valid_mask_for_target.repeat_interleave(num_q, dim=1)
                    elif valid_mask_for_target.shape[1] == target.shape[1]:
                        target_kpm = ~valid_mask_for_target
                    else: # Fallback
                        target_kpm = None
                else:
                    target_kpm = None

            targets_for_expander_stack.append(target)
            target_kpms_for_expander_stack.append(target_kpm)

        # 3. Decompress with teacher forcing, using prepared targets and their KPMs
        decompression_results = self.decompress(
            top_codes=top_codes,
            top_codes_key_padding_mask=kpm_for_top_expander_input,
            targets_for_teacher_forcing=targets_for_expander_stack,
            target_key_padding_masks=target_kpms_for_expander_stack
        )
        all_reconstruction_logits = decompression_results['all_logits'] # List of logits from each expander

        # 4. Calculate hierarchical reconstruction losses
        total_reconstruction_loss = torch.tensor(0., device=tokens.device, dtype=torch.float32)
        reconstruction_loss_details: Dict[str, torch.Tensor] = {}

        for i in range(self.num_levels): # i is index for expanders list and logits list
            logits_i = all_reconstruction_logits[i]  # (B, S_target_i, K_target_vocab_i)
            target_i = targets_for_expander_stack[i] # (B, S_target_i)
            target_kpm_i = target_kpms_for_expander_stack[i] # (B, S_target_i) or None

            # Reshape for CrossEntropyLoss:
            # Logits: (N, C, ...) -> (BatchSize * SeqLen, VocabSize)
            # Target: (N, ...)    -> (BatchSize * SeqLen)
            current_loss = F.cross_entropy(
                logits_i.reshape(-1, logits_i.size(-1)),
                target_i.reshape(-1),
                ignore_index=pad_token_id if pad_token_id is not None and target_kpm_i is not None else -100
                # If target_kpm_i is True for padding, those target tokens should be pad_token_id.
                # CrossEntropyLoss with ignore_index=-100 (default) handles this if target labels are -100.
                # A more robust way is to ensure padded positions in target_i have value `pad_token_id`
                # and then pass `ignore_index=pad_token_id`.
            )
            # TODO: Refine ignore_index based on how padded targets are represented (e.g. specific pad_token_id).
            # If target_kpm_i exists and pad_token_id is defined, ensure target_i uses pad_token_id for padded positions.

            expander_k_lo = self.expanders[i].K_lo
            # Naming for the level this expander is reconstructing
            # self.expanders[i] corresponds to target `targets_for_expander_stack[i]`
            # If i = 0 (TopExpander), target is codes_L(N-2) (input to compressor N-1)
            # If i = N-1 (BottomExpander), target is original tokens.
            target_level_desc = ""
            if expander_k_lo == self.initial_vocab_size:
                target_level_desc = "_bytes"
            else: # It's reconstructing codes from a previous compression level
                # Expander i reconstructs the input to compressor (num_levels - 1 - i).
                # That input was the output of compressor (num_levels - 2 - i).
                if self.num_levels - 2 - i >= 0:
                    target_level_desc = f"_codesL{self.num_levels - 2 - i}"
                else: # Should not happen if logic is correct, means K_lo is initial_vocab_size
                    target_level_desc = "_intermediate_codes"


            level_name = f"reconstruction_to_K{expander_k_lo}{target_level_desc}"
            reconstruction_loss_details[level_name] = current_loss
            total_reconstruction_loss += current_loss

        avg_reconstruction_loss = total_reconstruction_loss / self.num_levels if self.num_levels > 0 else torch.tensor(0.)
        final_total_loss = avg_reconstruction_loss + vq_loss

        return {
            'total_loss': final_total_loss,
            'vq_loss': vq_loss,
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
        Performs end-to-end compression and then autoregressive decompression
        to reconstruct byte tokens. Sets the model to evaluation mode.

        Args:
            tokens (torch.Tensor): Input byte tokens.
                                   Shape: (batch_size, initial_sequence_length).
            key_padding_mask (Optional[torch.Tensor]): KPM for input tokens.
            max_len_override (Optional[int]): Max generation length for expanders.

        Returns:
            torch.Tensor: Reconstructed byte tokens.
                          Shape: (batch_size, generated_sequence_length).
        """
        self.eval() # Set model to evaluation mode (disables dropout, etc.)
        compression_results = self.compress(tokens, key_padding_mask=key_padding_mask)
        top_codes = compression_results['top_codes']
        # KPM for the input to the top-most expander
        kpm_for_top_expander_input = compression_results['final_key_padding_mask']

        decompression_results = self.decompress(
            top_codes=top_codes,
            top_codes_key_padding_mask=kpm_for_top_expander_input,
            targets_for_teacher_forcing=None,  # Ensure inference mode
            max_len_override=max_len_override
        )
        return decompression_results['final_reconstructed_tokens']