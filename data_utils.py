"""Utility functions for dataset tokenization and processing."""

from typing import Dict, List, Any
import torch


def tokenize_and_process_examples(
    examples: Dict[str, List[str]],
    sequence_length: int,
    tokenizer: Any,
    text_column: str = "text",
) -> Dict[str, List[torch.Tensor]]:
    """Tokenize text examples and return padded/trimmed tensors.

    Args:
        examples: Mapping with a list of strings under ``text_column``.
        sequence_length: Desired fixed sequence length.
        tokenizer: Tokenizer instance providing ``encode`` and ``pad_id`` attrs.
        text_column: Name of the text field in ``examples``.

    Returns:
        Dict with ``input_ids``, ``labels`` (identical to ``input_ids``) and
        ``key_padding_mask`` tensors.
    """
    processed_input_ids_list: List[torch.Tensor] = []
    processed_kpm_list: List[torch.Tensor] = []

    for text_content in examples[text_column]:
        if not isinstance(text_content, str):
            text_content = str(text_content) if text_content is not None else ""

        encoded_tokens = tokenizer.encode(text_content)
        current_length = len(encoded_tokens)

        if current_length == 0 and sequence_length == 0:
            final_tokens = torch.tensor([], dtype=torch.int16)
        elif current_length == 0 and sequence_length > 0:
            final_tokens = torch.full((sequence_length,), tokenizer.pad_id, dtype=torch.int16)
        elif sequence_length == 0 and current_length > 0:
            final_tokens = torch.tensor([], dtype=torch.int16)
        elif current_length > sequence_length:
            if getattr(tokenizer, "add_eos", False):
                final_tokens = encoded_tokens[: sequence_length - 1]
                final_tokens = torch.cat(
                    (final_tokens, torch.tensor([tokenizer.eos_id], dtype=torch.int16))
                )
            else:
                final_tokens = encoded_tokens[:sequence_length]
        elif current_length < sequence_length:
            padding_needed = sequence_length - current_length
            padding_tensor = torch.full((padding_needed,), tokenizer.pad_id, dtype=torch.int16)
            final_tokens = torch.cat((encoded_tokens, padding_tensor))
        else:
            final_tokens = encoded_tokens

        if len(final_tokens) != sequence_length:
            if len(final_tokens) > sequence_length:
                final_tokens = final_tokens[:sequence_length]
            else:
                padding_needed = sequence_length - len(final_tokens)
                final_tokens = torch.cat(
                    (final_tokens, torch.full((padding_needed,), tokenizer.pad_id, dtype=torch.int16))
                )

        key_padding_mask = final_tokens == tokenizer.pad_id

        processed_input_ids_list.append(final_tokens)
        processed_kpm_list.append(key_padding_mask)

    return {
        "input_ids": processed_input_ids_list,
        "labels": processed_input_ids_list,
        "key_padding_mask": processed_kpm_list,
    }

