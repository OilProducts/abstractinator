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

        # 1) Tokenize the text to a tensor of token IDs
        tokens = tokenizer.encode(text_content).to(torch.int16)

        if sequence_length > 0:
            # 2) Truncate to ``sequence_length`` while preserving EOS semantics
            if tokens.size(0) > sequence_length:
                if getattr(tokenizer, "add_eos", False):
                    truncated = tokens[: sequence_length - 1]
                    tokens = torch.cat(
                        (truncated, torch.tensor([tokenizer.eos_id], dtype=torch.int16))
                    )
                else:
                    tokens = tokens[:sequence_length]

            # 3) Pad to ``sequence_length`` using the tokenizer's PAD ID
            if tokens.size(0) < sequence_length:
                pad_len = sequence_length - tokens.size(0)
                padding = torch.full((pad_len,), tokenizer.pad_id, dtype=torch.int16)
                tokens = torch.cat((tokens, padding))
        else:
            # ``sequence_length`` of zero means return an empty tensor
            tokens = torch.tensor([], dtype=torch.int16)

        key_padding_mask = tokens == tokenizer.pad_id

        processed_input_ids_list.append(tokens)
        processed_kpm_list.append(key_padding_mask)

    return {
        "input_ids": processed_input_ids_list,
        "labels": processed_input_ids_list,
        "key_padding_mask": processed_kpm_list,
    }

