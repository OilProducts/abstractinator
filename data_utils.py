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

        # 1. Tokenize the raw text
        tokens = tokenizer.encode(text_content).to(torch.int16)

        # 2. Truncate to ``sequence_length`` while ensuring the last token is EOS
        #    when the tokenizer adds EOS by default
        if len(tokens) > sequence_length:
            tokens = tokens[:sequence_length]
            if getattr(tokenizer, "add_eos", False) and sequence_length > 0:
                tokens[-1] = tokenizer.eos_id

        # 3. Pad sequences shorter than ``sequence_length``
        if len(tokens) < sequence_length:
            pad_len = sequence_length - len(tokens)
            pad_tensor = torch.full((pad_len,), tokenizer.pad_id, dtype=torch.int16)
            tokens = torch.cat((tokens, pad_tensor))

        key_padding_mask = tokens == tokenizer.pad_id

        processed_input_ids_list.append(tokens)
        processed_kpm_list.append(key_padding_mask)

    return {
        "input_ids": processed_input_ids_list,
        "labels": processed_input_ids_list,
        "key_padding_mask": processed_kpm_list,
    }

