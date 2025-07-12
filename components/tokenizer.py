"""Tokenizers used by Abstractinator."""

from __future__ import annotations

import logging
import torch

logger = logging.getLogger(__name__)


class ByteLevelTokenizer:
    """Simple byte-level tokenizer used for the demo training script.

    The tokenizer works directly on raw UTF-8 bytes. Optionally BOS and EOS
    tokens are inserted when encoding text. It keeps byte values (0-255) as
    their own token IDs and reserves additional IDs for BOS, EOS and padding.

    Attributes:
        bos_id (int): Token ID prepended at the start of a sequence when
            ``add_bos`` is ``True``.
        eos_id (int): Token ID appended at the end of a sequence when
            ``add_eos`` is ``True``.
        pad_id (int): Token ID used for padding sequences.
        add_bos (bool): Whether ``encode`` adds ``bos_id`` by default.
        add_eos (bool): Whether ``encode`` adds ``eos_id`` by default.
        vocab_size (int): Size of the tokenizer vocabulary.
    """

    def __init__(
        self,
        bos_id: int = 256,
        eos_id: int = 257,
        pad_id: int = 258,
        add_bos: bool = True,
        add_eos: bool = True,
        expected_vocab_size: int | None = None,
    ) -> None:
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.add_bos = add_bos
        self.add_eos = add_eos
        # Ensure initial_vocab_size in configuration matches this if provided.
        # vocab_size here would be max(bos,eos,pad) + 1 = 259 for defaults.
        self.vocab_size = max(bos_id, eos_id, pad_id) + 1
        if expected_vocab_size is not None and self.vocab_size != expected_vocab_size:
            logger.warning(
                "Tokenizer vocab_size (%s) does not match expected value (%s).",
                self.vocab_size,
                expected_vocab_size,
            )

    def encode(
        self, text: str, add_bos: bool | None = None, add_eos: bool | None = None
    ) -> torch.Tensor:
        if add_bos is None:
            add_bos = self.add_bos
        if add_eos is None:
            add_eos = self.add_eos
        raw_bytes = text.encode("utf-8", errors="ignore")
        tokens: list[int] = []
        if add_bos:
            tokens.append(self.bos_id)
        tokens.extend(raw_bytes)
        if add_eos:
            tokens.append(self.eos_id)
        return torch.tensor(tokens, dtype=torch.int)

    def decode(self, tokens: list[int] | torch.Tensor, cut_at_eos: bool = False) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        if cut_at_eos and (self.eos_id in tokens):
            try:
                eos_index = tokens.index(self.eos_id)
                tokens = tokens[:eos_index]
            except ValueError:
                pass
        byte_list = [t for t in tokens if 0 <= t < 256]
        return bytes(byte_list).decode("utf-8", errors="ignore")

