"""Checkpoint utility functions."""

from __future__ import annotations

import logging
import os
import torch

logger = logging.getLogger(__name__)

from .hierarchical_autoencoder import HierarchicalAutoencoder


def save_base_components(model: HierarchicalAutoencoder, path: str) -> None:
    """Save only the compressor and expander weights from ``model``.

    The resulting file can be used to initialize another model with pretrained
    base components.

    Args:
        model: The :class:`HierarchicalAutoencoder` containing the components.
        path: Destination file path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "compressors": model.compressors.state_dict(),
            "expanders": model.expanders.state_dict(),
        },
        path,
    )
    logger.info("Base components saved to %s", path)
