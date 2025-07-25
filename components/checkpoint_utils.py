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


def load_base_components(
    model: HierarchicalAutoencoder,
    path: str,
    *,
    freeze: bool = True,
    map_location: str | torch.device | None = "cpu",
) -> None:
    """Load pretrained compressors and expanders from ``path``.

    The checkpoint may be in the original ``save_base_components`` format::

        {"compressors": ..., "expanders": ...}

    or a full checkpoint containing ``"model_state"`` with a flattened
    ``state_dict``. Keys starting with ``"compressors."`` and
    ``"expanders."`` will be extracted automatically.

    Args:
        model: Model to load weights into.
        path: Checkpoint file path.
        freeze: If ``True`` the loaded modules will be frozen and set to eval
            mode.
        map_location: Device mapping for :func:`torch.load`.
    """

    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    state = ckpt.get("model_state", ckpt)

    if "compressors" in state and "expanders" in state:
        compressors_sd = state["compressors"]
        expanders_sd = state["expanders"]
    else:
        compressors_sd = {
            k[len("compressors.") :]: v
            for k, v in state.items()
            if k.startswith("compressors.")
        }
        expanders_sd = {
            k[len("expanders.") :]: v
            for k, v in state.items()
            if k.startswith("expanders.")
        }

    model.compressors.load_state_dict(compressors_sd, strict=False)
    model.expanders.load_state_dict(expanders_sd, strict=False)

    if freeze:
        model.compressors.requires_grad_(False)
        model.expanders.requires_grad_(False)
        model.compressors.eval()
        model.expanders.eval()

    logger.info("Loaded base components from %s", path)
