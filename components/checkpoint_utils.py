"""Checkpoint utility functions."""

from __future__ import annotations

import logging
import os

import torch

from components import AbstractinatorPyramid
from components.segment_compressor import SegmentCompressor

logger = logging.getLogger(__name__)


def save_base_components(model: AbstractinatorPyramid, path: str) -> None:
    """Save only the compressor and expander weights from ``model``.

    The resulting file can be used to initialize another model with pretrained
    base components.

    Args:
        model: The :class:`AbstractinatorPyramid` containing the components.
        path: Destination file path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Build flat state dicts keyed as "{level_idx}.{param_name}" so they can be
    # loaded without requiring aggregator attributes on the model.
    comp_sd: dict[str, torch.Tensor] = {}
    exp_sd: dict[str, torch.Tensor] = {}
    for i, lvl in enumerate(getattr(model, "levels", [])):
        for k, v in lvl.compressor.state_dict().items():
            comp_sd[f"{i}.{k}"] = v
        for k, v in lvl.expander.state_dict().items():
            exp_sd[f"{i}.{k}"] = v

    torch.save({"compressors": comp_sd, "expanders": exp_sd}, path)
    logger.info("Base components saved to %s", path)


def load_base_components(
    model: AbstractinatorPyramid,
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
        compressors_sd = {k[len("compressors.") :]: v for k, v in state.items() if k.startswith("compressors.")}
        expanders_sd = {k[len("expanders.") :]: v for k, v in state.items() if k.startswith("expanders.")}

    # Load compressors starting from the bottom level.  The saved checkpoint may
    # contain fewer levels than ``model`` when fine‑tuning a larger hierarchy.
    def _split_by_module(sd: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        modules: dict[int, dict[str, torch.Tensor]] = {}
        for k, v in sd.items():
            idx_str, subkey = k.split(".", 1)
            modules.setdefault(int(idx_str), {})[subkey] = v
        return [modules[i] for i in sorted(modules)]

    loaded_comp = _split_by_module(compressors_sd)
    for src, lvl in zip(loaded_comp, getattr(model, "levels", []), strict=False):
        lvl.compressor.load_state_dict(src, strict=False)
    n_loaded_comp = len(loaded_comp)

    loaded_exp = _split_by_module(expanders_sd)
    n_levels = len(getattr(model, "levels", []))
    offset = n_levels - len(loaded_exp)
    loaded_exp_indices: list[int] = []
    for i, src in enumerate(loaded_exp):
        lvl = model.levels[offset + i]
        lvl.expander.load_state_dict(src, strict=False)
        loaded_exp_indices.append(offset + i)

    if freeze:
        for i in range(n_loaded_comp):
            lvl = model.levels[i]
            lvl.compressor.requires_grad_(False)
            lvl.compressor.eval()
        for idx in loaded_exp_indices:
            lvl = model.levels[idx]
            lvl.expander.requires_grad_(False)
            lvl.expander.eval()

    logger.info("Loaded base components from %s", path)


def save_entropy_stack(compressor: SegmentCompressor, path: str) -> None:
    """
    Save the entropy stack (embedding + shared layers + entropy model) to a checkpoint.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "embedding": compressor.embedding.state_dict(),
        "shared_layers": compressor.shared_layers.state_dict(),
        "entropy_model": compressor.entropy_model.state_dict(),
    }
    # Include config when available (best-effort)
    try:
        from dataclasses import asdict

        cfg = getattr(compressor.entropy_model, "cfg", None)
        if cfg is not None:
            state["entropy_config"] = asdict(cfg)
    except Exception:
        pass
    torch.save(state, path)
    logger.info("Entropy stack saved to %s", path)


def load_entropy_stack(
    compressor: SegmentCompressor,
    path: str,
    *,
    freeze: bool = True,
    map_location: str | torch.device | None = "cpu",
) -> None:
    """
    Load embedding + shared_layers + entropy_model into the given compressor.
    If freeze=True, marks those parameters as non-trainable and sets eval().
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if "embedding" in ckpt:
        compressor.embedding.load_state_dict(ckpt["embedding"], strict=False)
    if "shared_layers" in ckpt:
        compressor.shared_layers.load_state_dict(ckpt["shared_layers"], strict=False)
    if "entropy_model" in ckpt:
        compressor.entropy_model.load_state_dict(ckpt["entropy_model"], strict=False)

    if freeze:
        for mod in (compressor.embedding, compressor.shared_layers, compressor.entropy_model):
            mod.requires_grad_(False)
            mod.eval()

    logger.info("Loaded entropy stack from %s (freeze=%s)", path, freeze)
