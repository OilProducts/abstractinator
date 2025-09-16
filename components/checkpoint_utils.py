"""Checkpoint utility functions."""

from __future__ import annotations

import logging
import os

import torch

from components import AbstractinatorPyramid
from components.segment_compressor import SegmentCompressor
from components.segment_compressor import FlexibleEntropyModel
from components.config_types import (
    AttentionConfig,
    EntropyModelConfig,
    EntropyLogitHeadConfig,
    EntropyGaussianHeadConfig,
    EntropyLossConfig,
    EntropySegmentationConfig,
)

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
    """Save the entropy stack (entropy embedding + entropy model) to a checkpoint."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "embedding": compressor.embedding.state_dict(),
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
    """Load entropy embedding and entropy model; rebuild to checkpoint config/dims if needed.

    - If the checkpoint's embedding dimension differs from the current compressor's
      entropy embedding, reconstruct the entropy embedding and entropy model to match
      the checkpoint (using saved config when available), then load weights.
    - Otherwise, load weights into the existing modules with ``strict=False``.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    emb_sd = ckpt.get("embedding", None)
    ent_sd = ckpt.get("entropy_model", None)
    saved_cfg_raw = ckpt.get("entropy_config", None)

    def _to_device_dtype(mod: torch.nn.Module, ref: torch.nn.Module | None) -> torch.nn.Module:
        if ref is None:
            return mod
        return mod.to(dtype=next(ref.parameters()).dtype, device=next(ref.parameters()).device)

    # Helper: rebuild EntropyModelConfig (and nested dataclasses) from a plain dict
    def _cfg_from_dict(d: dict | None, fallback: EntropyModelConfig) -> EntropyModelConfig:
        if not isinstance(d, dict):
            return fallback
        def _mk_ac(x: dict | None, fb: AttentionConfig | None) -> AttentionConfig | None:
            if x is None:
                return fb
            try:
                return AttentionConfig(**x)
            except Exception:
                return fb
        try:
            cfg = EntropyModelConfig(
                n_layers=int(d.get("n_layers", fallback.n_layers)),
                n_heads=int(d.get("n_heads", fallback.n_heads)),
                window=int(d.get("window", fallback.window)),
                attention_config=_mk_ac(d.get("attention_config"), fallback.attention_config),
                logit=EntropyLogitHeadConfig(**d.get("logit", {})) if isinstance(d.get("logit", None), dict) else fallback.logit,
                gaussian=EntropyGaussianHeadConfig(**d.get("gaussian", {})) if isinstance(d.get("gaussian", None), dict) else fallback.gaussian,
                loss=EntropyLossConfig(**d.get("loss", {})) if isinstance(d.get("loss", None), dict) else fallback.loss,
                segmentation=EntropySegmentationConfig(**d.get("segmentation", {})) if isinstance(d.get("segmentation", None), dict) else fallback.segmentation,
            )
            return cfg
        except Exception:
            return fallback

    # Infer checkpoint embedding shape if present
    emb_dim_ckpt = None
    vocab_ckpt = None
    if isinstance(emb_sd, dict) and "weight" in emb_sd and isinstance(emb_sd["weight"], torch.Tensor):
        w = emb_sd["weight"]
        if w.ndim == 2:
            vocab_ckpt, emb_dim_ckpt = int(w.shape[0]), int(w.shape[1])

    # Current shapes
    try:
        w_cur = compressor.embedding.weight
        vocab_cur, emb_dim_cur = int(w_cur.shape[0]), int(w_cur.shape[1])
    except Exception:
        vocab_cur, emb_dim_cur = None, None

    # Decide whether to rebuild
    rebuild = emb_dim_ckpt is not None and emb_dim_cur is not None and (emb_dim_ckpt != emb_dim_cur)

    if rebuild:
        # Use saved config when available; otherwise, fall back to current config
        fallback_cfg = getattr(compressor.entropy_model, "cfg", None)
        if fallback_cfg is None:
            # Minimal fallback
            fallback_cfg = EntropyModelConfig()
        cfg = _cfg_from_dict(saved_cfg_raw, fallback_cfg)

        # Device selection
        device = map_location if isinstance(map_location, torch.device) else torch.device(str(map_location)) if map_location is not None else next(compressor.parameters()).device

        # Rebuild embedding on requested device
        vocab = int(vocab_ckpt or vocab_cur or getattr(compressor, "vocab_size", 260))
        new_embedding = torch.nn.Embedding(vocab, int(emb_dim_ckpt))
        new_embedding.to(device)

        # Tie weights if requested by config
        tie_weight = None
        if getattr(cfg.logit, "use", True) and getattr(cfg.logit, "tie_to_embedding", False):
            tie_weight = new_embedding.weight

        # Build entropy model to match checkpoint dim and config
        attn_cfg = cfg.attention_config
        trunk_heads = int(getattr(cfg, "n_heads", 8))
        trunk_layers = int(getattr(cfg, "n_layers", 4))
        trunk_window = int(getattr(cfg, "window", 128))

        new_entropy = FlexibleEntropyModel(
            dim=int(emb_dim_ckpt),
            vocab_size=vocab,
            trunk_heads=trunk_heads,
            trunk_window=trunk_window,
            trunk_layers=trunk_layers,
            attn_cfg=attn_cfg if isinstance(attn_cfg, AttentionConfig) else AttentionConfig(**attn_cfg) if isinstance(attn_cfg, dict) else AttentionConfig(),
            cfg=cfg,
            tied_embedding_weight=tie_weight,
        ).to(device)

        # Swap modules in compressor
        compressor.embedding = new_embedding
        compressor.entropy_model = new_entropy

        # Now load weights (strict=False to be robust to minor naming diffs)
        if isinstance(emb_sd, dict):
            compressor.embedding.load_state_dict(emb_sd, strict=False)
        if isinstance(ent_sd, dict):
            compressor.entropy_model.load_state_dict(ent_sd, strict=False)

        logger.info(
            "Rebuilt entropy stack to match checkpoint (dim=%s, vocab=%s); loaded from %s",
            emb_dim_ckpt,
            vocab,
            path,
        )
    else:
        # No rebuild needed; load with graceful strict=False
        if isinstance(emb_sd, dict):
            compressor.embedding.load_state_dict(emb_sd, strict=False)
        if isinstance(ent_sd, dict):
            compressor.entropy_model.load_state_dict(ent_sd, strict=False)
        logger.info("Loaded entropy stack into existing modules from %s", path)

    if freeze:
        for mod in (compressor.embedding, compressor.entropy_model):
            mod.requires_grad_(False)
            mod.eval()

    logger.info("Entropy stack ready (freeze=%s)", freeze)
