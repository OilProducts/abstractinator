# Repository Guidelines

## Project Structure & Module Organization
- `components/`: Core model code and utilities
  - `abstractinator.py`: Single-level encoder/decoder module.
  - `abstractinator_pyramid.py`: Multi-level orchestration (`AbstractinatorPyramid`).
  - `segment_compressor.py`: Entropy-driven segmentation and `SegmentCompressor` (Gaussian or logit entropy, learned-query pooling, optional VQ).
  - `expander.py`: Decoder-only expanders (e.g., `DecoderOnlyExpanderRVQ`) and attention blocks.
  - `code_sequence_transformer.py`: Optional top-level `CodeSequenceTransformer` over high-level codes.
  - `vector_quantizer.py`: VQ and residual VQ implementations plus adapters.
  - Attention/utility modules: `learned_query_attention.py`, `sliding_window_attention.py`, `mla.py`, `attentions.py`, `rope.py`, `swiglu.py`.
  - Other utilities: `tokenizer.py`, `metrics.py`, `checkpoint_utils.py`, `utils.py`.
- `components/config_types.py`: Component configuration dataclasses (`AbstractinatorConfig`, `TopTransformerConfig`, `PyramidConfig`). Pure types, no runtime logic.
- `experiments/`: Experiment/runtime configuration and helpers
  - `exp_config.py`: `ExpConfig` dataclass plus device/dtype/flex-attention checks.
  - Other experiment profiles (optional) can live here (e.g., stage variants).
- `tests/`: Pytest suite for utils, vector quantizer, continuous transformers, and attention components.
- Top-level scripts: `train.py` (training), `evaluate.py` (benchmarks), `segment_dataset.py`, `expand_codebook.py`, `gaussian_train.py`.
- Artifacts (not for PRs): `checkpoints/`, `mlruns/`, `models/`.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install runtime deps: `pip install -r requirements.txt`
- Install dev tools (pytest, ruff): `pip install -r requirements-dev.txt`
- Train (default exp config): `python train.py --config experiments/exp_config.py`
- Example variants: `experiments/tiny.py`, `experiments/super_tiny.py`,
  `experiments/stage1_super_tiny.py`, `experiments/stage2_super_tiny.py`.
- Resume/load base: `python train.py --config experiments/stage2_super_tiny.py --load_base_from ./stage1_base.pt`
- Evaluate HF model: `python evaluate.py --model hf --model_args pretrained=facebook/opt-1.3b`
  
- Run tests: `pytest -q`

Makefile shortcuts:
- `make install-dev` – install runtime + dev deps.
- `make format` – apply Ruff formatter to the repo.
- `make lint` / `make lint-fix` – run Ruff lint, optionally with `--fix`.

## Coding Style & Formatting
- Python 3.12; follow PEP 8 with 4‑space indentation and type hints where practical.
- Naming: modules/functions `snake_case`; classes `CapWords`.
- Formatter and linter: Ruff is required.
  - Format: `ruff format .`
  - Lint: `ruff check .` (use `--fix` for autofixes)
  - Configuration lives in `pyproject.toml` (`target-version=py312`, `line-length=120`).
- Keep functions focused; extract shared helpers to `components/utils.py` when reused.

## Testing Guidelines
- Use `pytest`; name files `tests/test_<topic>.py`; name tests `test_<behavior>()`.
- Tests must run CPU‑only and deterministically; avoid network/downloads.
- Prefer tiny configs (copy `experiments/exp_config.py` to a minimal variant) and small tensors for unit tests.
- Include unit tests alongside new modules covering core behavior and edge cases.

## Commit & Pull Request Guidelines
- Commits: imperative, concise, scoped. Example: `components/metrics: batch MLflow logging`.
- Reference issues/PRs when relevant; separate mechanical changes from logic.
- PRs should include: summary, rationale, config changes, and any perf/behavior notes.
- Keep diffs focused; do not commit large artifacts (`checkpoints/`, `mlruns/`).

## Configuration Tips
- Prefer editing `exp_config` in `experiments/exp_config.py` (or copies under `experiments/`).
- Component config types live in `components/config_types.py`; avoid hard‑coding model hyperparameters in modules.
- `train.py` can resume from checkpoints that embed the `exp_config`.
- Two‑stage workflow: set `top_transformer_config=None` and `save_base_components_path` for Stage 1; then set `--load_base_from` and enable top LM settings for Stage 2.
- Local MLflow tracking: `mlflow ui --backend-store-uri ./mlruns/<project_name>`.
