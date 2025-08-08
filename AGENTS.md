# Repository Guidelines

## Project Structure & Module Organization
- `components/`: Core models and utilities (e.g., `hierarchical_autoencoder.py`, `byte_segment_compressor.py`, `expander.py`, `vector_quantizer.py`).
- `configs/`: Training configs (`*.py`) with `exp_config` dicts; start from `tiny_config.py`.
- `tests/`: Pytest suite (`test_*.py`) covering compression, expansion, generation, VQ, and masks.
- Top-level scripts: `train.py` (training), `evaluate.py` (benchmarks), `segment_dataset.py`, `expand_codebook.py`.
- Artifacts: `checkpoints/`, `mlruns/`, `models/`, `experiments/` (not required for PRs).

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Train (default tiny config): `python train.py --config configs/tiny_config.py`
- Resume/load base: `python train.py --config configs/stage2_super_tiny_config.py --load_base_from ./stage1_base.pt`
- Evaluate HF model: `python evaluate.py --model hf --model_args pretrained=facebook/opt-1.3b`
- Evaluate checkpoint: `python evaluate.py --model hier_ae --model_args checkpoint=./checkpoints/checkpoint_stepXXXX.pt`
- Run tests: `pytest -q` (or `python -m pytest -q`)

## Coding Style & Naming Conventions
- Python 3.12; follow PEP 8 with 4‑space indentation and type hints where practical.
- Modules and functions: `snake_case`; classes: `CapWords`.
- Keep functions focused; prefer small, composable helpers in `components/utils.py` when shared.
- No formatter is enforced; if used locally, keep output minimal and consistent.

## Testing Guidelines
- Use `pytest`; name files `tests/test_<topic>.py`; name tests `test_<behavior>()`.
- Add unit tests alongside new modules; prefer small, deterministic tests with tiny configs (`configs/super_tiny_config.py`).
- Ensure tests run CPU‑only and do not require network/downloads.

## Commit & Pull Request Guidelines
- Commits: imperative, concise, scoped. Example: `components/metrics: batch MLflow logging`.
- Reference issues/PRs when relevant; group mechanical changes separately from logic.
- PRs must include: summary, rationale, config changes, and any perf/behavior notes.
- Keep diffs focused; avoid committing large artifacts (`checkpoints/`, `mlruns/`).

## Configuration Tips
- Prefer editing `exp_config` in `configs/*.py`; avoid hard‑coding in modules.
- For local tracking, use `mlflow ui --backend-store-uri ./mlruns/<project_name>`.
