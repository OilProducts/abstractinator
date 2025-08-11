#!/usr/bin/env python3
"""
Bundle a Hugging Face dataset for offline use.

Creates a directory like:

  bundle/
    cache/            # portable HF datasets cache (use with HF_DATASETS_CACHE)
    exports/
      train/          # saved split -> load_from_disk(".../exports/train")
      validation/
      test/
    metadata.json     # reproducibility info
    README.txt        # quick offline instructions

Usage (examples):
  python bundle_hf_dataset.py --dataset wikitext --config wikitext-103-v1 --splits train validation --out ./wikitext_bundle
  python bundle_hf_dataset.py --dataset c4 --config en --splits train --trust-remote-code

On the offline box (option A: zero code changes):
  export HF_DATASETS_CACHE=/path/to/wikitext_bundle/cache
  export HF_DATASETS_OFFLINE=1
  # your existing `load_dataset(...)` line will work from cache.

Option B (explicit load):
  from datasets import load_from_disk
  raw_dataset = load_from_disk("/path/to/wikitext_bundle/exports/train")
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Dict

from datasets import load_dataset, Dataset, DatasetDict, DownloadConfig


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_readme(out_dir: Path, dataset: str, config: Optional[str], splits: List[str]) -> None:
    text = f"""Offline dataset bundle

Dataset: {dataset}
Config:  {config or "<none>"}
Splits:  {", ".join(splits)}

Two ways to use this bundle on an air-gapped machine:

A) No code changes (recommended)
   1) Set env vars before running your training script:
        export HF_DATASETS_CACHE="{(out_dir/'cache').resolve()}"
        export HF_DATASETS_OFFLINE=1
   2) Keep your existing `load_dataset("{dataset}", name={json.dumps(config)}, split=...)`.

B) Explicit load-from-disk (one-line change)
   from datasets import load_from_disk
   raw_dataset = load_from_disk("{(out_dir/'exports'/'<split>').resolve()}")

Notes:
- The exports/ are saved with `save_to_disk` per split.
- The cache/ contains the processed Arrow shards so `load_dataset` works offline.
"""
    (out_dir / "README.txt").write_text(text)


def main():
    pa = argparse.ArgumentParser(description="Bundle a HF dataset for offline use")
    pa.add_argument("--dataset", required=True, help="HF dataset name, e.g. 'wikitext' or 'c4'")
    pa.add_argument("--config", default=None, help="Optional dataset config/name")
    pa.add_argument("--splits", nargs="+", default=["train"], help="One or more splits, e.g. train validation test")
    pa.add_argument("--out", required=True, help="Output directory for the bundle")
    pa.add_argument("--hf-token", default=None, help="Optional HF token for gated datasets")
    pa.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to load_dataset")
    pa.add_argument("--local-files-only", action="store_true", help="Do not hit network (useful for re-bundling)")
    args = pa.parse_args()

    out_dir = _ensure_dir(Path(args.out).expanduser())
    cache_dir = _ensure_dir(out_dir / "cache")
    exports_dir = _ensure_dir(out_dir / "exports")

    # Force datasets to write the cache *inside* the bundle for portability.
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir.resolve())
    # Optional: minimize surprises if other envs are set
    os.environ.pop("HF_DATASETS_OFFLINE", None)

    # Download config (respects --local-files-only for re-runs)
    dl_cfg = DownloadConfig(
        token=args.hf_token,
        local_files_only=args.local_files_only,
        max_retries=5,
        num_proc=8,  # parallelize downloads/extractions when allowed
    )

    # Load per-split (so we materialize Arrow shards in cache) and export each to disk
    meta: Dict[str, Dict[str, object]] = {
        "dataset": args.dataset,
        "config": args.config,
        "splits": {},
        "notes": "Copy the entire directory to the offline host.",
        "versions": {},
    }

    for split in args.splits:
        print(f"[+] Loading {args.dataset} (config={args.config}) split={split} ...")
        ds = load_dataset(
            path=args.dataset,
            name=args.config,
            split=split,
            download_config=dl_cfg,
            trust_remote_code=args.trust_remote_code,
        )
        if isinstance(ds, DatasetDict):
            raise RuntimeError("Expected a single Dataset, got a DatasetDict. Specify a concrete split.")

        assert isinstance(ds, Dataset)
        print(f"    -> rows={ds.num_rows}, columns={list(ds.column_names)}")

        # Save split to disk
        split_dir = _ensure_dir(exports_dir / split)
        print(f"[+] Saving split to disk: {split_dir}")
        ds.save_to_disk(str(split_dir))

        # Record metadata
        info = ds.info or {}
        meta["splits"][split] = {
            "num_rows": ds.num_rows,
            "features": {k: str(v) for k, v in (ds.features or {}).items()},
            "fingerprint": getattr(ds, "fingerprint", None) or getattr(ds, "_fingerprint", None),
        }
        meta["versions"][split] = str(getattr(info, "version", ""))

    # Save metadata + README
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    _save_readme(out_dir, args.dataset, args.config, args.splits)

    print("\n[✓] Done.")
    print(f"    Bundle directory: {out_dir.resolve()}")
    print("    To use with *no code changes* on the offline box:")
    print(f'      export HF_DATASETS_CACHE="{(out_dir/"cache").resolve()}"')
    print( "      export HF_DATASETS_OFFLINE=1")
    print("      # your existing load_dataset(...) line will read from the cache.")
    print("\n    Or load a split explicitly:")
    print(f'      from datasets import load_from_disk; ds = load_from_disk("{(exports_dir/"train").resolve()}")')


if __name__ == "__main__":
    main()
