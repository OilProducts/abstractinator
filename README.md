# Abstractinator

An experimental hierarchical autoencoder for compressing and reconstructing raw
text at the byte level. The project explores whether long sequences can be
represented with a compact set of discrete codes while still allowing faithful
recovery. Training and dataset parameters are configured in `config.py`.

## Overview

The autoencoder is built from several custom modules that work together to
compress sequences of UTF‑8 bytes. Each compression level reduces the length of
the sequence and discretizes the result, enabling efficient storage or further
modeling. Decompression reverses this process to reconstruct the original text.

Why build this? Hierarchical discrete representations can decouple expensive
language models from the byte-level details of text. If a compact code sequence
faithfully represents a document, the codes themselves become a smaller,
cleaner target for downstream models or storage.

## Installation

```bash
pip install -r requirements.txt
```

## Training

Run the training loop:

```bash
python train.py
```

All experiment settings live in the `exp_config` dictionary within `config.py`.  Edit values there to change model size, dataset selection and training options. Training normally runs for `exp_config['num_epochs']`, but you can specify `exp_config['max_steps']` to cap the total number of optimizer steps instead.

Checkpoints are saved to `exp_config['checkpoint_dir']` every `exp_config['checkpoint_interval']` steps.  To resume training, set `exp_config['resume_from_checkpoint']` to the path of a saved checkpoint.

Training metrics and sample outputs are logged with [MLflow](https://mlflow.org/docs/latest/python_api/mlflow.html). If `exp_config['project_name']` is set, logs are stored under `./mlruns/<project_name>`.

## Components

Key modules under `components/` include:

- **ByteSegmentCompressor** – Encodes a byte sequence with a local sliding-window
  transformer, divides it into variable‑length segments using token entropy, and
  pools each segment with learned queries before vector quantization.  The
  result is a shorter sequence of discrete codes.
- **CodeExpander** – A sequence‑to‑sequence transformer that learns to map a
  sequence of higher-level codes back into the lower-level sequence from which
  they were produced. During generation it autoregressively expands codes.
- **CodeSequenceTransformer** – Optional encoder stack for modeling the top-level
  codes themselves.  It can predict the next code and provide contextual
  embeddings for the decoder.
- **HierarchicalAutoencoder** – Orchestrates multiple compressors and expanders.
  Each level compresses further than the last, and decompression reverses the
  chain to reconstruct the original bytes.  When enabled, the
  `CodeSequenceTransformer` operates on the highest-level codes.
- **LearnedQueryAttention** – Attention module with a bank of learnable query
  vectors used by the compressors to pool variable-length segments into fixed
  representations.
- **VectorQuantizer** – Discretizes segment embeddings with an EMA-updated
  codebook and provides the VQ loss used during training.
- **SlidingWindowAttention** – Provides efficient local attention kernels for the
  token encoder used inside each compressor.
- **HierarchicalAELM** – Adapter that exposes trained autoencoder checkpoints as
  language models for evaluation with the LM harness.


See `notes.md` for additional design thoughts.

## Generation

The `generate_bytes` method performs end‑to‑end text generation. A prompt is
first compressed to top‑level codes. These codes seed the
`CodeSequenceTransformer`, which autoregressively predicts additional codes
until an EOS token is produced or a maximum length is reached. Once the full
code sequence is ready, the expander stack decodes it in a single pass back
into bytes.

## Evaluation

Run a suite of standard LLM benchmarks using the evaluation harness:

```bash
python evaluate.py --model hf --model_args pretrained=facebook/opt-1.3b use_accelerate=True
```

By default this evaluates on tasks like `lambada_openai`, `hellaswag` and
`arc_easy`.  Additional tasks or model arguments can be specified via command
line flags.

To test a model you've trained with `train.py`, point the harness at the saved
checkpoint using the `hier_ae` model type:

```bash
python evaluate.py --model hier_ae --model_args checkpoint=./checkpoints/checkpoint_step1000.pt
```
