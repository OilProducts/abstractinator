# Abstractinator

An experimental hierarchical autoencoder for compressing and reconstructing raw
text at the byte level. The project explores whether long sequences can be
represented with a compact set of discrete codes while still allowing faithful
recovery. Training and dataset parameters are configured in a Python config file
(`configs/config.py` by default).

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

Run the training loop with the default config:

```bash
python train.py --config configs/config.py
```

Specify a different config file with `--config`:

```bash
python train.py --config configs/tiny_config.py
```

All experiment settings live in the `exp_config` dictionary within your chosen config file. Edit values there to change model size, dataset selection and training options. Training normally runs for `exp_config['num_epochs']`, but you can specify `exp_config['max_steps']` to cap the total number of optimizer steps instead.

Checkpoints are saved to `exp_config['checkpoint_dir']` every `exp_config['checkpoint_interval']` steps and include a copy of `exp_config`.  To resume training you can either set `exp_config['resume_from_checkpoint']` in the config file or pass `--resume_from_checkpoint <path>` on the command line.  If no `--config` is given, the configuration embedded in the checkpoint will be used.

Training metrics and sample outputs are logged with [MLflow](https://mlflow.org/docs/latest/python_api/mlflow.html). If `exp_config['project_name']` is set, logs are stored under `./mlruns/<project_name>`.
Logging to MLflow occurs in batches controlled by `exp_config['mlflow_batch_interval']`.
You can monitor a run locally by launching the MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns/<project_name>
```

Then open the shown URL in your browser to view metrics and artifacts as training progresses.

## Two-Stage Workflow

For large experiments it can be helpful to pretrain the compressor and expander
stack first and then freeze it while training a top-level language model.

**Stage 1: Pretrain compressors/expanders**

```python
exp_config = {
    # no top transformer during pretraining
    'top_transformer_config': None,
    'top_lm_loss_weight': 0.0,
    'save_base_components_path': './stage1_base.pt',
}
```

Run `python train.py --config configs/stage1_super_tiny_config.py` and a file at
`save_base_components_path` will contain the compressor and expander weights.

**Stage 2: Train the top LM**

```python
exp_config = {
    'top_transformer_config': {...},
    'top_lm_loss_weight': 1.0,
}
```

```bash
python train.py --config configs/stage2_super_tiny_config.py --load_base_from ./stage1_base.pt
```

The `configs` folder includes these stage 1 and stage 2 files so you can try the
two-step workflow with the tiny model. Run the stage 1 command first to create
`stage1_base.pt`, then launch stage 2 using `--load_base_from` to continue
training the top transformer while the lower layers remain frozen.

Only the top transformer parameters remain trainable while reconstruction loss
continues to flow through the frozen expanders.

## Loss Components

The autoencoder optimizes a `total_loss` that combines several terms:

1. **Reconstruction loss** – Cross-entropy between each expander's output and the target sequence from the level below (or the original bytes). The result is averaged across all levels.
2. **Vector-quantization loss** – Commitment and codebook losses from each `VectorQuantizer`.
3. **Auxiliary LM loss** – Optional next-token prediction loss on each compressor's input sequence, scaled by `aux_lm_loss_weight`.
4. **Top-level LM loss** – Optional language-modeling loss on the top-level codes when a `CodeSequenceTransformer` is used, scaled by `top_lm_loss_weight`.  When the top transformer is configured with `continuous=True` it predicts the next code embedding using mean squared error.  Otherwise it outputs logits over code indices and uses cross‑entropy.

These terms are summed to form `total_loss`, which is used for the backward pass.

## Components

Key modules under `components/` include:

- **ByteSegmentCompressor** – Encodes a byte sequence with a local sliding-window
  transformer, divides it into variable‑length segments using token entropy, and
  pools each segment with learned queries before vector quantization.  The
  segmentation behaviour can be tuned with ``entropy_delta`` and
  ``entropy_abs_threshold`` in ``compressor_level_configs``. The result is a
  shorter sequence of discrete codes.
- **CodeExpander** – A sequence‑to‑sequence transformer that learns to map a
  sequence of higher-level codes back into the lower-level sequence from which
  they were produced. During generation it autoregressively expands codes.
- **CodeSequenceTransformer** – Optional causal encoder stack for modeling the
  top-level codes themselves.  It predicts the next code and provides contextual
  embeddings for the decoder.
- **HierarchicalAutoencoder** – Orchestrates multiple compressors and expanders.
  Each level compresses further than the last, and decompression reverses the
  chain to reconstruct the original bytes.  When enabled, the
  `CodeSequenceTransformer` operates on the highest-level codes.
- **LearnedQueryAttention** – Attention module with a bank of learnable query
  vectors used by the compressors to pool variable-length segments into fixed
  representations.
- **VectorQuantizer** – Discretizes segment embeddings with an EMA-updated
  codebook and provides the VQ loss used during training. The
  frequency of dead-code resets is controlled by the
  `vq_reset_interval` value in each compressor's config.
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
The `--config` flag is optional when the checkpoint was created with `train.py` since the file stores the configuration used during training.
