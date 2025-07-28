# Abstractinator

An experimental hierarchical autoencoder for compressing and reconstructing arbitrary sequences of things. 

## Overview

I came up with the idea after reading the [Byte Latent Transformer](https://arxiv.org/pdf/2412.09871) paper.  They describe a method that could be described as "learned tokenization" which works pretty well, but I thought it could be improved (like all things in deep learning) by stacking them.

Lots of problems with that idea.

Eventually I came up with this model that I call the Abstractinator.  Inspired by the BLT paper it uses a small transformer based model at each level that learns to predict the next token.  Uses the logits from that prediction to calculate entropy, then uses the entropy of the next element to define patch boundaries which are then fed through an encoder.  The design of the encoder is the key innovation of the Abstractinator that allows them to be stacked.  An Abstractinator encoder consists of a sliding window transformer with a short window, followed by a *Learned Query, Multi-Head Attention* module with a *single* query that pools the variable length segments into a fixed length representation.  This is then passed through a vector quantizer to produce a discrete code that can be used as discrete inputs for the next token prediction transformer of the next level in the hierarchy.  

The continuous output from the encoder (not discrete) is also passed through to the next level, but goes direct to the compression path, not the entropy path. (at least, this is the plan)

Why build this?  Well in my grand plan, Abstractinator based compression can be used to enable transformer based models to reason over much higher level concepts.  The hypothesis is that the first level will create tokens similar to BPE tokens, like the Byte Latent Transformer paper showed, but the second level will create tokens that are more like phrases or sentence clauses, the next could be whole ideas that would take a full sentence to express.  Clearly, if your sequence is abstracted to that level then the model can reason over much bigger things, without getting to the problematic portion of the space complexity curve of attention.

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
