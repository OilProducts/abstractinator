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

Run the training loop with the default experiment config:

```bash
python train.py --config experiments/exp_config.py
```

Create a different config by copying `experiments/exp_config.py` (for example to `experiments/regular_flex.py`) and pass it with `--config`:

```bash
python train.py --config experiments/regular_flex.py
```

Sample experiment files are included:
- `experiments/regular_flex.py` – regular attention on Flex backend, single level.
- `experiments/regular_sdpa.py` – regular attention on SDPA backend.
- `experiments/mla_flex.py` – MLA attention form with Flex backend.
- `experiments/mla_sdpa.py` – MLA attention form with SDPA fallback backend.
- `experiments/regular_flex_512.py` – larger 512‑dimensional variant.
- `experiments/regular_flex_entropy_frozen.py` – loads a pretrained entropy stack and freezes it.

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

Run with the above settings (e.g., `experiments/exp_config.py`) and a file at
`save_base_components_path` will contain the compressor and expander weights.

**Stage 2: Train the top LM**

```python
exp_config = {
    'top_transformer_config': {...},
    'top_lm_loss_weight': 1.0,
}
```

```bash
python train.py --config experiments/exp_config.py --load_base_from ./stage1_base.pt
```

Use separate copies of `experiments/exp_config.py` if you prefer dedicated files
for each stage. Run Stage 1 to create `stage1_base.pt`, then launch Stage 2 using
`--load_base_from` to continue training the top transformer while the lower layers
remain frozen.

Only the top transformer parameters remain trainable while reconstruction loss
continues to flow through the frozen expanders.

## Loss Components

The autoencoder optimizes a `total_loss` that combines several terms:

1. **Reconstruction loss** – Cross-entropy between each expander's output and the target sequence from the level below (or the original bytes). The result is averaged across all levels.
2. **Vector-quantization loss** – Commitment and codebook losses from each `VectorQuantizer`.
3. **Entropy‑model loss** – Optional loss from the compressor's entropy_model (categorical CE and/or Gaussian NLL as configured by `EntropyModelConfig.loss`). This is combined into the level objective with the component weight `w_entropy_ce`.
4. **Top-level LM loss** – Optional language-modeling loss on the top-level codes when a `CodeSequenceTransformer` is used, scaled by `top_lm_loss_weight`.  When the top transformer is configured with `continuous=True` it predicts the next code embedding using mean squared error.  Otherwise it outputs logits over code indices and uses cross‑entropy.

These terms are summed to form `total_loss`, which is used for the backward pass.

## Components

Key modules under `components/` include:

- **SegmentCompressor** – Encodes a token sequence with local attention,
  segments by predictive entropy, pools each segment via learned queries, and
  optionally vector-quantizes to discrete codes.
- **DecoderOnlyExpanderRVQ** – Autoregressive decoder that maps higher‑level
  codes back to lower‑level tokens (teacher‑forced or generative), producing
  stage‑wise logits when using residual VQ.
- **Abstractinator** – One level combining a SegmentCompressor and a
  DecoderOnlyExpanderRVQ.
- **AbstractinatorPyramid** – Orchestrates multiple levels for hierarchical
  compression and expansion; provides `compress_all` and `generate_bytes`.
- **CodeSequenceTransformer** – Optional causal transformer operating on top‑level
  codes/embeddings for next‑step prediction or conditioning.
- **LearnedQueryAttention** – Pools variable‑length segments into fixed‑length
  representations via a bank of learned queries.
- **VectorQuantizer** – EMA codebook with dead‑code resets; discretizes segment
  embeddings and provides the VQ loss.
- **Code spaces (D vs d_c)** – The model's hidden width is `D`. Quantization and
  factorized heads operate in a smaller code space `d_c` for efficiency. The
  compressor's VQ uses `c_vq_d_c` (defaults to 64 when unset) for its internal
  code vectors; the decoder's bottom‑level adapter uses `d_lo_d_c` for byte‑level
  embeddings and logits when no low‑side VQ is present. Upper levels inherit
  `d_c` from the child VQ via the RVQ adapter. Keeping `d_c` small reduces
  compute in nearest‑neighbor search and K‑way logits without changing external
  tensor shapes (memory remains `D`‑dim).
- **Attention** – Pluggable attention along two axes:
  - Forms: Regular vs MLA (Multi‑Head Latent Attention)
  - Backends: SDPA vs FlexAttention
  - See `components/attention/factory.py` and `components/attention/forms/*`.


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

To evaluate models trained with this repository, prefer custom scripts using
`AbstractinatorPyramid` directly (harness adapter support was removed).
