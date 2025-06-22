# Abstractinator

A hierarchical autoencoder that compresses and reconstructs byte sequences using vector quantization.  Training and dataset parameters are configured in `config.py`.

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

## Components

Key modules under `components/` include:

- `ByteSegmentCompressor` – segments byte tokens and vector-quantizes them.
- `CodeExpander` – transformer that reconstructs lower level codes.
- `HierarchicalAutoencoder` – stacks compressors and expanders for end-to-end compression.

See `notes.md` for additional design thoughts.

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
