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
