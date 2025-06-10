# abstractinator

The training script now supports periodic checkpointing. Configure the
`checkpoint_interval` and `checkpoint_dir` fields in `train.py`'s
`exp_config` to control how often checkpoints are saved and where they are
stored. Set `resume_from_checkpoint` to the path of a saved checkpoint to
resume training from that point.
