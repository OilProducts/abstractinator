# abstractinator

The training script now supports periodic checkpointing. Configuration
options have been moved to `config.py`.  Edit the
`checkpoint_interval` and `checkpoint_dir` fields in `config.py`'s
`exp_config` dictionary to control how often checkpoints are saved and
where they are stored. Set `resume_from_checkpoint` to the path of a
saved checkpoint to resume training from that point.
