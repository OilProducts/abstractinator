from copy import deepcopy

from experiments.exp_config import ExpConfig
from experiments.exp_config import exp_config as _base_exp_config

# Use defaults from experiments.exp_config without modification
exp_config: ExpConfig = deepcopy(_base_exp_config)

# exp_config.top_transformer_config = None
exp_config.num_epochs = 2
exp_config.batch_size = 4
exp_config.gradient_accumulation_steps = 4
exp_config.sequence_length = 2048
