from copy import deepcopy
from configs.base_config import DEVICE, N_CPU, exp_config as _base_exp_config, ExpConfig

# Use defaults from base_config without modification
exp_config: ExpConfig = deepcopy(_base_exp_config)

# exp_config.top_transformer_config = None
exp_config.num_epochs=2
