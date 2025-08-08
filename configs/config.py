from copy import deepcopy

from configs.base_config import ExpConfig
from configs.base_config import exp_config as _base_exp_config

# Use defaults from base_config without modification
exp_config: ExpConfig = deepcopy(_base_exp_config)
