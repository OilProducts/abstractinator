from copy import deepcopy
from base_config import DEVICE, N_CPU, exp_config as _base_exp_config

# Use defaults from base_config without modification
exp_config = deepcopy(_base_exp_config)
