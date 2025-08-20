from copy import deepcopy

from configs.base_config import ExpConfig
from configs.base_config import exp_config as _base_exp_config

# Use defaults from base_config without modification
exp_config = _base_exp_config

exp_config.batch_size = 64
exp_config.gradient_accumulation_steps = 1
exp_config.sequence_length = 2048
exp_config.dataset_name = "HuggingFaceFW/fineweb-edu"
exp_config.dataset_config = "sample-10BT"  # Use a small subset for quick testing
exp_config.dataset_train_split = "train"  # Use only 10% of the training data for quick testing
exp_config.sample_prompt_for_generation = "The purpose of education is to"
