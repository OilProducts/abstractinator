import os
import math

import torch
from datasets import load_dataset, Dataset  # Import Dataset for the dummy data
from torch import optim
from torch.utils.data import DataLoader
from typing import List, Dict, Any

import aim  # Assuming 'aim' is installed and accessible
from tqdm import tqdm

# Assuming HierarchicalAutoencoder is in abstractinator.py and has KPM updates
from components import HierarchicalAutoencoder

torch.set_float32_matmul_precision("high")
torch.set_default_dtype(torch.bfloat16)

def short_num(n):
    n = float(n)
    millnames = ['', 'k', 'm', 'b', 't', 'q']

    if n == 0:
        return '0'
    millidx = max(0, min(len(millnames) - 1,
                         int(math.floor(math.log10(abs(n)) / 3))))
    scaled = n / 10 ** (3 * millidx)
    if scaled < 10:
        formatted = f"{scaled:.2f}"
    elif scaled < 100:
        formatted = f"{scaled:.1f}"
    else:
        formatted = f"{scaled:.0f}"
    return f"{formatted}{millnames[millidx]}"


N_CPU = int(os.cpu_count() / 2) if os.cpu_count() else 1  # Ensure at least 1 worker

# --- Device Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available() and DEVICE == "cpu":  # Prefer MPS if available and CUDA not
    DEVICE = torch.device("mps")
print(f"Using device: {DEVICE}")

# --- Experiment Configuration ---
exp_config = {
    "run_name": "HierarchicalAE_KPM_Run_v2",  # Updated run name
    "project_name": "TemporalAutoencodedLanguageModelling",
    "num_levels": 2,
    "initial_vocab_size": 259,  # Matches ByteLevelTokenizer default + 3 special tokens
    "compressor_level_configs": [
        {"dim": 512, "heads": 8, "window": 128,
         "num_encoder_layers": 2,
         'encoder_ffn_dim_multiplier': 4,
         'encoder_dropout': 0.1,
         'max_seq_len_encoder': 4096,
         "num_queries": 1,
         "pooler_dropout": 0.1,  # Added from previous HierarchicalAutoencoder init
         "codebook_size": 512,
         "beta": 0.05},
        {"dim": 1024, "heads": 16, "window": 64,
         "num_encoder_layers": 2,
         'encoder_ffn_dim_multiplier': 4,
         'encoder_dropout': 0.1,
         'max_seq_len_encoder': 4096,
         "num_queries": 1,
         "pooler_dropout": 0.1,  # Added from previous HierarchicalAutoencoder init
         "codebook_size": 1024,  # Was CODEBOOK_L0 * MULTIPLIER, direct value now
         "beta": 0.05}
    ],
    "expander_dim_scale": 1.0,
    "expander_num_enc_layers": 2,
    "expander_num_dec_layers": 2,
    "expander_heads_scale": 1.0,
    "expander_dropout": 0.1,
    "expander_eos_id": 1,  # As used in CodeExpander
    "expander_max_len": 2048,  # Default max generation length for CodeExpander
    "propagate_key_padding_mask": True,  # Crucial for KPM pipeline
    "aux_lm_loss_weight": 1.0,
    "learning_rate": 1e-4,
    "batch_size": 32,  # Centralized BATCH_SIZE
    "sequence_length": 1024,  # Centralized SEQUENCE_LENGTH
    "num_epochs": 10,
    "log_interval": 1,  # Log metrics to AIM every N steps (increased for less frequent logging)
    "gradient_clip_norm": 1.0,  # Optional gradient clipping
    # Dataset configurations
    "dataset_name": "HuggingFaceFW/fineweb-edu",
    "dataset_config": "sample-10BT",
    "dataset_train_split": "train[:200000]",  # Using a larger subset for demo
    "text_column_name": "text",

    # --- Generation during training settings ---
    "generation_interval": 50,  # Generate every 50 global steps
    "sample_prompt_for_generation": "This is a test sentence for the hierarchical autoencoder to compress and then reconstruct. Let's see how well it performs on this particular piece of text, including some special characters like é, ë, and numbers 12345.",
    "generation_max_len_override": 512  # Max length for the generated sample
}

# --- AIM Setup ---
print(f"Initializing AIM run: {exp_config.get('run_name', 'DefaultRun')}")
aim_run = aim.Run(experiment=exp_config.get("run_name", "DefaultRun"))
if exp_config.get("project_name"):
    # Ensure the AIM repo path is correctly specified if not using default location
    aim_run.repo = f"./aim_repo/{exp_config['project_name']}"
    print(f"AIM repository set to: {aim_run.repo}")

print("Tracking hyperparameters with AIM...")
# Track all hyperparameters (excluding potentially sensitive or very large ones if any)
# The loop below is an alternative way, track_hparams is generally preferred.
for k, v in exp_config.items():
    aim_run[k] = v
print("Hyperparameters tracked.")

# --- Model, Optimizer ---
print("Initializing HierarchicalAutoencoder model...")
model = HierarchicalAutoencoder(
    num_levels=exp_config["num_levels"],
    compressor_level_configs=exp_config["compressor_level_configs"],
    initial_vocab_size=exp_config["initial_vocab_size"],
    expander_dim_scale=exp_config["expander_dim_scale"],
    expander_num_enc_layers=exp_config["expander_num_enc_layers"],
    expander_num_dec_layers=exp_config["expander_num_dec_layers"],
    expander_heads_scale=exp_config["expander_heads_scale"],
    expander_dropout=exp_config["expander_dropout"],
    expander_eos_id=exp_config["expander_eos_id"],
    expander_max_len=exp_config["expander_max_len"],  # Pass expander_max_len
    propagate_key_padding_mask=exp_config["propagate_key_padding_mask"],
    aux_lm_loss_weight=exp_config["aux_lm_loss_weight"],
).to(DEVICE)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model initialized on {DEVICE} with {short_num(num_params)} trainable parameters.")

optimizer = optim.AdamW(model.parameters(), lr=exp_config["learning_rate"])
print(f"Optimizer AdamW initialized with learning rate: {exp_config['learning_rate']:.0e}")


# --- Tokenizer ---
class ByteLevelTokenizer:
    """Minimal Byte-level tokenizer (details in previous review)."""

    def __init__(self, bos_id: int = 256, eos_id: int = 257, pad_id: int = 258,
                 add_bos: bool = True, add_eos: bool = True):
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.add_bos = add_bos
        self.add_eos = add_eos
        # Ensure initial_vocab_size in exp_config matches this.
        # vocab_size here would be max(bos,eos,pad) + 1 = 259 for defaults.
        self.vocab_size = max(bos_id, eos_id, pad_id) + 1
        if self.vocab_size != exp_config["initial_vocab_size"]:
            print(f"Warning: Tokenizer vocab_size ({self.vocab_size}) does not match "
                  f"exp_config['initial_vocab_size'] ({exp_config['initial_vocab_size']}).")

    def encode(self, text: str, add_bos: bool | None = None,
               add_eos: bool | None = None) -> torch.Tensor:
        if add_bos is None: add_bos = self.add_bos
        if add_eos is None: add_eos = self.add_eos
        raw_bytes = text.encode("utf-8", errors="ignore")
        tokens = []
        if add_bos: tokens.append(self.bos_id)
        tokens.extend(raw_bytes)
        if add_eos: tokens.append(self.eos_id)
        return torch.tensor(tokens, dtype=torch.int)  # Using int16 for tokens

    def decode(self, tokens: list[int] | torch.Tensor, cut_at_eos: bool = False) -> str:
        if isinstance(tokens, torch.Tensor): tokens = tokens.tolist()
        if cut_at_eos and (self.eos_id in tokens):
            try:
                eos_index = tokens.index(self.eos_id)
                tokens = tokens[:eos_index]
            except ValueError:
                pass
        byte_list = [t for t in tokens if 0 <= t < 256]
        return bytes(byte_list).decode("utf-8", errors="ignore")


tokenizer = ByteLevelTokenizer(add_bos=True,
                               add_eos=True)  # Using defaults that match initial_vocab_size=259
print(f"Tokenizer initialized. BOS ID: {tokenizer.bos_id}, EOS ID: {tokenizer.eos_id}, "
      f"PAD ID: {tokenizer.pad_id}, Effective Vocab Size: {tokenizer.vocab_size}")


# --- Data Processing Function ---
def tokenize_and_process_examples(
        examples: Dict[str, List[str]],
        tokenizer_instance: ByteLevelTokenizer,  # Renamed for clarity
        sequence_length: int,  # From exp_config
        text_column="text"
) -> Dict[str, List[torch.Tensor]]:
    processed_input_ids_list = []
    # Labels are same as input_ids for autoencoding tasks
    # processed_labels_list = [] # Not explicitly needed if labels are input_ids
    processed_kpm_list = []

    for text_content in examples[text_column]:
        if not isinstance(text_content, str):
            text_content = str(text_content) if text_content is not None else ""

        # Encode tokens (BOS/EOS added by tokenizer based on its settings)
        encoded_tokens = tokenizer_instance.encode(text_content)
        current_length = len(encoded_tokens)
        final_tokens: torch.Tensor

        if current_length == 0 and sequence_length == 0:
            final_tokens = torch.tensor([], dtype=torch.int16)
        elif current_length == 0 and sequence_length > 0:
            final_tokens = torch.full((sequence_length,), tokenizer_instance.pad_id,
                                      dtype=torch.int16)
        elif sequence_length == 0 and current_length > 0:  # Truncate to empty
            final_tokens = torch.tensor([], dtype=torch.int16)
        elif current_length > sequence_length:
            # Truncate and ensure EOS if tokenizer adds it and there's space
            if tokenizer_instance.add_eos:
                # Truncate to sequence_length - 1 to make space for EOS
                final_tokens = encoded_tokens[:sequence_length - 1]
                final_tokens = torch.cat(
                    (final_tokens, torch.tensor([tokenizer_instance.eos_id], dtype=torch.int16))
                )
            else:
                final_tokens = encoded_tokens[:sequence_length]
        elif current_length < sequence_length:
            # Pad
            padding_needed = sequence_length - current_length
            padding_tensor = torch.full((padding_needed,), tokenizer_instance.pad_id,
                                        dtype=torch.int16)
            final_tokens = torch.cat((encoded_tokens, padding_tensor))
        else:  # current_length == sequence_length
            final_tokens = encoded_tokens

        # Ensure the final tensor has the exact sequence_length (important safety check)
        if len(final_tokens) != sequence_length:
            if len(final_tokens) > sequence_length:
                final_tokens = final_tokens[:sequence_length]
            else:  # len < sequence_length
                padding_needed = sequence_length - len(final_tokens)
                final_tokens = torch.cat((final_tokens,
                                          torch.full((padding_needed,), tokenizer_instance.pad_id,
                                                     dtype=torch.int16)))

        # Generate key_padding_mask (True for padded tokens)
        key_padding_mask = (final_tokens == tokenizer_instance.pad_id)

        processed_input_ids_list.append(final_tokens)
        # processed_labels_list.append(final_tokens.clone()) # Labels are same as input
        processed_kpm_list.append(key_padding_mask)

    return {
        "input_ids": processed_input_ids_list,
        "labels": processed_input_ids_list,  # For autoencoder, labels are inputs
        "key_padding_mask": processed_kpm_list
    }


# --- Dataset Loading and Processing ---
print(f"\nLoading dataset '{exp_config['dataset_name']}'...")
raw_dataset = None
try:
    raw_dataset = load_dataset(
        exp_config["dataset_name"],
        name=exp_config.get("dataset_config"),  # Use .get() for optional config name
        split=exp_config["dataset_train_split"]
    )
    print(f"Dataset loaded. Number of examples: {len(raw_dataset)}")
except Exception as e:
    print(f"Error loading dataset '{exp_config['dataset_name']}': {e}")
    print("Using a small dummy dataset for demonstration.")
    dummy_data = {"text": ["Sample text for testing tokenization and model."] * 100}
    raw_dataset = Dataset.from_dict(dummy_data)
    exp_config["text_column_name"] = "text"  # Ensure text column matches dummy
    print(f"Dummy dataset created with {len(raw_dataset)} examples.")

print(
    f"\nTokenizing and processing dataset with sequence length {exp_config['sequence_length']}...")
tokenized_dataset = raw_dataset.map(
    tokenize_and_process_examples,
    batched=True,
    fn_kwargs={
        "tokenizer_instance": tokenizer,  # Pass the instantiated tokenizer
        "sequence_length": exp_config["sequence_length"],
        "text_column": exp_config["text_column_name"]
    },
    remove_columns=raw_dataset.column_names,
    num_proc=N_CPU  # Use multiple processes for mapping
)
print("Tokenization complete.")

print("\nSetting dataset format to PyTorch tensors...")
tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "labels", "key_padding_mask"]
)
print("Dataset format set.")

# --- DataLoader ---
print(f"\nCreating PyTorch DataLoader with batch size {exp_config['batch_size']}...")
train_dataloader = DataLoader(
    tokenized_dataset,
    batch_size=exp_config["batch_size"],
    # shuffle=True,
    num_workers=N_CPU,  # Use N_CPU for DataLoader workers
    pin_memory=True if DEVICE == "cuda" else False  # pin_memory if using CUDA
)
print(f"DataLoader created with {N_CPU} workers.")

# --- Training Loop ---
print("\nStarting training loop...")
global_step = 0
model.train()  # Set model to training mode

for epoch in range(exp_config["num_epochs"]):
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{exp_config['num_epochs']}",
                        unit="batch")
    for batch in progress_bar:
        tokens = batch["input_ids"].to(DEVICE)
        # KPM is already boolean, just move to device
        kpm = batch["key_padding_mask"].to(DEVICE) if exp_config[
            "propagate_key_padding_mask"] else None

        optimizer.zero_grad()
        # HierarchicalAutoencoder.forward now handles loss calculation internally
        # and expects KPM for the initial tokens.
        output_dict = model(tokens, key_padding_mask=kpm)

        total_loss = output_dict['total_loss']
        total_loss.backward()
        if exp_config.get("gradient_clip_norm"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), exp_config["gradient_clip_norm"])
        optimizer.step()

        if global_step % exp_config["log_interval"] == 0:
            aim_run.track(total_loss.item(), name='loss/total', step=global_step, epoch=epoch,
                          context={"subset": "train"})
            aim_run.track(output_dict['vq_loss'].item(), name='loss/vq', step=global_step,
                          epoch=epoch, context={"subset": "train"})
            aim_run.track(output_dict['avg_reconstruction_loss'].item(),
                          name='loss/reconstruction_avg', step=global_step, epoch=epoch,
                          context={"subset": "train"})

            for name, val in output_dict['reconstruction_loss_details'].items():
                aim_run.track(val.item(), name=f'loss_detail/{name}', step=global_step, epoch=epoch,
                              context={"subset": "train"})

            if 'avg_aux_lm_loss' in output_dict and exp_config.get("aux_lm_loss_weight", 0.0) > 0:
                aim_run.track(output_dict['avg_aux_lm_loss'].item(), name='loss/aux_lm_avg', step=global_step,
                              epoch=epoch, context={"subset": "train"})
                for name, val in output_dict['aux_lm_loss_details'].items():
                    aim_run.track(val.item(), name=f'loss_detail/{name}', step=global_step, epoch=epoch,
                                  context={"subset": "train"})

            if 'compression_ratios' in output_dict:  # Check if metrics are present
                for i, ratio in enumerate(output_dict['compression_ratios']):
                    aim_run.track(ratio, name=f'compression/ratio_L{i}', step=global_step,
                                  epoch=epoch)
                for i, length in enumerate(output_dict['input_seq_lengths_compressors']):
                    aim_run.track(length, name=f'compression/input_len_L{i}', step=global_step,
                                  epoch=epoch)
                for i, length in enumerate(output_dict['output_seq_lengths_compressors']):
                    aim_run.track(length, name=f'compression/output_len_L{i}', step=global_step,
                                  epoch=epoch)

            aim_run.track(optimizer.param_groups[0]['lr'], name='learning_rate', step=global_step,
                          epoch=epoch)

            if 'all_codebook_perplexities' in output_dict:
                for i, perplexity_val in enumerate(output_dict['all_codebook_perplexities']):
                    aim_run.track(perplexity_val.item(), name=f'vq_metrics/perplexity_L{i}', step=global_step,
                                  epoch=epoch)
                    # Log codebook size for reference
                    # This requires actual_codebook_sizes to be accessible or passed to logger
                    # Or, get it from compressor_level_configs
                    codebook_size_L_i = exp_config["compressor_level_configs"][i]["codebook_size"]
                    aim_run.track(codebook_size_L_i, name=f'vq_metrics/codebook_size_L{i}', step=global_step,
                                  epoch=epoch, context={"type": "config"})

            postfix_dict = {
                "loss": f"{total_loss.item():.4f}",
                "vq": f"{output_dict['vq_loss'].item():.4f}",
                "reco": f"{output_dict['avg_reconstruction_loss'].item():.4f}",
                "ratios": ", ".join([f"{r:.2f}" for r in output_dict.get('compression_ratios', [])])
            }
            if 'avg_aux_lm_loss' in output_dict and exp_config.get("aux_lm_loss_weight", 0.0) > 0:
                postfix_dict["auxLM"] = f"{output_dict['avg_aux_lm_loss'].item():.4f}"
            progress_bar.set_postfix(postfix_dict)

        # --- Generation During Training ---
        if global_step > 0 and global_step % exp_config["generation_interval"] == 0:
            print(f"\nStep {global_step}: Generating sample...")
            model.eval()  # Switch to evaluation mode for generation
            with torch.no_grad():
                sample_text = exp_config["sample_prompt_for_generation"]
                # Encode the prompt without padding to a fixed length
                # BOS/EOS will be added by tokenizer.encode if configured
                input_gen_tokens = tokenizer.encode(sample_text).unsqueeze(0).to(DEVICE)

                # For unpadded, variable-length input to generate_bytes, KPM is all False
                input_gen_kpm = None
                if exp_config["propagate_key_padding_mask"]:
                    input_gen_kpm = torch.zeros_like(input_gen_tokens, dtype=torch.bool).to(DEVICE)

                reconstructed_tokens = model.generate_bytes(
                    tokens=input_gen_tokens,
                    key_padding_mask=input_gen_kpm,  # Pass KPM if propagation is on
                    max_len_override=exp_config["generation_max_len_override"]
                )
                reconstructed_text = tokenizer.decode(reconstructed_tokens.squeeze(0).cpu(),
                                                      cut_at_eos=True)

                print(f"--- Sample Generation at Step {global_step} ---")
                print(f"Original Prompt:\n{sample_text}")
                print(f"Reconstructed Text:\n{reconstructed_text}")
                print("------------------------------------------")

                # Log to AIM
                aim_text_log = f"Original:\n{sample_text}\n\nReconstructed:\n{reconstructed_text}"
                aim_run.track(aim.Text(aim_text_log), name='sample_generation', step=global_step,
                              epoch=epoch)

            model.train()  # Switch back to training mode
        global_step += 1

print("Training finished.")

# Example of saving the model (optional)
# final_model_path = f"{exp_config.get('run_name', 'model')}_final.pth"
# torch.save(model.state_dict(), final_model_path)
# print(f"Final model state dict saved to {final_model_path}")
