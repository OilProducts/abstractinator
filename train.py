"""
TODO (training cleanups):

1) Base-component freezing logic
   - When loading base components via `--load_base_from`, we currently call
     `model.compressors.eval()` and `model.expanders.eval()` in `train_loop`.
     The pyramid exposes levels under `model.levels`, each with
     `level.compressor` and `level.expander`. Replace the blanket calls with:
       for lvl in model.levels:
           lvl.compressor.eval()
           lvl.expander.eval()
     Optionally also set `requires_grad=False` for their parameters if we intend
     to fully freeze them (note: `load_base_components(..., freeze=True)` already
     handles `requires_grad=False`).

2) Top‑LM VQ/quantizer wiring
   - We set `top_lm = CodeSequenceTransformer(..., vq=None)` and later assign
     `model.top_lm.vq = model.levels[-1].compressor.vq`, but `SegmentCompressor`
     exposes `quantizer`, not `vq`. Options:
       a) If the compressor’s `quantizer` implements the same `(z) ->
          (z_q, vq_loss, indices, perplexity)` API as `VectorQuantizer`, set
          `model.top_lm.vq = model.levels[-1].compressor.quantizer`.
       b) Generalize `CodeSequenceTransformer` to accept a `QuantizerBase`, or
          add a small adapter/wrapper to match the expected interface.
   - Decide which quantizer the top LM should use (top level vs. shared), and
     document the choice.
"""

import argparse
import importlib.util
import logging
import math
import os
import time
from dataclasses import replace
from functools import partial

import mlflow  # Logging with MLflow
import torch
import torch._dynamo as dynamo
from datasets import load_dataset  # Import Dataset for the dummy data
from mlflow.tracking import MlflowClient
from torch import optim
from torch.utils.data import DataLoader
from transformers.optimization import get_scheduler

from components import AbstractinatorPyramid, CodeSequenceTransformer
from components.checkpoint_utils import load_base_components, save_base_components

# from components.expander import _cached_causal_mask as _cached_causal_mask_cpu
from components.metrics import MlflowBatchLogger, TrainingMetrics

# from components.sliding_window_attention import _cached_cross_window_mask as _cached_cross_window_mask_cpu
from components.tokenizer import ByteLevelTokenizer
from components.utils import short_num
from data_utils import tokenize_and_process_examples
from experiments.exp_config import (
    DEVICE as DEFAULT_DEVICE,
)
from experiments.exp_config import (
    N_CPU as DEFAULT_N_CPU,
)
from experiments.exp_config import (
    ExpConfig,
)

# Name for loggers created in this module so logs don't show '__main__'
LOGGER_NAME = "abstractinator.train"

torch.backends.cuda.enable_flash_sdp(True)  # FlashAttn‑2*
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
dynamo.config.capture_dynamic_output_shape_ops = True
# torch._dynamo.config.recompile_limit = 512
# torch._dynamo.config.capture_scalar_outputs = True


def parse_args() -> argparse.Namespace:
    """Return command line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Training script for AbstractinatorPyramid")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to the configuration Python file. If omitted, the config saved in a checkpoint "
            "will be used when resuming"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from",
    )
    parser.add_argument(
        "--load_base_from",
        type=str,
        default=None,
        help="Load pretrained compressor/expander checkpoint",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="Logging level (debug, info, warning, error, critical)",
    )
    return parser.parse_args()


def load_experiment_config(args: argparse.Namespace) -> tuple[str, int, ExpConfig]:
    """Load the experiment configuration from a file or checkpoint."""

    def _load_config(path: str):
        spec = importlib.util.spec_from_file_location("config_module", path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module

    if args.config:
        config = _load_config(args.config)
        exp_cfg: ExpConfig = config.exp_config
        # Prefer explicit attributes on the config module, else fall back to
        # values inside exp_config, else use library defaults.
        device = getattr(config, "DEVICE", None) or getattr(exp_cfg, "device", None) or DEFAULT_DEVICE
        n_cpu = getattr(config, "N_CPU", None) or DEFAULT_N_CPU
    else:
        if not args.resume_from_checkpoint:
            raise ValueError("--config or --resume_from_checkpoint must be provided")
        ckpt = torch.load(args.resume_from_checkpoint, map_location="cpu")
        if "exp_config" not in ckpt:
            raise ValueError("Checkpoint missing exp_config; provide --config")
        cfg = ckpt["exp_config"]
        exp_cfg = ExpConfig(**cfg) if isinstance(cfg, dict) else cfg
        device = DEFAULT_DEVICE
        n_cpu = DEFAULT_N_CPU

    return device, n_cpu, exp_cfg


def initialize_model(
    args: argparse.Namespace, exp_config: ExpConfig, device: str
) -> tuple[AbstractinatorPyramid, optim.Optimizer, int, int]:
    """Instantiate the model, optimizer and restore any checkpoints."""

    logger = logging.getLogger(LOGGER_NAME)

    # Optional Top LM instantiation
    top_lm = None
    try:
        use_top_lm = bool(getattr(exp_config.pyramid_config, "use_top_code_lm", False))
    except Exception:
        use_top_lm = False

    if use_top_lm and getattr(exp_config, "top_transformer_config", None) is not None:
        tcfg = exp_config.top_transformer_config
        # Match top level embedding dimension
        try:
            top_level_cfg = exp_config.pyramid_config.levels[-1]
            embed_dim = int(getattr(top_level_cfg, "D"))
        except Exception:
            embed_dim = int(getattr(tcfg, "embed_dim", 128))

        # Keep original config intact; construct a local copy with matched embed_dim
        tcfg_local = replace(tcfg, embed_dim=embed_dim)

        top_lm = CodeSequenceTransformer(
            tcfg_local,
            vq=None,  # wired after model is constructed
            use_flex_attention=bool(getattr(exp_config, "flex_attention", True)),
        )

    model = AbstractinatorPyramid(cfg=exp_config.pyramid_config, top_lm=top_lm).to(device)

    # If we constructed a top LM, attach the top-level compressor VQ so it can quantize/pick EOS
    if top_lm is not None and hasattr(model, "levels") and len(model.levels) > 0:
        try:
            model.top_lm.vq = model.levels[-1].compressor.vq
        except Exception:
            pass

    if args.load_base_from:
        load_base_components(
            model,
            args.load_base_from,
            freeze=True,
            map_location=device,
        )
    # Entropy stack may be loaded/frozen per-level via AbstractinatorConfig

    def _count_params(module: torch.nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    num_params = _count_params(model)
    logger.info(
        "Model initialized on %s with %s trainable parameters.",
        device,
        short_num(num_params),
    )

    # High-level component breakdown
    levels_params = _count_params(model.levels) if hasattr(model, "levels") else 0
    logger.info("  Levels (all): %s parameters", short_num(levels_params))

    # Optional top LM module (high level)
    top_lm = getattr(model, "top_lm", None)
    if top_lm is not None:
        top_lm_params = _count_params(top_lm)
        logger.info("  Top LM: %s parameters", short_num(top_lm_params))

        # Finer breakdown for CodeSequenceTransformer (avoid shadowing by aliasing)
        try:
            from components import CodeSequenceTransformer as _CST  # safe alias

            if isinstance(top_lm, _CST):
                in_proj = _count_params(getattr(top_lm, "in_proj", top_lm))
                enc = _count_params(getattr(top_lm, "encoder", top_lm))
                f_norm = _count_params(getattr(top_lm, "final_norm", top_lm))
                out_proj = _count_params(getattr(top_lm, "out_proj", top_lm))
                logger.info(
                    "    Top LM breakdown → in: %s | encoder: %s | norm: %s | out: %s",
                    short_num(in_proj),
                    short_num(enc),
                    short_num(f_norm),
                    short_num(out_proj),
                )
        except Exception:
            pass

    # Per-level breakdown with a finer split for compressor/expander internals
    if hasattr(model, "levels"):
        total_comp = 0
        total_exp = 0

        # Helper to safely count attributes that may not exist
        def _safe_count(mod: torch.nn.Module | None) -> int:
            try:
                return _count_params(mod) if mod is not None else 0
            except Exception:
                return 0

        for li, lvl in enumerate(model.levels):
            comp = getattr(lvl, "compressor", None)
            expn = getattr(lvl, "expander", None)

            comp_params = _safe_count(comp)
            exp_params = _safe_count(expn)
            total_comp += comp_params
            total_exp += exp_params
            logger.info(
                "  L%s: %s total | compress %s | decompress %s",
                li,
                short_num(comp_params + exp_params),
                short_num(comp_params),
                short_num(exp_params),
            )

            # Compressor breakdown
            if comp is not None:
                emb = _safe_count(getattr(comp, "embedding", None))
                shared = _safe_count(getattr(comp, "shared_layers", None))
                comp_layers = _safe_count(getattr(comp, "compression_layers", None))
                lm_layers = _safe_count(getattr(comp, "lm_layers", None))
                lm_norm = _safe_count(getattr(comp, "lm_final_norm", None))
                lm_head = _safe_count(getattr(comp, "logit_proj", None))
                pool = _safe_count(getattr(comp, "pooler", None))
                vq = getattr(comp, "vq", None)
                vq_total = _safe_count(vq)
                vq_down = _safe_count(getattr(vq, "down", None))
                vq_up = _safe_count(getattr(vq, "up", None))
                # sum stage codebooks (VectorQuantizer.codebook)
                vq_stage_params = 0
                try:
                    stages = getattr(vq, "stages", [])
                    for st in stages:
                        vq_stage_params += _safe_count(st)
                except Exception:
                    pass

                # Counts + small context
                try:
                    n_shared = len(getattr(comp, "shared_layers", []))
                except Exception:
                    n_shared = 0
                try:
                    n_comp = len(getattr(comp, "compression_layers", []))
                except Exception:
                    n_comp = 0
                try:
                    n_lm = len(getattr(comp, "lm_layers", []))
                except Exception:
                    n_lm = 0

                logger.info(
                    "    ├─ Compressor breakdown → embed: %s | shared(%d): %s | compression(%d): %s",
                    short_num(emb),
                    n_shared,
                    short_num(shared),
                    n_comp,
                    short_num(comp_layers),
                )
                logger.info(
                    "    │  LM branch → layers(%d): %s | norm: %s | logits: %s",
                    n_lm,
                    short_num(lm_layers),
                    short_num(lm_norm),
                    short_num(lm_head),
                )
                logger.info(
                    "    │  Pooler: %s | VQ total: %s (down: %s, up: %s, stages: %s)",
                    short_num(pool),
                    short_num(vq_total),
                    short_num(vq_down),
                    short_num(vq_up),
                    short_num(vq_stage_params),
                )

            # Expander breakdown
            if expn is not None:
                lo_adapt = _safe_count(getattr(expn, "lo_adapt", None))
                hi_adapt = _safe_count(getattr(expn, "hi_adapt", None))
                dec = _safe_count(getattr(expn, "decoder", None))
                head = _safe_count(getattr(expn, "head", None))

                try:
                    n_dec_layers = len(getattr(getattr(expn, "decoder", None), "layers", []))
                except Exception:
                    n_dec_layers = 0

                logger.info(
                    "    └─ Expander breakdown → lo_adapter: %s | hi_adapter: %s | decoder(%d): %s | head: %s",
                    short_num(lo_adapt),
                    short_num(hi_adapt),
                    n_dec_layers,
                    short_num(dec),
                    short_num(head),
                )

        logger.info(
            "  All compressors: %s | All decompressors: %s",
            short_num(total_comp),
            short_num(total_exp),
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=exp_config.learning_rate)

    start_epoch = 0
    global_step = 0
    resume_path = args.resume_from_checkpoint or exp_config.resume_from_checkpoint
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state", {}))
        optimizer.load_state_dict(ckpt.get("optimizer_state", {}))
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        logger.info(
            "Resumed from checkpoint '%s' at step %s, epoch %s",
            resume_path,
            global_step,
            start_epoch,
        )
    elif resume_path:
        logger.warning("Checkpoint path '%s' not found. Starting from scratch.", resume_path)

    return model, optimizer, start_epoch, global_step


def save_checkpoint(
    model: AbstractinatorPyramid,
    optimizer: optim.Optimizer,
    epoch: int,
    step: int,
    checkpoint_dir: str,
    exp_config: ExpConfig,
    logger: logging.Logger | None = None,
) -> None:
    """Persist training state to ``checkpoint_dir``."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step{step}.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": step,
            "exp_config": exp_config,
        },
        checkpoint_path,
    )
    if logger:
        logger.info("Checkpoint saved to %s", checkpoint_path)


class Trainer:
    """Encapsulate training state and loop."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device, self.n_cpu, self.exp_config = load_experiment_config(args)
        self._setup_logging()

        torch.set_float32_matmul_precision("high")
        torch.set_default_dtype(torch.bfloat16)
        torch.set_printoptions(threshold=100_000)
        torch._dynamo.config.capture_scalar_outputs = True
        # torch._dynamo.config.recompile_limit = 128
        # torch.autograd.set_detect_anomaly(True)

        (
            self.model,
            self.optimizer,
            self.start_epoch,
            self.global_step,
        ) = initialize_model(args, self.exp_config, self.device)

    def _setup_logging(self) -> None:
        """Configure the global logger."""
        log_level = getattr(logging, self.args.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        )
        self.logger = logging.getLogger(LOGGER_NAME)
        self.logger.info("Using device: %s", self.device)

    def train(self) -> None:
        """Run the training loop."""
        train_loop(
            model=self.model,
            optimizer=self.optimizer,
            exp_config=self.exp_config,
            device=self.device,
            n_cpu=self.n_cpu,
            start_epoch=self.start_epoch,
            global_step=self.global_step,
            args=self.args,
        )


def train_loop(
    model: AbstractinatorPyramid,
    optimizer: optim.Optimizer,
    exp_config: ExpConfig,
    device: str,
    n_cpu: int,
    start_epoch: int,
    global_step: int,
    args: argparse.Namespace,
) -> None:
    """Run the full training loop."""

    logger = logging.getLogger(LOGGER_NAME)

    mlflow.set_experiment(getattr(exp_config, "project_name", "DefaultExperiment"))
    mlflow_run = mlflow.start_run(run_name=getattr(exp_config, "run_name", "DefaultRun"))
    mlflow_client = MlflowClient()
    mlflow_run_id = mlflow_run.info.run_id
    mlflow_logger = MlflowBatchLogger(mlflow_client, mlflow_run_id, exp_config.mlflow_batch_interval)

    tokenizer = ByteLevelTokenizer(
        add_bos=True,
        add_eos=True,
        expected_vocab_size=exp_config.initial_vocab_size,
    )

    logger.info(
        "Tokenizer initialized. BOS ID: %s, EOS ID: %s, PAD ID: %s, Effective Vocab Size: %s",
        tokenizer.bos_id,
        tokenizer.eos_id,
        tokenizer.pad_id,
        tokenizer.vocab_size,
    )

    logger.info("\nLoading dataset '%s'...", exp_config.dataset_name)
    raw_dataset = load_dataset(
        exp_config.dataset_name,
        name=exp_config.dataset_config,
        split=exp_config.dataset_train_split,
    )
    logger.info("Dataset loaded. Number of examples: %s", len(raw_dataset))

    logger.info(
        "\nTokenizing and processing dataset with sequence length %s...",
        exp_config.sequence_length,
    )

    tokenized_dataset = raw_dataset.map(
        partial(
            tokenize_and_process_examples,
            sequence_length=exp_config.sequence_length,
            tokenizer=tokenizer,
            text_column=exp_config.text_column_name,
        ),
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=8,
        desc="Byte-tokenising",
    )

    tokenized_dataset.set_format(
        "torch", columns=["input_ids", "labels"], dtype=torch.int16
    )  # let datasets choose per-column dtype

    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "labels", "key_padding_mask"],
        output_all_columns=True,  # keeps prior dtype choices
    )

    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=exp_config.batch_size,
        num_workers=8,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True,
        prefetch_factor=2,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / exp_config.gradient_accumulation_steps)
    if exp_config.max_steps is not None:
        exp_config.num_training_steps = int(exp_config.max_steps)
    else:
        exp_config.num_training_steps = exp_config.num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=exp_config.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=exp_config.warmup_steps,
        num_training_steps=exp_config.num_training_steps,
        scheduler_specific_kwargs=exp_config.scheduler_specific_kwargs
        if hasattr(exp_config, "scheduler_specific_kwargs")
        else {},
    )

    mlflow.log_params(exp_config.as_dict())

    model.train()
    if args.load_base_from:
        model.compressors.eval()
        model.expanders.eval()
    optimizer.zero_grad()

    # if device == "cuda":
    #     torch.cuda.synchronize()
    training_start_time = time.time()
    metrics = TrainingMetrics(
        num_levels=len(getattr(model, "levels", []))
        if getattr(model, "levels", None) is not None
        else exp_config.num_levels,
        aux_lm_enabled=True,  # entropy model loss is always tracked for logging
        top_lm_enabled=getattr(exp_config.pyramid_config, "use_top_code_lm", False),
    )

    # Capture codebook sizes once for MLflow metadata
    codebook_sizes: list[int] | None = None
    if hasattr(model, "levels") and model.levels is not None:
        codebook_sizes = []
        try:
            for lvl in model.levels:
                vq = getattr(lvl.compressor, "vq", None)
                codebook_sizes.append(int(getattr(vq, "K", 0)) if vq is not None else 0)
        except Exception:
            codebook_sizes = None
    time_of_last_optimizer_step_event = training_start_time
    total_minibatches_in_epoch = len(train_dataloader)
    model = torch.compile(model)
    #
    # from torch.profiler import profile, record_function, ProfilerActivity
    #
    # with profile(
    #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #         # record_shapes=True,
    #         # with_stack=True,
    #         profile_memory=True,
    #         # with_modules=True,
    # ) as prof:
    #     for _ in range(50):  # run ~50 optimizer steps
    #         batch = next(iter(train_dataloader))
    #         tokens = batch["input_ids"].to(device, non_blocking=True)
    #         key_padding_mask = batch["key_padding_mask"].to(device, non_blocking=True)
    #         out = model(tokens, key_padding_mask=key_padding_mask)
    #         loss = out["loss_total"] / exp_config.gradient_accumulation_steps
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #
    # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
    # prof.export_chrome_trace("trace.json")

    # def nan_hook(name):
    #     def _hook(module, inputs, output):
    #         def has_bad(x):
    #             return torch.is_tensor(x) and (~torch.isfinite(x)).any().item()
    #
    #         bad_in = any(has_bad(x) for x in (inputs if isinstance(inputs, tuple) else (inputs,)))
    #         bad_out = has_bad(output)
    #         if bad_in or bad_out:
    #             print(f"[NaN/Inf] in {name}: bad_in={bad_in}, bad_out={bad_out}")
    #             if bad_out and torch.is_tensor(output):
    #                 with torch.no_grad():
    #                     print("  output stats:", output.float().min().item(), output.float().max().item())
    #             raise RuntimeError(f"NaN/Inf encountered in {name}")
    #
    #     return _hook
    #
    # # Register on sensitive blocks
    # for n, m in model.named_modules():
    #     if any(k in n.lower() for k in ["attention", "attn", "softmax", "layernorm", "vq", "loss"]):
    #         m.register_forward_hook(nan_hook(n))

    for epoch in range(start_epoch, exp_config.num_epochs):
        logger.info("\n--- Epoch %s/%s ---", epoch + 1, exp_config.num_epochs)
        # _cached_cross_window_mask_cpu.cache_clear()
        # _cached_causal_mask_cpu.cache_clear()
        epoch_start_time = time.time()
        model.train()
        if args.load_base_from:
            model.compressors.eval()
            model.expanders.eval()

        for i, batch in enumerate(train_dataloader):
            # Make H2D transfers asynchronous to overlap with compute
            tokens = batch["input_ids"].to(device, non_blocking=True)
            key_padding_mask = batch["key_padding_mask"].to(device, non_blocking=True)
            model_kpm = key_padding_mask if exp_config.propagate_key_padding_mask else None

            output_dict = model(tokens, key_padding_mask=model_kpm)
            total_loss = output_dict["loss_total"]

            # Add Top-LM loss if enabled and present
            if getattr(exp_config.pyramid_config, "use_top_code_lm", False) and "avg_top_code_lm_loss" in output_dict:
                w_top = float(getattr(exp_config, "top_lm_loss_weight", 1.0))
                w_mse = float(getattr(exp_config, "top_lm_mse_weight", 1.0))
                # CE path not currently modeled; include only MSE component here
                total_loss = total_loss + w_top * (w_mse * output_dict["avg_top_code_lm_loss"])

            loss_for_backward = total_loss / exp_config.gradient_accumulation_steps
            loss_for_backward.backward()

            metrics.update_from_batch(output_dict, key_padding_mask)

            if (i + 1) % exp_config.gradient_accumulation_steps == 0:
                if exp_config.gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), exp_config.gradient_clip_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # if device == "cuda":
                #     torch.cuda.synchronize()
                current_time = time.time()
                duration_accumulation_window = current_time - time_of_last_optimizer_step_event

                tp = metrics.throughput(duration_accumulation_window)
                tokens_per_second = tp.tokens_per_second
                avg_tok_s = tp.avg_tokens_per_second
                patches_per_second = tp.patches_per_second_per_level
                total_progress = (global_step + 1) / exp_config.num_training_steps
                total_elapsed = current_time - training_start_time
                total_eta_sec = (total_elapsed / total_progress - total_elapsed) if total_progress > 0 else 0

                steps_accumulated = metrics.count

                # Only do console + MLflow logging on the chosen interval to reduce CPU
                if steps_accumulated > 0 and (global_step % exp_config.log_interval == 0):
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    console_log_parts = metrics.console_parts(
                        timestamp=timestamp,
                        epoch=epoch + 1,
                        num_epochs=exp_config.num_epochs,
                        global_step=global_step,
                        minibatch_index=i + 1,
                        total_minibatches=total_minibatches_in_epoch,
                        avg_tok_s=avg_tok_s,
                        total_eta_sec=total_eta_sec,
                        patches_per_second=patches_per_second,
                    )
                    if console_log_parts:
                        logger.info(" | ".join(console_log_parts))

                    if metrics.count > 0:
                        metrics_dict = metrics.metrics_dict(
                            learning_rate=optimizer.param_groups[0]["lr"],
                            tokens_per_second=tokens_per_second,
                            patches_per_second=patches_per_second,
                            codebook_sizes=codebook_sizes,
                        )
                        mlflow_logger.log(global_step, metrics_dict)

                metrics.reset_window()
                time_of_last_optimizer_step_event = current_time

                if global_step > 0 and global_step % exp_config.generation_interval == 0:
                    logger.info("\nStep %s: Generating sample...", global_step)
                    model.eval()
                    with torch.no_grad():
                        sample_text = exp_config.sample_prompt_for_generation
                        input_gen_tokens = (
                            tokenizer.encode(sample_text, add_eos=False).unsqueeze(0).to(device).to(torch.int64)
                        )

                        input_gen_kpm = None
                        if exp_config.propagate_key_padding_mask:
                            input_gen_kpm = torch.zeros_like(input_gen_tokens, dtype=torch.bool).to(device)

                        reconstructed_tokens = model.generate_bytes(
                            prompt=input_gen_tokens,
                            prompt_kpm=input_gen_kpm,
                            max_top_steps=exp_config.generation_max_len_override,
                            max_child_len=8,
                            top_sample_fn=model.top_lm,
                        )
                        reconstructed_text = tokenizer.decode(reconstructed_tokens.squeeze(0).cpu())

                        logger.info("--- Sample Generation at Step %s ---", global_step)
                        logger.info("Original Prompt:\n%s", sample_text)
                        logger.info("Reconstructed Text:\n%s", reconstructed_text)
                        logger.info("------------------------------------------")

                        mlflow_text_log = f"Original:\n{sample_text}\n\nReconstructed:\n{reconstructed_text}"
                        mlflow.log_text(mlflow_text_log, f"sample_generation_step_{global_step}.txt")

                    model.train()
                    if args.load_base_from:
                        model.compressors.eval()
                        model.expanders.eval()
                if global_step > 0 and global_step % exp_config.checkpoint_interval == 0:
                    save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        global_step,
                        exp_config.checkpoint_dir,
                        exp_config,
                        logger,
                    )

                global_step += 1
                if exp_config.max_steps is not None and global_step >= exp_config.max_steps:
                    logger.info("Reached max_steps %s. Stopping training.", exp_config.max_steps)
                    break
        epoch_duration = time.time() - epoch_start_time
        logger.info("--- Epoch %s finished. Duration: %.2fs ---", epoch + 1, epoch_duration)
        save_checkpoint(
            model,
            optimizer,
            epoch,
            global_step,
            exp_config.checkpoint_dir,
            exp_config,
            logger,
        )

        if exp_config.max_steps is not None and global_step >= exp_config.max_steps:
            break

    logger.info("Training finished.")
    if exp_config.save_base_components_path:
        save_base_components(model, exp_config.save_base_components_path)
    mlflow_logger.flush()
    mlflow.end_run()


def main() -> None:
    """Entry point for command line execution."""

    args = parse_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
