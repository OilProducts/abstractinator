#  EXPERIMENTAL!!! THIS MODULE IS NOT IN USE.
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import get_scheduler

from components.gaussian_transformer import ByteGaussianProbe, GenCfg, generate_with_probe

# ---- your model pieces are assumed available in scope ----
# from your_model_file import ByteGaussianProbe, GenCfg, generate_with_probe


# ----------------------------
# Config
# ----------------------------
@dataclass
class ExpConfig:
    # data
    dataset_repo: str = "HuggingFaceFW/fineweb-edu"  # FineWeb‑Edu
    dataset_name: str = "sample-10BT"  # 10B‑token sample config
    split: str = "train"
    seq_len: int = 2048
    stream_shuffle_buffer: int = 50_000  # streaming shuffle buffer
    doc_separator: bytes = b"\n\n"  # between docs

    # training
    batch_size: int = 8
    grad_accum_steps: int = 4
    num_training_steps: int = 20_000
    lr: float = 3e-4
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    amp_dtype: Optional[str] = "bf16"  # "bf16", "fp16", or None
    compile_model: bool = True  # torch.compile if True

    # scheduler (exact call you asked for)
    scheduler_type: str = "cosine"  # e.g., "linear", "cosine", "cosine_with_restarts", ...
    warmup_steps: int = 1_000
    scheduler_specific_kwargs: Dict[str, Any] = field(default_factory=dict)

    # logging & sampling
    log_every: int = 10
    gen_every: int = 300
    max_new_tokens: int = 200
    gen_topk: Optional[int] = 100
    gen_topp: Optional[float] = 0.95
    gen_tau_proto: float = 0.8
    gen_prompt: str = "The quick brown fox"

    # model
    vocab_size: int = 256
    emb_dim: int = 512
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    dropout: float = 0.0
    n_components: int = 2  # Gaussian components

    # dual-objective knobs
    ce_weight: float = 1.0
    bpd_weight: float = 0.5  # try 0.05–0.2
    sigma2_floor: float = 1e-3  # stable mixture logits
    detach_bpd_targets: bool = True  # keep embedding path from being pulled by BPD


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_and_dtype(amp_dtype: Optional[str]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if amp_dtype == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return device, torch.bfloat16, True
    if amp_dtype == "fp16" and torch.cuda.is_available():
        return device, torch.float16, True
    return device, torch.float32, False


def bytes_encode(text: str) -> bytes:
    # UTF‑8 → bytes (0..255); replace errors to avoid loss in length accounting
    return text.encode("utf-8", errors="replace")


def bytes_decode(ids: torch.Tensor) -> str:
    # ids: (S,) in 0..255
    try:
        return bytes(ids.tolist()).decode("utf-8", errors="ignore")
    except Exception:
        return bytes([int(x) & 0xFF for x in ids.tolist()]).decode("utf-8", errors="ignore")


 


# --- numeric helper: stable logvar with a variance floor ---
def _effective_logvar(logvar: torch.Tensor, sigma2_floor: float = 1e-3) -> torch.Tensor:
    """Stable log(exp(logvar) + floor). Keeps mixtures from collapsing numerically."""
    if sigma2_floor <= 0:
        return logvar
    return torch.logaddexp(logvar, torch.tensor(sigma2_floor, device=logvar.device, dtype=logvar.dtype).log())


# --- vectorized, fp32-safe sequence logits for mixture-of-Gaussians ---
def mog_sequence_logits_safe(mu, logvar, pi_logit, E, ln, sigma2_floor: float = 1e-3):
    """
    Mixture Mahalanobis logits for every time-step in one shot.
    Shapes:
      mu, logvar:   (B, S1, K, D)
      pi_logit:     (B, S1, K)  (if K==1, may be None; treated as zeros)
      E.weight:     (V, D)
    Returns:
      logits:       (B, S1, V)   (float32; safe for CE)
    """
    B, S1, K, D = mu.shape
    with torch.amp.autocast(enabled=False, device_type='cuda'):  # force fp32 for stability
        P = ln(E.weight).float()  # (V, D)
        Q = P * P  # (V, D)

        mu32 = mu.float()  # (B,S1,K,D)
        lv32 = _effective_logvar(logvar.float(), sigma2_floor)  # (B,S1,K,D)
        invv = torch.exp(-lv32)  # (B,S1,K,D)

        a = mu32 * invv  # (B,S1,K,D)
        b = 0.5 * invv  # (B,S1,K,D)

        # Two GEMVs per component, no (B,S1,K,V,D) tensor materialization.
        term1 = torch.einsum('bskd,vd->bskv', a, P)  # (B,S1,K,V)
        term2 = torch.einsum('bskd,vd->bskv', b, Q)  # (B,S1,K,V)
        const = -0.5 * ((mu32 * mu32 * invv).sum(-1) + lv32.sum(-1))  # (B,S1,K)

        energy = term1 - term2 + const.unsqueeze(-1)  # (B,S1,K,V)

        if pi_logit is None:
            logpi = torch.zeros((B, S1, K, 1), device=mu.device, dtype=torch.float32)
        else:
            logpi = pi_logit.float().log_softmax(-1).unsqueeze(-1)  # (B,S1,K,1)

        logits32 = torch.logsumexp(logpi + energy, dim=2)  # (B,S1,V)
        logits32 = torch.nan_to_num(logits32, neginf=-1e9, posinf=1e9)
        return logits32


# --- mixture BPD (embedding space) on already computed stats ---
def mog_bpd(mu: torch.Tensor, logvar: torch.Tensor, pi_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mixture NLL -> bits-per-dim per step.
    mu/logvar: (B,S1,K,D) , pi_logit: (B,S1,K), target: (B,S1,D)
    Returns: (B,S1) bits/dim
    """
    B, S1, K, D = mu.shape
    invvar = torch.exp(-logvar)  # (B,S1,K,D)
    diff2 = (target.unsqueeze(2) - mu).pow(2) * invvar  # (B,S1,K,D)
    quad = diff2.sum(-1)  # (B,S1,K)
    sum_logvar = logvar.sum(-1)  # (B,S1,K)
    logpi = F.log_softmax(pi_logit, dim=-1) if pi_logit is not None else torch.zeros_like(quad)
    const = 0.5 * D * math.log(2 * math.pi)
    logp = torch.logsumexp(logpi - 0.5 * (quad + sum_logvar) - const, dim=-1)  # (B,S1)
    nll = -logp
    bpd = (nll / math.log(2.0)) / D
    return bpd


def gaussian_bpd(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Single-component (diagonal) Gaussian bits-per-dim per step: (B,S1,D)->(B,S1)."""
    const = 0.5 * math.log(2 * math.pi)
    nll = 0.5 * ((target - mu) ** 2 * torch.exp(-logvar) + logvar).sum(dim=-1) + const * target.size(-1)
    return (nll / math.log(2.0)) / target.size(-1)


def dual_objective_loss_once(
    probe,  # ByteGaussianProbe (with mixture core; K>=1)
    tokens: torch.Tensor,  # (B,S)
    attn_mask: torch.Tensor,  # (B,S) bool
    *,
    sigma2_floor: float = 1e-3,
    detach_bpd_targets: bool = True,
):
    """
    Returns:
      total_ce_loss (nats), total_bpd_loss (bits/dim), logits for logging
    Performs ONE embed + ONE core forward, reuses outputs for both objectives.
    """
    # Embed once
    x = probe._embed(tokens)  # (B,S,D)

    # Core once (μ/logσ²/π for positions 0..S-2 -> predicts steps 1..S-1)
    mu, logvar, pi_logit = probe.core(x, attn_mask=attn_mask)  # (B,S-1,K,D), (..), (B,S-1,K or None)

    # ----- CE over next token (byte) -----
    logits = mog_sequence_logits_safe(mu, logvar, pi_logit, probe.embedding, probe.norm, sigma2_floor)  # (B,S-1,V)
    tgt_tok = tokens[:, 1:]  # (B,S-1)
    ce_vec = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_tok.reshape(-1), reduction='none')  # nats

    if attn_mask is not None:
        vm = (attn_mask[:, 1:] & attn_mask[:, :-1]).reshape(-1)  # valid next-step positions
        ce_loss = (ce_vec * vm).sum() / vm.sum().clamp_min(1)
    else:
        ce_loss = ce_vec.mean()

    ce_bpb = ce_loss / math.log(2.0)  # bits-per-byte (for logging)

    # ----- BPD in embedding space (aux) -----
    tgt_emb = x[:, 1:, :]  # (B,S-1,D) from SAME embed pass
    if detach_bpd_targets:
        tgt_emb = tgt_emb.detach()

    if mu.dim() == 4 and mu.size(2) > 1:
        bpd_steps = mog_bpd(mu, logvar, pi_logit, tgt_emb)  # (B,S-1) bits/dim
    else:
        bpd_steps = gaussian_bpd(mu.squeeze(2), logvar.squeeze(2), tgt_emb)

    if attn_mask is not None:
        vm2 = attn_mask[:, 1:] & attn_mask[:, :-1]
        bpd_loss = (bpd_steps.masked_select(vm2)).mean()
    else:
        bpd_loss = bpd_steps.mean()

    return ce_loss, ce_bpb, bpd_loss


# ----------------------------
# Streaming byte sequence dataset
# ----------------------------
class HFByteSeqStream(IterableDataset):
    """
    Streams FineWeb‑Edu with datasets.load_dataset(..., streaming=True),
    concatenates docs with a separator, and yields fixed-length byte sequences.
    Each item is a dict: {"tokens": LongTensor[S], "attn_mask": BoolTensor[S]}
    """

    def __init__(self, repo: str, name: str, split: str, seq_len: int, shuffle_buffer: int, doc_sep: bytes, seed: int):
        super().__init__()
        self.repo = repo
        self.name = name
        self.split = split
        self.seq_len = seq_len
        self.shuffle_buffer = shuffle_buffer
        self.doc_sep = doc_sep
        self.seed = seed

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        ds = load_dataset(self.repo, name=self.name, split=self.split, streaming=True)  # streaming
        # Reservoir shuffle with buffer for stream order randomization
        ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)

        buf = bytearray()
        for row in ds:
            text = row.get("text", "") or ""
            if self.doc_sep:
                buf.extend(self.doc_sep)
            buf.extend(bytes_encode(text))
            # emit as many full chunks as we can
            while len(buf) >= self.seq_len:
                chunk = buf[: self.seq_len]
                del buf[: self.seq_len]
                tokens = torch.tensor(list(chunk), dtype=torch.long)
                attn = torch.ones(self.seq_len, dtype=torch.bool)
                yield {"tokens": tokens, "attn_mask": attn}


def collate(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    tokens = torch.stack([b["tokens"] for b in batch], dim=0)  # (B,S)
    attn = torch.stack([b["attn_mask"] for b in batch], dim=0)  # (B,S)
    return {"tokens": tokens, "attn_mask": attn}


# ----------------------------
# Training
# ----------------------------
def make_model(cfg: ExpConfig) -> ByteGaussianProbe:
    model = ByteGaussianProbe(
        vocab_size=cfg.vocab_size,
        dim=cfg.emb_dim,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        n_components=cfg.n_components,
    )
    return model


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def log_generation(model: ByteGaussianProbe, cfg: ExpConfig, device: torch.device):
    model.eval()
    with torch.no_grad():
        prefix = torch.tensor(list(bytes_encode(cfg.gen_prompt)), dtype=torch.long, device=device)[None, :]
        attn = torch.ones_like(prefix, dtype=torch.bool)
        gen = generate_with_probe(
            probe=model,
            E=model.embedding,
            prefix=prefix,
            cfg=GenCfg(
                max_new_tokens=cfg.max_new_tokens,
                tau_proto=cfg.gen_tau_proto,
                sample=True,
                topk=cfg.gen_topk,
                topp=cfg.gen_topp,
                gaussian_temp=1.0,
                n_components=cfg.n_components,
            ),
            attn_mask=attn,
        )
        out = bytes_decode(gen[0])
        print("\n" + "=" * 80)
        print(f"[sample @ step] prompt: {cfg.gen_prompt!r}")
        print(out)
        print("=" * 80 + "\n")
    model.train()


def train(exp_config: ExpConfig):
    set_seed(exp_config.seed)
    device, amp_dtype, amp_enabled = device_and_dtype(exp_config.amp_dtype)

    # Data
    ds = HFByteSeqStream(
        repo=exp_config.dataset_repo,
        name=exp_config.dataset_name,
        split=exp_config.split,
        seq_len=exp_config.seq_len,
        shuffle_buffer=exp_config.stream_shuffle_buffer,
        doc_sep=exp_config.doc_separator,
        seed=exp_config.seed,
    )
    # IterableDataset + automatic batching
    dl = DataLoader(
        ds,
        batch_size=exp_config.batch_size,
        collate_fn=collate,
        num_workers=0,  # keep 0 for deterministic streaming
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        prefetch_factor=None,  # not used with num_workers=0
    )
    data_iter = iter(dl)

    # Model
    model = make_model(exp_config).to(device)
    if exp_config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # optional, may help on long contexts

    print(f"Model params: {count_params(model) / 1e6:.2f}M")
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=exp_config.lr, betas=exp_config.betas, weight_decay=exp_config.weight_decay
    )
    # Scheduler (exact signature)
    lr_scheduler = get_scheduler(
        name=exp_config.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=exp_config.warmup_steps,
        num_training_steps=exp_config.num_training_steps,
        scheduler_specific_kwargs=exp_config.scheduler_specific_kwargs
        if hasattr(exp_config, "scheduler_specific_kwargs")
        else {},
    )

    scaler = torch.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    # Logging state
    model.train()
    global_step = 0
    running_loss = 0.0
    tokens_done = 0
    t0 = time.perf_counter()
    t_log = t0

    # Train loop
    while global_step < exp_config.num_training_steps:
        optimizer.zero_grad(set_to_none=True)

        micro_losses = []
        micro_ce_bpbs = []
        micro_bpd_bits = []

        for micro in range(exp_config.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                batch = next(data_iter)

            tokens = batch["tokens"].to(device, non_blocking=True)  # (B,S)
            attn = batch["attn_mask"].to(device, non_blocking=True)  # (B,S)
            valid_steps = (attn[:, 1:] & attn[:, :-1]).sum().item()
            tokens_done += int(valid_steps)

            # autocast for the backbone; mixture logits are forced to fp32 inside helper
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                ce_loss, ce_bpb, bpd_loss = dual_objective_loss_once(
                    model,
                    tokens,
                    attn,
                    sigma2_floor=exp_config.sigma2_floor,
                    detach_bpd_targets=exp_config.detach_bpd_targets,
                )
                total_loss = exp_config.ce_weight * ce_loss + exp_config.bpd_weight * bpd_loss

            # gradient accumulation (correctly scaled if fp16 scaler is enabled)
            micro_losses.append(total_loss.detach())
            micro_ce_bpbs.append(ce_bpb.detach())
            micro_bpd_bits.append(bpd_loss.detach())

            if scaler.is_enabled():
                scaler.scale(total_loss / exp_config.grad_accum_steps).backward()
            else:
                (total_loss / exp_config.grad_accum_steps).backward()

        if exp_config.max_grad_norm and exp_config.max_grad_norm > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), exp_config.max_grad_norm)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        lr_scheduler.step()
        global_step += 1

        # Stats
        step_loss = torch.stack(micro_losses).mean().item()
        step_ce_bpb = torch.stack(micro_ce_bpbs).mean().item()
        step_bpd_bits = torch.stack(micro_bpd_bits).mean().item()
        running_loss = 0.9 * running_loss + 0.1 * step_loss if global_step > 1 else step_loss

        if global_step % exp_config.log_every == 0:
            now = time.perf_counter()
            dt = max(now - t_log, 1e-9)
            tok_per_s = tokens_done / dt
            tokens_done = 0
            t_log = now
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step {global_step:6d} | loss {step_loss:.4f} (ema {running_loss:.4f}) | "
                f"CE bpb {step_ce_bpb:.4f} | BPD {step_bpd_bits:.4f} | "
                f"lr {lr:.6g} | tok/s {tok_per_s:,.0f}"
            )

        if global_step % exp_config.gen_every == 0:
            log_generation(model, exp_config, device)

    # Final sample
    log_generation(model, exp_config, device)
    print("Training complete.")


if __name__ == "__main__":
    cfg = ExpConfig(
        # tweak these as you like:
        batch_size=16,
        n_layers=12,
        grad_accum_steps=1,
        seq_len=2048,
        num_training_steps=200000,
        lr=3e-4,
        scheduler_type="cosine",  # e.g., "linear", "cosine", "cosine_with_restarts"
        warmup_steps=200,
        gen_every=300,
        max_new_tokens=200,
        gen_prompt="Once upon a time",
        n_components=2,  # Gaussian components
    )
    train(cfg)
