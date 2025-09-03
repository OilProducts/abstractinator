#  EXPERIMENTAL

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from components import SwiGLU
from components.rope import RoPECache, apply_rope


# ---------- math helpers ----------
def gaussian_bpd(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Diagonal-Gaussian bits-per-dim per step: (B,L,D)->(B,L)."""
    const = 0.5 * math.log(2 * math.pi)
    nll = 0.5 * ((target - mu) ** 2 * torch.exp(-logvar) + logvar) + const  # (B,L,D) in nats
    bpd = (nll / math.log(2.0)).mean(dim=-1)  # (B,L)
    return bpd


def gaussian_entropy_bits(logvar: torch.Tensor) -> torch.Tensor:
    """Predicted differential entropy per step in bits: (B,L,D)->(B,L)."""
    B, L, D = logvar.shape
    const = 0.5 * (D * math.log(2 * math.e * math.pi))
    H = const + 0.5 * logvar.sum(dim=-1)  # nats
    return H / math.log(2.0)  # bits


def l2_prototype_logits(z: torch.Tensor, E: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    Prototype (distance) logits over vocabulary from latent(s) z.
    z: (B,D) or (B,L,D); E: (V,D); logits ∝ -||z - E_i||^2 / tau.
    """
    (z * z).sum(dim=-1, keepdim=True)  # (B,1) or (B,L,1)
    E2 = (E * E).sum(dim=-1).view(1, *([1] * (z.dim() - 2)), -1)  # broadcast to (...,V)
    Ez = z @ E.T  # (...,V)
    logits = (2.0 * Ez - E2.squeeze(0)) / tau  # drop z2 (constant wrt i)
    return logits


def gaussian_token_logits(mu, logvar, E, ln_layer):  # mu, logvar: (B,D)
    # Apply the SAME LayerNorm to prototypes that you used during training.
    proto = ln_layer(E.weight)  # (V,D)
    invvar = torch.exp(-logvar).unsqueeze(1)  # (B,1,D)
    diff = proto.unsqueeze(0) - mu.unsqueeze(1)  # (B,V,D)
    # Drop constants in i (the -0.5*sum(logvar) term is common across tokens).
    logits = -0.5 * (diff.pow(2) * invvar).sum(-1)  # (B,V)
    return logits


# ---------- masks ----------
def _causal_mask(S: int, device) -> torch.Tensor:
    """Additive causal mask (S,S): 0 allowed, -inf future."""
    m = torch.zeros(S, S, device=device)
    m = m.masked_fill(torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), 1), float('-inf'))
    return m


def _valid_step_mask(attn_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    attn_mask: (B,S) bool where True means KEEP (valid token), False means PAD.
    Returns mask for positions 1..S-1 that have valid current AND previous token: (B,S-1) bool.
    """
    if attn_mask is None:
        return None
    prev = attn_mask[:, :-1]
    curr = attn_mask[:, 1:]
    return prev & curr


class MHA_RoPE(nn.Module):
    """
    Multi-head self-attention with RoPE on Q/K.
    - Pre-norm outside in block
    - Uses scaled_dot_product_attention with key-padding masking + causal
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert (self.head_dim % 2) == 0, "RoPE expects even head dim"

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # (B, L, H)
        attn_keep_mask: Optional[torch.Tensor],  # (B, L) bool, True=keep (valid), False=pad
        rope: "RoPECache",
    ) -> torch.Tensor:
        B, L, H = x.shape
        qkv = self.qkv(x)  # (B, L, 3H)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (B, h, L, d)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # RoPE on Q/K
        cos, sin = rope.slice(L)  # (1,1,L,d/2) views
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Key-padding mask for keys: mask True means "do NOT attend to this key"
        # Broadcast to (B, 1, 1, L) so it applies to all heads & all queries.
        attn_mask = None
        if attn_keep_mask is not None:
            key_mask = (~attn_keep_mask).view(B, 1, 1, L)  # True at pads along key axis
            attn_mask = key_mask

        # SDPA handles scaling internally; combine with causal masking implicitly.
        # (We pass is_causal=True; attn_mask masks keys; queries that are pads get zeroed after.)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,  # boolean mask, broadcastable
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=True,
        )  # (B, h, L, d)

        out = attn.transpose(1, 2).reshape(B, L, H)  # (B, L, H)

        # Zero out outputs for padded query rows (stability/consistency).
        if attn_keep_mask is not None:
            out = out.masked_fill((~attn_keep_mask).unsqueeze(-1), 0.0)

        return self.resid_drop(self.out_proj(out))


class TransformerBlockRoPE(nn.Module):
    """Pre-Norm block: LN -> MHA_RoPE -> residual; LN -> SwiGLU -> residual."""

    def __init__(self, d_model: int, n_heads: int, mlp_hidden_dim: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = MHA_RoPE(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.mlp = SwiGLU(d_model, hidden_dim=mlp_hidden_dim)
        self.mlp_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_keep_mask: Optional[torch.Tensor], rope: "RoPECache") -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_keep_mask, rope)
        x = x + self.mlp_drop(self.mlp(self.ln2(x)))
        return x


class CausalGaussianCoreRoPE(nn.Module):
    """
    Causal encoder predicting μ/logσ²/π for next-step embedding, with RoPE inside attention.
    Compatible with your previous core's API (now with mixtures by setting n_components > 1).
    """

    def __init__(
        self,
        dim: int,
        d_model: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        clamp_logvar: Tuple[float, float] = (-6.0, 8.0),
        dropout: float = 0.0,
        n_components: int = 1,
        max_seq_len: int = 8192,
        rope_base: float = 10000.0,
        rope_dtype: torch.dtype = torch.bfloat16,
        mlp_hidden_dim: Optional[int] = None,  # default ~LLaMA-style if None
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert (d_model // n_heads) % 2 == 0, "RoPE head_dim must be even"

        self.in_proj = nn.Linear(dim, d_model, bias=False)

        if mlp_hidden_dim is None:
            # LLaMA-style SwiGLU width ≈ 8/3 * d_model keeps params near GELU-4x MLP
            mlp_hidden_dim = int((8 * d_model) / 3)

        self.blocks = nn.ModuleList(
            [TransformerBlockRoPE(d_model, n_heads, mlp_hidden_dim, dropout) for _ in range(n_layers)]
        )

        # Heads for mixture stats
        self.K = int(n_components)
        self.D = dim
        self.mu = nn.Linear(d_model, dim * self.K)
        self.logvar = nn.Linear(d_model, dim * self.K)
        self.pi = nn.Linear(d_model, self.K) if self.K > 1 else None
        self.clamp = clamp_logvar

        # RoPE cache (buffers move with .to(device) on this module)
        self.rope = RoPECache(
            max_seqlen=max_seq_len,
            head_dim=d_model // n_heads,
            base=rope_base,
            dtype=rope_dtype,
            device="cuda",  # safe default; will follow module.to(device)
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, S, D) embeddings
        attn_mask: Optional[torch.Tensor] = None,  # (B, S) bool True=keep
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
          mu:      (B, S-1, K, D)
          logvar:  (B, S-1, K, D)
          pi_logit:(B, S-1, K) or None if K==1
        """
        B, S, D = x.shape
        assert S >= 2, "Need S>=2"
        # We predict next at positions 1..S-1; encode prefixes 0..S-2
        src = self.in_proj(x[:, :-1, :])  # (B, S-1, H)

        keep = attn_mask[:, :-1] if attn_mask is not None else None  # (B, S-1) or None
        h = src
        for blk in self.blocks:
            h = blk(h, keep, self.rope)  # (B, S-1, H)

        mu = self.mu(h).view(B, S - 1, self.K, self.D)
        logvar = self.logvar(h).view(B, S - 1, self.K, self.D).clamp(*self.clamp)
        pi_logit = self.pi(h) if self.K > 1 else None
        return mu, logvar, pi_logit


class ByteGaussianProbe(nn.Module):
    """
    Byte-level Gaussian entropy probe.
      • Input: tokens (B,S) int; attn_mask (B,S) bool, True=valid
      • Embedding → LayerNorm → CausalGaussianCore → μ, logσ²
    """

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 512,
        d_model: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        share_embedding: Optional[nn.Embedding] = None,
        dropout: float = 0.0,
        n_components: int = 1,
    ):
        super().__init__()
        self.embedding = share_embedding or nn.Embedding(vocab_size, dim)
        self.norm = nn.LayerNorm(dim, elementwise_affine=True)
        self.core = CausalGaussianCoreRoPE(dim, d_model, n_layers, n_heads, dropout=dropout, n_components=n_components)

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(self.embedding(tokens))

    # # ----- training loss -----
    def loss_bpd(
        self,
        tokens: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        detach_inputs: bool = False,
        detach_targets: bool = True,
    ) -> torch.Tensor:
        x = self._embed(tokens)
        if detach_inputs:
            x = x.detach()
        mu, logvar, pi_logit = self.core(x, attn_mask=attn_mask)  # (B,S-1,K,D), (..), (B,S-1,K) or None

        tgt = self._embed(tokens)[:, 1:, :]
        tgt = tgt.detach() if detach_targets else tgt

        if self.core.K == 1:
            bpd = gaussian_bpd(mu.squeeze(2), logvar.squeeze(2), tgt)  # (B,S-1)
        else:
            bpd = mog_bpd(mu, logvar, pi_logit, tgt)  # (B,S-1)

        vm = _valid_step_mask(attn_mask)  # (B,S-1) or None
        if vm is not None:
            bpd = bpd.masked_select(vm)
        return bpd.mean()

    @torch.no_grad()
    def step_bpd(self, tokens: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self._embed(tokens)
        mu, logvar, pi_logit = self.core(x, attn_mask=attn_mask)
        B, S = tokens.size()
        if self.core.K == 1:
            bpd = gaussian_bpd(mu.squeeze(2), logvar.squeeze(2), x[:, 1:, :])  # (B,S-1)
        else:
            bpd = mog_bpd(mu, logvar, pi_logit, x[:, 1:, :])  # (B,S-1)
        out = torch.zeros(B, S, device=tokens.device)
        out[:, 1:] = bpd
        out[:, 0] = bpd[:, 0] if bpd.size(1) > 0 else 0.0
        return out

    @torch.no_grad()
    def step_entropy(self, tokens: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self._embed(tokens)
        mu, logvar, pi_logit = self.core(x, attn_mask=attn_mask)
        B, S1, K, D = mu.shape
        out = torch.zeros(B, S1 + 1, device=tokens.device)
        if self.core.K == 1:
            H = gaussian_entropy_bits(logvar.squeeze(2))  # exact for single Gaussian
            out[:, 1:] = H
            out[:, 0] = H[:, 0] if H.size(1) > 0 else 0.0
            return out
        # Upper-ish bound: H(Z) ≤ E_k[H(N_k)] + H(pi)
        w = pi_logit.softmax(-1)  # (B,S1,K)
        Hk_nats = 0.5 * (D * (1.0 + math.log(2 * math.pi))) + 0.5 * logvar.sum(-1)  # (B,S1,K)
        Hmix_nats_ub = (w * Hk_nats).sum(-1) - (w * (w.clamp_min(1e-12)).log()).sum(-1)  # +H(pi)
        Hmix_bits_ub = Hmix_nats_ub / math.log(2.0)  # (B,S-1)
        out[:, 1:] = Hmix_bits_ub
        out[:, 0] = Hmix_bits_ub[:, 0] if Hmix_bits_ub.size(1) > 0 else 0.0
        return out


# ---------- segmentation ----------
@dataclass
class SegmentationCfg:
    threshold: float  # BPD (if use_bpd) or bits (if use_entropy)
    use_bpd: bool = True
    min_len: int = 1
    max_len: Optional[int] = None
    smooth: Optional[int] = None  # causal SMA window


@torch.no_grad()
@torch._dynamo.disable()
def segment_greedy(
    tokens: torch.Tensor, probe: ByteGaussianProbe, cfg: SegmentationCfg, attn_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Greedy entropy/BPD thresholding. Returns (end_mask, score) with shapes (B,S)."""
    score = probe.step_bpd(tokens, attn_mask) if cfg.use_bpd else probe.step_entropy(tokens, attn_mask)

    if cfg.smooth and cfg.smooth > 1:
        k = cfg.smooth
        pad = (k - 1, 0)  # causal
        w = torch.ones(1, 1, k, device=tokens.device) / k
        s = F.conv1d(score.unsqueeze(1), w, padding=pad)[..., : score.size(1)]
        score = s.squeeze(1)

    B, S = tokens.shape
    end_mask = torch.zeros(B, S, dtype=torch.bool, device=tokens.device)

    for b in range(B):
        start = 0
        while start < S:
            stop = min(S - 1, start + cfg.min_len - 1)
            while True:
                if cfg.max_len is not None and (stop - start + 1) >= cfg.max_len:
                    break
                nxt = stop + 1
                if nxt >= S or (attn_mask is not None and not attn_mask[b, nxt]):
                    break
                if score[b, nxt] > cfg.threshold:
                    stop = nxt
                    break
                stop = nxt
            end_mask[b, stop] = True
            start = stop + 1
    return end_mask, score


@torch.no_grad()
def segment_quantile(
    tokens: torch.Tensor,
    probe: ByteGaussianProbe,
    quantile: float = 0.70,
    use_bpd: bool = True,
    smooth: Optional[int] = 5,
    attn_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Quantile-based thresholding (auto-calibrated). Returns (end_mask, score, threshold).
    """
    score = probe.step_bpd(tokens, attn_mask) if use_bpd else probe.step_entropy(tokens, attn_mask)
    if smooth and smooth > 1:
        k = smooth
        pad = (k - 1, 0)
        w = torch.ones(1, 1, k, device=tokens.device) / k
        s = F.conv1d(score.unsqueeze(1), w, padding=pad)[..., : score.size(1)]
        score = s.squeeze(1)
    # compute quantile over valid steps only
    if attn_mask is not None:
        vm = _valid_step_mask(attn_mask)
        vals = score[:, 1:][vm].detach()
    else:
        vals = score[:, 1:].reshape(-1).detach()
    thr = torch.quantile(vals, q=quantile).item()
    cfg = SegmentationCfg(threshold=thr, use_bpd=use_bpd, min_len=1, max_len=None, smooth=None)
    end_mask, _ = segment_greedy(tokens, probe, cfg, attn_mask)
    return end_mask, score, thr


# ---------- generation ----------
@dataclass
class GenCfg:
    max_new_tokens: int
    tau_proto: float = 0.8  # prototype-logits temperature
    sample: bool = True  # True: sample; False: greedy
    topk: Optional[int] = None
    topp: Optional[float] = None
    gaussian_temp: float = 1.0  # scale σ (sqrt of var) → exploration in latent space
    entropy_cond_temp: Optional[Tuple[float, float]] = None
    n_components: int = 1
    # entropy_cond_temp=(a,b): τ_proto = clamp(a + b * (H_t / H_ref), τ_min, τ_max)


def _top_filter(logits: torch.Tensor, topk: Optional[int], topp: Optional[float]) -> torch.Tensor:
    if topk is not None and topk > 0:
        v, _ = torch.topk(logits, k=min(topk, logits.size(-1)))
        cutoff = v[..., -1, None]
        logits = torch.where(logits >= cutoff, logits, torch.full_like(logits, float('-inf')))
    if topp is not None and 0 < topp < 1:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumprob = sorted_logits.softmax(-1).cumsum(-1)
        mask = cumprob > topp
        mask[..., 0] = False
        to_mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_idx, mask)
        logits = logits.masked_fill(to_mask, float('-inf'))
    return logits


@torch.no_grad()
def generate_with_probe(
    probe: ByteGaussianProbe,
    E: nn.Embedding,
    prefix: torch.Tensor,  # (B,T0)
    cfg: GenCfg,
    attn_mask: Optional[torch.Tensor] = None,
    n_components: int = 1,
) -> torch.Tensor:
    """
    Autoregressive generation using the probe's predictive Gaussian:
      1) get μ_t, logσ²_t for next embedding
      2) z ~ N(μ_t, σ²) (or z = μ_t)
      3) map z → token via prototype logits (distance to E)
    """
    device = prefix.device
    tokens = prefix.clone()
    for _ in range(cfg.max_new_tokens):
        x = probe.norm(E(tokens))  # (B,T,D)
        mu, logvar, pi_logit = probe.core(x, attn_mask=attn_mask) if x.size(1) > 1 else (None, None, None)

        if x.size(1) == 1:
            # cold start defaults
            B, D = tokens.size(0), probe.embedding.embedding_dim
            K = getattr(probe.core, "K", 1)
            mu_next = torch.zeros(B, K, D, device=x.device)
            logvar_next = torch.zeros_like(mu_next)
            pi_next = torch.zeros(B, K, device=x.device)  # uniform after softmax
        else:
            if probe.core.K == 1:
                mu_next = mu[:, -1, 0, :]  # (B,D) -> handled below
                logvar_next = logvar[:, -1, 0, :]  # (B,D)
            else:
                mu_next = mu[:, -1, :, :]  # (B,K,D)
                logvar_next = logvar[:, -1, :, :]  # (B,K,D)
                pi_next = pi_logit[:, -1, :]  # (B,K)

        # Build logits
        if getattr(probe.core, "K", 1) == 1:
            logits = gaussian_token_logits(mu_next, logvar_next, E, probe.norm)  # (B,V)
        else:
            logits = mog_token_logits(mu_next, logvar_next, pi_next, E, probe.norm)  # (B,V)

        assert torch.isfinite(logits).all(), f"non-finite logits: min={logits.min().item()}, max={logits.max().item()}"

        # Inspect mixture stats at the last step:
        print("min/max logvar last step:", logvar_next.min().item(), logvar_next.max().item())

        if cfg.n_components > 1:
            w = pi_next.softmax(-1)
            H_nats = -(w * (w.clamp_min(1e-12)).log()).sum(-1)  # (B,)
            mix_perplexity = H_nats.exp()
            print("pi perplexity last step:", mix_perplexity.item())

        logits = _top_filter(logits, cfg.topk, cfg.topp)
        probs = logits.softmax(dim=-1)
        next_tok = torch.multinomial(probs, 1) if cfg.sample else probs.argmax(dim=-1, keepdim=True)

        tokens = torch.cat([tokens, next_tok], dim=1)
        if attn_mask is not None:
            pad_col = torch.ones(tokens.size(0), 1, dtype=attn_mask.dtype, device=device)
            attn_mask = torch.cat([attn_mask, pad_col.bool()], dim=1)
    return tokens


def mog_token_logits(
    mu: torch.Tensor, logvar: torch.Tensor, pi_logit: torch.Tensor, E: nn.Embedding, ln: nn.LayerNorm
) -> torch.Tensor:
    """
    Mixture Mahalanobis logits over tokens.
    mu:      (B,K,D)
    logvar:  (B,K,D)
    pi_logit:(B,K)
    Returns: logits (B,V) proportional to log p(token | mixture), up to a global constant.
    """
    out_dtype = mu.dtype
    with torch.amp.autocast(enabled=False, device_type='cuda'):  # disable autocast here
        proto = ln(E.weight).float()  # (V,D)
        mu32, lv32, pi32 = mu.float(), logvar.float(), pi_logit.float()
        invvar = torch.exp(-lv32)  # (B,K,D)
        diff = proto.unsqueeze(0).unsqueeze(0) - mu32.unsqueeze(2)  # (B,K,V,D)
        energy = -0.5 * (diff.pow(2) * invvar.unsqueeze(2)).sum(-1) - 0.5 * lv32.sum(-1, keepdim=True)  # (B,K,V)
        logpi = pi32.log_softmax(dim=-1).unsqueeze(-1)  # (B,K,1)
        logits32 = torch.logsumexp(logpi + energy, dim=1)  # (B,V)
    return logits32.to(out_dtype)


def mog_bpd(mu: torch.Tensor, logvar: torch.Tensor, pi_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mixture NLL → bits-per-dim per step.
    mu/logvar: (B,S-1,K,D) , pi_logit: (B,S-1,K), target: (B,S-1,D)
    Returns: (B,S-1)
    """
    B, S1, K, D = mu.shape
    invvar = torch.exp(-logvar)  # (B,S1,K,D)
    diff2 = (target.unsqueeze(2) - mu).pow(2) * invvar  # (B,S1,K,D)
    quad = diff2.sum(-1)  # (B,S1,K)
    sum_logvar = logvar.sum(-1)  # (B,S1,K)
    logpi = F.log_softmax(pi_logit, dim=-1)  # (B,S1,K)
    const = 0.5 * D * math.log(2 * math.pi)
    # log p(z) = logsumexp_k [ logpi_k - 0.5*(quad_k + sum_logvar_k) - const ]
    logp = torch.logsumexp(logpi - 0.5 * (quad + sum_logvar) - const, dim=-1)  # (B,S1)
    nll = -logp
    bpd = (nll / math.log(2.0)) / D
    return bpd


def gaussian_token_logp(mu: torch.Tensor, logvar: torch.Tensor, E: nn.Embedding, ln: nn.LayerNorm) -> torch.Tensor:
    """
    Single-component anisotropic log-probabilities up to a global constant.
    Includes the row-wise constant term (-0.5 * sum(logvar)). Use when actual
    log-likelihood magnitudes matter (e.g., reporting/calibration). For
    softmax/CE or ranking, prefer gaussian_token_logits defined above.

    mu/logvar: (B,D)
    Returns: (B,V) tensor of log-probabilities up to a constant per-row.
    """
    B, D = mu.shape
    proto = ln(E.weight)  # (V,D)
    diff = proto.unsqueeze(0) - mu.unsqueeze(1)  # (B,V,D)
    invvar = torch.exp(-logvar).unsqueeze(1)  # (B,1,D)
    logits = -0.5 * (diff.pow(2) * invvar).sum(-1) - 0.5 * logvar.sum(-1, keepdim=True)  # (B,V)
    return logits  # global const dropped (same across tokens)


def mog_sequence_logits(mu, logvar, pi_logit, E, ln):
    """
    Vectorized sequence logits for a mixture of diagonal Gaussians.
    Shapes:
      mu, logvar:   (B, S1, K, D)
      pi_logit:     (B, S1, K)   (if K==1, pass zeros)
      E.weight:     (V, D)
    Returns:
      logits:       (B, S1, V)
    """
    B, S1, K, D = mu.shape
    V = E.num_embeddings
    # Prototypes in the SAME space used for training
    P = ln(E.weight)  # (V, D)
    Q = P * P  # (V, D)

    invvar = torch.exp(-logvar)  # (B,S1,K,D)
    a = invvar * mu  # (B,S1,K,D)
    b = 0.5 * invvar  # (B,S1,K,D)

    # Flatten (B,S1,K, D) -> (BSK, D) and do two GEMVs per component
    A = a.reshape(-1, D)  # (BSK, D)
    Bv = b.reshape(-1, D)  # (BSK, D)

    # term1 = P @ a   and  term2 = Q @ (0.5*invvar)
    term1 = A @ P.t()  # (BSK, V)
    term2 = Bv @ Q.t()  # (BSK, V)

    const = -0.5 * ((mu * mu * invvar).sum(-1) + logvar.sum(-1))  # (B,S1,K)
    energy = term1 - term2 + const.reshape(-1, 1)  # (BSK, V)
    energy = energy.view(B, S1, K, V)  # (B,S1,K,V)

    logpi = F.log_softmax(pi_logit, dim=-1).unsqueeze(-1)  # (B,S1,K,1)
    logits = torch.logsumexp(logpi + energy, dim=2)  # (B,S1,V)
    return logits


def token_ce_loss_from_mog(probe, tokens, attn_mask=None):
    """
    Cross-entropy over next-token using mixture logits.
    Uses the same core forward pass; safe for K=1 by passing pi_logit=0.
    """
    x = probe._embed(tokens)  # (B,S,D)
    mu, logvar, pi_logit = probe.core(x, attn_mask=attn_mask)  # (B,S-1,K,D), (..), (B,S-1,K|None)

    if pi_logit is None:  # K==1
        pi_logit = torch.zeros(mu.shape[:3], device=mu.device)  # (B,S-1,1)

    logits = mog_sequence_logits(mu, logvar, pi_logit, probe.embedding, probe.norm)  # (B,S-1,V)
    tgt = tokens[:, 1:]  # (B,S-1)

    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction='none')
    if attn_mask is not None:
        vm = (attn_mask[:, 1:] & attn_mask[:, :-1]).reshape(-1)
        loss = (loss * vm).sum() / vm.sum().clamp_min(1)
    else:
        loss = loss.mean()
    return loss
