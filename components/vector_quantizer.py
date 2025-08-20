import logging
from typing import Tuple, List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch._dynamo as dynamo


class VectorQuantizer(nn.Module):
    """
    Compile‑friendly VQ with EMA and vectorized, guard‑stable code resets.

    Key properties:
      • No Python mirrors used in forward() – only device tensors.
      • No Python branching on GPU scalars (.item()).
      • Resets run with constant shapes (R_MAX), gated by a 0‑dim tensor.
      • Same API: forward(z) -> (z_q_ste, vq_loss, indices, perplexity)
    """

    def __init__(
        self,
        K: int,
        D: int,
        beta: float = 0.25,
        ema: bool = True,
        decay: float = 0.999,
        eps: float = 1e-5,
        reset_codes: bool = True,
        reset_interval: int = 250,
        max_codes_to_reset_pct: float = 0.1,
        replacement_buffer_size: int = 65536,
        vectors_per_step_to_buffer: int = 1024,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        padding_token_id: int = 2,
        eop_token_id: int = 3,
    ):
        super().__init__()
        self.K = int(K)
        self.D = int(D)
        self.beta = float(beta)
        self.ema = bool(ema)
        self.decay = float(decay)
        self.eps = float(eps)

        self.reset_codes = bool(reset_codes)
        self.reset_interval = int(reset_interval)
        self.max_codes_to_reset_pct = float(max_codes_to_reset_pct)
        self.replacement_buffer_size = int(replacement_buffer_size)
        self.vectors_per_step_to_buffer = int(vectors_per_step_to_buffer)

        # specials
        self.bos_token_id = int(bos_token_id)
        self.eos_token_id = int(eos_token_id)
        self.padding_token_id = int(padding_token_id)
        self.eop_token_id = int(eop_token_id)

        # Codebook and EMA stats
        self.codebook = nn.Parameter(torch.randn(self.K, self.D))
        if self.ema:
            self.register_buffer("ema_cluster_size", torch.zeros(self.K))
            self.register_buffer("ema_weight_sum", self.codebook.data.clone())

        # Replacement/reset machinery as DEVICE STATE (no Python mirrors)
        if self.reset_codes:
            # circular buffer of recent vectors
            self.register_buffer(
                "replacement_buffer",
                torch.empty(self.replacement_buffer_size, self.D)
            )
            self.replacement_buffer.zero_()
            # 0‑dim device counters/flags
            self.register_buffer("buffer_idx", torch.zeros((), dtype=torch.long))
            self.register_buffer("buffer_is_full", torch.zeros((), dtype=torch.bool))
            self.register_buffer("steps_since_last_reset", torch.zeros((), dtype=torch.long))
            # dead threshold
            self.min_usage_threshold = 1.0
            # constant reset budget (shape‑stable)
            self.R_MAX = int(max(0, round(self.K * self.max_codes_to_reset_pct)))

    # ---------------------------
    # Internal helpers (tensor‑only)
    # ---------------------------

    @torch.no_grad()
    def _update_replacement_buffer_tensor(self, flat_input: torch.Tensor) -> None:
        """Device‑only circular buffer update. No Python guards, no graph breaks."""
        N = flat_input.size(0)
        if N == 0:
            return

        # optional subsample to cap copy cost
        if N > self.vectors_per_step_to_buffer:
            idx = torch.randperm(N, device=flat_input.device)[: self.vectors_per_step_to_buffer]
            flat_input = flat_input.index_select(0, idx)
            N = flat_input.size(0)

        B = self.replacement_buffer_size  # Python constant
        idx0 = self.buffer_idx            # 0‑dim LongTensor on device

        write_idx = (idx0 + torch.arange(N, device=flat_input.device, dtype=torch.long)) % B
        self.replacement_buffer.index_copy_(0, write_idx, flat_input)

        # advance head and set 'full' flag if wrapped
        new_idx = (idx0 + N) % B
        self.buffer_idx.copy_(new_idx)

        wrapped = (idx0 + N) >= B                # 0‑dim bool
        self.buffer_is_full.logical_or_(wrapped) # in‑place OR

    @torch.no_grad()
    def _ema_update(self, encodings: torch.Tensor, flat_input: torch.Tensor) -> None:
        """EMA updates (tensor‑only)."""
        enc = encodings
        # mask specials in EMA stats
        if self.eop_token_id < enc.size(1):
            enc = enc.clone()
            enc[:, self.eop_token_id] = 0
        if self.padding_token_id < enc.size(1):
            if enc is encodings:
                enc = enc.clone()
            enc[:, self.padding_token_id] = 0

        dw = enc.transpose(0, 1) @ flat_input    # (K,D)
        cluster = enc.sum(0)                     # (K,)

        self.ema_cluster_size.mul_(self.decay).add_(cluster, alpha=1 - self.decay)
        self.ema_weight_sum.mul_(self.decay).add_(dw, alpha=1 - self.decay)

        n = self.ema_cluster_size.sum()
        stabilized = ((self.ema_cluster_size + self.eps) / (n + self.K * self.eps)) * n
        self.codebook.data.copy_(self.ema_weight_sum / stabilized.unsqueeze(1))


    @torch.no_grad()
    def _maybe_vectorized_reset(self) -> None:
        """
        Cheap, guard-stable resets:
          • runs every step (no Python guards), but O(R_MAX) work only
          • no Python .item() / .any() on device tensors
          • constant shapes: candidates = R_MAX
        """
        if not self.reset_codes or self.R_MAX <= 0:
            return

        # step counter & gate
        self.steps_since_last_reset.add_(1)
        do_reset = self.steps_since_last_reset >= self.reset_interval  # 0-dim bool on device

        # Quick exit via masking (no Python if): if not do_reset, all applies will be False
        # Replacement buffer “valid count”
        B = self.replacement_buffer_size
        valid_count = torch.where(self.buffer_is_full,
                                  torch.tensor(B, device=self.buffer_idx.device),
                                  self.buffer_idx)  # 0-dim long
        has_source = valid_count > 0  # 0-dim bool

        # Sample R_MAX candidate code rows (constant shape)
        rand_idx = torch.randint(0, self.K, (self.R_MAX,), device=self.codebook.device)

        # Which of those candidates are dead? (exclude specials)
        if self.ema:
            usage = self.ema_cluster_size
        else:
            usage = torch.ones(self.K, device=self.codebook.device)

        dead_mask = usage < self.min_usage_threshold
        if self.eop_token_id < self.K:
            dead_mask[self.eop_token_id] = False
        if self.padding_token_id < self.K:
            dead_mask[self.padding_token_id] = False

        # Apply mask for sampled rows only; also gate by do_reset & has_source
        apply_rows = dead_mask[rand_idx]
        apply_rows &= do_reset
        apply_rows &= has_source

        # Sample replacement vectors (constant shape), modulo valid_count (safe even if buffer not full)
        if self.R_MAX > 0:
            # randint high must be Python int; then mod by runtime valid_count
            rbuf_idx = torch.randint(0, B, (self.R_MAX,), device=self.replacement_buffer.device)
            valid_count_safe = torch.clamp(valid_count, min=1)  # avoid div by zero
            rbuf_idx = rbuf_idx % valid_count_safe
            repl = self.replacement_buffer.index_select(0, rbuf_idx)  # (R_MAX, D)
        else:
            repl = torch.empty(0, self.D, device=self.codebook.device)

        # Read current candidate rows and build their new values (row-wise where)
        cand_old = self.codebook.index_select(0, rand_idx)  # (R_MAX, D)
        cand_new = torch.where(apply_rows.unsqueeze(1), repl, cand_old)

        # Write back candidate rows only (constant-length index_copy_)
        self.codebook.index_copy_(0, rand_idx, cand_new)

        if self.ema:
            # ema_cluster_size: set to 1.0 for rows we actually reset
            old_cnt = self.ema_cluster_size.index_select(0, rand_idx)
            new_cnt = torch.where(apply_rows, torch.ones_like(old_cnt), old_cnt)
            self.ema_cluster_size.index_copy_(0, rand_idx, new_cnt)

            # ema_weight_sum: mirror codebook rows for rows we reset
            old_sum = self.ema_weight_sum.index_select(0, rand_idx)
            new_sum = torch.where(apply_rows.unsqueeze(1), cand_new, old_sum)
            self.ema_weight_sum.index_copy_(0, rand_idx, new_sum)

        # Zero the step counter when reset actually fired (tensor math; no Python if)
        keep = (~do_reset).to(self.steps_since_last_reset.dtype)
        self.steps_since_last_reset.mul_(keep)

    # ---------------------------
    # Forward
    # ---------------------------

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z: (B, Q, D)
        returns:
            z_q_ste: (B, Q, D) straight-through quantized vectors
            vq_loss: scalar tensor
            indices: (B, Q) argmin ids
            perplexity: scalar tensor (excludes EOP/PAD)
        """
        B, Q, Din = z.shape
        assert Din == self.D, f"Expected D={self.D}, got {Din}"

        flat = z.reshape(-1, self.D)

        # Update replacement buffer (tensor-only)
        if self.training and self.reset_codes:
            self._update_replacement_buffer_tensor(flat.detach())

        # Nearest neighbors (squared L2)
        x2 = (flat * flat).sum(dim=1, keepdim=True)                    # (N,1)
        e2 = (self.codebook * self.codebook).sum(dim=1)                # (K,)
        xe = F.linear(flat, self.codebook)                             # (N,K)
        distances = x2 - 2 * xe + e2.unsqueeze(0)                      # (N,K)
        indices = distances.argmin(dim=1)                              # (N,)

        z_q = F.embedding(indices, self.codebook).view(B, Q, self.D)

        # VQ losses
        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z, z_q.detach())
        vq_loss = codebook_loss + (self.beta * commitment_loss)

        # Straight-through
        z_q_ste = z + (z_q - z).detach()

        # EMA + vectorized reset (no Python mirrors)
        if self.training:
            if self.ema:
                onehot = F.one_hot(indices, self.K).type_as(flat)      # (N,K)
                self._ema_update(onehot, flat)
            if self.reset_codes:
                self._maybe_vectorized_reset()

        # Perplexity (usage), exclude EOP/PAD
        if self.ema:
            counts = self.ema_cluster_size.clone()
        else:
            counts = torch.bincount(indices, minlength=self.K).to(flat.dtype)

        if self.eop_token_id < counts.size(0):
            counts[self.eop_token_id] = 0
        if self.padding_token_id < counts.size(0):
            counts[self.padding_token_id] = 0

        counts = counts.clamp(min=self.eps)
        probs = counts / (counts.sum() + self.eps)
        entropy = -(probs * probs.log()).sum()
        perplexity = entropy.exp().to(z.dtype).detach()

        return z_q_ste, vq_loss, indices.view(B, Q), perplexity


class MultiStageResidualVQ(nn.Module):
    """
    Residual VQ with N stages in a compressed space.

    - Uses a shared down-projection (D -> d_c) and tied up-projection (d_c -> D).
    - Each stage quantizes the residual; outputs are summed before up-projection.
    - Exposes a composed integer index per position so downstream components can
      treat it as a single discrete token in a space of size K**depth.
    - Provides a ``pad_vector`` sentinel in D-space used to short-circuit to a
      special padding ID without running normal quantization.
    """

    def __init__(
        self,
        K: int,
        D: int,
        depth: int = 2,
        d_c: int = 64,
        beta: float = 0.25,
        ema: bool = True,
        decay: float = 0.999,
        eps: float = 1e-5,
        reset_codes: bool = True,
        reset_interval: int = 250,
        max_codes_to_reset_pct: float = 0.1,
        replacement_buffer_size: int = 65536,
        vectors_per_step_to_buffer: int = 1024,
        bos_token_id: int = 256,
        eos_token_id: int = 257,
        padding_token_id: int = 258,
        eop_token_id: int = 259,
    ) -> None:
        super().__init__()
        # if depth < 2:
        #     raise ValueError("MultiStageResidualVQ requires depth >= 2")

        self.D = D
        self.d_c = d_c
        self.depth = int(depth)
        self.K = int(K)
        self.K_eff = int(self.K ** self.depth)

        # Special token ids (kept consistent with VectorQuantizer)
        self.bos_token_id = int(bos_token_id)
        self.eos_token_id = int(eos_token_id)
        self.padding_token_id = int(padding_token_id)
        self.eop_token_id = int(eop_token_id)

        # expose a codec so downstream code can (de)compose indices
        self.codec = ComposedIndexCodec(
            K=self.K, depth=self.depth,
            bos=self.bos_token_id, eos=self.eos_token_id,
            pad=self.padding_token_id, eop=self.eop_token_id,
        )

        # Allow small effective K; special tokens may be unused for code VQs

        # Shared down/up with tied weights
        self.down = nn.Linear(D, d_c, bias=False)
        self.up = nn.Linear(d_c, D, bias=False)
        # Initialize after orthogonal init (done below)

        vq_kwargs = dict(
            beta=beta,
            ema=ema,
            decay=decay,
            eps=eps,
            reset_codes=reset_codes,
            reset_interval=reset_interval,
            max_codes_to_reset_pct=max_codes_to_reset_pct,
            replacement_buffer_size=replacement_buffer_size,
            vectors_per_step_to_buffer=vectors_per_step_to_buffer,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            padding_token_id=padding_token_id,
            eop_token_id=eop_token_id,
        )
        self.stages = nn.ModuleList([VectorQuantizer(K=self.K, D=d_c, **vq_kwargs) for _ in range(self.depth)])

        # Sentinel pad vector in the D-space
        self.register_buffer("pad_vector", torch.zeros(D))

        # bfloat16‑friendly init: Xavier uniform for down, then copy transpose to up
        with torch.no_grad():
            nn.init.xavier_uniform_(self.down.weight)
            self.up.weight.copy_(self.down.weight.T.to(self.up.weight.dtype))

    def _compose_indices(self, idx_list: list[torch.Tensor]) -> torch.Tensor:
        base = 1
        composed = torch.zeros_like(idx_list[0])
        for i, idx in enumerate(idx_list):
            if i > 0:
                base = base * self.K
            composed = composed + idx * base
        return composed

    def forward(self, z: torch.Tensor):
        B, Q, _ = z.shape

        # Detect pad sentinels
        pad_vec = self.pad_vector.view(1, 1, -1)
        is_pad = (z == pad_vec).all(dim=-1)  # (B, Q)

        y = self.down(z)

        total_loss = torch.tensor(0.0, device=z.device)
        idx_list: list[torch.Tensor] = []
        ppl_list: list[torch.Tensor] = []
        q_sum = torch.zeros_like(y)
        r = y
        for stage in self.stages:
            q_i, loss_i, idx_i, ppl_i = stage(r)
            q_sum = q_sum + q_i
            total_loss = total_loss + loss_i
            idx_list.append(idx_i.view(-1))
            ppl_list.append(ppl_i)
            r = r - q_i.detach()

        z_hat = self.up(q_sum)
        idx_comp = self._compose_indices(idx_list).view(B, Q)

        if is_pad.any():
            idx_comp = idx_comp.masked_fill(is_pad, self.padding_token_id)
            z_hat = torch.where(is_pad.unsqueeze(-1), self.pad_vector.view(1, 1, -1), z_hat)

        perplexity = torch.ones((), device=z.device)
        for p in ppl_list:
            perplexity = perplexity * p

        return z_hat, total_loss, idx_comp, perplexity.detach()

    # convenience accessors
    def stage_codebook(self, s: int) -> torch.Tensor:
        return self.stages[s].codebook  # (K, d_c)

    @property
    def stage_codebooks(self) -> list[torch.Tensor]:
        return [st.codebook for st in self.stages]

class ComposedIndexCodec:
    def __init__(self, K: int, depth: int, bos: int, eos: int, pad: int, eop: int):
        self.K = int(K)
        self.depth = int(depth)
        self.special = {int(bos), int(eos), int(pad), int(eop)}
        self.special_list = sorted(list(self.special))
        self.special_to_local = {sid: i for i, sid in enumerate(self.special_list)}

    def is_special(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(x, dtype=torch.bool)
        for sid in self.special:
            mask |= (x == sid)
        return mask

    # Keep eager to avoid Inductor rewriting integer division into float paths
    @dynamo.disable()
    def decompose(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        x: (B, L) nonnegative composed ids
        returns:
          digits: list length=depth of (B, L) in [0..K-1]
          special_mask: (B, L) bool
        """
        x = x.long()
        device = x.device
        # base = [K^0, K^1, ..., K^(D-1)] as LONG
        base = (self.K ** torch.arange(0, self.depth, device=device, dtype=torch.long))  # (D,)
        # Integer division with explicit rounding keeps LONG dtype
        q = torch.div(x.unsqueeze(-1), base, rounding_mode='trunc')                      # (B,L,D) long
        digits = torch.remainder(q, self.K)                                             # (B,L,D) long
        out = [digits[..., d] for d in range(self.depth)]                               # list of (B,L) long
        special = self.is_special(x)
        return out, special

    @dynamo.disable()
    def compose(self, digits: List[torch.Tensor]) -> torch.Tensor:
        """
        digits: list length=depth of (B, L) in [0..K-1], long/int
        returns composed: (B, L) long
        """
        assert len(digits) == self.depth
        device = digits[0].device
        D = self.depth
        stacked = torch.stack([d.long() for d in digits], dim=-1)                       # (B,L,D) long
        base = (self.K ** torch.arange(0, D, device=device, dtype=torch.long))          # (D,) long
        composed = (stacked * base).sum(dim=-1)                                         # (B,L) long
        return composed.long()

    @dynamo.disable()
    def special_local_index(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        idx = torch.empty_like(x, dtype=torch.long)
        for sid, j in self.special_to_local.items():
            idx[x == sid] = j
        return idx



class RVQEmbeddingAdapter(nn.Module):
    """
    Bridges MultiStageResidualVQ to the expander.

    * Embeds discrete composed indices by summing stage code vectors in d_c, then up to D.
    * Optionally embeds specials with a tiny (num_special × D) table.
    * Exposes stage codebooks for factorized output heads.
    """
    def __init__(self,
                 vq: "MultiStageResidualVQ",
                 target_D: int,
                 use_shared_proj: bool = True,
                 tie_up_down: bool = False,
                 specials_in_D: bool = True):
        super().__init__()
        self.vq = vq
        self.K = vq.K
        self.depth = vq.depth
        self.d_c = vq.d_c
        self.D_vq = vq.D
        self.D = int(target_D)

        # projections
        share_ok = False #use_shared_proj and (self.D == self.D_vq)
        if share_ok:
            self.up = vq.up                       # d_c -> Das
            self.down = vq.down                   # D -> d_c
        else:
            # One shared parameter for both directions
            dtype = vq.up.weight.dtype
            device = vq.up.weight.device
            Wdc = nn.Parameter(torch.empty(self.d_c, self.D, dtype=dtype, device=device))
            nn.init.xavier_uniform_(Wdc)
            self.down = DownProj(Wdc)  # D -> d_c
            self.up = UpProj(Wdc)  # d_c -> D

        # tiny special table in target D
        self.special_ids = sorted({int(vq.bos_token_id),
                                   int(vq.eos_token_id),
                                   int(vq.padding_token_id),
                                   int(vq.eop_token_id)})
        self.special_to_local = {sid: i for i, sid in enumerate(self.special_ids)}
        self.special_emb = (nn.Embedding(len(self.special_ids), self.D)
                            if specials_in_D else None)


    @torch.no_grad()
    def stage_codebook(self, s: int) -> torch.Tensor:
        # Return d_c codebook of stage s, detached so no grads leak into VQ from expander.
        return self.vq.stages[s].codebook.detach()



    def embed_composed(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Compile-friendly embedding:
        - No boolean masked assignment
        - No .any() branches
        - All indices passed to Embedding are always in-range
        """
        idx = idx.long()
        B, L = idx.shape
        dev = idx.device
        dtype = self.up.Wdc.dtype

        # 1) Build special mask (no branching)
        special_mask = torch.zeros_like(idx, dtype=torch.bool)
        for sid in self.special_ids:
            special_mask |= (idx == sid)

        # 2) Non-special path for ALL positions (sanitize specials to 0 first)
        idx_ns = torch.where(special_mask, idx.new_zeros(()), idx)
        # decompose always on sanitized indices (safe)
        digits, _ = self.vq.codec.decompose(idx_ns)  # list len=depth, each (B,L) in [0..K-1]
        y_dc = torch.zeros(B, L, self.d_c, device=dev, dtype=dtype)
        for s in range(self.depth):
            W = self.stage_codebook(s)  # (K, d_c), detached in your adapter
            y_dc = y_dc + F.embedding(digits[s], W)
        out_ns = self.up(y_dc)  # (B,L,D)

        # 3) Specials path computed for ALL positions (indices always valid)
        if self.special_emb is None:
            # No branch on special_mask.any(); just return merged = out_ns
            return out_ns
        local = torch.zeros_like(idx)  # 0 everywhere, fill specials to local ids
        for sid, j in self.special_to_local.items():
            local = torch.where(idx == sid, local.new_full((), j), local)
        out_sp = self.special_emb(local)  # (B,L,D) safe for all positions

        # 4) Merge without masked assignment (pure where)
        return torch.where(special_mask.unsqueeze(-1), out_sp, out_ns)


class RVQFactorizedHead(nn.Module):
    """
    Produces stage-wise logits for a MultiStageResidualVQ:
      - Project decoder states h: D → d_c (tie to VQ.down)
      - If residual_conditioning=True:
            stage 0 sees y0
            stage 1 sees (y0 - e[i0]),
            stage 2 sees (y0 - e[i0] - e[i1]), ...
        where e[i] are stage code vectors (d_c).
      - Logits per stage are dot(y_res, codebook_s^T) or -||y_res - e||^2.
      - Tiny special head (optional).
    """
    def __init__(self,
                 adapter: RVQEmbeddingAdapter,
                 residual_conditioning: bool = True,
                 use_sqdist_logits: bool = False,
                 predict_specials: bool = True):
        super().__init__()
        self.adapt = adapter
        self.depth = adapter.depth
        self.K = adapter.K
        self.d_c = adapter.d_c
        self.D = adapter.D
        self.residual_conditioning = residual_conditioning
        self.use_sqdist_logits = use_sqdist_logits
        self.predict_specials = predict_specials
        self.has_specials = adapter.special_emb is not None and predict_specials
        self.special_head = nn.Linear(self.D, len(adapter.special_ids), bias=False) if self.has_specials else None

    def _stage_logits(self, r_dc: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        # r_dc: (B, L, d_c), codebook: (K, d_c)
        if self.use_sqdist_logits:
            # -||r - e||^2 = - (||r||^2 - 2 r·e + ||e||^2) → same ordering as dot, but better geometry
            r2 = (r_dc * r_dc).sum(-1, keepdim=True)                    # (B, L, 1)
            e2 = (codebook * codebook).sum(-1).view(1, 1, -1)          # (1, 1, K)
            dot = F.linear(r_dc, codebook)                              # (B, L, K)
            return -(r2 - 2*dot + e2)
        else:
            return F.linear(r_dc, codebook)  # (B, L, K)

    def forward(self,
                h: torch.Tensor,                    # (B, L, D) decoder hidden
                teacher_digits: Optional[List[torch.Tensor]] = None
                ) -> Dict[str, List[torch.Tensor] | torch.Tensor]:
        """
        Returns:
            {
              "stage_logits": [ (B,L,K) for s in 0..depth-1 ],
              "special_logits": (B,L,num_special) or None
            }
        """
        y0 = self.adapt.down(h)                      # (B, L, d_c)
        r = y0
        out: List[torch.Tensor] = []

        for s in range(self.depth):
            W = self.adapt.stage_codebook(s)         # (K, d_c)
            logits_s = self._stage_logits(r, W)      # (B, L, K)
            out.append(logits_s)

            if self.residual_conditioning:
                if teacher_digits is not None:
                    e = F.embedding(teacher_digits[s], W)  # (B, L, d_c)
                else:
                    # greedy residual update at inference
                    e = F.embedding(logits_s.argmax(dim=-1), W)
                r = r - e.detach()                   # stop gradients across stages

        special_logits = self.special_head(h) if self.has_specials else None
        return {"stage_logits": out, "special_logits": special_logits}

class LearnedCodebookAdapter(nn.Module):
    """
    Bottom-level adapter that behaves like RVQ with depth=1:
      - codebook E: (K, d_c)
      - tied projections via a single shared matrix Wdc: d_c<->D
      - exposes 'stage_codebook(0)' and 'depth' for the same interface as MS-RVQ adapters.
    """
    def __init__(self, K: int, D: int, d_c: int = 64,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None):
        super().__init__()
        self.K, self.D, self.d_c = int(K), int(D), int(d_c)
        # Codebook lives in d_c (factorized embedding)
        self.codebook = nn.Embedding(self.K, self.d_c,
                                     dtype=dtype, device=device)
        # Shared projection parameter Wdc (d_c x D)
        self.Wdc = nn.Parameter(torch.empty(self.d_c, self.D,
                                            dtype=dtype or self.codebook.weight.dtype,
                                            device=device or self.codebook.weight.device))
        # Tied modules
        self.down = DownProj(self.Wdc)   # D -> d_c
        self.up   = UpProj(self.Wdc)     # d_c -> D

        # Inits
        nn.init.normal_(self.codebook.weight, std=0.02)
        nn.init.xavier_uniform_(self.Wdc)

        # For API parity with RVQ adapter/head
        self.depth = 1
        self.special_ids = []            # bottom-level: put EOS/EOP inside [0..K-1]
        self.special_to_local = {}
        self.special_emb = None

    # ---- RVQ-compatible surface ----
    def stage_codebook(self, s: int = 0) -> torch.Tensor:
        assert s == 0, "LearnedCodebookAdapter has depth=1"
        return self.codebook.weight      # (K, d_c)

    def embed_composed(self, ids: torch.Tensor) -> torch.Tensor:
        """
        ids: (B, L) int64 in [0..K-1].
        Returns (B, L, D).
        """
        # Optional guard in eager:
        # if not torch._dynamo.is_compiling():
        #     assert ids.min().item() >= 0 and ids.max().item() < self.K, "byte id OOB"
        y_dc = self.codebook(ids)        # (B, L, d_c)
        return self.up(y_dc)             # (B, L, D)


class DownProj(nn.Module):         # D -> d_c
    def __init__(self, Wdc: nn.Parameter):
        super().__init__(); self.Wdc = Wdc        # shape (d_c, D)
    def forward(self, x):                          # F.linear: x @ Wdc^T
        return F.linear(x, self.Wdc)

class UpProj(nn.Module):           # d_c -> D
    def __init__(self, Wdc: nn.Parameter):
        super().__init__(); self.Wdc = Wdc        # same shared Parameter
    def forward(self, x):                          # x @ (Wdc^T)^T = x @ Wdc
        return F.linear(x, self.Wdc.t())
