import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# import torch._dynamo as dynamo

@torch.compile
class VectorQuantizer(nn.Module):
    """
    An efficient Vector Quantizer (VQ) layer with EMA updates and robust,
    performant code resetting.
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
        vectors_per_step_to_buffer: int = 1024,  # Controls update overhead
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        padding_token_id: int = 2,
        eop_token_id: int = 3,
    ):
        super().__init__()
        self.K = K
        self.D = D
        self.beta = beta
        self.ema = ema
        self.decay = decay
        self.eps = eps

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.eop_token_id = eop_token_id
        self.padding_token_id = padding_token_id

        # Allow small K in tests; special tokens may be unused for code VQs

        self.reset_codes = reset_codes
        if self.reset_codes:
            self.reset_interval = reset_interval
            self.max_codes_to_reset_pct = max_codes_to_reset_pct
            self.min_usage_threshold = 1.0
            self.register_buffer("steps_since_last_reset", torch.tensor(0, dtype=torch.int64))
            # Python mirror to avoid CUDA scalar sync on condition checks
            self._steps_since_last_reset_py: int = 0

            # Buffer settings
            self.replacement_buffer_size = replacement_buffer_size
            self.vectors_per_step_to_buffer = vectors_per_step_to_buffer
            self.register_buffer(
                "replacement_buffer",
                torch.empty(replacement_buffer_size, D).zero_(),
            )
            self.register_buffer("buffer_idx", torch.tensor(0, dtype=torch.long))
            self.register_buffer("buffer_is_full", torch.tensor(False, dtype=torch.bool))
            # Python mirrors to avoid reading CUDA scalars on the host
            self._buffer_idx_py: int = 0
            self._buffer_full_py: bool = False

        self.codebook = nn.Parameter(torch.randn(K, D))
        if ema:
            self.register_buffer("ema_cluster_size", torch.zeros(K))
            self.register_buffer("ema_weight_sum", self.codebook.data.clone())

    @torch.no_grad()
    @torch._dynamo.disable()
    def _update_replacement_buffer(self, new_vectors: torch.Tensor):
        """Efficiently updates the circular buffer with a subset of new vectors."""
        # --- PERFORMANCE FIX: Subsample vectors to reduce copy overhead ---
        if new_vectors.size(0) > self.vectors_per_step_to_buffer:
            perm = torch.randperm(new_vectors.size(0), device=new_vectors.device)[: self.vectors_per_step_to_buffer]
            new_vectors = new_vectors[perm]

        n_new = new_vectors.size(0)
        # buff_size = self.replacement_buffer_size
        # Avoid reading CUDA scalar with .item(); use Python mirror
        # current_idx = int(self._buffer_idx_py)

        B = self.replacement_buffer_size
        idx0 = self.buffer_idx  # scalar LongTensor on correct device

        # Ensure device match for the copy to avoid implicit sync paths
        # dest_dev = self.replacement_buffer.device
        # src = new_vectors.to(dest_dev, non_blocking=True)

        write_idx = (idx0 + torch.arange(n_new, device=new_vectors.device, dtype=idx0.dtype)) % B  # (n_new,)

        # If you guarantee n_new << B, this is fine; index_copy_ handles repeated idx
        self.replacement_buffer.index_copy_(0, write_idx, new_vectors)

        # Advance head and mark full if we wrapped
        self.buffer_idx.add_(n_new).remainder_(B)
        if not bool(self.buffer_is_full):
            # safe to use Python here; this function is compile-disabled
            if int(idx0) + int(n_new) >= int(B):
                self.buffer_is_full.fill_(True)

        # if n_new >= buff_size:
        #     self.replacement_buffer.copy_(src)
        #     self.buffer_is_full.fill_(True)
        #     self._buffer_full_py = True
        #     self._buffer_idx_py = 0
        #     self.buffer_idx.zero_()
        # else:
        #     space_left = buff_size - current_idx
        #     if n_new < space_left:
        #         self.replacement_buffer[current_idx : current_idx + n_new] = src
        #         self._buffer_idx_py += n_new
        #         self.buffer_idx.fill_(self._buffer_idx_py)
        #     else:
        #         self.replacement_buffer[current_idx:] = src[:space_left]
        #         self.replacement_buffer[: n_new - space_left] = src[space_left:]
        #         self._buffer_idx_py = n_new - space_left
        #         self.buffer_idx.fill_(self._buffer_idx_py)
        #         self.buffer_is_full.fill_(True)
        #         self._buffer_full_py = True

    @torch.no_grad()
    @torch._dynamo.disable()
    def _maybe_reset_dead_codes(self):
        if not self.reset_codes:
            return
        self.steps_since_last_reset.add_(1)
        if int(self.steps_since_last_reset) >= int(self.reset_interval):
            self._reset_dead_codes()  # you can also mark this @disable if you want
            self.steps_since_last_reset.zero_()


    @torch.no_grad()
    def _reset_dead_codes(self):
        """Resets dead codes using fast random sampling from the buffer."""
        if not self._buffer_full_py and self._buffer_idx_py == 0:
            return

        source_vectors = (
            self.replacement_buffer if self._buffer_full_py else self.replacement_buffer[: self._buffer_idx_py]
        )
        num_available_replacements = source_vectors.size(0)

        dead_code_candidates_indices = (self.ema_cluster_size < self.min_usage_threshold).nonzero(as_tuple=True)[0]
        dead_code_candidates_indices = dead_code_candidates_indices[dead_code_candidates_indices != self.eop_token_id]
        num_dead_candidates = dead_code_candidates_indices.size(0)
        if num_dead_candidates == 0:
            return

        max_resettable_by_pct = int(self.K * self.max_codes_to_reset_pct)
        num_to_reset = min(num_dead_candidates, num_available_replacements, max_resettable_by_pct)
        if num_to_reset == 0:
            return

        perm = torch.randperm(num_dead_candidates, device=dead_code_candidates_indices.device)[:num_to_reset]
        actual_dead_indices = dead_code_candidates_indices[perm]

        # --- PERFORMANCE FIX: Replace expensive FPS with fast random sampling ---
        rand_indices = torch.randint(0, num_available_replacements, (num_to_reset,), device=source_vectors.device)
        replacement_vectors = source_vectors[rand_indices]

        if self.training:
            logger.info(
                "VQ: Resetting %s dead codes via random sampling from buffer out of %s dead.",
                num_to_reset,
                num_dead_candidates,
            )

        self.codebook.data[actual_dead_indices] = replacement_vectors
        if self.ema:
            self.ema_cluster_size[actual_dead_indices] = 1.0
            self.ema_weight_sum[actual_dead_indices] = replacement_vectors.clone()

    # _ema_update method remains the same...
    @torch.no_grad()
    def _ema_update(self, encodings: torch.Tensor, flat_input: torch.Tensor):
        if self.eop_token_id < encodings.size(1):
            encodings = encodings.clone()
            encodings[:, self.eop_token_id] = 0

        if self.padding_token_id < encodings.size(1):
            encodings[:, self.padding_token_id] = 0

        dw = encodings.T @ flat_input
        cluster_size = encodings.sum(0)
        self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.ema_weight_sum.mul_(self.decay).add_(dw, alpha=1 - self.decay)
        n_total_clusters = self.ema_cluster_size.sum()
        stabilized_cluster_size = (
            (self.ema_cluster_size + self.eps) / (n_total_clusters + self.K * self.eps)
        ) * n_total_clusters
        self.codebook.data.copy_(self.ema_weight_sum / stabilized_cluster_size.unsqueeze(1))


    # @dynamo.disable
    def forward(self, z: torch.Tensor) -> tuple[Tensor | Any, Tensor, Tensor, Any]:
        B, Q, D_input = z.shape
        flat_input = z.reshape(-1, self.D)

        if self.training and self.reset_codes:
            self._update_replacement_buffer(flat_input.detach())

        dist_sq_input = flat_input.pow(2).sum(dim=1, keepdim=True)
        dist_sq_codebook = self.codebook.pow(2).sum(dim=1)
        dist_dot_product = -2 * torch.matmul(flat_input, self.codebook.T)
        distances = dist_sq_input + dist_dot_product + dist_sq_codebook
        indices = distances.argmin(dim=1)
        z_q = F.embedding(indices, self.codebook).view(B, Q, self.D)

        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z, z_q.detach())
        vq_loss = codebook_loss + (self.beta * commitment_loss)
        z_q_ste = z + (z_q - z).detach()

        if self.training:
            if self.ema:
                encodings = F.one_hot(indices, self.K).type_as(flat_input)
                self._ema_update(encodings, flat_input)

            if self.reset_codes:
                self._maybe_reset_dead_codes()

                # # Maintain Python counter to avoid CUDA scalar sync in condition
                # self._steps_since_last_reset_py += 1
                # self.steps_since_last_reset.fill_(self._steps_since_last_reset_py)
                # if self._steps_since_last_reset_py >= self.reset_interval:
                #     self._reset_dead_codes()
                #     self._steps_since_last_reset_py = 0
                #     self.steps_since_last_reset.zero_()

        if self.ema:
            counts = self.ema_cluster_size.clone()
        else:
            counts = torch.bincount(indices, minlength=self.K).float()

        if self.eop_token_id < counts.size(0):
            counts[self.eop_token_id] = 0

        if self.padding_token_id < counts.size(0):
            counts[self.padding_token_id] = 0

        counts = counts.clamp(min=self.eps)
        probs = counts / (counts.sum() + self.eps)
        entropy = -(probs.float() * probs.float().log()).sum()
        perplexity = entropy.exp().float()

        return z_q_ste, vq_loss, indices.view(B, Q), perplexity.detach()


class ResidualVQ(nn.Module):
    """
    Two-stage residual vector quantiser in a compressed space (d_c).
    """

    def __init__(self, K0: int, K1: int, D: int = 256, d_c: int = 64, **vq_kwargs):
        super().__init__()
        self.down = nn.Linear(D, d_c, bias=False)
        self.up = nn.Linear(d_c, D, bias=False)

        # Stage-0 and Stage-1 VQs
        self.vq0 = VectorQuantizer(K=K0, D=d_c, **vq_kwargs)
        self.vq1 = VectorQuantizer(K=K1, D=d_c, **vq_kwargs)

        # bfloat16‑friendly init: Xavier uniform for down, then copy transpose to up
        with torch.no_grad():
            nn.init.xavier_uniform_(self.down.weight)
            self.up.weight.copy_(self.down.weight.T.to(self.up.weight.dtype))

    def forward(self, z):
        """
        z : (B, Q, D)
        returns ẑ, total_vq_loss, (idx0, idx1), perplexities
        """
        y0 = self.down(z)  # (B,Q,d_c)

        q0, loss0, idx0, ppl0 = self.vq0(y0)
        r1 = y0 - q0.detach()  # stop grad into q0 path

        q1, loss1, idx1, ppl1 = self.vq1(r1)
        z_hat = self.up(q0 + q1)

        total_loss = loss0 + loss1
        return z_hat, total_loss, (idx0, idx1), (ppl0, ppl1)


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
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        padding_token_id: int = 2,
        eop_token_id: int = 3,
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict

class ComposedIndexCodec:
    """
    Converts between composed indices (0..K**depth-1) and per-stage digits (depth, each 0..K-1).
    Handles special ids (bos/eos/pad/eop) by flagging them as special.
    """
    def __init__(self, K: int, depth: int, bos: int, eos: int, pad: int, eop: int):
        self.K = int(K)
        self.depth = int(depth)
        self.special = {int(bos), int(eos), int(pad), int(eop)}
        # map special -> [0..#special-1] for a tiny special head
        self.special_list = sorted(list(self.special))
        self.special_to_local = {sid: i for i, sid in enumerate(self.special_list)}

    def is_special(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(x, dtype=torch.bool)
        for sid in self.special:
            mask |= (x == sid)
        return mask

    def decompose(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        x: (B, L) composed ids (may include specials)
        returns:
            digits: list of length 'depth', each (B, L) in [0..K-1]; undefined at special positions
            special_mask: (B, L) True where x is special
        """
        special = self.is_special(x)
        x_work = x.clone()
        digits = []
        for _ in range(self.depth):
            digits.append((x_work % self.K).long())
            x_work = x_work // self.K
        return digits, special

    def compose(self, digits: List[torch.Tensor]) -> torch.Tensor:
        """
        digits: list of length 'depth' of (B, L) in [0..K-1]
        returns composed: (B, L)
        """
        base = 1
        out = torch.zeros_like(digits[0], dtype=torch.long)
        for i, d in enumerate(digits):
            if i > 0: base *= self.K
            out = out + d.long() * base
        return out

    def special_local_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map special ids to [0..#special-1], garbage for non-specials.
        """
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
        share_ok = use_shared_proj and (self.D == self.D_vq)
        if share_ok:
            self.up = vq.up                       # d_c -> D
            self.down = vq.down                   # D -> d_c
        else:
            # One shared parameter for both directions
            dtype = vq.up.weight.dtype
            device = vq.up.weight.device
            Wdc = nn.Parameter(torch.empty(self.d_c, self.D, dtype=dtype, device=device))
            nn.init.xavier_uniform_(Wdc)
            self.down = DownProj(Wdc)  # D -> d_c
            self.up = UpProj(Wdc)  # d_c -> D

            # self.up = nn.Linear(self.d_c, self.D, bias=False)
            # self.down = nn.Linear(self.D, self.d_c, bias=False)
            # # nice init
            # nn.init.xavier_uniform_(self.up.weight)
            # if tie_up_down:
            #     self.down.weight = self.up.weight.T  # tie (optional)
            # else:
            #     nn.init.xavier_uniform_(self.down.weight)

        # tiny special table in target D
        self.special_ids = sorted({int(vq.bos_token_id),
                                   int(vq.eos_token_id),
                                   int(vq.padding_token_id),
                                   int(vq.eop_token_id)})
        self.special_to_local = {sid: i for i, sid in enumerate(self.special_ids)}
        self.special_emb = (nn.Embedding(len(self.special_ids), self.D)
                            if specials_in_D else None)


        # # Tie projections to MS‑RVQ by default (no extra params)
        # self.down = vq.down if use_shared_proj else nn.Linear(self.D, self.d_c, bias=False)
        # self.up   = vq.up   if use_shared_proj else nn.Linear(self.d_c, self.D, bias=False)
        #
        # # Tiny learned table for specials; you can also keep zero vectors if you prefer.
        # self.special_ids = [vq.bos_token_id, vq.eos_token_id, vq.padding_token_id, vq.eop_token_id]
        # self.special_ids = sorted(list(set(int(x) for x in self.special_ids)))
        # self.special_to_local = {sid: i for i, sid in enumerate(self.special_ids)}
        # self.special_emb = nn.Embedding(len(self.special_ids), self.D) if specials_in_D else None

    @torch.no_grad()
    def stage_codebook(self, s: int) -> torch.Tensor:
        # Return d_c codebook of stage s, detached so no grads leak into VQ from expander.
        return self.vq.stages[s].codebook.detach()

    # def embed_composed(self, idx: torch.Tensor) -> torch.Tensor:
    #     """
    #     idx: (B, L) composed ids including specials.
    #     returns: (B, L, D)
    #     """
    #     B, L = idx.shape
    #     # specials
    #     special_mask = torch.zeros_like(idx, dtype=torch.bool)
    #     for sid in self.special_ids:
    #         special_mask |= (idx == sid)
    #
    #     # non-special: sum stage vectors in d_c, then up to D
    #     out = torch.zeros(B, L, self.D, device=idx.device, dtype=self.vq.up.weight.dtype)
    #
    #     if (~special_mask).any():
    #         digits, _ = ComposedIndexCodec(self.K, self.depth,
    #                                        self.vq.bos_token_id, self.vq.eos_token_id,
    #                                        self.vq.padding_token_id, self.vq.eop_token_id).decompose(idx)
    #         y_dc = torch.zeros(B, L, self.d_c, device=idx.device, dtype=self.vq.up.weight.dtype)
    #         for s in range(self.depth):
    #             W = self.stage_codebook(s)               # (K, d_c)
    #             y_dc = y_dc + F.embedding(digits[s], W)  # (B, L, d_c)
    #         out_ns = self.up(y_dc)                       # (B, L, D)
    #         out = torch.where((~special_mask).unsqueeze(-1), out_ns, out)
    #
    #     if self.special_emb is not None and special_mask.any():
    #         # map specials to [0..num_special-1]
    #         local = idx.clone()
    #         for sid, j in self.special_to_local.items():
    #             local[idx == sid] = j
    #         out_sp = self.special_emb(local.clamp_min(0))
    #         out = torch.where(special_mask.unsqueeze(-1), out_sp, out)
    #
    #     return out

    # def embed_composed(self, idx: torch.Tensor) -> torch.Tensor:
    #     """
    #     idx: (B, L) composed ids including specials.
    #     returns: (B, L, D)
    #     Safe under torch.compile: never passes OOB indices to Embedding.
    #     """
    #     B, L = idx.shape
    #     device = idx.device
    #     out = torch.zeros(B, L, self.D, device=device, dtype=self.vq.up.weight.dtype)
    #
    #     # ----- specials mask -----
    #     special_mask = torch.zeros_like(idx, dtype=torch.bool)
    #     for sid in self.special_ids:
    #         special_mask |= (idx == sid)
    #
    #     # ----- non-special path (sum stage vectors in d_c -> up to D) -----
    #     if (~special_mask).any():
    #         # Decompose once
    #         digits, _ = ComposedIndexCodec(self.K, self.depth,
    #                                        self.vq.bos_token_id, self.vq.eos_token_id,
    #                                        self.vq.padding_token_id, self.vq.eop_token_id).decompose(idx)
    #         y_dc = torch.zeros(B, L, self.d_c, device=device, dtype=self.vq.up.weight.dtype)
    #         for s in range(self.depth):
    #             W = self.stage_codebook(s)  # (K, d_c)
    #             # Ensure indices are valid at special positions too (map specials -> 0)
    #             idx_s = digits[s]
    #             if special_mask.any():
    #                 idx_s = torch.where(special_mask, idx_s.new_zeros(()), idx_s)
    #             y_dc = y_dc + F.embedding(idx_s, W)
    #         out_ns = self.up(y_dc)
    #         out = torch.where((~special_mask).unsqueeze(-1), out_ns, out)
    #
    #     # ----- specials path (tiny table) -----
    #     if self.special_emb is not None and special_mask.any():
    #         # Build a safe index tensor: 0 everywhere, mapped j at special positions
    #         local = torch.zeros_like(idx, dtype=torch.long)
    #         for sid, j in self.special_to_local.items():
    #             local = torch.where(idx == sid, local.new_full((), j), local)
    #         # This call is now in-range for *all* positions
    #         out_sp = self.special_emb(local)
    #         out = torch.where(special_mask.unsqueeze(-1), out_sp, out)
    #
    #     return out

    #
    # def embed_composed(self, idx: torch.Tensor) -> torch.Tensor:
    #     idx = idx.long()
    #     B, L = idx.shape
    #     dev = idx.device
    #     out = torch.zeros(B, L, self.D, device=dev, dtype=self.up.Wdc.dtype)
    #
    #     # specials mask
    #     special_mask = torch.zeros_like(idx, dtype=torch.bool)
    #     for sid in self.special_ids:
    #         special_mask |= (idx == sid)
    #
    #     # non-special path: sum stage vectors in d_c -> up to D
    #     if (~special_mask).any():
    #         idx_ns = idx.masked_fill(special_mask, 0)      # sanitize
    #         digits, _ = self.vq.codec.decompose(idx_ns)    # [ (B,L) ] len=depth
    #         y_dc = torch.zeros(B, L, self.d_c, device=dev, dtype=self.up.Wdc.dtype)
    #         for s in range(self.depth):
    #             W = self.stage_codebook(s)                 # (K, d_c)
    #             y_dc = y_dc + F.embedding(digits[s], W)
    #         out_ns = self.up(y_dc)                         # (B,L,D_tgt)
    #         out[~special_mask] = out_ns[~special_mask]
    #
    #     # specials path: small table, guaranteed in-range
    #     if self.special_emb is not None and special_mask.any():
    #         local = torch.zeros_like(idx)
    #         for sid, j in self.special_to_local.items():
    #             local[idx == sid] = j
    #         out_sp = self.special_emb(local)
    #         out[special_mask] = out_sp[special_mask]
    #
    #     return out

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


# class LearnedCodebookAdapter(nn.Module):
#     """
#     Bottom-level adapter with a *single* stage (depth=1).
#     - codebook: (K, d_c)  learned
#     - up: d_c -> D  (and down: D -> d_c) so we can reuse the same head
#     """
#     def __init__(self, K: int, D: int, d_c: int = 64, tie_up_down: bool = True):
#         super().__init__()
#         self.K, self.D, self.d_c, self.depth = int(K), int(D), int(d_c), 1
#
#         self.special_emb = None  # no special handling here
#
#         # Factorized byte embedding
#         self._codebook = nn.Embedding(self.K, self.d_c)     # stage 0
#         self.down = nn.Linear(self.D, self.d_c, bias=False) # for the head
#         self.up   = nn.Linear(self.d_c, self.D, bias=False) # for inputs
#
#         # Optional tying (like ALBERT)
#         if tie_up_down:
#             self.up.weight = self.down.weight.T  # weight sharing
#
#         nn.init.normal_(self._codebook.weight, std=0.02)
#         nn.init.xavier_uniform_(self.down.weight)
#
#     def stage_codebook(self, s: int = 0) -> torch.Tensor:
#         assert s == 0, "Depth=1 adapter: only stage 0 exists."
#         return self._codebook.weight  # (K, d_c)
#
#     @torch.no_grad()
#     def embed_ids_dc(self, ids: torch.Tensor) -> torch.Tensor:
#         return self._codebook(ids)    # (B,L,d_c)
#
#     def embed_composed(self, ids: torch.Tensor) -> torch.Tensor:
#         # bytes (and EOS/EOP) are predicted *in the same K*; no special head needed
#         return self.up(self._codebook(ids))  # (B,L,D)

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
