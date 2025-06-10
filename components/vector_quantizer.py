import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Any

from torch import Tensor


@torch.compile
class VectorQuantizer(nn.Module):
    """
    An efficient Vector Quantizer (VQ) layer with EMA updates and robust,
    performant code resetting.
    """

    def __init__(self,
                 K: int,
                 D: int,
                 beta: float = 0.25,
                 ema: bool = True,
                 decay: float = 0.999,
                 eps: float = 1e-5,
                 reset_codes: bool = True,
                 reset_interval: int = 2000,
                 max_codes_to_reset_pct: float = 0.1,
                 replacement_buffer_size: int = 65536,
                 vectors_per_step_to_buffer: int = 1024):  # New: Controls update overhead
        super().__init__()
        self.K = K
        self.D = D
        self.beta = beta
        self.ema = ema
        self.decay = decay
        self.eps = eps

        self.reset_codes = reset_codes
        if self.reset_codes:
            self.reset_interval = reset_interval
            self.max_codes_to_reset_pct = max_codes_to_reset_pct
            self.min_usage_threshold = 1.0
            self.register_buffer("steps_since_last_reset", torch.tensor(0, dtype=torch.int64))

            # Buffer settings
            self.replacement_buffer_size = replacement_buffer_size
            self.vectors_per_step_to_buffer = vectors_per_step_to_buffer
            self.register_buffer("replacement_buffer", torch.empty(replacement_buffer_size, D))
            self.register_buffer("buffer_idx", torch.tensor(0, dtype=torch.long))
            self.register_buffer("buffer_is_full", torch.tensor(False, dtype=torch.bool))

        self.codebook = nn.Parameter(torch.randn(K, D))
        if ema:
            self.register_buffer("ema_cluster_size", torch.zeros(K))
            self.register_buffer("ema_weight_sum", self.codebook.data.clone())

    @torch.no_grad()
    def _update_replacement_buffer(self, new_vectors: torch.Tensor):
        """Efficiently updates the circular buffer with a subset of new vectors."""
        # --- PERFORMANCE FIX: Subsample vectors to reduce copy overhead ---
        if new_vectors.size(0) > self.vectors_per_step_to_buffer:
            perm = torch.randperm(new_vectors.size(0), device=new_vectors.device)[:self.vectors_per_step_to_buffer]
            new_vectors = new_vectors[perm]

        n_new = new_vectors.size(0)
        buff_size = self.replacement_buffer_size
        current_idx = self.buffer_idx.item()

        if n_new >= buff_size:
            self.replacement_buffer.copy_(new_vectors)
            self.buffer_is_full.fill_(True)
        else:
            space_left = buff_size - current_idx
            if n_new < space_left:
                self.replacement_buffer[current_idx: current_idx + n_new] = new_vectors
                self.buffer_idx.add_(n_new)
            else:
                self.replacement_buffer[current_idx:] = new_vectors[:space_left]
                self.replacement_buffer[: n_new - space_left] = new_vectors[space_left:]
                self.buffer_idx.fill_(n_new - space_left)
                self.buffer_is_full.fill_(True)

    @torch.no_grad()
    def _reset_dead_codes(self):
        """Resets dead codes using fast random sampling from the buffer."""
        if not self.buffer_is_full and self.buffer_idx == 0:
            return

        source_vectors = self.replacement_buffer if self.buffer_is_full else self.replacement_buffer[:self.buffer_idx]
        num_available_replacements = source_vectors.size(0)

        dead_code_candidates_indices = (self.ema_cluster_size < self.min_usage_threshold).nonzero(as_tuple=True)[0]
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
            print(f"VQ: Resetting {num_to_reset} dead codes via random sampling from buffer out of {num_dead_candidates} dead.")

        self.codebook.data[actual_dead_indices] = replacement_vectors
        if self.ema:
            self.ema_cluster_size[actual_dead_indices] = 1.0
            self.ema_weight_sum[actual_dead_indices] = replacement_vectors.clone()

    # _ema_update method remains the same...
    @torch.no_grad()
    def _ema_update(self, encodings: torch.Tensor, flat_input: torch.Tensor):
        dw = encodings.T @ flat_input
        cluster_size = encodings.sum(0)
        self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.ema_weight_sum.mul_(self.decay).add_(dw, alpha=1 - self.decay)
        n_total_clusters = self.ema_cluster_size.sum()
        stabilized_cluster_size = ((self.ema_cluster_size + self.eps) /
                                   (n_total_clusters + self.K * self.eps)) * n_total_clusters
        self.codebook.data.copy_(self.ema_weight_sum / stabilized_cluster_size.unsqueeze(1))

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
                self.steps_since_last_reset.add_(1)
                if self.steps_since_last_reset >= self.reset_interval:
                    self._reset_dead_codes()
                    self.steps_since_last_reset.zero_()

        counts = self.ema_cluster_size.clamp(min=self.eps)
        probs = counts / (counts.sum() + self.eps)
        entropy = -(probs.double() * probs.double().log()).sum()
        perplexity = entropy.exp().float()

        return z_q_ste, vq_loss, indices.view(B, Q), perplexity.detach()