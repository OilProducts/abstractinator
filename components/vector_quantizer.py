import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Any

from torch import Tensor


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer (VQ) layer.

    This layer maps continuous input vectors to a discrete set of learned codebook
    vectors. It's a core component of VQ-VAEs and other models requiring
    discretization of latent representations.

    The quantization process involves finding the closest codebook vector (in terms
    of Euclidean distance) for each input vector. The layer supports training
    the codebook via two mechanisms:
    1. Standard gradient descent using a codebook loss.
    2. Exponential Moving Average (EMA) updates, which can lead to more stable
       training and better codebook utilization.

    Args:
        K (int): The number of codebook vectors (codes).
        D (int): The dimensionality of each codebook vector and the input vectors.
        beta (float, optional): The commitment loss weight. This loss encourages
            the encoder output to commit to a specific codebook vector.
            Defaults to 0.25.
        ema (bool, optional): If True, uses EMA updates for the codebook.
            If False, the codebook is updated via standard gradient descent using
            the codebook loss. Defaults to True.
        decay (float, optional): The decay rate for EMA updates. Only used if
            `ema` is True. Values closer to 1 result in slower updates.
            Defaults to 0.99.
        eps (float, optional): A small epsilon value added for numerical stability,
            particularly in EMA cluster size calculations to prevent division by zero.
            Defaults to 1e-5.
        reset_codes (bool, optional): Enable code resetting. Defaults to False.
        reset_interval (int, optional): Steps between reset checks. Defaults to 1000.
        min_usage_threshold (float, optional): Min usage count over interval to avoid reset.
                                            Defaults to 1.0 (meaning used at least once).
        max_codes_to_reset_pct (float, optional): Max % of codebook to reset at once.
                                               Defaults to 0.1 (10%).
        reset_ema_on_reset (bool, optional): Reset EMA stats for reset codes. Defaults to True.

    Attributes:
        K (int): Number of codebook vectors.
        D (int): Dimensionality of vectors.
        beta (float): Commitment loss weight.
        ema (bool): Flag for using EMA updates.
        decay (float): EMA decay rate.
        eps (float): Epsilon for numerical stability.
        codebook (nn.Parameter): The learnable codebook vectors of shape (K, D).
            Even with EMA, this is a nn.Parameter to be part of the model's state,
            though its updates are managed by EMA logic if `ema` is True.
        ema_cluster_size (torch.Tensor): Buffer for EMA of cluster sizes (number of
            inputs assigned to each code). Shape (K,). Only used if `ema` is True.
        ema_weight_sum (torch.Tensor): Buffer for EMA of sum of input vectors
            assigned to each code. Shape (K, D). Only used if `ema` is True.
    """
    def __init__(self,
                 K: int,
                 D: int,
                 beta: float = 0.25,
                 ema: bool = True,
                 decay: float = 0.99,
                 eps: float = 1e-5,
                 reset_codes: bool = True,
                 usage_decay: float = 0.95,
                 reset_interval: int = 100,
                 min_usage_threshold: float = 0.01,
                 max_codes_to_reset_pct: float = 0.1,
                 reset_ema_on_reset: bool = True):
        super().__init__()
        self.K = K  # Number of codebook vectors
        self.D = D  # Dimensionality of codebook vectors
        self.beta = beta  # Commitment loss weight
        self.ema = ema    # Whether to use EMA for codebook updates
        self.decay = decay  # EMA decay factor
        self.eps = eps    # Epsilon for numerical stability
        self.usage_decay = usage_decay

        self.reset_codes = reset_codes # bool: whether to enable code resetting
        self.reset_interval = reset_interval # int: steps between reset checks
        self.min_usage_threshold = min_usage_threshold # float: min usage to avoid reset
        self.max_codes_to_reset_pct = max_codes_to_reset_pct # float: max % of codes to reset at once
        self.reset_ema_on_reset = reset_ema_on_reset # bool: whether to reset EMA stats for reset codes

        if self.reset_codes:
            if not self.ema and self.reset_ema_on_reset:
                print("Warning: reset_ema_on_reset is True, but EMA is not enabled. EMA stats won't be reset.")
            # Buffer to track code usage count within a reset interval
            # self.register_buffer("code_usage_count", torch.zeros(K, dtype=torch.float32))
            # Buffer to track steps since last reset
            self.register_buffer("steps_since_last_reset", torch.tensor(0, dtype=torch.int64))
            self.min_usage_threshold = 1.0

        # Initialize the codebook as a learnable parameter
        self.codebook = nn.Parameter(torch.randn(K, D))

        if ema:
            # Buffers for EMA updates
            # These are not parameters to be optimized by gradient descent but are part
            # of the module's state and should be saved with the model.
            self.register_buffer("ema_cluster_size", torch.zeros(K))
            # Initialize ema_weight_sum with the initial codebook values for stability
            self.register_buffer("ema_weight_sum", self.codebook.data.clone())

    @torch.no_grad()
    def _ema_update(self, encodings: torch.Tensor, flat_input: torch.Tensor):
        """
        Updates the codebook using Exponential Moving Average.

        This method should not contribute to gradient computation, hence @torch.no_grad().

        Args:
            encodings (torch.Tensor): One-hot encoded tensor indicating which
                codebook vector each input was closest to. Shape (N, K), where N is
                the total number of vectors (batch_size * num_vectors_per_item).
            flat_input (torch.Tensor): The flattened input tensor to the VQ layer.
                Shape (N, D).
        """
        # Calculate the sum of input vectors assigned to each codebook vector
        dw = encodings.T @ flat_input  # Shape: (K, D)

        # Calculate the number of input vectors assigned to each codebook vector (cluster size)
        cluster_size = encodings.sum(0)  # Shape: (K,)

        # Update EMA for cluster sizes
        # new_ema_cluster_size = decay * old_ema_cluster_size + (1 - decay) * current_cluster_size
        self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

        # Update EMA for sum of encoded vectors (weights)
        # new_ema_weight_sum = decay * old_ema_weight_sum + (1 - decay) * current_dw
        self.ema_weight_sum.mul_(self.decay).add_(dw, alpha=1 - self.decay)

        # Normalize cluster sizes to ensure their sum reflects the true token count,
        # adding epsilon for numerical stability. This prevents division by zero for
        # unused codes and helps in maintaining a stable codebook.
        n_total_clusters = self.ema_cluster_size.sum()
        # Additive smoothing (Laplace smoothing style) for cluster sizes
        # This helps prevent any cluster size from becoming exactly zero.
        stabilized_cluster_size = (
            (self.ema_cluster_size + self.eps) /
            (n_total_clusters + self.K * self.eps)
        ) * n_total_clusters

        stabilized_cluster_size = torch.clamp(stabilized_cluster_size, min=self.eps)  # <-- NEW

        # Update the codebook by dividing the EMA of summed vectors by the EMA of cluster sizes.
        # unsqueeze(1) is for broadcasting (K,) to (K,1) for element-wise division.
        self.codebook.data.copy_(self.ema_weight_sum / stabilized_cluster_size.unsqueeze(1))

    @torch.no_grad()
    def _sample_farthest(self, xs, n):
        # xs : (N,D)
        picked = xs[torch.randint(0, xs.size(0), (1,))]
        for _ in range(n - 1):
            dist2 = torch.cdist(xs, picked).pow(2).min(dim=1).values
            picked = torch.cat([picked, xs[dist2.argmax()].unsqueeze(0)], 0)
        return picked  # (n,D)

    @torch.no_grad()
    def _reset_dead_codes(self, current_batch_encoder_outputs: torch.Tensor):
        """
        Identifies and resets dead/underutilized codebook vectors.

        Dead codes are replaced with randomly selected unique encoder outputs from
        the current batch. EMA statistics for these codes are also reset if enabled.

        Args:
            current_batch_encoder_outputs (torch.Tensor): Flattened encoder outputs
                from the current batch. Shape (N, D).
        """
        if not self.reset_codes:
            return

        self.ema_cluster_size.mul_(self.usage_decay)
        # Identify dead codes: those with usage count below the threshold
        dead_code_candidates_indices = (self.ema_cluster_size < self.min_usage_threshold).nonzero(as_tuple=True)[0]
        num_dead_candidates = dead_code_candidates_indices.size(0)

        if num_dead_candidates == 0:
            if self.training:  # Only print during training if verbose
                print(f"VQ: No codes below usage threshold {self.min_usage_threshold} to reset.")
            return

        # Limit the number of codes to reset based on max_codes_to_reset_pct
        max_resettable = int(self.K * self.max_codes_to_reset_pct)
        num_to_reset = min(num_dead_candidates, max_resettable)

        if num_to_reset == 0:
            if self.training and num_dead_candidates > 0:  # If candidates existed but % limit hit 0
                print(
                    f"VQ: Dead code candidates found ({num_dead_candidates}), but 0 allowed by max_codes_to_reset_pct ({self.max_codes_to_reset_pct * 100}%).")
            return

        # Select the actual codes to reset (e.g., those with lowest usage among candidates, or random subset)
        # For simplicity, we'll take the first `num_to_reset` candidates (often these are already sorted by some logic or are just the ones that fell below)
        # If a more sophisticated selection is needed (e.g. truly random among candidates, or lowest usage), this can be enhanced.
        # Here, we'll just take the first ones found by nonzero(). If these indices need to be chosen more carefully (e.g. truly random from candidates),
        # one could shuffle dead_code_candidates_indices and take the top num_to_reset.
        # For now, let's randomly select from the candidates if there are more candidates than we can reset.
        if num_dead_candidates > num_to_reset:
            perm = torch.randperm(num_dead_candidates, device=dead_code_candidates_indices.device)[:num_to_reset]
            actual_dead_indices = dead_code_candidates_indices[perm]
        else:
            actual_dead_indices = dead_code_candidates_indices[
                                  :num_to_reset]  # Take all candidates if fewer than num_to_reset

        num_actually_reset = actual_dead_indices.size(0)
        if num_actually_reset == 0:  # Should not happen if num_to_reset > 0, but as safeguard
            return

        if self.training:
            print(f"VQ: Resetting {num_actually_reset} dead codes (out of {num_dead_candidates} candidates).")

        replacement_vectors = self._sample_farthest(current_batch_encoder_outputs, actual_dead_indices.numel())

        # Select replacement vectors from the current batch's encoder outputs
        # Ensure enough unique samples from the batch, or sample with replacement
        # num_batch_vectors = current_batch_encoder_outputs.size(0)
        # if num_batch_vectors == 0:
        #     if self.training:
        #         print("VQ: Warning - Cannot reset codes, current batch encoder outputs are empty.")
        #     return
        #
        # # Get unique encoder outputs from the batch first
        # unique_batch_outputs = torch.unique(current_batch_encoder_outputs, dim=0)
        # num_unique_batch_outputs = unique_batch_outputs.size(0)
        #
        # if num_unique_batch_outputs >= num_actually_reset:
        #     # Sample without replacement from unique outputs
        #     perm = torch.randperm(num_unique_batch_outputs, device=unique_batch_outputs.device)[:num_actually_reset]
        #     replacement_vectors = unique_batch_outputs[perm]
        # else:
        #     # Not enough unique outputs, sample with replacement from all batch outputs
        #     # (or use all unique and then sample with replacement for the remainder)
        #     # Simpler: sample with replacement from all batch outputs to fill num_actually_reset
        #     indices_to_sample = torch.randint(0, num_batch_vectors, (num_actually_reset,),
        #                                       device=current_batch_encoder_outputs.device)
        #     replacement_vectors = current_batch_encoder_outputs[indices_to_sample]

        # Update the codebook with the replacement vectors
        self.codebook.data[actual_dead_indices] = replacement_vectors

        # Reset EMA statistics for the reset codes if EMA is active and configured to do so
        if self.ema and self.reset_ema_on_reset:
            # A common approach is to set ema_cluster_size to a small non-zero value
            # (e.g., 1.0 or self.eps) to give the new code a chance to accumulate.
            # Let's use 1.0 for simplicity, assuming it represents a single "hit".
            self.ema_cluster_size[actual_dead_indices] = 1.0
            # And set ema_weight_sum to the new code vector value itself
            # (scaled by the new cluster size, which is 1.0 here)
            self.ema_weight_sum[actual_dead_indices] = replacement_vectors.clone()  # Use .clone() for safety

        # Important: Reset the usage count for ALL codes after a reset operation,
        # so that usage accumulation starts fresh for the next interval.
        # self.code_usage_count[actual_dead_indices] = 0.0
        # self.code_usage_count.mul_(self.usage_decay)

    def forward(self, z: torch.Tensor) -> tuple[Tensor | Any, Tensor, Tensor, Any]:
        """
        Forward pass of the Vector Quantizer.

        Args:
            z (torch.Tensor): The input tensor from the encoder, expected to be of
                shape (B, Q, D), where B is batch size, Q is the number of
                vectors per batch item (e.g., sequence length or spatial dimensions),
                and D is the dimensionality of the vectors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - z_q (torch.Tensor): The quantized output tensor, with gradients passed
              through using the Straight-Through Estimator. Shape (B, Q, D).
            - vq_loss (torch.Tensor): The total VQ loss, combining commitment loss
              and codebook loss (if EMA is not solely relied upon). Scalar.
            - idx (torch.Tensor): The indices of the chosen codebook vectors for
              each input vector. Shape (B, Q).
        """
        B, Q, D_input = z.shape
        if D_input != self.D:
            raise ValueError(f"Input tensor feature dimension {D_input} "
                             f"does not match codebook dimension {self.D}")

        # Reshape input tensor z from (B, Q, D) to (N, D) where N = B * Q
        flat_input = z.reshape(-1, self.D)  # Shape: (N, D)

        # Calculate squared Euclidean distances between each input vector and each codebook vector
        # dist = ||flat_input||^2 - 2 * flat_input @ codebook.T + ||codebook||^2
        # ||x||^2 term, sum over D, keep dim for broadcasting. Shape: (N, 1)
        dist_sq_input = flat_input.pow(2).sum(dim=1, keepdim=True)
        # ||c||^2 term, sum over D. Shape: (K,) -> (1, K) for broadcasting
        dist_sq_codebook = self.codebook.pow(2).sum(dim=1)
        # -2 * x^T c term. Shape: (N, K)
        dist_dot_product = -2 * torch.matmul(flat_input, self.codebook.T)

        # Total squared distance. Shape: (N, K)
        distances = dist_sq_input + dist_dot_product + dist_sq_codebook

        # Find the index of the closest codebook vector for each input vector
        # argmin over K dimension. Shape: (N,)
        indices = distances.argmin(dim=1)

        # Retrieve the quantized vectors from the codebook using the indices
        # and reshape back to (B, Q, D)
        z_q = F.embedding(indices, self.codebook).view(B, Q, self.D)
        # Alternative: z_q = self.codebook[indices].view(B, Q, self.D)

        # --- Loss Calculation ---
        # 1. Codebook Loss (or dictionary learning loss):
        # Encourages codebook vectors (z_q) to move towards encoder outputs (z).
        # Detach z to ensure gradients only flow to z_q (and thus the codebook) for this loss.
        # This loss is primarily used if EMA is OFF. If EMA is ON, the codebook is
        # updated by the _ema_update rule, but this loss can still be computed.
        codebook_loss = F.mse_loss(z_q, z.detach())

        # 2. Commitment Loss:
        # Encourages encoder outputs (z) to commit to the chosen codebook vectors.
        # Detach z_q to ensure gradients only flow to z (encoder) for this loss.
        commitment_loss = F.mse_loss(z, z_q.detach())

        # Total VQ loss
        # The beta factor scales the commitment loss.
        vq_loss = codebook_loss + (self.beta * commitment_loss)

        # --- Straight-Through Estimator (STE) ---
        # This allows gradients to pass through the discrete quantization step (argmin).
        # In the backward pass, the gradient of z_q is copied to z.
        # z_q = z + (quantized_z - z).detach()
        # This means dL/dz = dL/dz_q
        z_q_ste = z + (z_q - z).detach()

        # --- Code Usage Tracking and Resetting (if enabled and in training mode) ---
        if self.training and self.reset_codes:
            # Update code usage counts from the current batch
            # `encodings` needs to be computed if not already available from EMA logic
            # If EMA is off, `encodings` isn't computed by default in the current EMA block.
            # Let's assume `indices` is always available.
            # current_batch_one_hot_encodings = F.one_hot(indices, self.K).type_as(flat_input)
            # self.code_usage_count.add_(current_batch_one_hot_encodings.sum(0).detach()) # Sum over N dim
            self.steps_since_last_reset.add_(1)

            if self.steps_since_last_reset >= self.reset_interval:
                self._reset_dead_codes(flat_input.detach()) # Pass encoder outputs
                self.steps_since_last_reset.zero_()

        # --- EMA Codebook Update (if enabled and in training mode) ---
        if self.training and self.ema:
            # Convert indices to one-hot encodings for EMA update
            # Shape: (N, K)
            encodings = F.one_hot(indices, self.K).type_as(flat_input)
            self._ema_update(encodings, flat_input)

        # Calculate codebook perplexity (average usage of codes)
        # This is a common metric to monitor codebook usage.
        # Perplexity = exp(-sum(p_i * log(p_i))), where p_i is usage probability of code i
        # A simpler related metric is active_codes / K
        # For perplexity, we need probabilities. Use EMA cluster size for a stable estimate.
        # if self.ema: # Use EMA cluster sizes if available for a more stable perplexity
        #     probs = self.ema_cluster_size / (self.ema_cluster_size.sum() + self.eps) # Normalized probabilities
        # else: # Use current batch usage if EMA is off (less stable, but an indicator)
        #      if self.training and self.reset_codes and self.code_usage_count.sum() > 0:
        #          probs = self.code_usage_count / self.code_usage_count.sum()
        #      else: # Fallback or if not training/no usage yet
        #          # Create a uniform distribution if no usage info, or a tensor of zeros
        #          # This might not be very meaningful if no codes have been used.
        #          # Let's make it NaN if no codes used, or handle it upstream.
        #          # For now, if sum is zero, perplexity calculation will result in NaN or error.
        #          # A simple active code count is often more robust initially.
        #          # Let's compute active codes for now.
        #          # active_codes_in_batch = torch.unique(indices).numel()
        #          # perplexity = torch.tensor(float('nan'), device=z.device) # Placeholder for now
        #          # Let's calculate effective number of codes used in the current batch as a proxy
        #          # if not self.ema and self.training:
        #          #    current_batch_usage = F.one_hot(indices, self.K).float().mean(dim=0)
        #          #    perplexity = torch.exp(-torch.sum(current_batch_usage * torch.log(current_batch_usage + 1e-10), dim=-1))
        #          # else: # Fallback for non-EMA / non-training
        #          #    perplexity = torch.tensor(float('nan'), device=z.device)
        #          probs = torch.ones(self.K, device=z.device) / self.K # Assume uniform if no better info
        #
        # # Filter out zero probabilities to avoid log(0) = -inf
        # probs_nz = probs[probs > 0]
        # if probs_nz.numel() > 0:
        #     entropy = -torch.sum(probs_nz * torch.log(probs_nz))
        #     perplexity = torch.exp(entropy)
        # else: # If all probabilities are zero (e.g., no codes used yet)
        #     perplexity = torch.tensor(0.0, device=z.device)
        # choose the raw counts that represent usage
        probs = self.ema_cluster_size  # (K,)
        # else:
        #     raw_counts = self.code_usage_count  # (K,)

        # 1) make sure nothing is exactly zero
        # probs = torch.clamp(raw_counts, min=self.eps)

        # 2) normalise to a true probability distribution
        denom = probs.sum() + self.eps  # add eps once more for safety
        probs = probs / denom  # shape (K,)

        # 3) perplexity
        entropy = -(probs.double() * probs.double().log()).sum()
        perplexity = entropy.exp().float()

        return z_q_ste, vq_loss, indices.view(B, Q), perplexity.detach()

        # # Return quantized output (with STE), VQ loss, and codebook indices
        # return z_q_ste, vq_loss, indices.view(B, Q)