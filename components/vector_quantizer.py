import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

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
    def __init__(self, K: int, D: int, beta: float = 0.25,
                 ema: bool = False, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.K = K  # Number of codebook vectors
        self.D = D  # Dimensionality of codebook vectors
        self.beta = beta  # Commitment loss weight
        self.ema = ema    # Whether to use EMA for codebook updates
        self.decay = decay  # EMA decay factor
        self.eps = eps    # Epsilon for numerical stability

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

        # Update the codebook by dividing the EMA of summed vectors by the EMA of cluster sizes.
        # unsqueeze(1) is for broadcasting (K,) to (K,1) for element-wise division.
        self.codebook.data.copy_(self.ema_weight_sum / stabilized_cluster_size.unsqueeze(1))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # --- EMA Codebook Update (if enabled and in training mode) ---
        if self.training and self.ema:
            # Convert indices to one-hot encodings for EMA update
            # Shape: (N, K)
            encodings = F.one_hot(indices, self.K).type_as(flat_input)
            self._ema_update(encodings, flat_input)

        # Return quantized output (with STE), VQ loss, and codebook indices
        return z_q_ste, vq_loss, indices.view(B, Q)