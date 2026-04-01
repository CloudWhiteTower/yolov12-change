"""PyTorch implementation of PolarQuant for YOLOv12 feature map compression.

This module provides a PyTorch-compatible version of PolarQuant algorithm from TurboQuant,
enabling GPU tensor operations and optional gradient flow for neural network integration.

Reference:
    TurboQuant: arXiv 2504.19874 (ICLR 2026)
    PolarQuant: arXiv 2502.02617 (AISTATS 2026)
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import stats

__all__ = ["PolarQuantTorch"]


def _optimal_centroids_numpy(bit_width: int, d: int) -> np.ndarray:
    """Compute optimal MSE centroids using NumPy/SciPy (called once during init)."""
    n_centroids = 1 << bit_width

    if bit_width == 1:
        c = np.sqrt(2.0 / (np.pi * d))
        return np.array([-c, c])

    if bit_width == 2:
        return np.array([-1.51, -0.453, 0.453, 1.51]) / np.sqrt(d)

    # For b >= 3, use Lloyd's algorithm on N(0, 1/d)
    sigma = 1.0 / np.sqrt(d)

    # Initialize boundaries from uniform quantiles
    boundaries = stats.norm.ppf(np.linspace(0, 1, n_centroids + 1)[1:-1], scale=sigma)
    centroids = np.zeros(n_centroids)

    # Initial centroids: conditional expectations
    def _gaussian_cond_exp(sigma, a, b):
        a_std = a / sigma if np.isfinite(a) else a
        b_std = b / sigma if np.isfinite(b) else b

        if not np.isfinite(a_std):
            prob = stats.norm.cdf(b_std)
        elif not np.isfinite(b_std):
            prob = stats.norm.sf(a_std)
        else:
            prob = stats.norm.cdf(b_std) - stats.norm.cdf(a_std)

        if prob < 1e-15:
            if np.isfinite(a) and not np.isfinite(b):
                return a + sigma
            elif not np.isfinite(a) and np.isfinite(b):
                return b - sigma
            elif np.isfinite(a) and np.isfinite(b):
                return (a + b) / 2.0
            return 0.0

        pdf_diff = stats.norm.pdf(a_std) - stats.norm.pdf(b_std)
        return sigma * pdf_diff / prob

    centroids[0] = _gaussian_cond_exp(sigma, -np.inf, boundaries[0])
    for i in range(1, n_centroids - 1):
        centroids[i] = _gaussian_cond_exp(sigma, boundaries[i - 1], boundaries[i])
    centroids[-1] = _gaussian_cond_exp(sigma, boundaries[-1], np.inf)

    # Lloyd iterations
    for _ in range(100):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        centroids[0] = _gaussian_cond_exp(sigma, -np.inf, boundaries[0])
        for i in range(1, n_centroids - 1):
            centroids[i] = _gaussian_cond_exp(sigma, boundaries[i - 1], boundaries[i])
        centroids[-1] = _gaussian_cond_exp(sigma, boundaries[-1], np.inf)

    return np.sort(centroids)


def _random_rotation_numpy(d: int, seed: int) -> np.ndarray:
    """Generate Haar-distributed random rotation matrix using NumPy (called once during init)."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(G)

    # Fix signs via diagonal of R
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs[np.newaxis, :]

    # Ensure det = +1 (proper rotation)
    sign, _ = np.linalg.slogdet(Q)
    if sign < 0:
        Q[:, 0] = -Q[:, 0]

    return Q


class PolarQuantTorch(nn.Module):
    """PyTorch-compatible PolarQuant for neural network feature compression.

    Implements the PolarQuant algorithm: random rotation + optimal scalar quantization.
    Supports GPU tensors and can be integrated into neural network pipelines.

    Args:
        d: Vector dimension (e.g., channel dimension or head_dim).
        bit_width: Bits per coordinate (2, 3, or 4 recommended).
        seed: Random seed for rotation matrix generation.
        norm_correction: Whether to renormalize during dequantization (improves accuracy).

    Example:
        >>> pq = PolarQuantTorch(d=128, bit_width=4, seed=42)
        >>> x = torch.randn(32, 128)  # batch of 32 vectors
        >>> indices, norms = pq.quantize(x)
        >>> x_hat = pq.dequantize(indices, norms)
        >>> mse = (x - x_hat).pow(2).mean()  # typically < 0.01 for 4-bit
    """

    def __init__(self, d: int, bit_width: int, seed: int = 42, norm_correction: bool = True):
        super().__init__()

        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        if bit_width < 1:
            raise ValueError(f"bit_width must be >= 1, got {bit_width}")

        self.d = d
        self.bit_width = bit_width
        self.n_centroids = 1 << bit_width
        self.norm_correction = norm_correction

        # Pre-compute centroids using NumPy, then convert to tensor
        centroids_np = _optimal_centroids_numpy(bit_width, d)
        self.register_buffer("centroids", torch.from_numpy(centroids_np).float())

        # Pre-compute rotation matrix using NumPy, then convert to tensor
        rotation_np = _random_rotation_numpy(d, seed)
        self.register_buffer("rotation", torch.from_numpy(rotation_np).float())

        # Pre-compute rotation transpose (same as rotation.T but explicit for clarity)
        self.register_buffer("rotation_t", self.rotation.T.contiguous())

        # Pre-compute boundaries for fast nearest centroid search
        # boundaries[i] = midpoint between centroids[i] and centroids[i+1]
        boundaries_np = (centroids_np[:-1] + centroids_np[1:]) / 2.0
        self.register_buffer("boundaries", torch.from_numpy(boundaries_np).float())

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize input vectors.

        Args:
            x: Input tensor of shape (d,) or (batch, d).

        Returns:
            indices: Integer indices into centroids, same shape as x.
            norms: L2 norms of input vectors, scalar or (batch,).
        """
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        # 1. Extract norms and normalize
        norms = torch.norm(x, dim=1)  # (batch,)
        safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        x_normalized = x / safe_norms.unsqueeze(1)  # (batch, d)

        # 2. Apply rotation: y = R @ x^T, then transpose back
        # x_normalized: (batch, d), rotation: (d, d)
        # y = x_normalized @ rotation.T = (batch, d) @ (d, d) = (batch, d)
        y = x_normalized @ self.rotation_t  # (batch, d)

        # 3. Find nearest centroid for each coordinate
        # Use searchsorted on boundaries for O(n log k) complexity
        indices = torch.searchsorted(self.boundaries, y)  # (batch, d)

        if single:
            return indices.squeeze(0), norms.squeeze(0)
        return indices, norms

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """Dequantize indices back to approximate vectors.

        Args:
            indices: Integer indices from quantize(), shape (d,) or (batch, d).
            norms: L2 norms from quantize(), scalar or (batch,).

        Returns:
            Reconstructed vectors, same shape as original input.
        """
        single = indices.dim() == 1
        if single:
            indices = indices.unsqueeze(0)
            norms = norms.unsqueeze(0) if norms.dim() == 0 else norms.unsqueeze(0)

        # 1. Look up centroids
        y_hat = self.centroids[indices]  # (batch, d)

        # 2. Optional norm correction (improves reconstruction)
        if self.norm_correction:
            y_hat_norms = torch.norm(y_hat, dim=1, keepdim=True)
            y_hat_norms = torch.where(y_hat_norms > 1e-10, y_hat_norms, torch.ones_like(y_hat_norms))
            y_hat = y_hat / y_hat_norms

        # 3. Apply inverse rotation: x_hat = R^T @ y_hat^T
        # y_hat: (batch, d), rotation: (d, d)
        # x_hat_unit = y_hat @ rotation = (batch, d) @ (d, d) = (batch, d)
        x_hat_unit = y_hat @ self.rotation  # (batch, d)

        # 4. Rescale by original norms
        x_hat = x_hat_unit * norms.unsqueeze(1)

        if single:
            return x_hat.squeeze(0)
        return x_hat

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize and dequantize in one step (useful for training with STE).

        This method allows gradient flow through the dequantization step while
        introducing quantization error as a form of regularization.

        Args:
            x: Input tensor of shape (d,) or (batch, d).

        Returns:
            Reconstructed tensor with quantization error injected.
        """
        indices, norms = self.quantize(x)
        return self.dequantize(indices, norms)

    def compression_ratio(self, original_bits: int = 16) -> float:
        """Compute compression ratio vs original precision.

        Args:
            original_bits: Bits per value in original representation (16 for fp16).

        Returns:
            Compression ratio (e.g., 4.0 means 4x smaller).
        """
        # Each coordinate: bit_width bits
        # Plus one float32 norm per vector
        original_per_vector = self.d * original_bits
        compressed_per_vector = self.d * self.bit_width + 32
        return original_per_vector / compressed_per_vector

    def extra_repr(self) -> str:
        return f"d={self.d}, bit_width={self.bit_width}, n_centroids={self.n_centroids}"