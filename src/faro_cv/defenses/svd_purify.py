"""Differentiable Truncated SVD & Singular Value Energy Thresholding purification."""

from __future__ import annotations

import time

import torch
from torch import Tensor

from faro_cv.core.types import (
    ComputeDevice,
    DefenseConfig,
    OperatorDeviceProfile,
    SpectralSummary,
    SVDParams,
)
from faro_cv.defenses.base import BaseDefense, PurificationResult
from faro_cv.defenses.fft_purify import compute_high_freq_energy_ratio


def _rgb_to_ycbcr(rgb: Tensor) -> Tensor:
    """Differentiable BT.601 RGB -> YCbCr conversion on ``[B, 3, H, W]`` in ``[0, 1]``."""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.cat([y, cb, cr], dim=1)


def _ycbcr_to_rgb(ycbcr: Tensor) -> Tensor:
    """Differentiable BT.601 YCbCr -> RGB conversion on ``[B, 3, H, W]``."""
    y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2] - 0.5, ycbcr[:, 2:3] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.cat([r, g, b], dim=1)


def _select_rank_k(
    singular_values: Tensor,
    *,
    rank_k: int | None,
    energy_threshold: float | None,
) -> tuple[int, list[float], float]:
    """Resolve truncation rank ``k`` and compute cumulative singular value energy curve."""
    max_rank = int(singular_values.shape[-1])
    energy = singular_values.detach().pow(2)
    total_energy = float(energy.sum().item())

    if total_energy <= 1e-12:
        cum_list = [1.0] * max_rank
        return 1, cum_list, 1.0

    cum_tensor = torch.cumsum(energy, dim=0) / total_energy
    cum_list = [max(0.0, min(1.0, float(v))) for v in cum_tensor.cpu().tolist()]

    if rank_k is not None:
        k = max(1, min(rank_k, max_rank))
    else:
        assert energy_threshold is not None
        # Smallest k such that cumulative energy >= threshold
        indices = torch.nonzero(cum_tensor >= energy_threshold, as_tuple=False)
        k = int(indices[0].item()) + 1 if indices.numel() > 0 else max_rank

    retained = cum_list[k - 1]
    return k, cum_list, retained


def _stabilize_for_svd_grad(matrix: Tensor) -> Tensor:
    """Break exact singular value degeneracies (sigma_i == sigma_j == 0) during autograd.

    The exact SVD Jacobian contains ``1 / (sigma_j^2 - sigma_i^2)`` terms which produce NaNs
    on rank-deficient regions (e.g., flat sky or synthetic rank-1 channels). Adding a fixed
    1e-5 deterministic jitter strictly when gradients are enabled ensures finite backward
    passes with negligible (<0.003 LSB) forward impact.
    """
    if not (torch.is_grad_enabled() and matrix.requires_grad):
        return matrix
    gen = torch.Generator(device="cpu").manual_seed(0)
    jitter = torch.randn(matrix.shape, generator=gen, dtype=matrix.dtype) * 1e-5
    return matrix + jitter.to(device=matrix.device)


def _truncate_matrix_svd(matrix: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Differentiable rank-k Eckart-Young-Mirsky projection of ``[..., H, W]``."""
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    k_clamped = max(1, min(k, int(s.shape[-1])))
    u_k = u[..., :, :k_clamped]
    s_k = s[..., :k_clamped]
    vh_k = vh[..., :k_clamped, :]
    reconstructed = (u_k * s_k.unsqueeze(-2)) @ vh_k
    return reconstructed, s


class SVDPurification(BaseDefense):
    """Differentiable Truncated SVD matrix purification (per-channel or luminance Y)."""

    name = "svd"

    def __init__(
        self,
        config: DefenseConfig,
        *,
        device_profile: OperatorDeviceProfile | None = None,
    ) -> None:
        super().__init__(config)
        if config.svd is None:
            raise ValueError("SVDPurification requires config.svd to be set.")
        self.params: SVDParams = config.svd
        self.op_device: ComputeDevice = device_profile.svd_op.device if device_profile else "cpu"
        self.fft_device: ComputeDevice = device_profile.fft_op.device if device_profile else "cpu"

    def purify(self, image_tensor: Tensor) -> PurificationResult:
        orig_device = image_tensor.device
        was_unbatched = image_tensor.ndim == 3
        batched = image_tensor.unsqueeze(0) if was_unbatched else image_tensor
        if batched.ndim != 4 or batched.shape[1] != 3:
            raise ValueError(
                f"Expected [3, H, W] or [B, 3, H, W] tensor, got {tuple(image_tensor.shape)}."
            )

        start = time.perf_counter()
        x = batched.to(device=self.op_device, dtype=torch.float32)

        hf_before = compute_high_freq_energy_ratio(x, fft_device=self.fft_device)

        if self.params.channel_mode == "per_channel":
            u, s, vh = torch.linalg.svd(_stabilize_for_svd_grad(x), full_matrices=False)
            mean_s = s[0].mean(dim=0)
            k, cum_energy, retained = _select_rank_k(
                mean_s,
                rank_k=self.params.rank_k,
                energy_threshold=self.params.energy_threshold,
            )
            reconstructed = (u[..., :, :k] * s[..., :k].unsqueeze(-2)) @ vh[..., :k, :]
            singular_values_list = [float(v) for v in mean_s.detach().cpu().tolist()]
        else:
            # Luminance-channel SVD (preserves chrominance fidelity while denoising structure)
            ycbcr = _rgb_to_ycbcr(x)
            y_channel = ycbcr[:, 0:1]
            u, s, vh = torch.linalg.svd(_stabilize_for_svd_grad(y_channel), full_matrices=False)
            y_s = s[0, 0]
            k, cum_energy, retained = _select_rank_k(
                y_s,
                rank_k=self.params.rank_k,
                energy_threshold=self.params.energy_threshold,
            )
            y_recon = (u[..., :, :k] * s[..., :k].unsqueeze(-2)) @ vh[..., :k, :]
            ycbcr_recon = torch.cat([y_recon, ycbcr[:, 1:]], dim=1)
            reconstructed = _ycbcr_to_rgb(ycbcr_recon)
            singular_values_list = [float(v) for v in y_s.detach().cpu().tolist()]

        purified = reconstructed.clamp(0.0, 1.0)
        hf_after = compute_high_freq_energy_ratio(purified, fft_device=self.fft_device)

        out = purified.to(orig_device)
        if was_unbatched:
            out = out.squeeze(0)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        summary = SpectralSummary(
            singular_values=singular_values_list,
            cumulative_energy=cum_energy,
            effective_rank=k,
            retained_energy_ratio=round(retained, 6),
            high_freq_energy_before=round(hf_before, 6),
            high_freq_energy_after=round(hf_after, 6),
        )
        return PurificationResult(
            purified_tensor=out,
            spectral_summary=summary,
            latency_ms=round(elapsed_ms, 3),
        )
