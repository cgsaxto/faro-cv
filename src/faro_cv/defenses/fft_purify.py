"""Differentiable 2D-FFT frequency-domain purification and spectral telemetry."""

from __future__ import annotations

import math
import time

import torch
from torch import Tensor

from faro_cv.core.types import (
    ComputeDevice,
    DefenseConfig,
    FFTFilterType,
    FFTParams,
    OperatorDeviceProfile,
    SpectralSummary,
)
from faro_cv.defenses.base import BaseDefense, PurificationResult


def build_radial_frequency_mask(
    height: int,
    width: int,
    *,
    filter_type: FFTFilterType,
    cutoff_radius: float,
    band_width: float = 0.1,
    butterworth_order: int = 2,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Build a centered 2D radial frequency transfer function ``H(u, v)`` of shape ``[H, W]``.

    Uses exact DFT bin frequencies via ``fftshift(fftfreq(n)) / 0.5`` (normalized by the
    Nyquist frequency of 0.5 cycles/pixel) so the DC bin at ``(H // 2, W // 2)`` sits at
    exact radius ``0.0`` (unit DC gain) and positive/negative frequency pairs ``(+f, -f)``
    receive identical weights for both even and odd image dimensions.
    """
    fy = (
        torch.fft.fftshift(torch.fft.fftfreq(height, d=1.0, device=device, dtype=torch.float32))
        / 0.5
    )
    fx = (
        torch.fft.fftshift(torch.fft.fftfreq(width, d=1.0, device=device, dtype=torch.float32))
        / 0.5
    )
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(grid_x.pow(2) + grid_y.pow(2))

    rc = max(cutoff_radius, 1e-4)

    if filter_type == "gaussian":
        return torch.exp(-radius.pow(2) / (2.0 * rc * rc))
    if filter_type == "butterworth":
        return 1.0 / (1.0 + (radius / rc).pow(2 * butterworth_order))
    if filter_type == "ideal":
        return (radius <= rc).to(dtype=torch.float32)
    if filter_type == "bandstop":
        bw = max(band_width, 1e-4)
        return 1.0 - torch.exp(-((radius - rc).pow(2)) / (2.0 * bw * bw))

    raise ValueError(f"Unsupported FFT filter_type '{filter_type}'.")


def compute_log_magnitude_spectrum(
    image_tensor: Tensor,
    *,
    fft_device: ComputeDevice = "cpu",
) -> Tensor:
    """Return centered 2D log-magnitude spectrum normalized to ``[0, 1]`` of shape ``[H, W]``."""
    x = image_tensor.detach().to(device=fft_device, dtype=torch.float32)
    if x.ndim == 4:
        x = x[0]
    spec = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
    mag = torch.log1p(torch.abs(spec)).mean(dim=0)
    min_val = mag.min()
    max_val = mag.max()
    return (mag - min_val) / (max_val - min_val + 1e-12)


def compute_high_freq_energy_ratio(
    image_tensor: Tensor,
    *,
    cutoff_radius: float = 0.35,
    fft_device: ComputeDevice = "cpu",
) -> float:
    """Fraction of total 2D spectral power ``|F(u, v)|^2`` lying strictly outside ``cutoff_radius``."""
    x = image_tensor.detach().to(device=fft_device, dtype=torch.float32)
    if x.ndim == 4:
        x = x[0]
    height, width = int(x.shape[-2]), int(x.shape[-1])
    spec = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
    power = torch.abs(spec).pow(2)

    low_mask = build_radial_frequency_mask(
        height,
        width,
        filter_type="ideal",
        cutoff_radius=cutoff_radius,
        device=x.device,
    )
    high_mask = 1.0 - low_mask
    total_power = float(power.sum().item())
    if total_power <= 1e-12:
        return 0.0
    high_power = float((power * high_mask).sum().item())
    return max(0.0, min(1.0, high_power / total_power))


class FFTPurification(BaseDefense):
    """Differentiable 2D-FFT radial/band-stop frequency purification operator."""

    name = "fft"

    def __init__(
        self,
        config: DefenseConfig,
        *,
        device_profile: OperatorDeviceProfile | None = None,
    ) -> None:
        super().__init__(config)
        if config.fft is None:
            raise ValueError("FFTPurification requires config.fft to be set.")
        self.params: FFTParams = config.fft
        self.op_device: ComputeDevice = device_profile.fft_op.device if device_profile else "cpu"

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
        height, width = int(x.shape[-2]), int(x.shape[-1])

        hf_before = compute_high_freq_energy_ratio(
            x,
            cutoff_radius=self.params.cutoff_radius,
            fft_device=self.op_device,
        )

        # Forward 2D-FFT and center DC component
        spectrum = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        mask = build_radial_frequency_mask(
            height,
            width,
            filter_type=self.params.filter_type,
            cutoff_radius=self.params.cutoff_radius,
            band_width=self.params.band_width,
            butterworth_order=self.params.butterworth_order,
            device=x.device,
        )

        # Multiplying complex spectrum by real non-negative H(u,v) preserves phase exactly;
        # preserve_phase=False runs zero-phase magnitude-only ablation.
        filtered = spectrum * mask if self.params.preserve_phase else torch.abs(spectrum) * mask

        unshifted = torch.fft.ifftshift(filtered, dim=(-2, -1))
        reconstructed = torch.fft.ifft2(unshifted, dim=(-2, -1)).real
        purified = reconstructed.clamp(0.0, 1.0)

        hf_after = compute_high_freq_energy_ratio(
            purified,
            cutoff_radius=self.params.cutoff_radius,
            fft_device=self.op_device,
        )

        # Compute singular value diagnostics on the purified image for unified UI telemetry
        with torch.no_grad():
            svals = torch.linalg.svdvals(purified[0]).mean(dim=0)
            energy = svals.pow(2)
            total_e = float(energy.sum().item())
            cum_energy = (
                (torch.cumsum(energy, dim=0) / total_e).cpu().tolist()
                if total_e > 1e-12
                else [1.0] * int(svals.shape[0])
            )
            cum_energy = [max(0.0, min(1.0, float(v))) for v in cum_energy]
            retained_ratio = max(0.0, min(1.0, 1.0 - max(0.0, hf_before - hf_after)))
            effective_rank = max(1, math.ceil(min(height, width) * self.params.cutoff_radius))

        out = purified.to(orig_device)
        if was_unbatched:
            out = out.squeeze(0)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        summary = SpectralSummary(
            singular_values=[float(v) for v in svals.cpu().tolist()],
            cumulative_energy=cum_energy,
            effective_rank=min(effective_rank, int(svals.shape[0])),
            retained_energy_ratio=round(retained_ratio, 6),
            high_freq_energy_before=round(hf_before, 6),
            high_freq_energy_after=round(hf_after, 6),
        )
        return PurificationResult(
            purified_tensor=out,
            spectral_summary=summary,
            latency_ms=round(elapsed_ms, 3),
        )
