"""Composite purification pipeline chaining 2D-FFT and SVD operators."""

from __future__ import annotations

from torch import Tensor

from faro_cv.core.types import DefenseConfig, OperatorDeviceProfile, SpectralSummary
from faro_cv.defenses.base import BaseDefense, PurificationResult
from faro_cv.defenses.fft_purify import FFTPurification
from faro_cv.defenses.svd_purify import SVDPurification


class CompositePurification(BaseDefense):
    """Chains SVD and 2D-FFT purification operators in a user-specified order."""

    name = "pipeline"

    def __init__(
        self,
        config: DefenseConfig,
        *,
        device_profile: OperatorDeviceProfile | None = None,
    ) -> None:
        super().__init__(config)
        if config.svd is None or config.fft is None:
            raise ValueError("CompositePurification requires both svd and fft parameters.")

        self.stages: list[BaseDefense] = []
        for stage_kind in config.pipeline_order:
            if stage_kind == "svd":
                self.stages.append(
                    SVDPurification(
                        DefenseConfig(kind="svd", svd=config.svd),
                        device_profile=device_profile,
                    )
                )
            elif stage_kind == "fft":
                self.stages.append(
                    FFTPurification(
                        DefenseConfig(kind="fft", fft=config.fft),
                        device_profile=device_profile,
                    )
                )

    def purify(self, image_tensor: Tensor) -> PurificationResult:
        current = image_tensor
        total_latency_ms = 0.0
        first_summary: SpectralSummary | None = None
        last_summary: SpectralSummary | None = None

        for stage in self.stages:
            res = stage.purify(current)
            current = res.purified_tensor
            total_latency_ms += res.latency_ms
            if first_summary is None:
                first_summary = res.spectral_summary
            last_summary = res.spectral_summary

        assert first_summary is not None and last_summary is not None
        merged = SpectralSummary(
            singular_values=last_summary.singular_values,
            cumulative_energy=last_summary.cumulative_energy,
            effective_rank=last_summary.effective_rank,
            retained_energy_ratio=last_summary.retained_energy_ratio,
            high_freq_energy_before=first_summary.high_freq_energy_before,
            high_freq_energy_after=last_summary.high_freq_energy_after,
        )
        return PurificationResult(
            purified_tensor=current,
            spectral_summary=merged,
            latency_ms=round(total_latency_ms, 3),
        )


def build_defense(
    config: DefenseConfig,
    *,
    device_profile: OperatorDeviceProfile | None = None,
) -> BaseDefense:
    """Factory creating a concrete BaseDefense from a validated DefenseConfig."""
    if config.kind == "svd":
        return SVDPurification(config, device_profile=device_profile)
    if config.kind == "fft":
        return FFTPurification(config, device_profile=device_profile)
    if config.kind == "pipeline":
        return CompositePurification(config, device_profile=device_profile)
    raise ValueError(f"Unsupported defense kind '{config.kind}'.")
