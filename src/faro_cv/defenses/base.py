"""Abstract defense interface for differentiable signal and matrix purification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from faro_cv.core.types import DefenseConfig, SpectralSummary


@dataclass(frozen=True, slots=True)
class PurificationResult:
    """Output bundle produced by a purification operator or chained pipeline."""

    purified_tensor: Any
    spectral_summary: SpectralSummary
    latency_ms: float


class BaseDefense(ABC):
    """Contract for SVD, 2D-FFT, and composite purification pipelines."""

    name: str

    def __init__(self, config: DefenseConfig) -> None:
        self.config = config

    @abstractmethod
    def purify(self, image_tensor: Any) -> PurificationResult:
        """Purify an input image tensor and return reconstructed tensor + spectral telemetry."""
