"""Signal and matrix purification defenses (SVD, 2D-FFT, and composite pipelines)."""

from faro_cv.defenses.base import BaseDefense, PurificationResult
from faro_cv.defenses.fft_purify import (
    FFTPurification,
    build_radial_frequency_mask,
    compute_high_freq_energy_ratio,
    compute_log_magnitude_spectrum,
)
from faro_cv.defenses.pipeline import CompositePurification, build_defense
from faro_cv.defenses.svd_purify import SVDPurification

__all__ = [
    "BaseDefense",
    "CompositePurification",
    "FFTPurification",
    "PurificationResult",
    "SVDPurification",
    "build_defense",
    "build_radial_frequency_mask",
    "compute_high_freq_energy_ratio",
    "compute_log_magnitude_spectrum",
]
