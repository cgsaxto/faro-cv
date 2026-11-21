"""Shared pytest fixtures for faro-cv unit and integration tests."""

from __future__ import annotations

import pytest

from faro_cv.core.types import (
    AttackConfig,
    DefenseConfig,
    EvaluationMetrics,
    FFTParams,
    InferenceSummary,
    PredictionEntry,
    SpectralSummary,
    SVDParams,
    TensorMetadata,
)


@pytest.fixture
def sample_tensor_meta() -> TensorMetadata:
    return TensorMetadata(
        width=224,
        height=224,
        channels=3,
        dtype="float32",
        sha256="a1b2c3d4e5f60718",
        min_value=0.0,
        max_value=1.0,
    )


@pytest.fixture
def clean_inference() -> InferenceSummary:
    return InferenceSummary(
        top_k=[
            PredictionEntry(class_index=281, label="tabby cat", probability=0.94),
            PredictionEntry(class_index=282, label="tiger cat", probability=0.04),
        ],
        entropy=0.28,
        latency_ms=4.1,
        device="mps",
    )


@pytest.fixture
def adversarial_inference() -> InferenceSummary:
    return InferenceSummary(
        top_k=[
            PredictionEntry(class_index=859, label="toaster", probability=0.89),
            PredictionEntry(class_index=281, label="tabby cat", probability=0.06),
        ],
        entropy=0.41,
        latency_ms=4.2,
        device="mps",
    )


@pytest.fixture
def purified_inference() -> InferenceSummary:
    return InferenceSummary(
        top_k=[
            PredictionEntry(class_index=281, label="tabby cat", probability=0.86),
            PredictionEntry(class_index=282, label="tiger cat", probability=0.08),
        ],
        entropy=0.52,
        latency_ms=4.3,
        device="mps",
    )


@pytest.fixture
def pgd_config() -> AttackConfig:
    return AttackConfig(
        kind="pgd",
        norm="linf",
        epsilon=8.0 / 255.0,
        step_size=2.0 / 255.0,
        steps=10,
    )


@pytest.fixture
def svd_defense_config() -> DefenseConfig:
    return DefenseConfig(
        kind="svd",
        svd=SVDParams(rank_k=28),
    )


@pytest.fixture
def pipeline_defense_config() -> DefenseConfig:
    return DefenseConfig(
        kind="pipeline",
        svd=SVDParams(rank_k=32),
        fft=FFTParams(filter_type="gaussian", cutoff_radius=0.35),
    )


@pytest.fixture
def spectral_summary() -> SpectralSummary:
    return SpectralSummary(
        singular_values=[120.5, 42.1, 18.4, 9.2, 3.1],
        cumulative_energy=[0.84, 0.94, 0.98, 0.995, 1.0],
        effective_rank=28,
        retained_energy_ratio=0.964,
        high_freq_energy_before=0.31,
        high_freq_energy_after=0.11,
    )


@pytest.fixture
def evaluation_metrics() -> EvaluationMetrics:
    return EvaluationMetrics(
        attack_l2_norm=3.42,
        attack_linf_norm=8.0 / 255.0,
        purified_psnr_db=31.8,
        purified_ssim=0.92,
        attack_succeeded=True,
        prediction_consistent=True,
        conditional_recovered=True,
        clean_confidence=0.94,
        adversarial_confidence=0.89,
        purified_confidence=0.86,
        purification_latency_ms=5.8,
    )
