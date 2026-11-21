"""Validation and serialization tests for core domain schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from faro_cv.core.types import AttackConfig, DefenseConfig, FFTParams, SVDParams


def test_attack_config_validation_rules() -> None:
    # FGSM must enforce steps == 1
    with pytest.raises(ValidationError):
        AttackConfig(kind="fgsm", steps=5)

    # Targeted attack must include target_class
    with pytest.raises(ValidationError):
        AttackConfig(kind="pgd", targeted=True, target_class=None)

    # Patch EoT attack requires patch_size
    with pytest.raises(ValidationError):
        AttackConfig(kind="patch_eot", patch_size=None)

    valid_patch = AttackConfig(
        kind="patch_eot",
        steps=50,
        patch_size=48,
        patch_x=20,
        patch_y=20,
        eot_samples=8,
    )
    round_trip = AttackConfig.model_validate_json(valid_patch.model_dump_json())
    assert round_trip == valid_patch


def test_defense_config_validation_rules() -> None:
    # SVDParams requires rank_k or energy_threshold
    with pytest.raises(ValidationError):
        SVDParams()

    # DefenseConfig kind='svd' requires svd params
    with pytest.raises(ValidationError):
        DefenseConfig(kind="svd")

    # DefenseConfig kind='pipeline' requires both svd and fft
    with pytest.raises(ValidationError):
        DefenseConfig(kind="pipeline", svd=SVDParams(rank_k=16))

    valid_pipeline = DefenseConfig(
        kind="pipeline",
        svd=SVDParams(energy_threshold=0.95),
        fft=FFTParams(filter_type="butterworth", cutoff_radius=0.4, butterworth_order=3),
        pipeline_order=("fft", "svd"),
    )
    assert valid_pipeline.pipeline_order == ("fft", "svd")
