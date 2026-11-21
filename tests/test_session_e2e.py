"""End-to-end tests for LabSession integrating FSM, ResNet-18, PGD, SVD/FFT, and MPS."""

from __future__ import annotations

import torch

from faro_cv.core.fsm import LabState
from faro_cv.core.runner import LabSession
from faro_cv.core.types import AttackConfig, DefenseConfig, FFTParams, SVDParams


def _assert_state(session: LabSession, expected: LabState) -> None:
    assert session.fsm.state is expected


def test_lab_session_end_to_end_with_resnet18_and_repurification() -> None:
    session = LabSession(seed=42, prefer_mps=True)
    _assert_state(session, LabState.IDLE)
    assert session.device_profile.torch_available is True

    # Load offline randomly-initialized ResNet-18 for fast CI/local integration verification
    loaded = session.load_model("resnet18", pretrained=False)
    assert loaded.spec.model_id == "resnet18"
    _assert_state(session, LabState.MODEL_READY)

    # Load synthetic 64x64 RGB image in [0, 1]
    torch.manual_seed(123)
    clean = torch.rand(3, 64, 64)
    session.load_image(clean)
    _assert_state(session, LabState.INPUT_LOADED)
    assert session.fsm.context.clean_inference is not None

    # Execute PGD attack
    pgd = AttackConfig(kind="pgd", norm="linf", epsilon=8.0 / 255.0, step_size=2.0 / 255.0, steps=4)
    attack_res = session.run_attack(pgd)
    assert attack_res.step == 4
    _assert_state(session, LabState.ATTACKED)
    assert session.adversarial_tensor is not None

    # Purify via Truncated SVD
    svd_cfg = DefenseConfig(kind="svd", svd=SVDParams(rank_k=16))
    pur_res = session.purify(svd_cfg)
    assert pur_res.spectral_summary.effective_rank == 16
    _assert_state(session, LabState.PURIFIED)

    # Evaluate comparative metrics
    metrics = session.evaluate()
    _assert_state(session, LabState.EVALUATED)
    assert metrics.attack_linf_norm <= 8.0 / 255.0 + 1e-5
    assert -1.0 <= metrics.purified_ssim <= 1.0

    # Simulate dragging the SVD/FFT controls directly from EVALUATED state
    pipe_cfg = DefenseConfig(
        kind="pipeline",
        svd=SVDParams(rank_k=10),
        fft=FFTParams(filter_type="gaussian", cutoff_radius=0.35),
    )
    session.purify(pipe_cfg)
    _assert_state(session, LabState.PURIFIED)
    metrics_2 = session.evaluate()
    _assert_state(session, LabState.EVALUATED)
    assert metrics_2.purification_latency_ms >= 0.0

    # Reset back to IDLE
    session.reset()
    _assert_state(session, LabState.IDLE)
    assert session.clean_tensor is None
