"""Tests for Apple Silicon MPS-aware device capability probing."""

from __future__ import annotations

from faro_cv.core.device import probe_device_capabilities


def test_probe_device_capabilities_returns_valid_profile() -> None:
    profile = probe_device_capabilities(prefer_mps=True)
    assert profile.preferred_accelerator in ("mps", "cuda", "cpu")
    assert profile.model_op.device in ("mps", "cuda", "cpu")
    assert profile.svd_op.device in ("mps", "cuda", "cpu")
    assert profile.fft_op.device in ("mps", "cuda", "cpu")
    assert profile.model_op.probe_latency_ms >= 0.0
    assert profile.svd_op.probe_latency_ms >= 0.0
    assert profile.fft_op.probe_latency_ms >= 0.0
