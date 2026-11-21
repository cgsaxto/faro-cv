"""Tests for FGSM, PGD, differentiable SVD/2D-FFT purification, and fidelity metrics."""

from __future__ import annotations

from typing import Literal

import pytest
import torch
from torch import Tensor, nn

from faro_cv.analysis.metrics import (
    compute_evaluation_metrics,
    compute_perturbation_norms,
    compute_psnr,
    compute_ssim,
)
from faro_cv.attacks.perturbation import FGSMAttack, PGDAttack, build_attack
from faro_cv.core.types import (
    AttackConfig,
    DefenseConfig,
    FFTFilterType,
    FFTParams,
    InferenceSummary,
    NormKind,
    PredictionEntry,
    SVDParams,
)
from faro_cv.defenses.fft_purify import (
    FFTPurification,
    build_radial_frequency_mask,
    compute_log_magnitude_spectrum,
)
from faro_cv.defenses.pipeline import CompositePurification, build_defense
from faro_cv.defenses.svd_purify import SVDPurification


class _TinyConvClassifier(nn.Module):
    """Fast deterministic vision classifier for unit testing attacks without weight downloads."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(8 * 4 * 4, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.net(x)
        return out


def _synthetic_image(height: int = 64, width: int = 64) -> Tensor:
    """Smooth low-rank gradient + structured pattern image in [0, 1]."""
    y = torch.linspace(0.0, 1.0, height).unsqueeze(1).expand(height, width)
    x = torch.linspace(0.0, 1.0, width).unsqueeze(0).expand(height, width)
    r = 0.5 + 0.35 * torch.sin(3.0 * x) * torch.cos(2.0 * y)
    g = 0.4 + 0.4 * (x * y)
    b = 0.3 + 0.4 * torch.cos(4.0 * (x + y))
    return torch.stack([r, g, b], dim=0).clamp(0.0, 1.0)


@pytest.mark.parametrize("norm", ["linf", "l2"])
def test_fgsm_respects_epsilon_and_pixel_bounds(norm: NormKind) -> None:
    model = _TinyConvClassifier().eval()
    img = _synthetic_image(32, 32)
    eps = 8.0 / 255.0 if norm == "linf" else 0.5
    cfg = AttackConfig(kind="fgsm", norm=norm, epsilon=eps, steps=1)

    attack = build_attack(cfg)
    assert isinstance(attack, FGSMAttack)
    res = attack.run(model, img, label_index=0)

    assert res.step == 1
    assert res.total_steps == 1
    assert res.adversarial_tensor.shape == img.shape
    assert float(res.adversarial_tensor.min().item()) >= 0.0
    assert float(res.adversarial_tensor.max().item()) <= 1.0

    l2, linf = compute_perturbation_norms(img, res.adversarial_tensor)
    if norm == "linf":
        assert linf <= eps + 1e-5
    else:
        assert l2 <= eps + 1e-4


def test_pgd_streams_steps_deterministically_and_raises_loss() -> None:
    model = _TinyConvClassifier().eval()
    img = _synthetic_image(32, 32)
    eps = 16.0 / 255.0
    cfg = AttackConfig(kind="pgd", norm="linf", epsilon=eps, step_size=4.0 / 255.0, steps=6)

    steps_a = list(PGDAttack(cfg, seed=7).iterate(model, img, label_index=0))
    steps_b = list(PGDAttack(cfg, seed=7).iterate(model, img, label_index=0))

    assert len(steps_a) == 6
    assert [s.step for s in steps_a] == [1, 2, 3, 4, 5, 6]
    assert torch.allclose(steps_a[-1].adversarial_tensor, steps_b[-1].adversarial_tensor)

    # Untargeted PGD should increase cross-entropy loss over optimization steps
    assert steps_a[-1].loss > steps_a[0].loss
    _, linf = compute_perturbation_norms(img, steps_a[-1].adversarial_tensor)
    assert linf <= eps + 1e-5


def test_pgd_targeted_decreases_target_loss() -> None:
    model = _TinyConvClassifier().eval()
    img = _synthetic_image(32, 32)
    cfg = AttackConfig(
        kind="pgd",
        norm="linf",
        epsilon=16.0 / 255.0,
        step_size=4.0 / 255.0,
        steps=8,
        targeted=True,
        target_class=3,
    )
    steps = list(PGDAttack(cfg, seed=11, random_start=False).iterate(model, img, label_index=0))
    assert steps[-1].loss < steps[0].loss


def test_svd_purification_rank_energy_luminance_and_differentiability() -> None:
    img = _synthetic_image(48, 48)
    torch.manual_seed(42)
    noisy = (img + 0.08 * torch.randn_like(img)).clamp(0.0, 1.0)

    # Fixed rank-k per-channel
    svd_fixed = SVDPurification(DefenseConfig(kind="svd", svd=SVDParams(rank_k=8)))
    res_fixed = svd_fixed.purify(noisy)
    assert res_fixed.purified_tensor.shape == noisy.shape
    assert res_fixed.spectral_summary.effective_rank == 8
    assert res_fixed.spectral_summary.high_freq_energy_after < (
        res_fixed.spectral_summary.high_freq_energy_before
    )

    # Adaptive energy threshold
    svd_energy = SVDPurification(DefenseConfig(kind="svd", svd=SVDParams(energy_threshold=0.98)))
    res_energy = svd_energy.purify(noisy)
    assert res_energy.spectral_summary.retained_energy_ratio >= 0.98

    # Luminance-channel mode
    svd_luma = SVDPurification(
        DefenseConfig(kind="svd", svd=SVDParams(rank_k=10, channel_mode="luminance"))
    )
    res_luma = svd_luma.purify(noisy)
    assert res_luma.purified_tensor.shape == noisy.shape

    # End-to-end differentiability check (crucial for white-box adaptive attacks)
    x_var = noisy.clone().detach().requires_grad_(True)
    purified = svd_fixed.purify(x_var).purified_tensor
    purified.pow(2).mean().backward()
    assert x_var.grad is not None
    assert torch.isfinite(x_var.grad).all()
    assert float(x_var.grad.abs().sum().item()) > 0.0


@pytest.mark.parametrize("filter_type", ["gaussian", "butterworth", "ideal", "bandstop"])
def test_fft_purification_filters_and_differentiability(filter_type: FFTFilterType) -> None:
    img = _synthetic_image(48, 48)
    torch.manual_seed(99)
    noisy = (img + 0.08 * torch.randn_like(img)).clamp(0.0, 1.0)

    cfg = DefenseConfig(
        kind="fft",
        fft=FFTParams(
            filter_type=filter_type,
            cutoff_radius=0.35,
            band_width=0.12,
            butterworth_order=2,
        ),
    )
    defense = FFTPurification(cfg)
    x_var = noisy.clone().detach().requires_grad_(True)
    res = defense.purify(x_var)

    assert res.purified_tensor.shape == noisy.shape
    assert float(res.purified_tensor.min().item()) >= 0.0
    assert float(res.purified_tensor.max().item()) <= 1.0

    if filter_type != "bandstop":
        assert res.spectral_summary.high_freq_energy_after < (
            res.spectral_summary.high_freq_energy_before
        )

    res.purified_tensor.pow(2).mean().backward()
    assert x_var.grad is not None
    assert torch.isfinite(x_var.grad).all()
    assert float(x_var.grad.abs().sum().item()) > 0.0

    spectrum = compute_log_magnitude_spectrum(noisy)
    assert spectrum.shape == (48, 48)
    assert float(spectrum.min().item()) >= 0.0
    assert float(spectrum.max().item()) <= 1.0 + 1e-6


@pytest.mark.parametrize("size", [224, 225])
@pytest.mark.parametrize("filter_type", ["gaussian", "butterworth", "ideal"])
def test_fft_mask_exact_dc_unity_and_positive_negative_symmetry(
    size: int,
    filter_type: FFTFilterType,
) -> None:
    mask = build_radial_frequency_mask(
        size,
        size,
        filter_type=filter_type,
        cutoff_radius=0.35,
    )
    cy, cx = size // 2, size // 2
    # Exact DC bin must have unit gain (1.0), even for even-length grids like 224
    assert float(mask[cy, cx].item()) == pytest.approx(1.0, abs=1e-7)

    # Positive and negative frequencies (+k, -k) must have identical gain
    max_sym = (size - 1) // 2
    for offset in (1, 2, 7, 31, max_sym):
        assert float(mask[cy, cx + offset].item()) == pytest.approx(
            float(mask[cy, cx - offset].item()), abs=1e-7
        )
        assert float(mask[cy + offset, cx].item()) == pytest.approx(
            float(mask[cy - offset, cx].item()), abs=1e-7
        )
        assert float(mask[cy + offset, cx + offset].item()) == pytest.approx(
            float(mask[cy - offset, cx - offset].item()), abs=1e-7
        )


def test_fft_gaussian_preserves_constant_image_luminance() -> None:
    const_img = torch.full((3, 224, 224), 0.62, dtype=torch.float32)
    defense = FFTPurification(
        DefenseConfig(
            kind="fft",
            fft=FFTParams(filter_type="gaussian", cutoff_radius=0.35, preserve_phase=True),
        )
    )
    res = defense.purify(const_img)
    assert torch.allclose(res.purified_tensor, const_img, atol=1e-6)


def test_composite_pipeline_both_orders() -> None:
    img = _synthetic_image(48, 48)
    orders: tuple[tuple[Literal["svd", "fft"], ...], ...] = (("fft", "svd"), ("svd", "fft"))
    for order in orders:
        cfg = DefenseConfig(
            kind="pipeline",
            svd=SVDParams(rank_k=12),
            fft=FFTParams(filter_type="gaussian", cutoff_radius=0.4),
            pipeline_order=order,
        )
        defense = build_defense(cfg)
        assert isinstance(defense, CompositePurification)
        res = defense.purify(img)
        assert res.purified_tensor.shape == img.shape
        assert res.latency_ms >= 0.0


def test_metrics_psnr_ssim_and_norms() -> None:
    img = _synthetic_image(48, 48)
    l2_zero, linf_zero = compute_perturbation_norms(img, img)
    assert l2_zero == 0.0
    assert linf_zero == 0.0
    assert compute_psnr(img, img) == 100.0
    assert compute_ssim(img, img) == pytest.approx(1.0, abs=1e-4)

    perturbed = (img + 0.05).clamp(0.0, 1.0)
    l2, linf = compute_perturbation_norms(img, perturbed)
    assert l2 > 0.0
    assert linf <= 0.05 + 1e-5
    assert 15.0 < compute_psnr(img, perturbed) < 40.0
    assert 0.5 < compute_ssim(img, perturbed) < 1.0


def test_split_consistency_conditional_recovery_and_ground_truth_accuracy() -> None:
    img = _synthetic_image(32, 32)
    inf_a = InferenceSummary(
        top_k=[PredictionEntry(class_index=10, label="class_10", probability=0.8)],
        entropy=0.5,
        latency_ms=1.0,
        device="cpu",
    )
    inf_b = InferenceSummary(
        top_k=[PredictionEntry(class_index=20, label="class_20", probability=0.7)],
        entropy=0.6,
        latency_ms=1.0,
        device="cpu",
    )

    # Case 1: Attack did NOT flip prediction (clean=10, adv=10, pur=10), ground truth=99 (clean was WRONG!)
    ev_unflipped = compute_evaluation_metrics(
        clean_tensor=img,
        adversarial_tensor=img,
        purified_tensor=img,
        clean_inference=inf_a,
        adversarial_inference=inf_a,
        purified_inference=inf_a,
        purification_latency_ms=2.0,
        ground_truth_class=99,
    )
    assert ev_unflipped.attack_succeeded is False
    assert ev_unflipped.prediction_consistent is True
    assert ev_unflipped.conditional_recovered is None
    assert ev_unflipped.clean_correct is False
    assert ev_unflipped.purified_correct is False

    # Case 2: Attack flipped prediction (clean=10, adv=20, pur=10), ground truth=10
    ev_recovered = compute_evaluation_metrics(
        clean_tensor=img,
        adversarial_tensor=img,
        purified_tensor=img,
        clean_inference=inf_a,
        adversarial_inference=inf_b,
        purified_inference=inf_a,
        purification_latency_ms=2.0,
        ground_truth_class=10,
    )
    assert ev_recovered.attack_succeeded is True
    assert ev_recovered.prediction_consistent is True
    assert ev_recovered.conditional_recovered is True
    assert ev_recovered.clean_correct is True
    assert ev_recovered.adversarial_correct is False
    assert ev_recovered.purified_correct is True

    # Case 3: Attack flipped prediction and purification failed to recover (clean=10, adv=20, pur=20)
    ev_unrecovered = compute_evaluation_metrics(
        clean_tensor=img,
        adversarial_tensor=img,
        purified_tensor=img,
        clean_inference=inf_a,
        adversarial_inference=inf_b,
        purified_inference=inf_b,
        purification_latency_ms=2.0,
        ground_truth_class=10,
    )
    assert ev_unrecovered.attack_succeeded is True
    assert ev_unrecovered.prediction_consistent is False
    assert ev_unrecovered.conditional_recovered is False
    assert ev_unrecovered.purified_correct is False
