"""Quantitative image fidelity and adversarial robustness evaluation metrics."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from faro_cv.core.types import EvaluationMetrics, InferenceSummary


def _to_bchw(tensor: Tensor) -> Tensor:
    t = tensor.detach().to(device="cpu", dtype=torch.float32)
    if t.ndim == 3:
        return t.unsqueeze(0)
    if t.ndim == 4:
        return t
    raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(t.shape)}.")


def compute_perturbation_norms(clean: Tensor, adversarial: Tensor) -> tuple[float, float]:
    """Return ``(l2_norm, linf_norm)`` of ``adversarial - clean``."""
    diff = _to_bchw(adversarial) - _to_bchw(clean)
    l2 = float(torch.linalg.vector_norm(diff.reshape(-1), ord=2).item())
    linf = float(diff.abs().max().item())
    return round(l2, 6), round(linf, 6)


def compute_psnr(reference: Tensor, candidate: Tensor, *, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio (dB) between two ``[0, 1]`` images."""
    ref = _to_bchw(reference)
    cand = _to_bchw(candidate)
    mse = float(F.mse_loss(cand, ref).item())
    if mse <= 1e-12:
        return 100.0
    psnr = 10.0 * math.log10((max_val * max_val) / mse)
    return round(psnr, 4)


def _gaussian_window(window_size: int, sigma: float, channels: int) -> Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    g1d = torch.exp(-(coords.pow(2)) / (2.0 * sigma * sigma))
    g1d = g1d / g1d.sum()
    g2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)
    return g2d.expand(channels, 1, window_size, window_size).contiguous()


def compute_ssim(
    reference: Tensor,
    candidate: Tensor,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Compute mean Structural Similarity Index (Wang et al., 2004) in ``[-1, 1]``."""
    ref = _to_bchw(reference)
    cand = _to_bchw(candidate)
    channels = int(ref.shape[1])
    effective_window = min(window_size, int(ref.shape[-2]), int(ref.shape[-1]))
    if effective_window % 2 == 0:
        effective_window = max(1, effective_window - 1)

    kernel = _gaussian_window(effective_window, sigma, channels)
    pad = effective_window // 2

    mu_x = F.conv2d(ref, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(cand, kernel, padding=pad, groups=channels)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = (F.conv2d(ref * ref, kernel, padding=pad, groups=channels) - mu_x_sq).clamp(
        min=0.0
    )
    sigma_y_sq = (F.conv2d(cand * cand, kernel, padding=pad, groups=channels) - mu_y_sq).clamp(
        min=0.0
    )
    sigma_xy = F.conv2d(ref * cand, kernel, padding=pad, groups=channels) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return round(max(-1.0, min(1.0, float(ssim_map.mean().item()))), 6)


def compute_evaluation_metrics(
    *,
    clean_tensor: Tensor,
    adversarial_tensor: Tensor,
    purified_tensor: Tensor,
    clean_inference: InferenceSummary,
    adversarial_inference: InferenceSummary,
    purified_inference: InferenceSummary,
    purification_latency_ms: float,
    targeted_class: int | None = None,
    ground_truth_class: int | None = None,
    attention_similarity_adv: float | None = None,
    attention_similarity_purified: float | None = None,
) -> EvaluationMetrics:
    """Aggregate perturbation norms, reconstruction fidelity, and split recovery metrics."""
    l2_norm, linf_norm = compute_perturbation_norms(clean_tensor, adversarial_tensor)
    psnr_db = compute_psnr(clean_tensor, purified_tensor)
    ssim = compute_ssim(clean_tensor, purified_tensor)

    clean_class = clean_inference.top1.class_index
    adv_class = adversarial_inference.top1.class_index
    pur_class = purified_inference.top1.class_index

    attack_succeeded = (
        (adv_class == targeted_class) if targeted_class is not None else (adv_class != clean_class)
    )
    prediction_consistent = pur_class == clean_class
    conditional_recovered: bool | None = prediction_consistent if attack_succeeded else None

    clean_correct = (clean_class == ground_truth_class) if ground_truth_class is not None else None
    adv_correct = (adv_class == ground_truth_class) if ground_truth_class is not None else None
    pur_correct = (pur_class == ground_truth_class) if ground_truth_class is not None else None

    return EvaluationMetrics(
        attack_l2_norm=l2_norm,
        attack_linf_norm=linf_norm,
        purified_psnr_db=psnr_db,
        purified_ssim=ssim,
        attack_succeeded=attack_succeeded,
        prediction_consistent=prediction_consistent,
        conditional_recovered=conditional_recovered,
        clean_confidence=clean_inference.top1.probability,
        adversarial_confidence=adversarial_inference.top1.probability,
        purified_confidence=purified_inference.top1.probability,
        purification_latency_ms=purification_latency_ms,
        attention_similarity_adv=attention_similarity_adv,
        attention_similarity_purified=attention_similarity_purified,
        ground_truth_class=ground_truth_class,
        clean_correct=clean_correct,
        adversarial_correct=adv_correct,
        purified_correct=pur_correct,
    )
