"""Adversarial Patch synthesis with Expectation over Transformation (EoT).

Implements localized high-frequency adversarial patches (Brown et al., 2017) optimized
under stochastic physical transformations (Athalye et al., ICML 2018):
    - Random in-plane rotation (+/- 15 deg) and scale jitter (0.92x - 1.08x)
    - Small spatial translation jitter (+/- 4 px)
    - Brightness & contrast perturbations (+/- 10%)
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from faro_cv.attacks.base import AttackStepResult, BaseAttack
from faro_cv.core.types import AttackConfig


def _transform_patch(
    patch: Tensor,
    *,
    angle_rad: float,
    scale: float,
    brightness: float,
    contrast: float,
) -> Tensor:
    """Apply a differentiable affine + photometric transformation to a ``[1, 3, S, S]`` patch."""
    inv_scale = 1.0 / max(scale, 1e-3)
    cos_a = math.cos(angle_rad) * inv_scale
    sin_a = math.sin(angle_rad) * inv_scale
    theta = torch.tensor(
        [[[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]]],
        dtype=patch.dtype,
        device=patch.device,
    )
    grid = F.affine_grid(theta, size=list(patch.shape), align_corners=False)
    warped = F.grid_sample(
        patch,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )
    adjusted = (warped - 0.5) * contrast + 0.5 + brightness
    return adjusted.clamp(0.0, 1.0)


def _apply_patch_onto_image(
    clean: Tensor,
    patch: Tensor,
    *,
    x: int,
    y: int,
) -> Tensor:
    """Composite a ``[1, 3, S, S]`` patch onto ``[1, 3, H, W]`` via a differentiable binary mask."""
    _, _, h, w = clean.shape
    ps = int(patch.shape[-1])
    x0 = max(0, min(w - ps, x))
    y0 = max(0, min(h - ps, y))

    padded_patch = F.pad(patch, (x0, w - (x0 + ps), y0, h - (y0 + ps)))
    mask = torch.zeros((1, 1, h, w), dtype=clean.dtype, device=clean.device)
    mask[:, :, y0 : y0 + ps, x0 : x0 + ps] = 1.0
    return clean * (1.0 - mask) + padded_patch * mask


class AdversarialPatchEoTAttack(BaseAttack):
    """Localized physical-style Adversarial Patch optimized with EoT."""

    name = "patch_eot"

    def __init__(
        self,
        config: AttackConfig,
        *,
        seed: int = 42,
        learning_rate: float = 0.08,
    ) -> None:
        super().__init__(config)
        if config.patch_size is None:
            raise ValueError("AdversarialPatchEoTAttack requires config.patch_size.")
        self.seed = seed
        self.learning_rate = learning_rate

    def iterate(
        self,
        model: nn.Module,
        image_tensor: Tensor,
        label_index: int,
    ) -> Iterator[AttackStepResult]:
        was_unbatched = image_tensor.ndim == 3
        clean = image_tensor.detach().unsqueeze(0) if was_unbatched else image_tensor.detach()
        if clean.ndim != 4 or clean.shape[0] != 1:
            raise ValueError(
                f"Expected [3, H, W] or [1, 3, H, W] tensor, got {tuple(clean.shape)}."
            )

        device = next(model.parameters()).device
        clean = clean.to(device=device, dtype=torch.float32)
        _, _, h, w = clean.shape

        assert self.config.patch_size is not None
        ps = max(8, min(self.config.patch_size, min(h, w)))
        base_x = (
            (w - ps) // 2
            if self.config.patch_x is None
            else max(0, min(w - ps, self.config.patch_x))
        )
        base_y = (
            (h - ps) // 2
            if self.config.patch_y is None
            else max(0, min(h - ps, self.config.patch_y))
        )

        target_idx = (
            self.config.target_class
            if (self.config.targeted and self.config.target_class is not None)
            else label_index
        )
        label = torch.tensor([target_idx], dtype=torch.long, device=device)

        # Deterministic CPU generator so patch evolution is reproducible across MPS/CPU
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        init_patch = torch.rand((1, 3, ps, ps), generator=gen, dtype=torch.float32).to(device)
        patch = init_patch.clone().detach()

        # Momentum buffer for smooth, high-contrast patch synthesis
        momentum = torch.zeros_like(patch)
        beta = 0.85
        lr = max(self.learning_rate, self.config.step_size * 4.0)
        direction = -1.0 if self.config.targeted else 1.0

        total_steps = self.config.steps
        eot_samples = self.config.eot_samples

        for step_idx in range(1, total_steps + 1):
            patch_var = patch.detach().requires_grad_(True)
            total_loss = torch.zeros((), dtype=torch.float32, device=device)

            for sample_idx in range(eot_samples):
                if sample_idx == 0:
                    # Anchor sample at canonical placement
                    warped = patch_var
                    jx, jy = base_x, base_y
                else:
                    angle_deg = float((torch.rand((), generator=gen).item() * 2.0 - 1.0) * 15.0)
                    scale = 0.92 + float(torch.rand((), generator=gen).item()) * 0.16
                    brightness = float((torch.rand((), generator=gen).item() * 2.0 - 1.0) * 0.10)
                    contrast = 0.90 + float(torch.rand((), generator=gen).item()) * 0.20
                    dx = round((torch.rand((), generator=gen).item() * 2.0 - 1.0) * 4.0)
                    dy = round((torch.rand((), generator=gen).item() * 2.0 - 1.0) * 4.0)
                    jx = max(0, min(w - ps, base_x + dx))
                    jy = max(0, min(h - ps, base_y + dy))
                    warped = _transform_patch(
                        patch_var,
                        angle_rad=math.radians(angle_deg),
                        scale=scale,
                        brightness=brightness,
                        contrast=contrast,
                    )

                composed = _apply_patch_onto_image(clean, warped, x=jx, y=jy)
                logits = model(composed)
                total_loss = total_loss + F.cross_entropy(logits, label)

            mean_loss = total_loss / float(eot_samples)
            grad = torch.autograd.grad(
                mean_loss, patch_var, retain_graph=False, create_graph=False
            )[0]

            # Normalized momentum ascent/descent keeps patch updates crisp and scale-invariant
            grad_norm = grad.abs().mean().clamp(min=1e-8)
            momentum = beta * momentum + (grad / grad_norm)
            patch = (patch + direction * lr * momentum.sign()).clamp(0.0, 1.0).detach()

            # Canonical composed image for streaming inspection
            adv = _apply_patch_onto_image(clean, patch, x=base_x, y=base_y).detach()
            perturbation = (adv - clean).detach()

            yield AttackStepResult(
                step=step_idx,
                total_steps=total_steps,
                loss=float(mean_loss.item()),
                adversarial_tensor=adv.squeeze(0) if was_unbatched else adv,
                perturbation_tensor=perturbation.squeeze(0) if was_unbatched else perturbation,
            )
