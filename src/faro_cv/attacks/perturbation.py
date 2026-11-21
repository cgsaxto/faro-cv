"""Gradient-based adversarial perturbation attacks (FGSM and PGD) in pixel space [0, 1]."""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from faro_cv.attacks.base import AttackStepResult, BaseAttack
from faro_cv.core.types import AttackConfig


def _prepare_batch(image_tensor: Tensor) -> tuple[Tensor, bool]:
    """Ensure shape ``[1, 3, H, W]`` for model forward pass and record whether it was unbatched."""
    if image_tensor.ndim == 3:
        return image_tensor.unsqueeze(0), True
    if image_tensor.ndim == 4 and image_tensor.shape[0] == 1:
        return image_tensor, False
    raise ValueError(f"Expected [3, H, W] or [1, 3, H, W] tensor, got {tuple(image_tensor.shape)}.")


def _project_perturbation(delta: Tensor, *, epsilon: float, norm: str) -> Tensor:
    """Project perturbation ``delta`` onto the L_inf or L_2 epsilon-ball."""
    if norm == "linf":
        return delta.clamp(-epsilon, epsilon)
    if norm == "l2":
        flat_norm = torch.linalg.vector_norm(delta.reshape(delta.shape[0], -1), ord=2, dim=1)
        scale = torch.clamp(epsilon / (flat_norm + 1e-12), max=1.0).view(-1, 1, 1, 1)
        return delta * scale
    raise ValueError(f"Unsupported norm '{norm}'. Expected 'linf' or 'l2'.")


def _normalized_gradient_step(grad: Tensor, *, step_size: float, norm: str) -> Tensor:
    """Compute a single signed (L_inf) or normalized (L_2) gradient ascent/descent step."""
    if norm == "linf":
        return step_size * grad.sign()
    if norm == "l2":
        grad_norm = torch.linalg.vector_norm(grad.reshape(grad.shape[0], -1), ord=2, dim=1)
        unit_grad: Tensor = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-12)
        return step_size * unit_grad
    raise ValueError(f"Unsupported norm '{norm}'. Expected 'linf' or 'l2'.")


class FGSMAttack(BaseAttack):
    """Fast Gradient Sign Method (Goodfellow et al., 2015) with L_inf and L_2 support."""

    name = "fgsm"

    def iterate(
        self,
        model: nn.Module,
        image_tensor: Tensor,
        label_index: int,
    ) -> Iterator[AttackStepResult]:
        clean, was_unbatched = _prepare_batch(image_tensor.detach())
        device = next(model.parameters()).device
        clean = clean.to(device=device, dtype=torch.float32)

        target_idx = (
            self.config.target_class
            if (self.config.targeted and self.config.target_class is not None)
            else label_index
        )
        label = torch.tensor([target_idx], dtype=torch.long, device=device)

        x_in = clean.clone().detach().requires_grad_(True)
        logits = model(x_in)
        loss = F.cross_entropy(logits, label)
        grad = torch.autograd.grad(loss, x_in, retain_graph=False, create_graph=False)[0]

        direction = -1.0 if self.config.targeted else 1.0
        step = direction * _normalized_gradient_step(
            grad,
            step_size=self.config.epsilon,
            norm=self.config.norm,
        )
        adv = (clean + step).clamp(0.0, 1.0).detach()
        perturbation = (adv - clean).detach()

        yield AttackStepResult(
            step=1,
            total_steps=1,
            loss=float(loss.item()),
            adversarial_tensor=adv.squeeze(0) if was_unbatched else adv,
            perturbation_tensor=perturbation.squeeze(0) if was_unbatched else perturbation,
        )


class PGDAttack(BaseAttack):
    """Projected Gradient Descent (Madry et al., 2018) with seeded init and live step streaming."""

    name = "pgd"

    def __init__(self, config: AttackConfig, *, seed: int = 42, random_start: bool = True) -> None:
        super().__init__(config)
        self.seed = seed
        self.random_start = random_start

    def iterate(
        self,
        model: nn.Module,
        image_tensor: Tensor,
        label_index: int,
    ) -> Iterator[AttackStepResult]:
        clean, was_unbatched = _prepare_batch(image_tensor.detach())
        device = next(model.parameters()).device
        clean = clean.to(device=device, dtype=torch.float32)

        target_idx = (
            self.config.target_class
            if (self.config.targeted and self.config.target_class is not None)
            else label_index
        )
        label = torch.tensor([target_idx], dtype=torch.long, device=device)

        if self.random_start and self.config.epsilon > 0.0:
            # Generate deterministic noise on CPU so results match across MPS and CPU.
            gen = torch.Generator(device="cpu").manual_seed(self.seed)
            raw_noise = torch.empty(clean.shape, dtype=torch.float32).uniform_(
                -self.config.epsilon,
                self.config.epsilon,
                generator=gen,
            )
            delta = _project_perturbation(
                raw_noise.to(device),
                epsilon=self.config.epsilon,
                norm=self.config.norm,
            )
            adv = (clean + delta).clamp(0.0, 1.0).detach()
        else:
            adv = clean.clone().detach()

        direction = -1.0 if self.config.targeted else 1.0
        total_steps = self.config.steps

        for step_idx in range(1, total_steps + 1):
            adv_var = adv.detach().requires_grad_(True)
            logits = model(adv_var)
            loss = F.cross_entropy(logits, label)
            grad = torch.autograd.grad(loss, adv_var, retain_graph=False, create_graph=False)[0]

            step = direction * _normalized_gradient_step(
                grad,
                step_size=self.config.step_size,
                norm=self.config.norm,
            )
            candidate = adv_var.detach() + step
            delta = _project_perturbation(
                candidate - clean,
                epsilon=self.config.epsilon,
                norm=self.config.norm,
            )
            adv = (clean + delta).clamp(0.0, 1.0).detach()
            perturbation = (adv - clean).detach()

            yield AttackStepResult(
                step=step_idx,
                total_steps=total_steps,
                loss=float(loss.item()),
                adversarial_tensor=adv.squeeze(0) if was_unbatched else adv,
                perturbation_tensor=perturbation.squeeze(0) if was_unbatched else perturbation,
            )


def build_attack(config: AttackConfig, *, seed: int = 42) -> BaseAttack:
    """Factory creating a concrete BaseAttack from a validated AttackConfig."""
    if config.kind == "fgsm":
        return FGSMAttack(config)
    if config.kind == "pgd":
        return PGDAttack(config, seed=seed)
    if config.kind == "patch_eot":
        from faro_cv.attacks.patch import AdversarialPatchEoTAttack

        return AdversarialPatchEoTAttack(config, seed=seed)
    raise ValueError(f"Unsupported attack kind '{config.kind}'.")
