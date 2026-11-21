"""White-box adaptive attack wrapper for evaluating defense penetration (Exact & BPDA).

Implements the standard robustness evaluation methodology from Athalye et al. (ICML 2018,
"Obfuscated Gradients Give a False Sense of Security"):
    - Exact Adaptive Mode: Differentiates directly through PyTorch SVD / 2D-FFT layers
      in the end-to-end composite model ``f(g(x))``.
    - BPDA (Backward Pass Differentiable Approximation): Forward pass executes exact ``g(x)``,
      while backward pass applies the Straight-Through Estimator ``x + (g(x) - x).detach()``
      to bypass discontinuous/ill-conditioned gradients (e.g., ideal frequency cutoffs).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from faro_cv.core.types import AdaptiveMode
from faro_cv.defenses.base import BaseDefense


class AdaptiveDefendedModel(nn.Module):
    """Wraps a classifier ``f`` and purification operator ``g`` into ``f(g(x))``."""

    def __init__(
        self,
        base_model: nn.Module,
        defense: BaseDefense,
        *,
        mode: AdaptiveMode = "exact",
    ) -> None:
        super().__init__()
        if mode == "none":
            raise ValueError("AdaptiveDefendedModel requires mode='exact' or mode='bpda'.")
        self.base_model = base_model
        self.defense = defense
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == "bpda":
            # Forward uses exact g(x); backward uses identity straight-through d(g(x))/dx = I
            with torch.no_grad():
                purified_detached: Tensor = self.defense.purify(x.detach()).purified_tensor
            ste_input = x + (purified_detached.to(x.device) - x).detach()
            out_bpda: Tensor = self.base_model(ste_input)
            return out_bpda

        # Exact differentiable white-box pass through SVD / 2D-FFT
        purified: Tensor = self.defense.purify(x).purified_tensor
        out_exact: Tensor = self.base_model(purified.to(x.device))
        return out_exact
