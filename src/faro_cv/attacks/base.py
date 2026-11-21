"""Abstract attack interface supporting step-by-step streaming synthesis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from faro_cv.core.types import AttackConfig


@dataclass(frozen=True, slots=True)
class AttackStepResult:
    """Intermediate or final output yielded by an iterative adversarial attack."""

    step: int
    total_steps: int
    loss: float
    adversarial_tensor: Any
    perturbation_tensor: Any


class BaseAttack(ABC):
    """Contract for all perturbation and adversarial patch attacks in faro-cv."""

    name: str

    def __init__(self, config: AttackConfig) -> None:
        self.config = config

    @abstractmethod
    def iterate(
        self,
        model: Any,
        image_tensor: Any,
        label_index: int,
    ) -> Iterator[AttackStepResult]:
        """Yield intermediate optimization steps so the FSM & UI can stream live updates."""

    def run(self, model: Any, image_tensor: Any, label_index: int) -> AttackStepResult:
        """Execute the full attack to completion and return the final step result."""
        last: AttackStepResult | None = None
        for step_result in self.iterate(model, image_tensor, label_index):
            last = step_result
        if last is None:
            raise RuntimeError(f"Attack '{self.name}' yielded zero optimization steps.")
        return last
