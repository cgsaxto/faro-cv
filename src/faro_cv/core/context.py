"""Mutable experiment session context with deterministic deep-snapshot cloning."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from faro_cv.core.types import (
    AttackConfig,
    AttackProgress,
    DefenseConfig,
    EvaluationMetrics,
    InferenceSummary,
    OperatorDeviceProfile,
    SpectralSummary,
    TensorMetadata,
)


def _new_session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:12]}"


class ExperimentContext(BaseModel):
    """Structured state container governed by ExperimentFSM.

    Raw image tensors are stored outside this lightweight context (in an artifact/tensor
    store keyed by ``TensorMetadata.sha256``) so that snapshots, audit logs, and WebSocket
    broadcasts remain compact, deterministic, and JSON-serializable.
    """

    session_id: str = Field(default_factory=_new_session_id)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    seed: int = Field(default=42, ge=0)

    # Stage 1: Model & Hardware
    model_id: str | None = None
    device_profile: OperatorDeviceProfile | None = None

    # Stage 2: Clean Input
    input_meta: TensorMetadata | None = None
    clean_inference: InferenceSummary | None = None

    # Stage 3: Attack Synthesis
    attack_config: AttackConfig | None = None
    attack_progress: AttackProgress | None = None
    adversarial_meta: TensorMetadata | None = None
    adversarial_inference: InferenceSummary | None = None

    # Stage 4: Signal / Matrix Purification
    defense_config: DefenseConfig | None = None
    spectral_summary: SpectralSummary | None = None
    purified_meta: TensorMetadata | None = None
    purified_inference: InferenceSummary | None = None

    # Stage 5: Comparative Evaluation
    evaluation: EvaluationMetrics | None = None

    # Fault diagnostics
    last_error: str | None = None

    def clear_downstream_from_input(self) -> None:
        """Invalidate attack, purification, and evaluation artifacts when a new input loads."""
        self.clear_downstream_from_attack()
        self.attack_config = None

    def clear_downstream_from_attack(self) -> None:
        """Invalidate adversarial outputs and downstream purification/evaluation."""
        self.attack_progress = None
        self.adversarial_meta = None
        self.adversarial_inference = None
        self.clear_downstream_from_purification()

    def clear_downstream_from_purification(self) -> None:
        """Invalidate purification outputs and comparative evaluation."""
        self.spectral_summary = None
        self.purified_meta = None
        self.purified_inference = None
        self.evaluation = None

    def clone(self) -> ExperimentContext:
        """Return an isolated deep copy for immutable snapshot storage."""
        return self.model_copy(deep=True)
