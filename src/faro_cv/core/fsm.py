"""Formal Finite State Machine (FSM) governing the faro-cv experiment lifecycle.

Formally modeled as a deterministic transition system with guards and transactional context:
    M = (Q, Sigma, delta, q_0, F)
where:
    - Q     : finite set of LabState values
    - Sigma : finite set of LabEvent triggers
    - delta : partial transition function Q x Sigma -> Q guarded by domain predicates
    - q_0   : LabState.IDLE
    - F     : {LabState.EVALUATED} (accepting state; supports interactive re-purification loops)

Every transition executes transactionally:
    1. Resolve rule in TRANSITION_TABLE for (current_state, event).
    2. Stage a deep clone of ExperimentContext and apply optional mutator + cleanup hooks.
    3. Evaluate all transition guards against the staged context.
    4. Atomically commit state + context, append immutable TransitionRecord, and notify observers.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from faro_cv.core.context import ExperimentContext
from faro_cv.core.exceptions import (
    GuardRejectedError,
    InvalidTransitionError,
    SnapshotNotFoundError,
)


class LabState(StrEnum):
    """Finite state set Q of the interactive adversarial robustness laboratory."""

    IDLE = "IDLE"
    MODEL_READY = "MODEL_READY"
    INPUT_LOADED = "INPUT_LOADED"
    ATTACKING = "ATTACKING"
    PAUSED = "PAUSED"
    ATTACKED = "ATTACKED"
    PURIFYING = "PURIFYING"
    PURIFIED = "PURIFIED"
    EVALUATED = "EVALUATED"
    ERROR = "ERROR"


class LabEvent(StrEnum):
    """Input alphabet Sigma of valid laboratory control events."""

    LOAD_MODEL = "LOAD_MODEL"
    LOAD_INPUT = "LOAD_INPUT"
    START_ATTACK = "START_ATTACK"
    PAUSE_ATTACK = "PAUSE_ATTACK"
    RESUME_ATTACK = "RESUME_ATTACK"
    CANCEL_ATTACK = "CANCEL_ATTACK"
    COMPLETE_ATTACK = "COMPLETE_ATTACK"
    START_PURIFY = "START_PURIFY"
    COMPLETE_PURIFY = "COMPLETE_PURIFY"
    EVALUATE = "EVALUATE"
    FAIL = "FAIL"
    RECOVER = "RECOVER"
    RESET = "RESET"


ACCEPTING_STATES: frozenset[LabState] = frozenset({LabState.EVALUATED})
TRANSIENT_STATES: frozenset[LabState] = frozenset({LabState.ATTACKING, LabState.PURIFYING})


# ---------------------------------------------------------------------------
# Guard Predicates
# ---------------------------------------------------------------------------

GuardPredicate = Callable[[ExperimentContext], tuple[bool, str]]


@dataclass(frozen=True, slots=True)
class TransitionGuard:
    """Named predicate evaluated against a staged ExperimentContext before committing."""

    name: str
    predicate: GuardPredicate

    def check(self, ctx: ExperimentContext) -> tuple[bool, str]:
        return self.predicate(ctx)


def _guard_has_model(ctx: ExperimentContext) -> tuple[bool, str]:
    if not ctx.model_id:
        return False, "model_id must be set in ExperimentContext."
    return True, ""


def _guard_has_input(ctx: ExperimentContext) -> tuple[bool, str]:
    if ctx.input_meta is None or ctx.clean_inference is None:
        return False, "Both input_meta and clean_inference must be populated."
    return True, ""


def _guard_has_attack_config(ctx: ExperimentContext) -> tuple[bool, str]:
    if ctx.attack_config is None:
        return False, "attack_config must be provided before starting an attack."
    return True, ""


def _guard_has_adversarial_sample(ctx: ExperimentContext) -> tuple[bool, str]:
    if ctx.adversarial_meta is None or ctx.adversarial_inference is None:
        return False, "adversarial_meta and adversarial_inference must be populated."
    return True, ""


def _guard_has_defense_config(ctx: ExperimentContext) -> tuple[bool, str]:
    if ctx.defense_config is None:
        return False, "defense_config must be provided before starting purification."
    return True, ""


def _guard_has_purified_sample(ctx: ExperimentContext) -> tuple[bool, str]:
    if ctx.purified_meta is None or ctx.purified_inference is None or ctx.spectral_summary is None:
        return (
            False,
            "purified_meta, purified_inference, and spectral_summary must be populated.",
        )
    return True, ""


def _guard_has_evaluation(ctx: ExperimentContext) -> tuple[bool, str]:
    if ctx.evaluation is None:
        return False, "evaluation metrics must be computed before entering EVALUATED."
    return True, ""


def _guard_has_error_message(ctx: ExperimentContext) -> tuple[bool, str]:
    if not ctx.last_error:
        return False, "last_error must describe the failure reason."
    return True, ""


G_HAS_MODEL = TransitionGuard("has_model", _guard_has_model)
G_HAS_INPUT = TransitionGuard("has_input", _guard_has_input)
G_HAS_ATTACK_CONFIG = TransitionGuard("has_attack_config", _guard_has_attack_config)
G_HAS_ADVERSARIAL = TransitionGuard("has_adversarial_sample", _guard_has_adversarial_sample)
G_HAS_DEFENSE_CONFIG = TransitionGuard("has_defense_config", _guard_has_defense_config)
G_HAS_PURIFIED = TransitionGuard("has_purified_sample", _guard_has_purified_sample)
G_HAS_EVALUATION = TransitionGuard("has_evaluation", _guard_has_evaluation)
G_HAS_ERROR = TransitionGuard("has_error_message", _guard_has_error_message)


# ---------------------------------------------------------------------------
# Declarative Transition Table (Single Source of Truth)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TransitionRule:
    """Declarative specification for a single edge in delta(state, event)."""

    source: LabState
    event: LabEvent
    target: LabState | None  # None indicates dynamic target (used only by RECOVER)
    guards: tuple[TransitionGuard, ...] = ()
    description: str = ""


def _build_transition_table() -> dict[tuple[LabState, LabEvent], TransitionRule]:
    rules: list[TransitionRule] = [
        # Model loading (allowed from IDLE or hot-swapping from any stable state)
        TransitionRule(
            LabState.IDLE,
            LabEvent.LOAD_MODEL,
            LabState.MODEL_READY,
            guards=(G_HAS_MODEL,),
            description="Initialize vision backbone and probe device capabilities.",
        ),
        *[
            TransitionRule(
                state,
                LabEvent.LOAD_MODEL,
                LabState.MODEL_READY,
                guards=(G_HAS_MODEL,),
                description="Switch vision backbone and reset sample pipeline.",
            )
            for state in (
                LabState.MODEL_READY,
                LabState.INPUT_LOADED,
                LabState.ATTACKED,
                LabState.PURIFIED,
                LabState.EVALUATED,
            )
        ],
        # Input loading (re-loading an input clears downstream attack/purification results)
        *[
            TransitionRule(
                state,
                LabEvent.LOAD_INPUT,
                LabState.INPUT_LOADED,
                guards=(G_HAS_MODEL, G_HAS_INPUT),
                description="Load clean image tensor and compute baseline inference.",
            )
            for state in (
                LabState.MODEL_READY,
                LabState.INPUT_LOADED,
                LabState.ATTACKED,
                LabState.PURIFIED,
                LabState.EVALUATED,
            )
        ],
        # Attack synthesis lifecycle
        *[
            TransitionRule(
                state,
                LabEvent.START_ATTACK,
                LabState.ATTACKING,
                guards=(G_HAS_MODEL, G_HAS_INPUT, G_HAS_ATTACK_CONFIG),
                description="Begin adversarial perturbation or patch optimization.",
            )
            for state in (
                LabState.INPUT_LOADED,
                LabState.ATTACKED,
                LabState.PURIFIED,
                LabState.EVALUATED,
            )
        ],
        TransitionRule(
            LabState.ATTACKING,
            LabEvent.PAUSE_ATTACK,
            LabState.PAUSED,
            guards=(),
            description="Pause iterative optimization for live inspection.",
        ),
        TransitionRule(
            LabState.PAUSED,
            LabEvent.RESUME_ATTACK,
            LabState.ATTACKING,
            guards=(G_HAS_ATTACK_CONFIG,),
            description="Resume paused iterative optimization.",
        ),
        TransitionRule(
            LabState.ATTACKING,
            LabEvent.CANCEL_ATTACK,
            LabState.INPUT_LOADED,
            guards=(G_HAS_INPUT,),
            description="Abort active attack and revert to clean input state.",
        ),
        TransitionRule(
            LabState.PAUSED,
            LabEvent.CANCEL_ATTACK,
            LabState.INPUT_LOADED,
            guards=(G_HAS_INPUT,),
            description="Abort paused attack and revert to clean input state.",
        ),
        TransitionRule(
            LabState.ATTACKING,
            LabEvent.COMPLETE_ATTACK,
            LabState.ATTACKED,
            guards=(G_HAS_ADVERSARIAL,),
            description="Finalize adversarial sample and evaluate attacked prediction.",
        ),
        # Purification lifecycle (supports interactive cutoff scrubbing from PURIFIED / EVALUATED)
        *[
            TransitionRule(
                state,
                LabEvent.START_PURIFY,
                LabState.PURIFYING,
                guards=(G_HAS_ADVERSARIAL, G_HAS_DEFENSE_CONFIG),
                description="Run SVD / 2D-FFT purification on the adversarial sample.",
            )
            for state in (
                LabState.ATTACKED,
                LabState.PURIFIED,
                LabState.EVALUATED,
            )
        ],
        TransitionRule(
            LabState.PURIFYING,
            LabEvent.COMPLETE_PURIFY,
            LabState.PURIFIED,
            guards=(G_HAS_PURIFIED,),
            description="Store reconstructed tensor, spectral diagnostics, and purified inference.",
        ),
        # Comparative evaluation
        TransitionRule(
            LabState.PURIFIED,
            LabEvent.EVALUATE,
            LabState.EVALUATED,
            guards=(G_HAS_INPUT, G_HAS_ADVERSARIAL, G_HAS_PURIFIED, G_HAS_EVALUATION),
            description="Compute comparative PSNR, SSIM, perturbation norms, and recovery status.",
        ),
        # Fault handling (any non-ERROR state may transition to ERROR)
        *[
            TransitionRule(
                state,
                LabEvent.FAIL,
                LabState.ERROR,
                guards=(G_HAS_ERROR,),
                description="Enter structured fault state while preserving recovery anchor.",
            )
            for state in LabState
            if state is not LabState.ERROR
        ],
        TransitionRule(
            LabState.ERROR,
            LabEvent.RECOVER,
            None,  # Dynamic recovery to last known stable state
            guards=(),
            description="Recover from fault back to the last stable pre-error state.",
        ),
        # Hard reset from any state back to IDLE
        *[
            TransitionRule(
                state,
                LabEvent.RESET,
                LabState.IDLE,
                guards=(),
                description="Reset laboratory session back to IDLE.",
            )
            for state in LabState
        ],
    ]

    table: dict[tuple[LabState, LabEvent], TransitionRule] = {}
    for rule in rules:
        key = (rule.source, rule.event)
        if key in table:
            raise RuntimeError(f"Duplicate FSM transition rule detected for {key}.")
        table[key] = rule
    return table


TRANSITION_TABLE: dict[tuple[LabState, LabEvent], TransitionRule] = _build_transition_table()


# ---------------------------------------------------------------------------
# Audit Records, Snapshots, and UI Availability Models
# ---------------------------------------------------------------------------


class TransitionRecord(BaseModel):
    """Immutable audit log entry recorded for every committed FSM transition."""

    model_config = ConfigDict(frozen=True)

    step_index: int = Field(ge=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    from_state: LabState
    event: LabEvent
    to_state: LabState
    metadata: dict[str, Any] = Field(default_factory=dict)


class EventAvailability(BaseModel):
    """Describes whether an event is currently permitted and ready for UI binding."""

    model_config = ConfigDict(frozen=True)

    event: LabEvent
    target_state: LabState
    guard_passed: bool
    blocked_reason: str | None = None


class StateSnapshot(BaseModel):
    """Point-in-time frozen snapshot enabling deterministic time-travel & A/B comparison."""

    model_config = ConfigDict(frozen=True)

    snapshot_id: str
    label: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    state: LabState
    recovery_state: LabState
    context: ExperimentContext


class FSMWireState(BaseModel):
    """Serializable FSM payload streamed to the workbench over REST/WebSocket."""

    model_config = ConfigDict(frozen=True)

    state: LabState
    is_accepting: bool
    recovery_state: LabState
    step_count: int = Field(ge=0)
    allowed_events: list[EventAvailability]
    context: ExperimentContext
    recent_history: list[TransitionRecord]


TransitionObserver = Callable[[TransitionRecord, ExperimentContext], None]
ContextMutator = Callable[[ExperimentContext], None]


# ---------------------------------------------------------------------------
# Finite State Machine Engine
# ---------------------------------------------------------------------------


class ExperimentFSM:
    """Deterministic, transactional FSM controller for a single faro-cv lab session."""

    def __init__(
        self,
        context: ExperimentContext | None = None,
        *,
        initial_state: LabState = LabState.IDLE,
    ) -> None:
        self._state: LabState = initial_state
        self._recovery_state: LabState = (
            LabState.IDLE if initial_state in TRANSIENT_STATES else initial_state
        )
        self._context: ExperimentContext = context.clone() if context else ExperimentContext()
        self._history: list[TransitionRecord] = []
        self._snapshots: dict[str, StateSnapshot] = {}
        self._observers: list[TransitionObserver] = []

    @property
    def state(self) -> LabState:
        return self._state

    @property
    def is_accepting(self) -> bool:
        return self._state in ACCEPTING_STATES

    @property
    def recovery_state(self) -> LabState:
        return self._recovery_state

    @property
    def context(self) -> ExperimentContext:
        return self._context

    @property
    def history(self) -> tuple[TransitionRecord, ...]:
        return tuple(self._history)

    @property
    def snapshots(self) -> tuple[StateSnapshot, ...]:
        return tuple(self._snapshots.values())

    def subscribe(self, observer: TransitionObserver) -> Callable[[], None]:
        """Register an observer called after each committed transition; returns unsubscribe fn."""
        self._observers.append(observer)

        def _unsubscribe() -> None:
            if observer in self._observers:
                self._observers.remove(observer)

        return _unsubscribe

    def allowed_events(self) -> list[EventAvailability]:
        """Return all events structurally valid from the current state with guard status."""
        results: list[EventAvailability] = []
        for (source, event), rule in TRANSITION_TABLE.items():
            if source is not self._state:
                continue
            target = self._resolve_target(rule)
            passed = True
            reason: str | None = None
            for guard in rule.guards:
                ok, msg = guard.check(self._context)
                if not ok:
                    passed = False
                    reason = f"{guard.name}: {msg}"
                    break
            results.append(
                EventAvailability(
                    event=event,
                    target_state=target,
                    guard_passed=passed,
                    blocked_reason=reason,
                )
            )
        return results

    def dispatch(
        self,
        event: LabEvent,
        *,
        mutate: ContextMutator | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TransitionRecord:
        """Atomically transition the FSM upon receiving ``event``.

        Args:
            event: Control event from ``LabEvent``.
            mutate: Optional callable applied to a staged clone of ``ExperimentContext``
                prior to guard evaluation. If any error or guard rejection occurs, the
                live FSM state and context remain untouched.
            metadata: Optional structured key-value telemetry recorded in the audit log.

        Raises:
            InvalidTransitionError: If ``(self.state, event)`` is absent from ``delta``.
            GuardRejectedError: If any domain guard rejects the staged context.
        """
        rule = TRANSITION_TABLE.get((self._state, event))
        if rule is None:
            valid = tuple(ev.value for (src, ev) in TRANSITION_TABLE if src is self._state)
            raise InvalidTransitionError(
                current_state=self._state.value,
                event=event.value,
                allowed_events=valid,
            )

        staged = self._prepare_staged_context(event)
        if mutate is not None:
            mutate(staged)

        for guard in rule.guards:
            ok, reason = guard.check(staged)
            if not ok:
                raise GuardRejectedError(
                    current_state=self._state.value,
                    event=event.value,
                    guard_name=guard.name,
                    reason=reason,
                )

        previous_state = self._state
        target_state = self._resolve_target(rule)

        # Commit atomically
        self._state = target_state
        self._context = staged
        if target_state not in (LabState.ERROR, *TRANSIENT_STATES, LabState.PAUSED):
            self._recovery_state = target_state

        record = TransitionRecord(
            step_index=len(self._history) + 1,
            from_state=previous_state,
            event=event,
            to_state=target_state,
            metadata=dict(metadata) if metadata else {},
        )
        self._history.append(record)

        for observer in list(self._observers):
            observer(record, self._context)

        return record

    def create_snapshot(self, label: str = "") -> StateSnapshot:
        """Capture an immutable snapshot of the current FSM state and context."""
        snap_id = f"snap_{uuid.uuid4().hex[:8]}"
        snapshot = StateSnapshot(
            snapshot_id=snap_id,
            label=label or f"{self._state.value} @ step {len(self._history)}",
            state=self._state,
            recovery_state=self._recovery_state,
            context=self._context.clone(),
        )
        self._snapshots[snap_id] = snapshot
        return snapshot

    def restore_snapshot(self, snapshot_id: str) -> StateSnapshot:
        """Restore FSM state and context to a previously captured snapshot."""
        snapshot = self._snapshots.get(snapshot_id)
        if snapshot is None:
            raise SnapshotNotFoundError(snapshot_id)
        self._state = snapshot.state
        self._recovery_state = snapshot.recovery_state
        self._context = snapshot.context.clone()
        return snapshot

    def to_wire_state(self, *, history_limit: int = 20) -> FSMWireState:
        """Serialize current FSM status and recent transitions for API/WebSocket transport."""
        recent = self._history[-history_limit:] if history_limit > 0 else []
        return FSMWireState(
            state=self._state,
            is_accepting=self.is_accepting,
            recovery_state=self._recovery_state,
            step_count=len(self._history),
            allowed_events=self.allowed_events(),
            context=self._context.clone(),
            recent_history=recent,
        )

    def _resolve_target(self, rule: TransitionRule) -> LabState:
        if rule.target is not None:
            return rule.target
        # Dynamic target for RECOVER -> last stable anchor state
        return self._recovery_state

    def _prepare_staged_context(self, event: LabEvent) -> ExperimentContext:
        """Create staged context with deterministic lifecycle cleanup rules."""
        if event is LabEvent.RESET:
            return ExperimentContext(
                session_id=self._context.session_id,
                seed=self._context.seed,
            )

        staged = self._context.clone()

        if event is LabEvent.LOAD_MODEL:
            # Hot-swapping a model invalidates everything downstream of model loading.
            staged = ExperimentContext(
                session_id=self._context.session_id,
                seed=self._context.seed,
                device_profile=self._context.device_profile,
            )
        elif event is LabEvent.LOAD_INPUT:
            staged.clear_downstream_from_input()
        elif event in (LabEvent.START_ATTACK, LabEvent.CANCEL_ATTACK):
            staged.clear_downstream_from_attack()
        elif event is LabEvent.START_PURIFY:
            staged.clear_downstream_from_purification()
        elif event is LabEvent.RECOVER:
            staged.last_error = None

        if event is not LabEvent.FAIL:
            staged.last_error = None

        return staged

    @classmethod
    def export_mermaid(cls, *, include_global_edges: bool = False) -> str:
        """Generate a canonical Mermaid stateDiagram-v2 directly from TRANSITION_TABLE."""
        lines: list[str] = ["stateDiagram-v2", "    [*] --> IDLE"]
        for rule in TRANSITION_TABLE.values():
            if not include_global_edges and rule.event in (LabEvent.FAIL, LabEvent.RESET):
                continue
            target_label = rule.target.value if rule.target is not None else "LAST_STABLE_STATE"
            lines.append(f"    {rule.source.value} --> {target_label} : {rule.event.value}")
        if not include_global_edges:
            lines.append("    note right of ERROR")
            lines.append("        Any state --(FAIL)--> ERROR")
            lines.append("        Any state --(RESET)--> IDLE")
            lines.append("    end note")
        return "\n".join(lines)
