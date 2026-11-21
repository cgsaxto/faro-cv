"""Core primitives for faro-cv: FSM, context, schemas, exceptions, and device probing."""

from faro_cv.core.context import ExperimentContext
from faro_cv.core.device import probe_device_capabilities
from faro_cv.core.exceptions import (
    FaroError,
    GuardRejectedError,
    InvalidTransitionError,
    SnapshotNotFoundError,
    StateInvariantError,
)
from faro_cv.core.fsm import (
    ACCEPTING_STATES,
    TRANSITION_TABLE,
    EventAvailability,
    ExperimentFSM,
    FSMWireState,
    LabEvent,
    LabState,
    StateSnapshot,
    TransitionRecord,
    TransitionRule,
)

__all__ = [
    "ACCEPTING_STATES",
    "TRANSITION_TABLE",
    "EventAvailability",
    "ExperimentContext",
    "ExperimentFSM",
    "FSMWireState",
    "FaroError",
    "GuardRejectedError",
    "InvalidTransitionError",
    "LabEvent",
    "LabState",
    "SnapshotNotFoundError",
    "StateInvariantError",
    "StateSnapshot",
    "TransitionRecord",
    "TransitionRule",
    "probe_device_capabilities",
]
