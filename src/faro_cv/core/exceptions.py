"""Domain-specific exception hierarchy for faro-cv."""

from __future__ import annotations


class FaroError(Exception):
    """Base exception for all faro-cv domain and runtime errors."""


class InvalidTransitionError(FaroError):
    """Raised when an event is dispatched that is illegal in the current FSM state."""

    def __init__(
        self,
        *,
        current_state: str,
        event: str,
        allowed_events: tuple[str, ...],
    ) -> None:
        self.current_state = current_state
        self.event = event
        self.allowed_events = allowed_events
        allowed = ", ".join(allowed_events) if allowed_events else "<none>"
        super().__init__(
            f"Event '{event}' is not permitted in state '{current_state}'. "
            f"Allowed events: [{allowed}]."
        )


class GuardRejectedError(FaroError):
    """Raised when a transition rule's domain guard condition evaluates to False."""

    def __init__(
        self,
        *,
        current_state: str,
        event: str,
        guard_name: str,
        reason: str,
    ) -> None:
        self.current_state = current_state
        self.event = event
        self.guard_name = guard_name
        self.reason = reason
        super().__init__(
            f"Guard '{guard_name}' rejected transition '{current_state}' --({event})-->: {reason}"
        )


class StateInvariantError(FaroError):
    """Raised when the ExperimentContext violates invariants required by the target state."""


class SnapshotNotFoundError(FaroError):
    """Raised when attempting to restore a nonexistent state snapshot."""

    def __init__(self, snapshot_id: str) -> None:
        self.snapshot_id = snapshot_id
        super().__init__(f"Snapshot '{snapshot_id}' was not found in FSM history.")
