"""Comprehensive tests for the transactional ExperimentFSM core."""

from __future__ import annotations

import pytest

from faro_cv.core.context import ExperimentContext
from faro_cv.core.exceptions import (
    GuardRejectedError,
    InvalidTransitionError,
    SnapshotNotFoundError,
)
from faro_cv.core.fsm import ExperimentFSM, LabEvent, LabState, TransitionRecord
from faro_cv.core.types import (
    AttackConfig,
    DefenseConfig,
    EvaluationMetrics,
    InferenceSummary,
    SpectralSummary,
    SVDParams,
    TensorMetadata,
)


def _assert_state(fsm: ExperimentFSM, expected: LabState) -> None:
    assert fsm.state is expected


def _load_model(fsm: ExperimentFSM, model_id: str = "resnet18") -> TransitionRecord:
    def _mutate(ctx: ExperimentContext) -> None:
        ctx.model_id = model_id

    return fsm.dispatch(LabEvent.LOAD_MODEL, mutate=_mutate)


def _load_input(
    fsm: ExperimentFSM,
    meta: TensorMetadata,
    inference: InferenceSummary,
) -> TransitionRecord:
    def _mutate(ctx: ExperimentContext) -> None:
        ctx.input_meta = meta
        ctx.clean_inference = inference

    return fsm.dispatch(LabEvent.LOAD_INPUT, mutate=_mutate)


def _start_attack(fsm: ExperimentFSM, config: AttackConfig) -> TransitionRecord:
    def _mutate(ctx: ExperimentContext) -> None:
        ctx.attack_config = config

    return fsm.dispatch(LabEvent.START_ATTACK, mutate=_mutate)


def _complete_attack(
    fsm: ExperimentFSM,
    meta: TensorMetadata,
    inference: InferenceSummary,
) -> TransitionRecord:
    def _mutate(ctx: ExperimentContext) -> None:
        ctx.adversarial_meta = meta
        ctx.adversarial_inference = inference

    return fsm.dispatch(LabEvent.COMPLETE_ATTACK, mutate=_mutate)


def _start_purify(fsm: ExperimentFSM, config: DefenseConfig) -> TransitionRecord:
    def _mutate(ctx: ExperimentContext) -> None:
        ctx.defense_config = config

    return fsm.dispatch(LabEvent.START_PURIFY, mutate=_mutate)


def _complete_purify(
    fsm: ExperimentFSM,
    meta: TensorMetadata,
    inference: InferenceSummary,
    spectral: SpectralSummary,
) -> TransitionRecord:
    def _mutate(ctx: ExperimentContext) -> None:
        ctx.purified_meta = meta
        ctx.purified_inference = inference
        ctx.spectral_summary = spectral

    return fsm.dispatch(LabEvent.COMPLETE_PURIFY, mutate=_mutate)


def _evaluate(fsm: ExperimentFSM, metrics: EvaluationMetrics) -> TransitionRecord:
    def _mutate(ctx: ExperimentContext) -> None:
        ctx.evaluation = metrics

    return fsm.dispatch(LabEvent.EVALUATE, mutate=_mutate)


def _drive_to_evaluated(
    fsm: ExperimentFSM,
    *,
    sample_tensor_meta: TensorMetadata,
    clean_inference: InferenceSummary,
    adversarial_inference: InferenceSummary,
    purified_inference: InferenceSummary,
    pgd_config: AttackConfig,
    svd_defense_config: DefenseConfig,
    spectral_summary: SpectralSummary,
    evaluation_metrics: EvaluationMetrics,
) -> None:
    _load_model(fsm, "resnet18")
    _load_input(fsm, sample_tensor_meta, clean_inference)
    _start_attack(fsm, pgd_config)
    _complete_attack(fsm, sample_tensor_meta, adversarial_inference)
    _start_purify(fsm, svd_defense_config)
    _complete_purify(fsm, sample_tensor_meta, purified_inference, spectral_summary)
    _evaluate(fsm, evaluation_metrics)


def test_full_happy_path_lifecycle(
    sample_tensor_meta: TensorMetadata,
    clean_inference: InferenceSummary,
    adversarial_inference: InferenceSummary,
    purified_inference: InferenceSummary,
    pgd_config: AttackConfig,
    svd_defense_config: DefenseConfig,
    spectral_summary: SpectralSummary,
    evaluation_metrics: EvaluationMetrics,
) -> None:
    fsm = ExperimentFSM()
    _assert_state(fsm, LabState.IDLE)
    assert not fsm.is_accepting

    _drive_to_evaluated(
        fsm,
        sample_tensor_meta=sample_tensor_meta,
        clean_inference=clean_inference,
        adversarial_inference=adversarial_inference,
        purified_inference=purified_inference,
        pgd_config=pgd_config,
        svd_defense_config=svd_defense_config,
        spectral_summary=spectral_summary,
        evaluation_metrics=evaluation_metrics,
    )

    _assert_state(fsm, LabState.EVALUATED)
    assert fsm.is_accepting
    assert len(fsm.history) == 7
    assert fsm.history[-1].from_state is LabState.PURIFIED
    assert fsm.history[-1].to_state is LabState.EVALUATED


def test_attack_pause_resume_and_cancel_flow(
    sample_tensor_meta: TensorMetadata,
    clean_inference: InferenceSummary,
    pgd_config: AttackConfig,
) -> None:
    fsm = ExperimentFSM()
    _load_model(fsm)
    _load_input(fsm, sample_tensor_meta, clean_inference)
    _start_attack(fsm, pgd_config)
    _assert_state(fsm, LabState.ATTACKING)

    fsm.dispatch(LabEvent.PAUSE_ATTACK)
    _assert_state(fsm, LabState.PAUSED)

    fsm.dispatch(LabEvent.RESUME_ATTACK)
    _assert_state(fsm, LabState.ATTACKING)

    fsm.dispatch(LabEvent.CANCEL_ATTACK)
    _assert_state(fsm, LabState.INPUT_LOADED)
    assert fsm.context.adversarial_meta is None


def test_interactive_svd_cutoff_repurification_loop(
    sample_tensor_meta: TensorMetadata,
    clean_inference: InferenceSummary,
    adversarial_inference: InferenceSummary,
    purified_inference: InferenceSummary,
    pgd_config: AttackConfig,
    svd_defense_config: DefenseConfig,
    spectral_summary: SpectralSummary,
    evaluation_metrics: EvaluationMetrics,
) -> None:
    fsm = ExperimentFSM()
    _drive_to_evaluated(
        fsm,
        sample_tensor_meta=sample_tensor_meta,
        clean_inference=clean_inference,
        adversarial_inference=adversarial_inference,
        purified_inference=purified_inference,
        pgd_config=pgd_config,
        svd_defense_config=svd_defense_config,
        spectral_summary=spectral_summary,
        evaluation_metrics=evaluation_metrics,
    )

    # Dragging the SVD cutoff slider from EVALUATED re-enters PURIFYING without re-attacking
    new_svd = DefenseConfig(kind="svd", svd=SVDParams(rank_k=16))
    _start_purify(fsm, new_svd)
    _assert_state(fsm, LabState.PURIFYING)
    assert fsm.context.adversarial_meta is not None
    assert fsm.context.purified_meta is None
    assert fsm.context.evaluation is None

    _complete_purify(fsm, sample_tensor_meta, purified_inference, spectral_summary)
    _assert_state(fsm, LabState.PURIFIED)
    assert fsm.context.defense_config == new_svd


def test_invalid_transition_raises_with_allowed_events() -> None:
    fsm = ExperimentFSM()
    with pytest.raises(InvalidTransitionError) as exc_info:
        fsm.dispatch(LabEvent.START_ATTACK)

    err = exc_info.value
    assert err.current_state == "IDLE"
    assert err.event == "START_ATTACK"
    assert "LOAD_MODEL" in err.allowed_events


def test_guard_rejection_is_transactional_and_does_not_mutate_context() -> None:
    fsm = ExperimentFSM()

    # Missing model_id must reject transition and leave context + state untouched
    def _bad_mutator(ctx: ExperimentContext) -> None:
        ctx.seed = 999  # Should be rolled back when guard fails

    with pytest.raises(GuardRejectedError) as exc_info:
        fsm.dispatch(LabEvent.LOAD_MODEL, mutate=_bad_mutator)

    assert exc_info.value.guard_name == "has_model"
    _assert_state(fsm, LabState.IDLE)
    assert fsm.context.seed == 42
    assert len(fsm.history) == 0


def test_error_state_and_deterministic_recovery_anchor(
    sample_tensor_meta: TensorMetadata,
    clean_inference: InferenceSummary,
    pgd_config: AttackConfig,
) -> None:
    fsm = ExperimentFSM()
    _load_model(fsm)
    _load_input(fsm, sample_tensor_meta, clean_inference)
    _start_attack(fsm, pgd_config)
    _assert_state(fsm, LabState.ATTACKING)
    # Recovery anchor stays at last stable state (INPUT_LOADED), not transient ATTACKING
    assert fsm.recovery_state is LabState.INPUT_LOADED

    def _set_error(ctx: ExperimentContext) -> None:
        ctx.last_error = "MPS out-of-memory during PGD step 4"

    fsm.dispatch(LabEvent.FAIL, mutate=_set_error)
    _assert_state(fsm, LabState.ERROR)
    assert fsm.context.last_error is not None

    fsm.dispatch(LabEvent.RECOVER)
    _assert_state(fsm, LabState.INPUT_LOADED)
    assert fsm.context.last_error is None


def test_reset_from_any_state_preserves_session_identity_and_clears_artifacts(
    sample_tensor_meta: TensorMetadata,
    clean_inference: InferenceSummary,
) -> None:
    fsm = ExperimentFSM()
    original_session_id = fsm.context.session_id
    _load_model(fsm, "vit_b_16")
    _load_input(fsm, sample_tensor_meta, clean_inference)

    fsm.dispatch(LabEvent.RESET)
    _assert_state(fsm, LabState.IDLE)
    assert fsm.context.session_id == original_session_id
    assert fsm.context.model_id is None
    assert fsm.context.input_meta is None


def test_observer_subscription_and_unsubscribe() -> None:
    fsm = ExperimentFSM()
    received: list[TransitionRecord] = []
    unsubscribe = fsm.subscribe(lambda rec, _ctx: received.append(rec))

    def _set_model(ctx: ExperimentContext) -> None:
        ctx.model_id = "resnet18"

    fsm.dispatch(
        LabEvent.LOAD_MODEL,
        mutate=_set_model,
        metadata={"source": "unit_test"},
    )
    assert len(received) == 1
    assert received[0].metadata == {"source": "unit_test"}

    unsubscribe()
    fsm.dispatch(LabEvent.RESET)
    assert len(received) == 1


def test_snapshot_time_travel_and_missing_snapshot_error(
    sample_tensor_meta: TensorMetadata,
    clean_inference: InferenceSummary,
) -> None:
    fsm = ExperimentFSM()
    _load_model(fsm)
    snap_model = fsm.create_snapshot("after_model")

    _load_input(fsm, sample_tensor_meta, clean_inference)
    _assert_state(fsm, LabState.INPUT_LOADED)

    restored = fsm.restore_snapshot(snap_model.snapshot_id)
    assert restored.label == "after_model"
    _assert_state(fsm, LabState.MODEL_READY)
    assert fsm.context.input_meta is None

    with pytest.raises(SnapshotNotFoundError):
        fsm.restore_snapshot("snap_does_not_exist")


def test_wire_state_and_mermaid_export() -> None:
    fsm = ExperimentFSM()
    wire = fsm.to_wire_state()
    assert wire.state is LabState.IDLE
    assert wire.step_count == 0
    assert any(ev.event is LabEvent.LOAD_MODEL for ev in wire.allowed_events)

    diagram = ExperimentFSM.export_mermaid()
    assert diagram.startswith("stateDiagram-v2")
    assert "IDLE --> MODEL_READY : LOAD_MODEL" in diagram
    assert "EVALUATED --> PURIFYING : START_PURIFY" in diagram
