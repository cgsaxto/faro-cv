"""High-level FSM-governed laboratory session orchestrator."""

from __future__ import annotations

import time
from collections.abc import Iterator

from torch import Tensor, nn

from faro_cv.analysis.explainability import compute_attention_similarity, compute_gradcam
from faro_cv.analysis.metrics import compute_evaluation_metrics
from faro_cv.attacks.adaptive import AdaptiveDefendedModel
from faro_cv.attacks.base import AttackStepResult
from faro_cv.attacks.perturbation import build_attack
from faro_cv.core.context import ExperimentContext
from faro_cv.core.device import probe_device_capabilities
from faro_cv.core.fsm import ExperimentFSM, LabEvent
from faro_cv.core.types import (
    AttackConfig,
    AttackProgress,
    DefenseConfig,
    EvaluationMetrics,
    OperatorDeviceProfile,
)
from faro_cv.defenses.base import PurificationResult
from faro_cv.defenses.pipeline import build_defense
from faro_cv.models.registry import (
    LoadedVisionModel,
    ensure_chw_tensor,
    infer_tensor,
    load_vision_model,
    tensor_to_metadata,
)


class LabSession:
    """Binds ExperimentFSM to live tensors, vision models, attacks, Grad-CAM, and defenses.

    Every mutating operation routes through ``self.fsm.dispatch(...)`` so state invariants,
    audit records, observer notifications, and fault recovery remain guaranteed.
    """

    def __init__(
        self,
        *,
        seed: int = 42,
        prefer_mps: bool = True,
        device_profile: OperatorDeviceProfile | None = None,
    ) -> None:
        self.fsm = ExperimentFSM(ExperimentContext(seed=seed))
        self.device_profile: OperatorDeviceProfile = device_profile or probe_device_capabilities(
            prefer_mps=prefer_mps
        )
        self._model: LoadedVisionModel | None = None
        self._clean_tensor: Tensor | None = None
        self._adversarial_tensor: Tensor | None = None
        self._purified_tensor: Tensor | None = None
        self._clean_cam: Tensor | None = None
        self._adversarial_cam: Tensor | None = None
        self._purified_cam: Tensor | None = None
        self._tensor_store: dict[str, Tensor] = {}
        self._cam_store: dict[str, Tensor] = {}
        self._last_purification_latency_ms: float = 0.0

    @property
    def model(self) -> LoadedVisionModel | None:
        return self._model

    @property
    def clean_tensor(self) -> Tensor | None:
        return self._clean_tensor

    @property
    def adversarial_tensor(self) -> Tensor | None:
        return self._adversarial_tensor

    @property
    def purified_tensor(self) -> Tensor | None:
        return self._purified_tensor

    @property
    def clean_cam(self) -> Tensor | None:
        return self._clean_cam

    @property
    def adversarial_cam(self) -> Tensor | None:
        return self._adversarial_cam

    @property
    def purified_cam(self) -> Tensor | None:
        return self._purified_cam

    def load_model(self, model_id: str, *, pretrained: bool = True) -> LoadedVisionModel:
        """Load a vision backbone on the probed accelerator and advance FSM to MODEL_READY."""
        try:
            loaded = load_vision_model(
                model_id,
                pretrained=pretrained,
                device=self.device_profile.model_op.device,
                seed=self.fsm.context.seed,
            )

            def _mutate(ctx: ExperimentContext) -> None:
                ctx.model_id = loaded.spec.model_id
                ctx.device_profile = self.device_profile

            self.fsm.dispatch(
                LabEvent.LOAD_MODEL,
                mutate=_mutate,
                metadata={"pretrained": pretrained, "device": loaded.device},
            )
            self._model = loaded
            self._clean_tensor = None
            self._adversarial_tensor = None
            self._purified_tensor = None
            self._clean_cam = None
            self._adversarial_cam = None
            self._purified_cam = None
            return loaded
        except Exception as exc:
            self._fail_safely(exc)
            raise

    def load_image(self, image_tensor: Tensor) -> Tensor:
        """Register a ``[3, H, W]`` clean image in ``[0, 1]`` and compute baseline inference + Grad-CAM."""
        if self._model is None:
            raise RuntimeError("No model loaded; call load_model() before load_image().")
        try:
            chw = ensure_chw_tensor(image_tensor).cpu()
            meta = tensor_to_metadata(chw)
            clean_inf = infer_tensor(self._model, chw)
            clean_cam = compute_gradcam(
                self._model,
                chw,
                target_class=clean_inf.top1.class_index,
            )

            def _mutate(ctx: ExperimentContext) -> None:
                ctx.input_meta = meta
                ctx.clean_inference = clean_inf

            self.fsm.dispatch(
                LabEvent.LOAD_INPUT,
                mutate=_mutate,
                metadata={"sha256": meta.sha256, "top1": clean_inf.top1.label},
            )
            self._clean_tensor = chw
            self._clean_cam = clean_cam
            self._tensor_store[meta.sha256] = chw
            self._cam_store[meta.sha256] = clean_cam
            self._adversarial_tensor = None
            self._purified_tensor = None
            self._adversarial_cam = None
            self._purified_cam = None
            return chw
        except Exception as exc:
            self._fail_safely(exc)
            raise

    def iter_attack(self, config: AttackConfig) -> Iterator[AttackStepResult]:
        """Stream iterative attack steps (oblivious or white-box adaptive) under FSM control."""
        if self._model is None or self._clean_tensor is None:
            raise RuntimeError("Model and clean input must be loaded before starting an attack.")

        clean_inf = self.fsm.context.clean_inference
        if clean_inf is None:
            raise RuntimeError("Missing clean_inference in FSM context.")

        def _start_mutate(ctx: ExperimentContext) -> None:
            ctx.attack_config = config

        self.fsm.dispatch(
            LabEvent.START_ATTACK,
            mutate=_start_mutate,
            metadata={
                "kind": config.kind,
                "steps": config.steps,
                "epsilon": config.epsilon,
                "adaptive_mode": config.adaptive_mode,
            },
        )
        self._adversarial_tensor = None
        self._purified_tensor = None
        self._adversarial_cam = None
        self._purified_cam = None

        attack = build_attack(config, seed=self.fsm.context.seed)

        # Wrap classifier with differentiable defense when evaluating white-box adaptive attacks
        target_model: nn.Module = self._model.model
        if config.adaptive_mode != "none" and config.adaptive_defense is not None:
            defense_for_attack = build_defense(
                config.adaptive_defense,
                device_profile=self.device_profile,
            )
            target_model = AdaptiveDefendedModel(
                self._model.model,
                defense_for_attack,
                mode=config.adaptive_mode,
            )

        start_time = time.perf_counter()
        last_step: AttackStepResult | None = None

        try:
            for step_result in attack.iterate(
                target_model,
                self._clean_tensor,
                clean_inf.top1.class_index,
            ):
                last_step = step_result
                self.fsm.context.attack_progress = AttackProgress(
                    current_step=step_result.step,
                    total_steps=step_result.total_steps,
                    current_loss=step_result.loss,
                    elapsed_ms=round((time.perf_counter() - start_time) * 1000.0, 3),
                )
                yield step_result

            if last_step is None:
                raise RuntimeError("Attack produced no optimization steps.")

            adv_chw = ensure_chw_tensor(last_step.adversarial_tensor).cpu()
            adv_meta = tensor_to_metadata(adv_chw)
            adv_inf = infer_tensor(self._model, adv_chw)
            adv_cam = compute_gradcam(
                self._model,
                adv_chw,
                target_class=adv_inf.top1.class_index,
            )

            def _complete_mutate(ctx: ExperimentContext) -> None:
                ctx.adversarial_meta = adv_meta
                ctx.adversarial_inference = adv_inf

            self.fsm.dispatch(
                LabEvent.COMPLETE_ATTACK,
                mutate=_complete_mutate,
                metadata={"sha256": adv_meta.sha256, "top1": adv_inf.top1.label},
            )
            self._adversarial_tensor = adv_chw
            self._adversarial_cam = adv_cam
            self._tensor_store[adv_meta.sha256] = adv_chw
            self._cam_store[adv_meta.sha256] = adv_cam
        except Exception as exc:
            self._fail_safely(exc)
            raise

    def run_attack(self, config: AttackConfig) -> AttackStepResult:
        """Execute an adversarial attack to completion."""
        last: AttackStepResult | None = None
        for step_result in self.iter_attack(config):
            last = step_result
        if last is None:
            raise RuntimeError("Attack produced no optimization steps.")
        return last

    def purify(self, config: DefenseConfig) -> PurificationResult:
        """Run SVD, 2D-FFT, or chained purification on the adversarial sample."""
        if self._model is None or self._adversarial_tensor is None:
            raise RuntimeError("Adversarial sample must exist before purification.")

        def _start_mutate(ctx: ExperimentContext) -> None:
            ctx.defense_config = config

        self.fsm.dispatch(
            LabEvent.START_PURIFY,
            mutate=_start_mutate,
            metadata={"kind": config.kind},
        )
        self._purified_tensor = None
        self._purified_cam = None

        try:
            defense = build_defense(config, device_profile=self.device_profile)
            result = defense.purify(self._adversarial_tensor)
            pur_chw = ensure_chw_tensor(result.purified_tensor).cpu()
            pur_meta = tensor_to_metadata(pur_chw)
            pur_inf = infer_tensor(self._model, pur_chw)
            pur_cam = compute_gradcam(
                self._model,
                pur_chw,
                target_class=pur_inf.top1.class_index,
            )

            def _complete_mutate(ctx: ExperimentContext) -> None:
                ctx.purified_meta = pur_meta
                ctx.purified_inference = pur_inf
                ctx.spectral_summary = result.spectral_summary

            self.fsm.dispatch(
                LabEvent.COMPLETE_PURIFY,
                mutate=_complete_mutate,
                metadata={
                    "effective_rank": result.spectral_summary.effective_rank,
                    "latency_ms": result.latency_ms,
                    "top1": pur_inf.top1.label,
                },
            )
            self._purified_tensor = pur_chw
            self._purified_cam = pur_cam
            self._tensor_store[pur_meta.sha256] = pur_chw
            self._cam_store[pur_meta.sha256] = pur_cam
            self._last_purification_latency_ms = result.latency_ms
            return PurificationResult(
                purified_tensor=pur_chw,
                spectral_summary=result.spectral_summary,
                latency_ms=result.latency_ms,
            )
        except Exception as exc:
            self._fail_safely(exc)
            raise

    def evaluate(self, *, ground_truth_class: int | None = None) -> EvaluationMetrics:
        """Compute quantitative metrics across Clean, Adversarial, and Purified passes."""
        if (
            self._clean_tensor is None
            or self._adversarial_tensor is None
            or self._purified_tensor is None
        ):
            raise RuntimeError("Clean, adversarial, and purified tensors are required to evaluate.")

        ctx = self.fsm.context
        if (
            ctx.clean_inference is None
            or ctx.adversarial_inference is None
            or ctx.purified_inference is None
        ):
            raise RuntimeError("Missing inference summaries in FSM context.")

        targeted_cls = (
            ctx.attack_config.target_class
            if (ctx.attack_config and ctx.attack_config.targeted)
            else None
        )
        attn_adv = (
            compute_attention_similarity(self._clean_cam, self._adversarial_cam)
            if (self._clean_cam is not None and self._adversarial_cam is not None)
            else None
        )
        attn_pur = (
            compute_attention_similarity(self._clean_cam, self._purified_cam)
            if (self._clean_cam is not None and self._purified_cam is not None)
            else None
        )

        metrics = compute_evaluation_metrics(
            clean_tensor=self._clean_tensor,
            adversarial_tensor=self._adversarial_tensor,
            purified_tensor=self._purified_tensor,
            clean_inference=ctx.clean_inference,
            adversarial_inference=ctx.adversarial_inference,
            purified_inference=ctx.purified_inference,
            purification_latency_ms=self._last_purification_latency_ms,
            targeted_class=targeted_cls,
            ground_truth_class=ground_truth_class,
            attention_similarity_adv=attn_adv,
            attention_similarity_purified=attn_pur,
        )

        def _mutate(staged: ExperimentContext) -> None:
            staged.evaluation = metrics

        self.fsm.dispatch(
            LabEvent.EVALUATE,
            mutate=_mutate,
            metadata={
                "attack_succeeded": metrics.attack_succeeded,
                "prediction_consistent": metrics.prediction_consistent,
                "conditional_recovered": metrics.conditional_recovered,
                "psnr_db": metrics.purified_psnr_db,
            },
        )
        return metrics

    def restore_snapshot(self, snapshot_id: str) -> None:
        """Restore FSM state, context, and corresponding cached image tensors and CAMs."""
        self.fsm.restore_snapshot(snapshot_id)
        ctx = self.fsm.context
        self._clean_tensor = (
            self._tensor_store.get(ctx.input_meta.sha256) if ctx.input_meta else None
        )
        self._adversarial_tensor = (
            self._tensor_store.get(ctx.adversarial_meta.sha256) if ctx.adversarial_meta else None
        )
        self._purified_tensor = (
            self._tensor_store.get(ctx.purified_meta.sha256) if ctx.purified_meta else None
        )
        self._clean_cam = self._cam_store.get(ctx.input_meta.sha256) if ctx.input_meta else None
        self._adversarial_cam = (
            self._cam_store.get(ctx.adversarial_meta.sha256) if ctx.adversarial_meta else None
        )
        self._purified_cam = (
            self._cam_store.get(ctx.purified_meta.sha256) if ctx.purified_meta else None
        )

    def reset(self) -> None:
        """Reset session and clear cached tensors."""
        self.fsm.dispatch(LabEvent.RESET)
        self._model = None
        self._clean_tensor = None
        self._adversarial_tensor = None
        self._purified_tensor = None
        self._clean_cam = None
        self._adversarial_cam = None
        self._purified_cam = None
        self._last_purification_latency_ms = 0.0

    def _fail_safely(self, exc: Exception) -> None:
        msg = f"{type(exc).__name__}: {exc}"
        if self.fsm.state.value != "ERROR":
            self.fsm.dispatch(
                LabEvent.FAIL,
                mutate=lambda ctx: setattr(ctx, "last_error", msg),
            )
