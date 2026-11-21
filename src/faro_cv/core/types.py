"""Strongly typed Pydantic schemas shared across the core engine, API, and UI."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ComputeDevice = Literal["mps", "cuda", "cpu"]
AttackKind = Literal["fgsm", "pgd", "patch_eot"]
NormKind = Literal["linf", "l2"]
AdaptiveMode = Literal["none", "exact", "bpda"]
DefenseKind = Literal["svd", "fft", "pipeline"]
FFTFilterType = Literal["gaussian", "butterworth", "ideal", "bandstop"]


class OperatorDeviceStatus(BaseModel):
    """Execution device and probe telemetry for a single mathematical operator."""

    model_config = ConfigDict(frozen=True)

    device: ComputeDevice
    supported_on_accelerator: bool
    probe_latency_ms: float = Field(ge=0.0)
    fallback_reason: str | None = None


class OperatorDeviceProfile(BaseModel):
    """Hardware capability profile discovered at startup (Apple Silicon MPS aware)."""

    model_config = ConfigDict(frozen=True)

    torch_available: bool
    torch_version: str | None = None
    preferred_accelerator: ComputeDevice
    model_op: OperatorDeviceStatus
    svd_op: OperatorDeviceStatus
    fft_op: OperatorDeviceStatus


class TensorMetadata(BaseModel):
    """Lightweight, serializable summary of an image tensor for auditability."""

    model_config = ConfigDict(frozen=True)

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    channels: int = Field(default=3, gt=0)
    dtype: str = "float32"
    sha256: str = Field(min_length=8)
    min_value: float = 0.0
    max_value: float = 1.0


class PredictionEntry(BaseModel):
    """Single top-k classification prediction."""

    model_config = ConfigDict(frozen=True)

    class_index: int = Field(ge=0)
    label: str = Field(min_length=1)
    probability: float = Field(ge=0.0, le=1.0)


class InferenceSummary(BaseModel):
    """Classification output and telemetry for a single image pass."""

    model_config = ConfigDict(frozen=True)

    top_k: list[PredictionEntry] = Field(min_length=1)
    entropy: float = Field(ge=0.0)
    latency_ms: float = Field(ge=0.0)
    device: ComputeDevice

    @property
    def top1(self) -> PredictionEntry:
        return self.top_k[0]


class SVDParams(BaseModel):
    """Parameters governing Truncated SVD / Singular Value Thresholding."""

    model_config = ConfigDict(frozen=True)

    rank_k: int | None = Field(default=None, ge=1)
    energy_threshold: float | None = Field(default=None, gt=0.0, le=1.0)
    channel_mode: Literal["per_channel", "luminance"] = "per_channel"

    @model_validator(mode="after")
    def _validate_rank_or_energy(self) -> SVDParams:
        if self.rank_k is None and self.energy_threshold is None:
            raise ValueError("Specify either rank_k or energy_threshold for SVD purification.")
        return self


class FFTParams(BaseModel):
    """Parameters governing 2D-FFT frequency-domain attenuation."""

    model_config = ConfigDict(frozen=True)

    filter_type: FFTFilterType = "gaussian"
    cutoff_radius: float = Field(default=0.35, gt=0.0, le=1.0)
    band_width: float = Field(default=0.1, gt=0.0, le=0.5)
    butterworth_order: int = Field(default=2, ge=1, le=8)
    preserve_phase: bool = True


class DefenseConfig(BaseModel):
    """Configuration for SVD, 2D-FFT, or chained purification pipelines."""

    model_config = ConfigDict(frozen=True)

    kind: DefenseKind
    svd: SVDParams | None = None
    fft: FFTParams | None = None
    pipeline_order: tuple[Literal["svd", "fft"], ...] = ("fft", "svd")

    @model_validator(mode="after")
    def _validate_defense_params(self) -> DefenseConfig:
        if self.kind == "svd" and self.svd is None:
            raise ValueError("svd parameters are required when kind='svd'.")
        if self.kind == "fft" and self.fft is None:
            raise ValueError("fft parameters are required when kind='fft'.")
        if self.kind == "pipeline" and (self.svd is None or self.fft is None):
            raise ValueError("Both svd and fft parameters are required when kind='pipeline'.")
        return self


class AttackConfig(BaseModel):
    """Validated configuration for perturbation, patch+EoT, or white-box adaptive attacks."""

    model_config = ConfigDict(frozen=True)

    kind: AttackKind
    norm: NormKind = "linf"
    epsilon: float = Field(default=8.0 / 255.0, ge=0.0, le=1.0)
    step_size: float = Field(default=2.0 / 255.0, ge=0.0, le=1.0)
    steps: int = Field(default=10, ge=1, le=500)
    targeted: bool = False
    target_class: int | None = Field(default=None, ge=0)
    # Adversarial Patch + EoT parameters
    patch_size: int | None = Field(default=None, ge=8, le=224)
    patch_x: int | None = Field(default=None, ge=0)
    patch_y: int | None = Field(default=None, ge=0)
    eot_samples: int = Field(default=4, ge=1, le=32)
    # White-box adaptive attack (differentiable defense penetration / BPDA)
    adaptive_mode: AdaptiveMode = "none"
    adaptive_defense: DefenseConfig | None = None

    @model_validator(mode="after")
    def _validate_attack_constraints(self) -> AttackConfig:
        if self.targeted and self.target_class is None:
            raise ValueError("target_class must be provided when targeted=True.")
        if self.kind == "fgsm" and self.steps != 1:
            raise ValueError("FGSM is a single-step attack; steps must equal 1.")
        if self.kind == "patch_eot" and self.patch_size is None:
            raise ValueError("patch_size must be provided for patch_eot attacks.")
        if self.adaptive_mode != "none" and self.adaptive_defense is None:
            raise ValueError(
                "adaptive_defense must be specified when adaptive_mode is 'exact' or 'bpda'."
            )
        return self


class AttackProgress(BaseModel):
    """Live progress snapshot emitted during iterative attack optimization."""

    model_config = ConfigDict(frozen=True)

    current_step: int = Field(ge=0)
    total_steps: int = Field(ge=1)
    current_loss: float
    elapsed_ms: float = Field(ge=0.0)


class SpectralSummary(BaseModel):
    """Precomputed spectral diagnostics driving the interactive Scree Plot & FFT viewer."""

    model_config = ConfigDict(frozen=True)

    singular_values: list[float] = Field(min_length=1)
    cumulative_energy: list[float] = Field(min_length=1)
    effective_rank: int = Field(ge=1)
    retained_energy_ratio: float = Field(ge=0.0, le=1.0)
    high_freq_energy_before: float = Field(ge=0.0, le=1.0)
    high_freq_energy_after: float = Field(ge=0.0, le=1.0)


class EvaluationMetrics(BaseModel):
    """Quantitative comparison across Clean, Adversarial, and Purified passes.

    Separates unconditional consistency with the clean prediction from conditional
    recovery on samples whose prediction actually flipped under attack, and optionally
    tracks ground-truth accuracy when a human/dataset label is available.
    """

    model_config = ConfigDict(frozen=True)

    attack_l2_norm: float = Field(ge=0.0)
    attack_linf_norm: float = Field(ge=0.0)
    purified_psnr_db: float
    purified_ssim: float = Field(ge=-1.0, le=1.0)
    attack_succeeded: bool
    # True iff purified top-1 equals clean top-1 (regardless of whether attack flipped it)
    prediction_consistent: bool
    # None (N/A) when attack_succeeded is False; otherwise True iff purified recovers clean top-1
    conditional_recovered: bool | None = None
    clean_confidence: float = Field(ge=0.0, le=1.0)
    adversarial_confidence: float = Field(ge=0.0, le=1.0)
    purified_confidence: float = Field(ge=0.0, le=1.0)
    purification_latency_ms: float = Field(ge=0.0)
    # Phase 2: Grad-CAM spatial attention cosine similarity vs. Clean baseline [0, 1]
    attention_similarity_adv: float | None = Field(default=None, ge=0.0, le=1.0)
    attention_similarity_purified: float | None = Field(default=None, ge=0.0, le=1.0)
    # Optional ground-truth evaluation (reveals whether consistent predictions were actually right)
    ground_truth_class: int | None = Field(default=None, ge=0)
    clean_correct: bool | None = None
    adversarial_correct: bool | None = None
    purified_correct: bool | None = None
