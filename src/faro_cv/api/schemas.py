"""Request and response schemas for the faro-cv FastAPI & WebSocket workbench bridge."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from faro_cv.core.fsm import FSMWireState, LabState
from faro_cv.core.types import AttackConfig, DefenseConfig, OperatorDeviceProfile


class ModelOption(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    display_name: str
    architecture: str
    input_size: int


class PresetOption(BaseModel):
    model_config = ConfigDict(frozen=True)

    preset_id: str
    label: str
    description: str


class SnapshotSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    snapshot_id: str
    label: str
    state: LabState
    created_at: datetime


class ImageBundle(BaseModel):
    """Lossless PNG data URLs powering the 3-column pixel loupe, Grad-CAM, and 2D-FFT viewer."""

    model_config = ConfigDict(frozen=True)

    clean_png: str | None = None
    adversarial_png: str | None = None
    residual_png: str | None = None
    purified_png: str | None = None
    clean_cam_png: str | None = None
    adversarial_cam_png: str | None = None
    purified_cam_png: str | None = None
    fft_before_png: str | None = None
    fft_after_png: str | None = None


class WorkbenchPayload(BaseModel):
    """Complete synchronized instrument state pushed over REST and WebSocket."""

    model_config = ConfigDict(frozen=True)

    fsm: FSMWireState
    device_profile: OperatorDeviceProfile
    images: ImageBundle
    available_models: list[ModelOption]
    available_presets: list[PresetOption]
    snapshots: list[SnapshotSummary]


class LoadModelRequest(BaseModel):
    model_id: str = "resnet18"
    pretrained: bool = False


class LoadPresetRequest(BaseModel):
    preset_id: str = "beacon_target"


class RunAttackRequest(BaseModel):
    config: AttackConfig


class RunPurifyRequest(BaseModel):
    config: DefenseConfig
    auto_evaluate: bool = True


class CreateSnapshotRequest(BaseModel):
    label: str = Field(default="", max_length=64)
