"""Vision model registry and pixel-space [0, 1] normalization wrapper."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torchvision import models

from faro_cv.core.types import ComputeDevice, InferenceSummary, PredictionEntry, TensorMetadata

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


class NormalizedVisionModel(nn.Module):
    """Wraps a torchvision classifier with differentiable [0, 1] -> ImageNet normalization.

    By folding normalization inside ``forward()``, all adversarial attacks (FGSM, PGD,
    Patch+EoT) and signal/matrix defenses (SVD, 2D-FFT) operate strictly in pixel space
    ``[0, 1]``, eliminating epsilon-scaling mismatches.
    """

    mean: Tensor
    std: Tensor

    def __init__(
        self,
        backbone: nn.Module,
        *,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        batched = x.unsqueeze(0) if x.ndim == 3 else x
        normalized = (batched - self.mean) / self.std
        out: Tensor = self.backbone(normalized)
        return out


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Metadata describing a supported vision backbone."""

    model_id: str
    display_name: str
    architecture: str
    input_size: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "resnet18": ModelSpec(
        model_id="resnet18",
        display_name="ResNet-18",
        architecture="CNN",
        input_size=224,
    ),
    "mobilenet_v3_small": ModelSpec(
        model_id="mobilenet_v3_small",
        display_name="MobileNetV3-Small",
        architecture="CNN (Lightweight)",
        input_size=224,
    ),
    "vit_b_16": ModelSpec(
        model_id="vit_b_16",
        display_name="ViT-B/16",
        architecture="Vision Transformer",
        input_size=224,
    ),
}


@dataclass(frozen=True)
class LoadedVisionModel:
    """Loaded, evaluation-mode vision model bound to a target device."""

    spec: ModelSpec
    model: NormalizedVisionModel
    labels: tuple[str, ...]
    device: ComputeDevice


def _default_imagenet_labels(num_classes: int = 1000) -> tuple[str, ...]:
    return tuple(f"class_{idx}" for idx in range(num_classes))


def _sync_device(device: ComputeDevice) -> None:
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_vision_model(
    model_id: str,
    *,
    pretrained: bool = True,
    device: ComputeDevice = "cpu",
    seed: int | None = None,
) -> LoadedVisionModel:
    """Instantiate and place a supported torchvision backbone in eval mode."""
    if model_id not in MODEL_SPECS:
        supported = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unsupported model_id '{model_id}'. Supported: [{supported}].")

    spec = MODEL_SPECS[model_id]
    weights: Any = None
    labels = _default_imagenet_labels()

    if not pretrained and seed is not None:
        torch.manual_seed(seed)

    if model_id == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone: nn.Module = models.resnet18(weights=weights)
    elif model_id == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
    elif model_id == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        backbone = models.vit_b_16(weights=weights)
    else:  # pragma: no cover - guarded by MODEL_SPECS lookup above
        raise ValueError(f"Unhandled model_id '{model_id}'.")

    if weights is not None and "categories" in weights.meta:
        labels = tuple(str(cat) for cat in weights.meta["categories"])

    wrapped = NormalizedVisionModel(backbone).to(device)
    wrapped.eval()
    for param in wrapped.parameters():
        param.requires_grad_(False)

    return LoadedVisionModel(
        spec=spec,
        model=wrapped,
        labels=labels,
        device=device,
    )


def ensure_chw_tensor(image: Tensor) -> Tensor:
    """Normalize input shape to float32 ``[3, H, W]`` in ``[0, 1]``."""
    tensor = image.detach().to(dtype=torch.float32)
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for single-image tensor, got {tensor.shape}.")
        tensor = tensor.squeeze(0)
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected [3, H, W] RGB tensor, got shape {tuple(tensor.shape)}.")
    return tensor.clamp(0.0, 1.0)


def tensor_to_metadata(image: Tensor) -> TensorMetadata:
    """Compute compact, deterministic TensorMetadata (including sha256) for FSM context."""
    chw = ensure_chw_tensor(image).cpu().contiguous()
    digest = hashlib.sha256(chw.numpy().tobytes()).hexdigest()[:16]
    return TensorMetadata(
        width=int(chw.shape[2]),
        height=int(chw.shape[1]),
        channels=int(chw.shape[0]),
        dtype=str(chw.dtype).replace("torch.", ""),
        sha256=digest,
        min_value=float(chw.min().item()),
        max_value=float(chw.max().item()),
    )


@torch.no_grad()
def infer_tensor(
    loaded: LoadedVisionModel,
    image: Tensor,
    *,
    top_k: int = 5,
) -> InferenceSummary:
    """Run evaluation-mode inference and return top-k predictions, entropy, and latency."""
    chw = ensure_chw_tensor(image).to(loaded.device)
    _sync_device(loaded.device)
    start = time.perf_counter()
    logits = loaded.model(chw.unsqueeze(0)).squeeze(0)
    probs = torch.softmax(logits, dim=-1)
    _sync_device(loaded.device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    k = min(top_k, int(probs.shape[-1]))
    top_probs, top_indices = torch.topk(probs, k=k)

    # Shannon entropy H(p) = -sum(p * log(p))
    clamped = probs.clamp(min=1e-12)
    entropy = float(-(clamped * torch.log(clamped)).sum().item())
    if math.isnan(entropy) or entropy < 0.0:
        entropy = 0.0

    entries: list[PredictionEntry] = []
    for prob_val, idx_val in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist(), strict=True):
        idx = int(idx_val)
        label = loaded.labels[idx] if idx < len(loaded.labels) else f"class_{idx}"
        entries.append(
            PredictionEntry(
                class_index=idx,
                label=label,
                probability=max(0.0, min(1.0, float(prob_val))),
            )
        )

    return InferenceSummary(
        top_k=entries,
        entropy=round(entropy, 6),
        latency_ms=round(elapsed_ms, 3),
        device=loaded.device,
    )
