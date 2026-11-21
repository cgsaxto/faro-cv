"""Architecture-aware Grad-CAM explainability and spatial attention shift analysis."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from faro_cv.models.registry import LoadedVisionModel, ensure_chw_tensor


def _resolve_cam_target_layer(loaded: LoadedVisionModel) -> tuple[nn.Module, bool]:
    """Return ``(target_module, is_vit_tokens)`` for supported CNN & ViT backbones."""
    backbone = loaded.model.backbone
    model_id = loaded.spec.model_id

    if model_id == "resnet18":
        layer4 = getattr(backbone, "layer4", None)
        if isinstance(layer4, nn.Sequential) and len(layer4) > 0:
            return layer4[-1], False
    elif model_id == "mobilenet_v3_small":
        features = getattr(backbone, "features", None)
        if isinstance(features, nn.Sequential) and len(features) > 0:
            return features[-1], False
    elif model_id == "vit_b_16":
        encoder = getattr(backbone, "encoder", None)
        layers = getattr(encoder, "layers", None) if encoder is not None else None
        if isinstance(layers, nn.Sequential) and len(layers) > 0:
            ln1 = getattr(layers[-1], "ln_1", None)
            if isinstance(ln1, nn.Module):
                return ln1, True

    # Fallback: pick last Conv2d layer for arbitrary CNN backbones
    last_conv: nn.Module | None = None
    for module in backbone.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is not None:
        return last_conv, False

    raise ValueError(f"Could not locate a Grad-CAM target layer for model '{model_id}'.")


def compute_gradcam(
    loaded: LoadedVisionModel,
    image: Tensor,
    *,
    target_class: int | None = None,
) -> Tensor:
    """Compute a normalized ``[H, W]`` Grad-CAM spatial saliency map in ``[0, 1]``.

    Supports CNN feature maps (ResNet-18, MobileNetV3) and ViT patch token grids (ViT-B/16).
    """
    chw = ensure_chw_tensor(image).to(loaded.device)
    height, width = int(chw.shape[1]), int(chw.shape[2])
    target_layer, is_vit = _resolve_cam_target_layer(loaded)

    activations: list[Tensor] = []
    gradients: list[Tensor] = []

    def _forward_hook(_module: nn.Module, _inp: tuple[Tensor, ...], output: Tensor) -> None:
        activations.append(output)

    def _backward_hook(
        _module: nn.Module,
        _grad_in: tuple[Tensor | None, ...],
        grad_out: tuple[Tensor | None, ...],
    ) -> None:
        if grad_out and grad_out[0] is not None:
            gradients.append(grad_out[0])

    fwd_handle = target_layer.register_forward_hook(_forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(_backward_hook)

    try:
        x_in = chw.unsqueeze(0).clone().detach().requires_grad_(True)
        logits = loaded.model(x_in)
        cls_idx = int(logits.argmax(dim=-1).item()) if target_class is None else target_class
        score = logits[0, cls_idx]
        loaded.model.zero_grad(set_to_none=True)
        score.backward()
    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    if not activations or not gradients:
        return torch.zeros((height, width), dtype=torch.float32)

    act = activations[0].detach()
    grad = gradients[0].detach()

    if is_vit:
        # ViT token layout: [B, 1 + N_patches, D] -> strip CLS token and reshape to [B, D, Gh, Gw]
        patch_act = act[:, 1:, :]
        patch_grad = grad[:, 1:, :]
        n_tokens = int(patch_act.shape[1])
        grid_side = round(math.sqrt(n_tokens))
        if grid_side * grid_side != n_tokens:
            return torch.zeros((height, width), dtype=torch.float32)
        act_2d = patch_act.transpose(1, 2).reshape(1, -1, grid_side, grid_side)
        grad_2d = patch_grad.transpose(1, 2).reshape(1, -1, grid_side, grid_side)
    else:
        act_2d = act
        grad_2d = grad

    # Channel importance weights alpha_k = GAP(d y^c / d A^k)
    weights = grad_2d.mean(dim=(-2, -1), keepdim=True)
    cam = F.relu((weights * act_2d).sum(dim=1, keepdim=True))
    cam_up = (
        F.interpolate(
            cam,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
    )

    cam_cpu = cam_up.to(device="cpu", dtype=torch.float32)
    cam_min = float(cam_cpu.min().item())
    cam_max = float(cam_cpu.max().item())
    if cam_max - cam_min <= 1e-8:
        return torch.zeros((height, width), dtype=torch.float32)
    return ((cam_cpu - cam_min) / (cam_max - cam_min)).clamp(0.0, 1.0)


def compute_attention_similarity(cam_reference: Tensor, cam_candidate: Tensor) -> float:
    """Compute cosine similarity in ``[0, 1]`` between two non-negative Grad-CAM maps.

    Values near ``1.0`` indicate preserved spatial focus; low values indicate attention
    hijacking (e.g., an adversarial patch pulling saliency away from the true object).
    """
    ref = cam_reference.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    cand = cam_candidate.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    denom = float(torch.linalg.vector_norm(ref, ord=2).item()) * float(
        torch.linalg.vector_norm(cand, ord=2).item()
    )
    if denom <= 1e-12:
        return 0.0
    sim = float(torch.dot(ref, cand).item()) / denom
    return round(max(0.0, min(1.0, sim)), 6)
