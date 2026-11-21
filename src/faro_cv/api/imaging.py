"""Lossless PNG encoding, residual/FFT spectrum colormaps, and built-in test targets."""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor

from faro_cv.core.types import ComputeDevice
from faro_cv.defenses.fft_purify import compute_log_magnitude_spectrum


@dataclass(frozen=True, slots=True)
class SamplePreset:
    """Metadata for a built-in deterministic 224x224 calibration/test image."""

    preset_id: str
    label: str
    description: str


SAMPLE_PRESETS: dict[str, SamplePreset] = {
    "beacon_target": SamplePreset(
        preset_id="beacon_target",
        label="Faro Beacon Scene",
        description="High-contrast architectural lighthouse geometry + horizon gradient (ideal for edge & SVD rank analysis).",
    ),
    "harmonic_grid": SamplePreset(
        preset_id="harmonic_grid",
        label="Harmonic Calibration Grid",
        description="Multi-frequency sinusoidal bands + sharp optical reticle (ideal for 2D-FFT cutoff & ringing inspection).",
    ),
    "chromatic_spheres": SamplePreset(
        preset_id="chromatic_spheres",
        label="Chromatic Shading Spheres",
        description="Smooth Phong-lit geometric primitives (ideal for luminance vs. RGB SVD comparison).",
    ),
}


def pil_to_chw_tensor(image: Image.Image, *, size: int = 224) -> Tensor:
    """Center-crop and resize a PIL image to ``[3, size, size]`` float32 in ``[0, 1]``."""
    rgb = image.convert("RGB")
    w, h = rgb.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    cropped = rgb.crop((left, top, left + side, top + side))
    resized = cropped.resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor.clamp(0.0, 1.0)


def bytes_to_chw_tensor(raw_bytes: bytes, *, size: int = 224) -> Tensor:
    """Decode uploaded PNG/JPEG bytes into a ``[3, size, size]`` float32 tensor."""
    with Image.open(io.BytesIO(raw_bytes)) as img:
        return pil_to_chw_tensor(img, size=size)


def tensor_to_png_data_url(image: Tensor) -> str:
    """Encode a ``[3, H, W]`` tensor in ``[0, 1]`` as a lossless base64 PNG data URL."""
    chw = image.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
    if chw.ndim == 4:
        chw = chw[0]
    hwc_u8 = (chw.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    pil_img = Image.fromarray(hwc_u8, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", compress_level=1)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def residual_to_png_data_url(clean: Tensor, adversarial: Tensor, *, gain: float = 12.0) -> str:
    """Render amplified ``|x_adv - x_clean|`` residual onto an instrument charcoal-vermilion map."""
    c = clean.detach().to(device="cpu", dtype=torch.float32)
    a = adversarial.detach().to(device="cpu", dtype=torch.float32)
    diff = (a - c).abs().mean(dim=0)  # [H, W]
    scaled = (diff * gain).clamp(0.0, 1.0).numpy()

    # Scientific dark-charcoal -> amber -> vermilion -> bone-white ramp (no neon purple)
    r = np.clip(14 + scaled * 241, 0, 255)
    g = np.clip(16 + np.power(scaled, 1.6) * 165, 0, 255)
    b = np.clip(20 + np.power(scaled, 3.0) * 140, 0, 255)
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    pil_img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", compress_level=1)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def fft_spectrum_to_png_data_url(
    image: Tensor,
    *,
    fft_device: ComputeDevice = "cpu",
) -> str:
    """Render centered 2D-FFT log-magnitude spectrum in an oscilloscope slate-cyan colormap."""
    spec = compute_log_magnitude_spectrum(image, fft_device=fft_device).cpu().numpy()
    # Crisp dark slate -> steel blue -> instrument cyan -> white peak
    v = np.power(np.clip(spec, 0.0, 1.0), 0.85)
    r = np.clip(10 + np.power(v, 2.2) * 235, 0, 255)
    g = np.clip(13 + np.power(v, 1.15) * 215, 0, 255)
    b = np.clip(18 + np.power(v, 0.8) * 232, 0, 255)
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    pil_img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", compress_level=1)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def gradcam_overlay_to_png_data_url(
    image: Tensor,
    cam: Tensor,
    *,
    alpha: float = 0.54,
) -> str:
    """Blend a ``[H, W]`` Grad-CAM map onto ``[3, H, W]`` using an instrument thermal ramp."""
    chw = image.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
    if chw.ndim == 4:
        chw = chw[0]
    base = chw.permute(1, 2, 0).numpy()  # [H, W, 3] in [0, 1]
    heat = cam.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0).numpy()

    # Scientific instrument thermal colormap: charcoal -> amber -> vermilion -> bone-white
    hr = np.clip(np.power(heat, 0.65) * 1.15, 0.0, 1.0)
    hg = np.clip(np.power(heat, 1.35) * 0.92, 0.0, 1.0)
    hb = np.clip(np.power(heat, 2.8) * 0.78, 0.0, 1.0)
    heat_rgb = np.stack([hr, hg, hb], axis=-1)

    # Spatially weighted alpha keeps zero-saliency regions crisp and readable
    weight = (heat[..., None] ** 0.75) * alpha
    blended = base * (1.0 - weight) + heat_rgb * weight
    rgb_u8 = (np.clip(blended, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    pil_img = Image.fromarray(rgb_u8, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", compress_level=1)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def generate_preset_tensor(preset_id: str, *, size: int = 224) -> Tensor:
    """Synthesize a crisp deterministic 224x224 calibration target image tensor."""
    if preset_id not in SAMPLE_PRESETS:
        supported = ", ".join(sorted(SAMPLE_PRESETS))
        raise ValueError(f"Unknown preset_id '{preset_id}'. Supported: [{supported}].")

    if preset_id == "beacon_target":
        return _render_beacon_target(size)
    if preset_id == "harmonic_grid":
        return _render_harmonic_grid(size)
    return _render_chromatic_spheres(size)


def _render_beacon_target(size: int) -> Tensor:
    """Architectural coastal lighthouse & horizon scene with crisp structural edges."""
    img = Image.new("RGB", (size, size), (18, 28, 42))
    draw = ImageDraw.Draw(img)

    # Sky & twilight gradient
    horizon = int(size * 0.64)
    for y in range(horizon):
        t = y / max(horizon - 1, 1)
        r = int(20 + 62 * t)
        g = int(34 + 68 * t)
        b = int(58 + 56 * t)
        draw.line([(0, y), (size, y)], fill=(r, g, b))

    # Sea reflections
    for y in range(horizon, size):
        t = (y - horizon) / max(size - horizon - 1, 1)
        wave = int(6 * math.sin(y * 0.65))
        r = max(0, min(255, int(24 + 14 * (1 - t)) + wave))
        g = max(0, min(255, int(44 + 28 * (1 - t)) + wave))
        b = max(0, min(255, int(68 + 42 * (1 - t)) + wave))
        draw.line([(0, y), (size, y)], fill=(r, g, b))

    # Lighthouse beam cone
    cx = int(size * 0.50)
    lantern_y = int(size * 0.27)
    draw.polygon(
        [(cx, lantern_y), (size, int(size * 0.10)), (size, int(size * 0.42))],
        fill=(228, 214, 174),
    )

    # Rocky headland base
    draw.polygon(
        [
            (int(size * 0.16), horizon + 18),
            (int(size * 0.34), horizon - 8),
            (int(size * 0.68), horizon - 6),
            (int(size * 0.84), horizon + 20),
        ],
        fill=(46, 52, 60),
    )

    # Striped lighthouse tower
    tower_top = int(size * 0.31)
    tower_bot = horizon - 4
    stripes = 5
    for i in range(stripes):
        y0 = tower_top + int((tower_bot - tower_top) * (i / stripes))
        y1 = tower_top + int((tower_bot - tower_top) * ((i + 1) / stripes))
        w0 = int(12 + 12 * (i / stripes))
        w1 = int(12 + 12 * ((i + 1) / stripes))
        color = (232, 234, 238) if i % 2 == 0 else (188, 58, 48)
        draw.polygon([(cx - w0, y0), (cx + w0, y0), (cx + w1, y1), (cx - w1, y1)], fill=color)

    # Lantern room & cupola
    draw.rectangle([cx - 14, lantern_y - 8, cx + 14, tower_top], fill=(246, 224, 152))
    draw.polygon(
        [(cx - 16, lantern_y - 8), (cx, lantern_y - 22), (cx + 16, lantern_y - 8)],
        fill=(38, 44, 52),
    )
    # Window slits
    draw.rectangle([cx - 3, int(size * 0.42), cx + 3, int(size * 0.47)], fill=(26, 30, 36))
    draw.rectangle([cx - 3, int(size * 0.52), cx + 3, int(size * 0.57)], fill=(26, 30, 36))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _render_harmonic_grid(size: int) -> Tensor:
    """Multi-frequency sinusoidal pattern + sharp geometric calibration rings."""
    coords = torch.linspace(-1.0, 1.0, steps=size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    rr = torch.sqrt(xx.pow(2) + yy.pow(2))

    low = 0.5 + 0.32 * torch.cos(3.0 * math.pi * xx) * torch.cos(3.0 * math.pi * yy)
    mid = 0.5 + 0.25 * torch.sin(10.0 * math.pi * rr)
    ring = ((rr > 0.42) & (rr < 0.46)).to(torch.float32) * 0.85
    cross = ((xx.abs() < 0.015) | (yy.abs() < 0.015)).to(torch.float32) * 0.65

    r = (0.45 * low + 0.35 * mid + ring).clamp(0.0, 1.0)
    g = (0.50 * low + 0.25 * mid + 0.5 * cross).clamp(0.0, 1.0)
    b = (0.35 * low + 0.45 * (1.0 - mid) + 0.7 * cross).clamp(0.0, 1.0)
    return torch.stack([r, g, b], dim=0).contiguous()


def _render_chromatic_spheres(size: int) -> Tensor:
    """Smooth Phong-shaded overlapping geometric spheres on a gridded lab surface."""
    coords = torch.linspace(-1.0, 1.0, steps=size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")

    # Subtle studio grid background
    grid = (((xx * 10).abs() % 1.0 < 0.04) | ((yy * 10).abs() % 1.0 < 0.04)).to(torch.float32)
    canvas = torch.zeros(3, size, size, dtype=torch.float32)
    canvas[0] = 0.12 + 0.05 * grid
    canvas[1] = 0.14 + 0.05 * grid
    canvas[2] = 0.18 + 0.06 * grid

    spheres = [
        (-0.28, -0.10, 0.48, (0.85, 0.32, 0.26)),
        (0.32, 0.16, 0.42, (0.22, 0.68, 0.58)),
        (0.08, -0.34, 0.28, (0.88, 0.74, 0.32)),
    ]
    lx, ly, lz = -0.45, -0.55, 0.70

    for cx, cy, rad, (cr, cg, cb) in spheres:
        dx = (xx - cx) / rad
        dy = (yy - cy) / rad
        dist_sq = dx.pow(2) + dy.pow(2)
        mask = dist_sq <= 1.0
        nz = torch.sqrt((1.0 - dist_sq).clamp(min=0.0))
        diffuse = (dx * lx + dy * ly + nz * lz).clamp(min=0.0)
        spec = torch.pow((diffuse).clamp(min=0.0), 18.0) * 0.45
        shade = 0.18 + 0.82 * diffuse

        canvas[0] = torch.where(mask, (cr * shade + spec).clamp(0.0, 1.0), canvas[0])
        canvas[1] = torch.where(mask, (cg * shade + spec).clamp(0.0, 1.0), canvas[1])
        canvas[2] = torch.where(mask, (cb * shade + spec).clamp(0.0, 1.0), canvas[2])

    return canvas.contiguous()
