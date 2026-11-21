"""Controlled detail-loss vs. denoising comparison: Unprocessed vs. SVD-only vs. 2D-FFT-only.

Adds fixed-seed Gaussian noise (sigma=0.08) to synthetic calibration targets and feeds the
exact same noisy tensor into:
  1. Unprocessed (Noisy Baseline)
  2. SVD-only (Truncated SVD, rank k=28, RGB per-channel)
  3. 2D-FFT-only (Gaussian radial low-pass, cutoff r=0.35, exact DFT grid, phase-preserving)

Measures whole-image fidelity (PSNR, SSIM), reference-ROI luminance residual standard
deviation (output - clean reference), luminance contrast (Delta L) and retained contrast (%)
of a pre-selected thin line or sharp edge, and pipeline latency (CPU, including workbench
spectral/SVD diagnostics). Renders a lossless pixel-exact comparison plate for the primary
seed (seed=42) and records additional noise seeds (123, 999) in the CSV and Markdown summary.

Usage:
    .venv/bin/python experiments/detail_tradeoff_study.py
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

from faro_cv.analysis.metrics import compute_psnr, compute_ssim
from faro_cv.api.imaging import generate_preset_tensor
from faro_cv.core.types import DefenseConfig, FFTParams, SVDParams
from faro_cv.defenses.fft_purify import FFTPurification
from faro_cv.defenses.svd_purify import SVDPurification


@dataclass(frozen=True, slots=True)
class BoxROI:
    """Pixel bounding box ``[x0, x1) x [y0, y1)`` on a 224x224 calibration image."""

    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """Pre-specified Reference ROI and thin-feature contrast definitions."""

    preset_id: str
    display_name: str
    ref_roi: BoxROI
    detail_roi: BoxROI
    feature_label: str
    feature_fg: BoxROI
    feature_bg: tuple[BoxROI, ...]


@dataclass(frozen=True, slots=True)
class DetailTradeoffRow:
    """Measured denoising vs. detail-preservation record for one (seed, image, method)."""

    seed: int
    image: str
    feature_label: str
    method: str
    psnr_db: float
    ssim: float
    ref_roi_residual_std: float
    feature_contrast_luma: float
    feature_contrast_retained_pct: float
    pipeline_latency_ms: float


TARGET_SPECS: tuple[TargetSpec, ...] = (
    TargetSpec(
        preset_id="harmonic_grid",
        display_name="Harmonic Calibration Grid",
        ref_roi=BoxROI(16, 16, 52, 52),
        detail_roi=BoxROI(168, 94, 204, 130),
        feature_label="Horizontal reticle line (y=110..113)",
        feature_fg=BoxROI(168, 110, 204, 114),
        feature_bg=(BoxROI(168, 104, 204, 108), BoxROI(168, 116, 204, 120)),
    ),
    TargetSpec(
        preset_id="beacon_target",
        display_name="Faro Beacon Scene",
        ref_roi=BoxROI(16, 16, 52, 52),
        detail_roi=BoxROI(94, 84, 130, 120),
        feature_label="Lighthouse window slit (x=109..115, y=94..105)",
        feature_fg=BoxROI(109, 94, 115, 105),
        feature_bg=(BoxROI(103, 94, 108, 105), BoxROI(116, 94, 121, 105)),
    ),
)


def _to_luma(rgb_chw: Tensor) -> Tensor:
    """Compute BT.601 luminance ``[H, W]`` in ``[0, 1]`` from ``[3, H, W]`` RGB."""
    x = rgb_chw.detach().to(device="cpu", dtype=torch.float32)
    return 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]


def _slice_roi(img_hw: Tensor, box: BoxROI) -> Tensor:
    return img_hw[box.y0 : box.y1, box.x0 : box.x1]


def compute_ref_roi_residual_std(clean: Tensor, candidate: Tensor, ref_roi: BoxROI) -> float:
    """Std of luminance residual ``(candidate - clean)`` inside the pre-specified Reference ROI."""
    diff = _slice_roi(_to_luma(candidate) - _to_luma(clean), ref_roi)
    return round(float(diff.std(unbiased=False).item()), 5)


def compute_feature_contrast(image: Tensor, fg: BoxROI, bg_boxes: tuple[BoxROI, ...]) -> float:
    """Absolute BT.601 luminance difference between a pre-selected thin feature and background."""
    luma = _to_luma(image)
    fg_mean = float(_slice_roi(luma, fg).mean().item())
    bg_vals = torch.cat([_slice_roi(luma, b).reshape(-1) for b in bg_boxes])
    bg_mean = float(bg_vals.mean().item())
    return round(abs(fg_mean - bg_mean), 5)


def _add_fixed_seed_noise(clean: Tensor, *, sigma: float, seed: int) -> Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(clean.shape, generator=gen, dtype=torch.float32) * sigma
    return (clean + noise).clamp(0.0, 1.0)


def _benchmark_method_latency(
    fn: object,
    noisy: Tensor,
    *,
    warmup: int = 5,
    repeats: int = 20,
) -> tuple[Tensor, float]:
    """Run identical warmup + repeated in-memory CPU timing (including workbench diagnostics)."""
    assert callable(fn)
    out: Tensor = noisy
    for _ in range(warmup):
        out = fn(noisy)
    start = time.perf_counter()
    for _ in range(repeats):
        out = fn(noisy)
    elapsed_ms = ((time.perf_counter() - start) / float(repeats)) * 1000.0
    return out, round(elapsed_ms, 3)


def _tensor_to_pil(rgb_chw: Tensor) -> Image.Image:
    hwc = (
        (
            rgb_chw.detach()
            .to(device="cpu", dtype=torch.float32)
            .clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .numpy()
            * 255.0
        )
        .round()
        .astype(np.uint8)
    )
    return Image.fromarray(hwc, mode="RGB")


def _load_mono_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _render_comparison_plate(
    panels: list[tuple[TargetSpec, dict[str, Tensor], list[DetailTradeoffRow]]],
    *,
    sigma: float,
    seed: int,
    svd_k: int,
    fft_r: float,
    output_path: Path,
) -> Path:
    """Render a crisp, pixel-exact Swiss-industrial side-by-side comparison figure."""
    bg_color = (9, 10, 12)
    surface_color = (16, 18, 22)
    border_color = (38, 43, 54)
    text_primary = (230, 232, 236)
    text_muted = (139, 147, 163)
    ref_accent = (16, 185, 129)  # emerald
    detail_accent = (245, 158, 11)  # amber
    cutoff_accent = (56, 189, 248)  # cyan

    col_keys = [
        ("Clean Reference", "Clean"),
        (f"Noisy Input (\u03c3={sigma:.2f})", "Noisy (Unprocessed)"),
        (f"SVD-only (k={svd_k})", f"SVD-only (k={svd_k})"),
        (f"2D-FFT Gaussian (r={fft_r:.2f})", f"2D-FFT Gaussian (r={fft_r:.2f})"),
    ]

    full_px = 224
    zoom_px = 180  # 36x36 ROI rendered at exact 5x nearest-neighbor scale
    col_w = 248
    gap = 12
    pad_x = 20
    header_h = 54
    section_header_h = 30
    col_label_h = 24
    zoom_label_h = 20
    metrics_h = 24
    footer_note_h = 26

    total_w = pad_x * 2 + len(col_keys) * col_w + (len(col_keys) - 1) * gap
    section_h = (
        section_header_h
        + col_label_h
        + full_px
        + 10
        + zoom_label_h
        + zoom_px
        + 10
        + zoom_label_h
        + zoom_px
        + metrics_h
        + 16
    )
    total_h = header_h + len(panels) * section_h + footer_note_h + 12

    canvas = Image.new("RGB", (total_w, total_h), bg_color)
    draw = ImageDraw.Draw(canvas)

    font_title = _load_mono_font(15)
    font_sub = _load_mono_font(11)
    font_label = _load_mono_font(11)
    font_small = _load_mono_font(9)

    # Top instrument header
    draw.rectangle([0, 0, total_w, header_h - 1], fill=surface_color)
    draw.line([(0, header_h - 1), (total_w, header_h - 1)], fill=border_color, width=1)
    draw.text(
        (pad_x, 10),
        "FARO-CV · CONTROLLED DENOISING vs. DETAIL PRESERVATION (SYNTHETIC CALIBRATION IMAGES)",
        fill=text_primary,
        font=font_title,
    )
    draw.text(
        (pad_x, 31),
        f"Identical noisy input (Gaussian \u03c3={sigma:.2f}, seed={seed}) · 5\u00d7 nearest-neighbor crops · GREEN = Reference ROI · AMBER = Target Detail ROI",
        fill=text_muted,
        font=font_sub,
    )

    y_cursor = header_h + 12

    for spec, tensors, rows in panels:
        row_by_method = {r.method: r for r in rows}
        draw.rectangle(
            [pad_x, y_cursor, total_w - pad_x, y_cursor + section_header_h - 4],
            fill=surface_color,
            outline=border_color,
        )
        draw.text(
            (pad_x + 10, y_cursor + 7),
            f"{spec.display_name.upper()} (Synthetic calibration image, 224\u00d7224)  ·  Target Feature: {spec.feature_label}",
            fill=cutoff_accent,
            font=font_label,
        )
        y_row = y_cursor + section_header_h

        for col_idx, (col_title, method_key) in enumerate(col_keys):
            x0 = pad_x + col_idx * (col_w + gap)
            img_x = x0 + (col_w - full_px) // 2

            draw.text((x0 + 4, y_row + 4), col_title, fill=text_primary, font=font_label)

            # Full 224x224 image
            img_y = y_row + col_label_h
            pil_full = _tensor_to_pil(tensors[method_key])
            canvas.paste(pil_full, (img_x, img_y))
            draw.rectangle(
                [img_x - 1, img_y - 1, img_x + full_px, img_y + full_px],
                outline=border_color,
            )

            # Draw ROI callout boxes on Clean Reference column
            if col_idx == 0:
                rr = spec.ref_roi
                dr = spec.detail_roi
                draw.rectangle(
                    [img_x + rr.x0, img_y + rr.y0, img_x + rr.x1, img_y + rr.y1],
                    outline=ref_accent,
                    width=2,
                )
                draw.rectangle(
                    [img_x + dr.x0, img_y + dr.y0, img_x + dr.x1, img_y + dr.y1],
                    outline=detail_accent,
                    width=2,
                )

            # Row 2: Reference ROI 5x pixel zoom (luminance residual std vs. clean)
            ref_label_y = img_y + full_px + 10
            row_stats = row_by_method.get(method_key)
            ref_caption = (
                "Reference ROI 5\u00d7 (Clean)"
                if row_stats is None
                else f"Reference ROI · residual \u03c3={row_stats.ref_roi_residual_std:.4f}"
            )
            draw.text((x0 + 4, ref_label_y + 2), ref_caption, fill=ref_accent, font=font_small)

            ref_img_y = ref_label_y + zoom_label_h
            ref_crop = pil_full.crop(
                (spec.ref_roi.x0, spec.ref_roi.y0, spec.ref_roi.x1, spec.ref_roi.y1)
            ).resize((zoom_px, zoom_px), Image.Resampling.NEAREST)
            zoom_x = x0 + (col_w - zoom_px) // 2
            canvas.paste(ref_crop, (zoom_x, ref_img_y))
            draw.rectangle(
                [zoom_x - 1, ref_img_y - 1, zoom_x + zoom_px, ref_img_y + zoom_px],
                outline=ref_accent,
            )

            # Row 3: Thin Line / Edge ROI 5x pixel zoom (contrast retained % vs. clean)
            det_label_y = ref_img_y + zoom_px + 10
            det_caption = (
                "Detail ROI 5\u00d7 (Clean: 100.0%)"
                if row_stats is None
                else f"Contrast retained: {row_stats.feature_contrast_retained_pct:.1f}%"
            )
            draw.text((x0 + 4, det_label_y + 2), det_caption, fill=detail_accent, font=font_small)

            det_img_y = det_label_y + zoom_label_h
            det_crop = pil_full.crop(
                (spec.detail_roi.x0, spec.detail_roi.y0, spec.detail_roi.x1, spec.detail_roi.y1)
            ).resize((zoom_px, zoom_px), Image.Resampling.NEAREST)
            canvas.paste(det_crop, (zoom_x, det_img_y))
            draw.rectangle(
                [zoom_x - 1, det_img_y - 1, zoom_x + zoom_px, det_img_y + zoom_px],
                outline=detail_accent,
            )

            # Compact quantitative footer per column
            foot_y = det_img_y + zoom_px + 6
            if row_stats is None:
                foot_text = "Clean Reference · PSNR \u221e"
            else:
                foot_text = f"PSNR {row_stats.psnr_db:.2f}dB · SSIM {row_stats.ssim:.3f} · {row_stats.pipeline_latency_ms:.1f}ms*"
            draw.text((x0 + 4, foot_y), foot_text, fill=text_muted, font=font_small)

        y_cursor += section_h

    # Bottom methodology note
    draw.line(
        [(pad_x, total_h - footer_note_h), (total_w - pad_x, total_h - footer_note_h)],
        fill=border_color,
    )
    draw.text(
        (pad_x, total_h - footer_note_h + 7),
        "*Pipeline latency (CPU, incl. workbench spectral/SVD diagnostics). Exact DFT grid via fftshift(fftfreq(n))/0.5 (DC gain = 1.0).",
        fill=text_muted,
        font=font_small,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", compress_level=6)
    return output_path


def _export_web_study_bundle(
    seed_panels: dict[int, list[tuple[TargetSpec, dict[str, Tensor], list[DetailTradeoffRow]]]],
    all_rows: list[DetailTradeoffRow],
    *,
    sigma: float,
    primary_seed: int,
    extra_seeds: tuple[int, ...],
    svd_k: int,
    fft_r: float,
    web_study_dir: Path,
) -> Path:
    """Export per-seed lossless 224x224 PNGs (shared clean) and study-data.json."""
    web_study_dir.mkdir(parents=True, exist_ok=True)
    key_to_slug = {
        "Noisy (Unprocessed)": "noisy",
        f"SVD-only (k={svd_k})": "svd",
        f"2D-FFT Gaussian (r={fft_r:.2f})": "fft",
    }

    all_seeds = (primary_seed, *tuple(s for s in extra_seeds if s != primary_seed))
    primary_panels = seed_panels[primary_seed]
    targets_payload: list[dict[str, object]] = []

    for spec_idx, (spec, primary_tensors, _primary_rows) in enumerate(primary_panels):
        # Save shared clean reference PNG once per target
        clean_filename = f"{spec.preset_id}_clean.png"
        _tensor_to_pil(primary_tensors["Clean"]).save(
            web_study_dir / clean_filename,
            format="PNG",
            compress_level=6,
        )
        clean_url = f"/study/{clean_filename}"

        # Save per-seed noisy, svd, and fft PNGs so images always match the selected seed
        images_by_seed: dict[str, dict[str, str]] = {}
        for seed in all_seeds:
            _, seed_tensors, _ = seed_panels[seed][spec_idx]
            seed_urls: dict[str, str] = {"clean": clean_url}
            for method_key, slug in key_to_slug.items():
                filename = f"{spec.preset_id}_s{seed}_{slug}.png"
                _tensor_to_pil(seed_tensors[method_key]).save(
                    web_study_dir / filename,
                    format="PNG",
                    compress_level=6,
                )
                seed_urls[slug] = f"/study/{filename}"
            images_by_seed[str(seed)] = seed_urls

        clean_contrast = compute_feature_contrast(
            primary_tensors["Clean"],
            spec.feature_fg,
            spec.feature_bg,
        )
        targets_payload.append(
            {
                "preset_id": spec.preset_id,
                "display_name": spec.display_name,
                "feature_label": spec.feature_label,
                "clean_feature_contrast_luma": clean_contrast,
                "ref_roi": asdict(spec.ref_roi),
                "detail_roi": asdict(spec.detail_roi),
                "feature_fg": asdict(spec.feature_fg),
                "feature_bg": [asdict(b) for b in spec.feature_bg],
                "images": images_by_seed[str(primary_seed)],
                "images_by_seed": images_by_seed,
            }
        )

    bundle = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "sigma": sigma,
        "primary_seed": primary_seed,
        "seeds": list(all_seeds),
        "svd_k": svd_k,
        "fft_r": fft_r,
        "fft_grid_note": "Exact centered DFT grid via fftshift(fftfreq(n))/0.5 (unit DC gain).",
        "latency_note": "Pipeline latency (CPU, incl. workbench spectral/SVD diagnostics).",
        "targets": targets_payload,
        "rows": [asdict(r) for r in all_rows],
    }

    json_path = web_study_dir / "study-data.json"
    json_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    return json_path


def run_detail_tradeoff_study(
    *,
    sigma: float = 0.08,
    primary_seed: int = 42,
    extra_seeds: tuple[int, ...] = (123, 999),
    svd_k: int = 28,
    fft_r: float = 0.35,
    output_dir: Path = Path("experiments/results"),
    figure_path: Path = Path("docs/assets/detail-tradeoff-comparison.png"),
    web_study_dir: Path | None = Path("web/public/study"),
) -> tuple[list[DetailTradeoffRow], Path, Path, Path]:
    """Execute the controlled noise-reduction vs. detail-preservation study across seeds."""
    svd_op = SVDPurification(DefenseConfig(kind="svd", svd=SVDParams(rank_k=svd_k)))
    fft_op = FFTPurification(
        DefenseConfig(
            kind="fft",
            fft=FFTParams(filter_type="gaussian", cutoff_radius=fft_r, preserve_phase=True),
        )
    )

    seeds = (primary_seed, *tuple(s for s in extra_seeds if s != primary_seed))
    all_rows: list[DetailTradeoffRow] = []
    seed_panels: dict[int, list[tuple[TargetSpec, dict[str, Tensor], list[DetailTradeoffRow]]]] = {
        s: [] for s in seeds
    }

    for seed in seeds:
        for spec in TARGET_SPECS:
            clean = generate_preset_tensor(spec.preset_id, size=224)
            noisy = _add_fixed_seed_noise(clean, sigma=sigma, seed=seed)
            clean_contrast = compute_feature_contrast(clean, spec.feature_fg, spec.feature_bg)

            method_specs: list[tuple[str, object]] = [
                ("Noisy (Unprocessed)", lambda t: t),
                (f"SVD-only (k={svd_k})", lambda t: svd_op.purify(t).purified_tensor),
                (
                    f"2D-FFT Gaussian (r={fft_r:.2f})",
                    lambda t: fft_op.purify(t).purified_tensor,
                ),
            ]

            tensors_for_plate: dict[str, Tensor] = {"Clean": clean}
            spec_rows: list[DetailTradeoffRow] = []

            for method_name, fn in method_specs:
                if method_name == "Noisy (Unprocessed)":
                    out_tensor = noisy
                    latency_ms = 0.0
                else:
                    # Full timing sweep on primary seed; light timing on extra seeds
                    repeats = 20 if seed == primary_seed else 5
                    out_tensor, latency_ms = _benchmark_method_latency(
                        fn, noisy, warmup=5, repeats=repeats
                    )

                tensors_for_plate[method_name] = out_tensor
                psnr_db = compute_psnr(clean, out_tensor)
                ssim_val = compute_ssim(clean, out_tensor)
                ref_res_std = compute_ref_roi_residual_std(clean, out_tensor, spec.ref_roi)
                feat_contrast = compute_feature_contrast(
                    out_tensor, spec.feature_fg, spec.feature_bg
                )
                retained_pct = (
                    round((feat_contrast / clean_contrast) * 100.0, 1)
                    if clean_contrast > 1e-8
                    else 0.0
                )

                row = DetailTradeoffRow(
                    seed=seed,
                    image=spec.preset_id,
                    feature_label=spec.feature_label,
                    method=method_name,
                    psnr_db=psnr_db,
                    ssim=ssim_val,
                    ref_roi_residual_std=ref_res_std,
                    feature_contrast_luma=feat_contrast,
                    feature_contrast_retained_pct=retained_pct,
                    pipeline_latency_ms=latency_ms,
                )
                spec_rows.append(row)
                all_rows.append(row)

            seed_panels[seed].append((spec, tensors_for_plate, spec_rows))

    # Export CSV (all seeds)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "detail_tradeoff.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(all_rows[0]).keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(asdict(row))

    # Export Markdown summary (primary seed table + multi-seed consistency table)
    md_path = output_dir / "detail_tradeoff_summary.md"
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    primary_rows = [r for r in all_rows if r.seed == primary_seed]
    extra_rows = [r for r in all_rows if r.seed != primary_seed]

    lines = [
        "# SVD vs. 2D-FFT Denoising & Detail-Loss Study (Synthetic Calibration Images)",
        "",
        f"- **Generated**: {timestamp}",
        f"- **Noise model**: Identical fixed-seed Gaussian noise (`sigma={sigma:.2f}`, primary `seed={primary_seed}`, verification seeds `{list(extra_seeds)}`)",
        f"- **Methods compared**: Unprocessed Noisy vs. `SVD-only (k={svd_k}, RGB)` vs. `2D-FFT Gaussian (r={fft_r:.2f}, exact DFT grid via fftshift(fftfreq)/0.5, preserve_phase=True)`",
        "- **Reference ROI residual**: Standard deviation of `(output - clean)` BT.601 luminance inside `[16:52, 16:52]` (captures residual noise plus any local smoothing distortion on smooth/textured regions).",
        "- **Pipeline latency (CPU, incl. diagnostics)**: Measured on the same CPU device after 5 warmup passes (mean of 20 runs for primary seed), reflecting full workbench operator cost including spectral/SVD diagnostic telemetry.",
        "- **Scope**: Synthetic calibration images (`harmonic_grid`, `beacon_target`, 224\u00d7224).",
        "",
        f"## 1. Primary Controlled Comparison (`seed={primary_seed}`)",
        "",
        "| Calibration Image | Pre-Selected Detail Feature | Method | PSNR (dB) | SSIM | Ref ROI Residual \u03c3 | Feature \u0394L (Y) | Contrast Retained (%) | Pipeline Latency (ms, CPU incl. diagnostics) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in primary_rows:
        lines.append(
            f"| `{r.image}` | {r.feature_label} | {r.method} | "
            f"{r.psnr_db:.2f} | {r.ssim:.4f} | {r.ref_roi_residual_std:.4f} | "
            f"{r.feature_contrast_luma:.4f} | {r.feature_contrast_retained_pct:.1f}% | "
            f"{r.pipeline_latency_ms:.2f} |"
        )

    if extra_rows:
        lines.extend(
            [
                "",
                "## 2. Multi-Seed Robustness Check (Additional Noise Seeds)",
                "",
                "| Seed | Calibration Image | Method | PSNR (dB) | SSIM | Ref ROI Residual \u03c3 | Feature \u0394L (Y) | Contrast Retained (%) |",
                "|---:|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for r in extra_rows:
            lines.append(
                f"| `{r.seed}` | `{r.image}` | {r.method} | "
                f"{r.psnr_db:.2f} | {r.ssim:.4f} | {r.ref_roi_residual_std:.4f} | "
                f"{r.feature_contrast_luma:.4f} | {r.feature_contrast_retained_pct:.1f}% |"
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Export side-by-side lossless comparison figure for primary seed
    saved_fig = _render_comparison_plate(
        seed_panels[primary_seed],
        sigma=sigma,
        seed=primary_seed,
        svd_k=svd_k,
        fft_r=fft_r,
        output_path=figure_path,
    )

    if web_study_dir is not None:
        _export_web_study_bundle(
            seed_panels,
            all_rows,
            sigma=sigma,
            primary_seed=primary_seed,
            extra_seeds=extra_seeds,
            svd_k=svd_k,
            fft_r=fft_r,
            web_study_dir=web_study_dir,
        )

    return all_rows, csv_path, md_path, saved_fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run controlled SVD vs. 2D-FFT denoising and detail-preservation comparison."
    )
    parser.add_argument("--sigma", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--svd-k", type=int, default=28)
    parser.add_argument("--fft-r", type=float, default=0.35)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=Path("docs/assets/detail-tradeoff-comparison.png"),
    )
    args = parser.parse_args()

    rows, csv_path, md_path, fig_path = run_detail_tradeoff_study(
        sigma=args.sigma,
        primary_seed=args.seed,
        output_dir=args.output_dir,
        figure_path=args.figure_path,
        svd_k=args.svd_k,
        fft_r=args.fft_r,
    )
    print(f"Detail trade-off study completed ({len(rows)} rows across 3 seeds).")
    print(f"  CSV     : {csv_path}")
    print(f"  Summary : {md_path}")
    print(f"  Figure  : {fig_path}")


if __name__ == "__main__":
    main()
