"""Reproducible ablation & benchmark suite for faro-cv.

Runs a deterministic attack x defense x threat-model evaluation matrix plus an SVD rank
trade-off sweep, exporting both machine-readable CSV and publication-ready Markdown tables.

Metric Definitions (strictly separated to prevent denominator conflation):
    - Attack Flip Rate (ASR)       = flipped_count / samples
    - Prediction Consistency Rate  = consistent_count / samples
      (purified top-1 == clean top-1 across all samples)
    - Conditional Recovery Rate    = conditional_recovered_count / flipped_count
      (purified top-1 == clean top-1 strictly among samples flipped by the attack;
      reported as N/A when flipped_count == 0)
    - Ground-Truth Accuracy        = correct_count / labeled_samples
      (optional, enabled via --labels-csv with columns `image,class_index`, reporting
      Clean / Adversarial / Purified accuracy against human/dataset ground truth)

Usage:
    # Fast deterministic benchmark on built-in calibration targets (offline weights)
    .venv/bin/python experiments/benchmark.py --model resnet18 --offline

    # Pretrained ImageNet benchmark on custom labeled photos
    .venv/bin/python experiments/benchmark.py --model resnet18 --pretrained \\
        --image-dir data/photos --labels-csv data/labels.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image
from torch import Tensor

from faro_cv.api.imaging import SAMPLE_PRESETS, generate_preset_tensor, pil_to_chw_tensor
from faro_cv.core.runner import LabSession
from faro_cv.core.types import AttackConfig, DefenseConfig, FFTParams, SVDParams


@dataclass(frozen=True, slots=True)
class MatrixRow:
    """Single aggregated benchmark cell with explicit numerator/denominator counts."""

    attack: str
    threat_model: str
    defense: str
    samples: int
    flipped_count: int
    consistent_count: int
    conditional_recovered_count: int
    attack_success_rate: float
    prediction_consistency_rate: float
    conditional_recovery_rate: float | None
    labeled_samples: int
    clean_accuracy: float | None
    adversarial_accuracy: float | None
    purified_accuracy: float | None
    mean_psnr_db: float
    mean_ssim: float
    mean_linf: float
    mean_l2: float
    mean_hf_before: float
    mean_hf_after: float
    mean_cam_sim_adv: float
    mean_cam_sim_pur: float
    mean_purify_ms: float


@dataclass(frozen=True, slots=True)
class RankSweepRow:
    """Single point on the SVD truncation rank k trade-off curve."""

    rank_k: int
    samples: int
    retained_energy_pct: float
    mean_psnr_db: float
    mean_ssim: float
    mean_hf_after_pct: float
    mean_cam_sim_pur: float
    mean_purify_ms: float


def _load_ground_truth_csv(labels_csv: Path | None) -> dict[str, int]:
    """Load optional ``image,class_index`` ground-truth mapping (keyed by filename & stem)."""
    if labels_csv is None:
        return {}
    mapping: dict[str, int] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            img_key = (row.get("image") or "").strip()
            cls_raw = (row.get("class_index") or "").strip()
            if not img_key or not cls_raw:
                continue
            cls_idx = int(cls_raw)
            mapping[img_key] = cls_idx
            mapping[Path(img_key).stem] = cls_idx
    return mapping


def _load_evaluation_images(
    image_dir: Path | None,
    *,
    labels_map: dict[str, int],
    size: int = 224,
) -> list[tuple[str, Tensor, int | None]]:
    if image_dir is None:
        return [
            (preset_id, generate_preset_tensor(preset_id, size=size), labels_map.get(preset_id))
            for preset_id in SAMPLE_PRESETS
        ]

    valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
    paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts)
    if not paths:
        raise ValueError(f"No PNG/JPG/WEBP images found in '{image_dir}'.")

    loaded: list[tuple[str, Tensor, int | None]] = []
    for path in paths:
        gt = labels_map.get(path.name, labels_map.get(path.stem))
        with Image.open(path) as img:
            loaded.append((path.stem, pil_to_chw_tensor(img, size=size), gt))
    return loaded


def _default_defense_suite() -> list[tuple[str, DefenseConfig]]:
    return [
        ("SVD (k=28, RGB)", DefenseConfig(kind="svd", svd=SVDParams(rank_k=28))),
        (
            "SVD (k=28, Luma Y)",
            DefenseConfig(kind="svd", svd=SVDParams(rank_k=28, channel_mode="luminance")),
        ),
        (
            "2D-FFT Gaussian (r=0.35)",
            DefenseConfig(kind="fft", fft=FFTParams(filter_type="gaussian", cutoff_radius=0.35)),
        ),
        (
            "2D-FFT Butterworth (r=0.35)",
            DefenseConfig(
                kind="fft",
                fft=FFTParams(filter_type="butterworth", cutoff_radius=0.35, butterworth_order=2),
            ),
        ),
        (
            "2D-FFT Ideal (r=0.35)",
            DefenseConfig(kind="fft", fft=FFTParams(filter_type="ideal", cutoff_radius=0.35)),
        ),
        (
            "Pipeline FFT->SVD",
            DefenseConfig(
                kind="pipeline",
                svd=SVDParams(rank_k=28),
                fft=FFTParams(filter_type="gaussian", cutoff_radius=0.35),
                pipeline_order=("fft", "svd"),
            ),
        ),
    ]


def _default_attack_suite(
    adaptive_svd: DefenseConfig,
    adaptive_fft_ideal: DefenseConfig,
    *,
    fast: bool = False,
) -> list[tuple[str, str, AttackConfig]]:
    pgd_steps = 4 if fast else 10
    patch_steps = 6 if fast else 16
    eot_n = 2 if fast else 4
    return [
        (
            "FGSM (Linf 12/255)",
            "Oblivious",
            AttackConfig(kind="fgsm", norm="linf", epsilon=12.0 / 255.0, steps=1),
        ),
        (
            f"PGD-{pgd_steps} (Linf 12/255)",
            "Oblivious",
            AttackConfig(
                kind="pgd",
                norm="linf",
                epsilon=12.0 / 255.0,
                step_size=3.0 / 255.0,
                steps=pgd_steps,
            ),
        ),
        (
            f"PGD-{pgd_steps} (Linf 12/255)",
            "Adaptive Exact (SVD)",
            AttackConfig(
                kind="pgd",
                norm="linf",
                epsilon=12.0 / 255.0,
                step_size=3.0 / 255.0,
                steps=pgd_steps,
                adaptive_mode="exact",
                adaptive_defense=adaptive_svd,
            ),
        ),
        (
            f"PGD-{pgd_steps} (Linf 12/255)",
            "BPDA (FFT Ideal)",
            AttackConfig(
                kind="pgd",
                norm="linf",
                epsilon=12.0 / 255.0,
                step_size=3.0 / 255.0,
                steps=pgd_steps,
                adaptive_mode="bpda",
                adaptive_defense=adaptive_fft_ideal,
            ),
        ),
        (
            f"Patch+EoT (48px, {eot_n}x)",
            "Oblivious",
            AttackConfig(
                kind="patch_eot",
                steps=patch_steps,
                patch_size=48,
                patch_x=88,
                patch_y=76,
                eot_samples=eot_n,
            ),
        ),
        (
            f"Patch+EoT (48px, {eot_n}x)",
            "Adaptive Exact (SVD)",
            AttackConfig(
                kind="patch_eot",
                steps=patch_steps,
                patch_size=48,
                patch_x=88,
                patch_y=76,
                eot_samples=eot_n,
                adaptive_mode="exact",
                adaptive_defense=adaptive_svd,
            ),
        ),
    ]


def run_benchmark(
    *,
    model_id: str = "resnet18",
    pretrained: bool = False,
    seed: int = 42,
    image_dir: Path | None = None,
    labels_csv: Path | None = None,
    fast: bool = False,
) -> tuple[list[MatrixRow], list[RankSweepRow], str]:
    """Execute the full attack x defense matrix and SVD rank sweep."""
    session = LabSession(seed=seed, prefer_mps=True)
    loaded = session.load_model(model_id, pretrained=pretrained)
    labels_map = _load_ground_truth_csv(labels_csv)
    images = _load_evaluation_images(
        image_dir,
        labels_map=labels_map,
        size=loaded.spec.input_size,
    )

    defenses = _default_defense_suite()
    adaptive_svd = defenses[0][1]
    adaptive_fft_ideal = defenses[4][1]
    attacks = _default_attack_suite(adaptive_svd, adaptive_fft_ideal, fast=fast)

    # 1. Main Attack x Defense Ablation Matrix
    matrix_rows: list[MatrixRow] = []
    for attack_name, threat_label, attack_cfg in attacks:
        if threat_label.startswith("Adaptive"):
            target_defenses = [defenses[0], defenses[5]]
        elif threat_label.startswith("BPDA"):
            target_defenses = [defenses[4], defenses[2]]
        else:
            target_defenses = defenses

        per_defense_acc: dict[str, list[dict[str, float | int | bool | None]]] = {
            name: [] for name, _ in target_defenses
        }

        for _img_name, tensor, gt_class in images:
            session.load_image(tensor)
            session.run_attack(attack_cfg)

            for def_name, def_cfg in target_defenses:
                pur = session.purify(def_cfg)
                ev = session.evaluate(ground_truth_class=gt_class)
                per_defense_acc[def_name].append(
                    {
                        "flipped": ev.attack_succeeded,
                        "consistent": ev.prediction_consistent,
                        "cond_rec": ev.conditional_recovered,
                        "clean_correct": ev.clean_correct,
                        "adv_correct": ev.adversarial_correct,
                        "pur_correct": ev.purified_correct,
                        "psnr": ev.purified_psnr_db,
                        "ssim": ev.purified_ssim,
                        "linf": ev.attack_linf_norm,
                        "l2": ev.attack_l2_norm,
                        "hf_b": pur.spectral_summary.high_freq_energy_before * 100.0,
                        "hf_a": pur.spectral_summary.high_freq_energy_after * 100.0,
                        "cam_adv": ev.attention_similarity_adv or 0.0,
                        "cam_pur": ev.attention_similarity_purified or 0.0,
                        "ms": ev.purification_latency_ms,
                    }
                )

        for def_name, _ in target_defenses:
            records = per_defense_acc[def_name]
            n_int = len(records)
            n = float(n_int)

            flipped_count = sum(1 for r in records if r["flipped"] is True)
            consistent_count = sum(1 for r in records if r["consistent"] is True)
            cond_rec_count = sum(1 for r in records if r["cond_rec"] is True)

            cond_rec_rate: float | None = (
                round(cond_rec_count / float(flipped_count) * 100.0, 1)
                if flipped_count > 0
                else None
            )

            labeled = [r for r in records if r["clean_correct"] is not None]
            labeled_n = len(labeled)
            clean_acc = (
                round(sum(1 for r in labeled if r["clean_correct"] is True) / labeled_n * 100.0, 1)
                if labeled_n > 0
                else None
            )
            adv_acc = (
                round(sum(1 for r in labeled if r["adv_correct"] is True) / labeled_n * 100.0, 1)
                if labeled_n > 0
                else None
            )
            pur_acc = (
                round(sum(1 for r in labeled if r["pur_correct"] is True) / labeled_n * 100.0, 1)
                if labeled_n > 0
                else None
            )

            matrix_rows.append(
                MatrixRow(
                    attack=attack_name,
                    threat_model=threat_label,
                    defense=def_name,
                    samples=n_int,
                    flipped_count=flipped_count,
                    consistent_count=consistent_count,
                    conditional_recovered_count=cond_rec_count,
                    attack_success_rate=round(flipped_count / n * 100.0, 1),
                    prediction_consistency_rate=round(consistent_count / n * 100.0, 1),
                    conditional_recovery_rate=cond_rec_rate,
                    labeled_samples=labeled_n,
                    clean_accuracy=clean_acc,
                    adversarial_accuracy=adv_acc,
                    purified_accuracy=pur_acc,
                    mean_psnr_db=round(sum(float(r["psnr"] or 0.0) for r in records) / n, 2),
                    mean_ssim=round(sum(float(r["ssim"] or 0.0) for r in records) / n, 4),
                    mean_linf=round(sum(float(r["linf"] or 0.0) for r in records) / n, 4),
                    mean_l2=round(sum(float(r["l2"] or 0.0) for r in records) / n, 2),
                    mean_hf_before=round(sum(float(r["hf_b"] or 0.0) for r in records) / n, 2),
                    mean_hf_after=round(sum(float(r["hf_a"] or 0.0) for r in records) / n, 2),
                    mean_cam_sim_adv=round(sum(float(r["cam_adv"] or 0.0) for r in records) / n, 3),
                    mean_cam_sim_pur=round(sum(float(r["cam_pur"] or 0.0) for r in records) / n, 3),
                    mean_purify_ms=round(sum(float(r["ms"] or 0.0) for r in records) / n, 2),
                )
            )

    # 2. SVD Truncation Rank Sweep under PGD (Linf 12/255)
    ranks = [8, 16, 28, 48, 80, 128]
    pgd_sweep_cfg = AttackConfig(
        kind="pgd",
        norm="linf",
        epsilon=12.0 / 255.0,
        step_size=3.0 / 255.0,
        steps=4 if fast else 10,
    )
    sweep_acc: dict[int, list[dict[str, float]]] = {k: [] for k in ranks}
    for _img_name, tensor, gt_class in images:
        session.load_image(tensor)
        session.run_attack(pgd_sweep_cfg)
        for k in ranks:
            pur = session.purify(DefenseConfig(kind="svd", svd=SVDParams(rank_k=k)))
            ev = session.evaluate(ground_truth_class=gt_class)
            sweep_acc[k].append(
                {
                    "energy": pur.spectral_summary.retained_energy_ratio * 100.0,
                    "psnr": ev.purified_psnr_db,
                    "ssim": ev.purified_ssim,
                    "hf_a": pur.spectral_summary.high_freq_energy_after * 100.0,
                    "cam_pur": ev.attention_similarity_purified or 0.0,
                    "ms": ev.purification_latency_ms,
                }
            )

    sweep_rows: list[RankSweepRow] = []
    for k in ranks:
        recs = sweep_acc[k]
        n = float(len(recs))
        sweep_rows.append(
            RankSweepRow(
                rank_k=k,
                samples=len(recs),
                retained_energy_pct=round(sum(r["energy"] for r in recs) / n, 2),
                mean_psnr_db=round(sum(r["psnr"] for r in recs) / n, 2),
                mean_ssim=round(sum(r["ssim"] for r in recs) / n, 4),
                mean_hf_after_pct=round(sum(r["hf_a"] for r in recs) / n, 2),
                mean_cam_sim_pur=round(sum(r["cam_pur"] for r in recs) / n, 3),
                mean_purify_ms=round(sum(r["ms"] for r in recs) / n, 2),
            )
        )

    device_str = (
        f"model={session.device_profile.model_op.device}, "
        f"svd={session.device_profile.svd_op.device}, "
        f"fft={session.device_profile.fft_op.device}"
    )
    return matrix_rows, sweep_rows, device_str


def _format_cond_recovery(rate: float | None, rec_count: int, flipped_count: int) -> str:
    if rate is None or flipped_count == 0:
        return "N/A (0/0)"
    return f"{rate:.1f}% ({rec_count}/{flipped_count})"


def export_results(
    matrix_rows: list[MatrixRow],
    sweep_rows: list[RankSweepRow],
    *,
    output_dir: Path,
    model_id: str,
    pretrained: bool,
    device_str: str,
    seed: int,
) -> tuple[Path, Path]:
    """Write CSV and Markdown summary artifacts into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_results.csv"
    md_path = output_dir / "benchmark_summary.md"

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = list(asdict(matrix_rows[0]).keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in matrix_rows:
            raw = asdict(row)
            for key, val in raw.items():
                if val is None:
                    raw[key] = "N/A"
            writer.writerow(raw)

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    weights_tag = "ImageNet Pretrained" if pretrained else "Offline Deterministic Init (seed=42)"
    n_samples = matrix_rows[0].samples if matrix_rows else 0
    has_ground_truth = any(r.labeled_samples > 0 for r in matrix_rows)

    lines: list[str] = [
        "# faro-cv Benchmark & Ablation Summary",
        "",
        f"- **Generated**: {timestamp}",
        f"- **Model**: `{model_id}` ({weights_tag})",
        f"- **Samples**: `n={n_samples}` ({'labeled' if has_ground_truth else 'unlabeled synthetic calibration targets'})",
        f"- **Devices**: `{device_str}`",
        f"- **Seed**: `{seed}`",
        "",
        "## 1. Attack x Threat Model x Purification Defense Matrix",
        "",
        "> **Metric definitions**: `Flip Rate (ASR)` = flipped / total; "
        "`Consistency` = (purified == clean) / total; "
        "`Cond. Recovery` = (flipped AND purified == clean) / flipped (`N/A` when 0 samples flipped).",
        "",
        "| Attack | Threat Model | Defense | Flip Rate (ASR) | Consistency (vs. Clean) | Cond. Recovery (on Flipped) | PSNR (dB) | SSIM | L_inf / L2 | HF (%) Before->After | CAM Sim (Adv->Pur) | Latency (ms) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in matrix_rows:
        cond_str = _format_cond_recovery(
            r.conditional_recovery_rate,
            r.conditional_recovered_count,
            r.flipped_count,
        )
        lines.append(
            f"| {r.attack} | {r.threat_model} | {r.defense} | "
            f"{r.attack_success_rate:.1f}% ({r.flipped_count}/{r.samples}) | "
            f"{r.prediction_consistency_rate:.1f}% ({r.consistent_count}/{r.samples}) | "
            f"{cond_str} | "
            f"{r.mean_psnr_db:.2f} | {r.mean_ssim:.4f} | "
            f"{r.mean_linf:.4f} / {r.mean_l2:.2f} | "
            f"{r.mean_hf_before:.2f} -> {r.mean_hf_after:.2f} | "
            f"{r.mean_cam_sim_adv:.3f} -> {r.mean_cam_sim_pur:.3f} | "
            f"{r.mean_purify_ms:.2f} |"
        )

    if has_ground_truth:
        lines.extend(
            [
                "",
                "## 2. Ground-Truth Classification Accuracy (Human-Labeled Samples)",
                "",
                "| Attack | Threat Model | Defense | Labeled n | Clean Acc (%) | Adv Acc (%) | Purified Acc (%) |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for r in matrix_rows:
            if r.labeled_samples == 0:
                continue
            c_acc = f"{r.clean_accuracy:.1f}" if r.clean_accuracy is not None else "N/A"
            a_acc = f"{r.adversarial_accuracy:.1f}" if r.adversarial_accuracy is not None else "N/A"
            p_acc = f"{r.purified_accuracy:.1f}" if r.purified_accuracy is not None else "N/A"
            lines.append(
                f"| {r.attack} | {r.threat_model} | {r.defense} | {r.labeled_samples} | "
                f"{c_acc} | {a_acc} | {p_acc} |"
            )

    sweep_heading = "3" if has_ground_truth else "2"
    lines.extend(
        [
            "",
            f"## {sweep_heading}. SVD Truncation Rank (k) Trade-Off Sweep (PGD Linf 12/255, n={n_samples})",
            "",
            "| Rank k | Retained Energy (%) | Mean PSNR (dB) | Mean SSIM | Post-Purify HF Power (%) | CAM Sim (Pur) | Latency (ms) |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for s in sweep_rows:
        lines.append(
            f"| {s.rank_k} | {s.retained_energy_pct:.2f} | {s.mean_psnr_db:.2f} | "
            f"{s.mean_ssim:.4f} | {s.mean_hf_after_pct:.2f} | "
            f"{s.mean_cam_sim_pur:.3f} | {s.mean_purify_ms:.2f} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run faro-cv reproducible ablation benchmarks.")
    parser.add_argument(
        "--model",
        default="resnet18",
        choices=["resnet18", "mobilenet_v3_small", "vit_b_16"],
    )
    weight_group = parser.add_mutually_exclusive_group()
    weight_group.add_argument(
        "--pretrained", action="store_true", help="Load ImageNet pretrained weights."
    )
    weight_group.add_argument(
        "--offline", action="store_true", help="Use deterministic offline weights (default)."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Optional folder of PNG/JPG evaluation images.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Optional CSV (`image,class_index`) for ground-truth Clean/Adv/Purified accuracy.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument(
        "--fast", action="store_true", help="Use shorter iteration counts for rapid checks."
    )
    args = parser.parse_args()

    matrix_rows, sweep_rows, device_str = run_benchmark(
        model_id=args.model,
        pretrained=args.pretrained,
        seed=args.seed,
        image_dir=args.image_dir,
        labels_csv=args.labels_csv,
        fast=args.fast,
    )
    csv_path, md_path = export_results(
        matrix_rows,
        sweep_rows,
        output_dir=args.output_dir,
        model_id=args.model,
        pretrained=args.pretrained,
        device_str=device_str,
        seed=args.seed,
    )
    print(
        f"Benchmark completed ({len(matrix_rows)} matrix cells, {len(sweep_rows)} rank sweep points)."
    )
    print(f"  CSV     : {csv_path}")
    print(f"  Summary : {md_path}")


if __name__ == "__main__":
    main()
