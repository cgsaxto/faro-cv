"""Smoke & metric-integrity tests for the reproducible benchmark suite."""

from __future__ import annotations

from pathlib import Path

from experiments.benchmark import export_results, run_benchmark


def test_benchmark_runner_splits_consistency_conditional_recovery_and_ground_truth(
    tmp_path: Path,
) -> None:
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text(
        "image,class_index\nbeacon_target,10\nharmonic_grid,20\nchromatic_spheres,30\n",
        encoding="utf-8",
    )

    matrix_rows, sweep_rows, device_str = run_benchmark(
        model_id="resnet18",
        pretrained=False,
        seed=42,
        labels_csv=labels_csv,
        fast=True,
    )
    assert len(matrix_rows) > 0
    assert len(sweep_rows) == 6
    assert "model=" in device_str

    for row in matrix_rows:
        assert row.samples == 3
        assert row.labeled_samples == 3
        assert row.clean_accuracy is not None
        assert row.adversarial_accuracy is not None
        assert row.purified_accuracy is not None
        if row.flipped_count == 0:
            assert row.conditional_recovery_rate is None
        else:
            assert row.conditional_recovery_rate is not None

    csv_path, md_path = export_results(
        matrix_rows,
        sweep_rows,
        output_dir=tmp_path,
        model_id="resnet18",
        pretrained=False,
        device_str=device_str,
        seed=42,
    )
    assert csv_path.exists()
    assert md_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert "Consistency (vs. Clean)" in content
    assert "Cond. Recovery (on Flipped)" in content
    assert "Ground-Truth Classification Accuracy" in content
    if any(r.flipped_count == 0 for r in matrix_rows):
        assert "N/A (0/0)" in content
