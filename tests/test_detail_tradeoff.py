"""Tests for the controlled SVD vs. 2D-FFT denoising & detail-loss study."""

from __future__ import annotations

from pathlib import Path

from experiments.detail_tradeoff_study import run_detail_tradeoff_study


def test_detail_tradeoff_study_reproducible_and_exports_artifacts(tmp_path: Path) -> None:
    fig_path = tmp_path / "comparison.png"
    rows_a, csv_path, md_path, saved_fig = run_detail_tradeoff_study(
        sigma=0.08,
        primary_seed=42,
        extra_seeds=(123,),
        svd_k=28,
        fft_r=0.35,
        output_dir=tmp_path,
        figure_path=fig_path,
        web_study_dir=tmp_path / "web_study",
    )
    rows_b, _, _, _ = run_detail_tradeoff_study(
        sigma=0.08,
        primary_seed=42,
        extra_seeds=(123,),
        svd_k=28,
        fft_r=0.35,
        output_dir=tmp_path,
        figure_path=fig_path,
        web_study_dir=None,
    )

    assert len(rows_a) == 12  # 2 seeds x 2 images x 3 methods
    assert csv_path.exists()
    assert md_path.exists()
    assert saved_fig.exists()
    assert (tmp_path / "web_study" / "study-data.json").exists()
    assert (tmp_path / "web_study" / "harmonic_grid_clean.png").exists()
    s42_fft = tmp_path / "web_study" / "harmonic_grid_s42_fft.png"
    s123_fft = tmp_path / "web_study" / "harmonic_grid_s123_fft.png"
    assert s42_fft.exists()
    assert s123_fft.exists()
    assert s42_fft.read_bytes() != s123_fft.read_bytes()

    # Exact deterministic reproducibility of quality/detail metrics across runs
    for a, b in zip(rows_a, rows_b, strict=True):
        assert a.seed == b.seed
        assert a.image == b.image
        assert a.method == b.method
        assert a.psnr_db == b.psnr_db
        assert a.ssim == b.ssim
        assert a.ref_roi_residual_std == b.ref_roi_residual_std
        assert a.feature_contrast_luma == b.feature_contrast_luma
        assert a.feature_contrast_retained_pct == b.feature_contrast_retained_pct

    # Both SVD and 2D-FFT reduce reference-ROI residual std relative to unprocessed noisy input
    by_key = {(r.seed, r.image, r.method): r for r in rows_a}
    for seed in (42, 123):
        for img_id in ("harmonic_grid", "beacon_target"):
            noisy_std = by_key[(seed, img_id, "Noisy (Unprocessed)")].ref_roi_residual_std
            svd_std = by_key[(seed, img_id, "SVD-only (k=28)")].ref_roi_residual_std
            fft_std = by_key[(seed, img_id, "2D-FFT Gaussian (r=0.35)")].ref_roi_residual_std
            assert svd_std < noisy_std
            assert fft_std < noisy_std
