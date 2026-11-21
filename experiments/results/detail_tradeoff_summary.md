# SVD vs. 2D-FFT Denoising & Detail-Loss Study (Synthetic Calibration Images)

- **Generated**: 2026-11-20 14:41 UTC
- **Noise model**: Identical fixed-seed Gaussian noise (`sigma=0.08`, primary `seed=42`, verification seeds `[123, 999]`)
- **Methods compared**: Unprocessed Noisy vs. `SVD-only (k=28, RGB)` vs. `2D-FFT Gaussian (r=0.35, exact DFT grid via fftshift(fftfreq)/0.5, preserve_phase=True)`
- **Reference ROI residual**: Standard deviation of `(output - clean)` BT.601 luminance inside `[16:52, 16:52]` (captures residual noise plus any local smoothing distortion on smooth/textured regions).
- **Pipeline latency (CPU, incl. diagnostics)**: Measured on the same CPU device after 5 warmup passes (mean of 20 runs for primary seed), reflecting full workbench operator cost including spectral/SVD diagnostic telemetry.
- **Scope**: Synthetic calibration images (`harmonic_grid`, `beacon_target`, 224×224).

## 1. Primary Controlled Comparison (`seed=42`)

| Calibration Image | Pre-Selected Detail Feature | Method | PSNR (dB) | SSIM | Ref ROI Residual σ | Feature ΔL (Y) | Contrast Retained (%) | Pipeline Latency (ms, CPU incl. diagnostics) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `harmonic_grid` | Horizontal reticle line (y=110..113) | Noisy (Unprocessed) | 21.92 | 0.3630 | 0.0528 | 0.2540 | 100.0% | 0.00 |
| `harmonic_grid` | Horizontal reticle line (y=110..113) | SVD-only (k=28) | 26.51 | 0.5940 | 0.0303 | 0.2560 | 100.9% | 8.38 |
| `harmonic_grid` | Horizontal reticle line (y=110..113) | 2D-FFT Gaussian (r=0.35) | 29.37 | 0.8329 | 0.0156 | 0.2148 | 84.6% | 5.34 |
| `beacon_target` | Lighthouse window slit (x=109..115, y=94..105) | Noisy (Unprocessed) | 22.08 | 0.2512 | 0.0521 | 0.6309 | 96.5% | 0.00 |
| `beacon_target` | Lighthouse window slit (x=109..115, y=94..105) | SVD-only (k=28) | 26.25 | 0.4647 | 0.0325 | 0.6309 | 96.5% | 8.38 |
| `beacon_target` | Lighthouse window slit (x=109..115, y=94..105) | 2D-FFT Gaussian (r=0.35) | 28.16 | 0.7266 | 0.0156 | 0.5610 | 85.8% | 5.33 |

## 2. Multi-Seed Robustness Check (Additional Noise Seeds)

| Seed | Calibration Image | Method | PSNR (dB) | SSIM | Ref ROI Residual σ | Feature ΔL (Y) | Contrast Retained (%) |
|---:|---|---|---:|---:|---:|---:|---:|
| `123` | `harmonic_grid` | Noisy (Unprocessed) | 21.95 | 0.3642 | 0.0521 | 0.2631 | 103.6% |
| `123` | `harmonic_grid` | SVD-only (k=28) | 26.56 | 0.5962 | 0.0291 | 0.2596 | 102.3% |
| `123` | `harmonic_grid` | 2D-FFT Gaussian (r=0.35) | 29.37 | 0.8343 | 0.0158 | 0.2207 | 86.9% |
| `123` | `beacon_target` | Noisy (Unprocessed) | 22.10 | 0.2511 | 0.0514 | 0.6421 | 98.2% |
| `123` | `beacon_target` | SVD-only (k=28) | 26.27 | 0.4664 | 0.0321 | 0.6392 | 97.8% |
| `123` | `beacon_target` | 2D-FFT Gaussian (r=0.35) | 28.15 | 0.7273 | 0.0156 | 0.5679 | 86.9% |
| `999` | `harmonic_grid` | Noisy (Unprocessed) | 21.96 | 0.3652 | 0.0526 | 0.2520 | 99.3% |
| `999` | `harmonic_grid` | SVD-only (k=28) | 26.59 | 0.5979 | 0.0293 | 0.2520 | 99.3% |
| `999` | `harmonic_grid` | 2D-FFT Gaussian (r=0.35) | 29.37 | 0.8346 | 0.0159 | 0.2126 | 83.8% |
| `999` | `beacon_target` | Noisy (Unprocessed) | 22.12 | 0.2521 | 0.0520 | 0.6502 | 99.5% |
| `999` | `beacon_target` | SVD-only (k=28) | 26.26 | 0.4653 | 0.0318 | 0.6511 | 99.6% |
| `999` | `beacon_target` | 2D-FFT Gaussian (r=0.35) | 28.18 | 0.7279 | 0.0158 | 0.5744 | 87.9% |
