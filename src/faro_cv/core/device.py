"""Apple Silicon MPS-aware device capability probing and operator routing.

PyTorch's MPS backend accelerates convolutional and transformer forward/backward passes
substantially on Apple Silicon, but complex-valued 2D-FFT and certain SVD routines can
fall back or raise backend errors depending on the installed PyTorch build. Probing each
operator once at startup lets faro-cv route every operator safely while surfacing honest
hardware telemetry in the workbench status bar.
"""

from __future__ import annotations

import time
from typing import Any

from faro_cv.core.types import ComputeDevice, OperatorDeviceProfile, OperatorDeviceStatus


def _probe_callable(
    fn: Any,
    *,
    preferred_device: ComputeDevice,
) -> OperatorDeviceStatus:
    """Execute a tiny micro-probe on the preferred device and fall back to CPU on failure."""
    start = time.perf_counter()
    try:
        fn(preferred_device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return OperatorDeviceStatus(
            device=preferred_device,
            supported_on_accelerator=(preferred_device != "cpu"),
            probe_latency_ms=round(elapsed_ms, 3),
            fallback_reason=None,
        )
    except Exception as exc:
        if preferred_device == "cpu":
            raise
        fallback_start = time.perf_counter()
        fn("cpu")
        fallback_ms = (time.perf_counter() - fallback_start) * 1000.0
        return OperatorDeviceStatus(
            device="cpu",
            supported_on_accelerator=False,
            probe_latency_ms=round(fallback_ms, 3),
            fallback_reason=f"{preferred_device} unsupported ({type(exc).__name__}: {exc})",
        )


def probe_device_capabilities(*, prefer_mps: bool = True) -> OperatorDeviceProfile:
    """Probe available compute backends and per-operator MPS/CUDA compatibility."""
    try:
        import torch
    except ImportError:
        cpu_status = OperatorDeviceStatus(
            device="cpu",
            supported_on_accelerator=False,
            probe_latency_ms=0.0,
            fallback_reason="PyTorch is not installed in this environment.",
        )
        return OperatorDeviceProfile(
            torch_available=False,
            torch_version=None,
            preferred_accelerator="cpu",
            model_op=cpu_status,
            svd_op=cpu_status,
            fft_op=cpu_status,
        )

    preferred: ComputeDevice = "cpu"
    if torch.cuda.is_available():
        preferred = "cuda"
    elif prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        preferred = "mps"

    def _probe_model(dev: ComputeDevice) -> None:
        x = torch.randn(1, 3, 16, 16, device=dev)
        conv = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1).to(dev)
        y = conv(x)
        if dev == "mps":
            torch.mps.synchronize()
        elif dev == "cuda":
            torch.cuda.synchronize()
        _ = float(y.mean().item())

    def _probe_svd(dev: ComputeDevice) -> None:
        mat = torch.randn(3, 16, 16, device=dev)
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        if dev == "mps":
            torch.mps.synchronize()
        elif dev == "cuda":
            torch.cuda.synchronize()
        _ = float((u.sum() + s.sum() + vh.sum()).item())

    def _probe_fft(dev: ComputeDevice) -> None:
        img = torch.randn(3, 16, 16, device=dev)
        spec = torch.fft.fft2(img)
        recon = torch.fft.ifft2(spec).real
        if dev == "mps":
            torch.mps.synchronize()
        elif dev == "cuda":
            torch.cuda.synchronize()
        _ = float(recon.mean().item())

    return OperatorDeviceProfile(
        torch_available=True,
        torch_version=str(torch.__version__),
        preferred_accelerator=preferred,
        model_op=_probe_callable(_probe_model, preferred_device=preferred),
        svd_op=_probe_callable(_probe_svd, preferred_device=preferred),
        fft_op=_probe_callable(_probe_fft, preferred_device=preferred),
    )
