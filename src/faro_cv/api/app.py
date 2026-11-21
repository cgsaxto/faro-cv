"""FastAPI + WebSocket workbench server exposing LabSession and ExperimentFSM."""

from __future__ import annotations

import asyncio
import threading
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from faro_cv.api.imaging import (
    SAMPLE_PRESETS,
    bytes_to_chw_tensor,
    fft_spectrum_to_png_data_url,
    generate_preset_tensor,
    gradcam_overlay_to_png_data_url,
    residual_to_png_data_url,
    tensor_to_png_data_url,
)
from faro_cv.api.schemas import (
    CreateSnapshotRequest,
    ImageBundle,
    LoadModelRequest,
    LoadPresetRequest,
    ModelOption,
    PresetOption,
    RunAttackRequest,
    RunPurifyRequest,
    SnapshotSummary,
    WorkbenchPayload,
)
from faro_cv.core.exceptions import (
    GuardRejectedError,
    InvalidTransitionError,
    SnapshotNotFoundError,
)
from faro_cv.core.runner import LabSession
from faro_cv.models.registry import MODEL_SPECS


class _ConnectionHub:
    """Thread-safe WebSocket connection manager for live workbench telemetry."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)

    async def broadcast_json(self, payload: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for ws in list(self._connections):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections.discard(ws)


class WorkbenchController:
    """Stateful controller wrapping LabSession with PNG caching and thread safety."""

    def __init__(self, *, prefer_mps: bool = True) -> None:
        self.session = LabSession(seed=42, prefer_mps=prefer_mps)
        self.lock = threading.RLock()
        self.hub = _ConnectionHub()
        # Cached PNG data URLs indexed by tensor sha256 to avoid redundant re-encoding
        self._png_cache: dict[str, str] = {}
        self._fft_cache: dict[str, str] = {}

    def _cached_png(self, sha: str, tensor: Any) -> str:
        cached = self._png_cache.get(sha)
        if cached is None:
            cached = tensor_to_png_data_url(tensor)
            self._png_cache[sha] = cached
        return cached

    def _cached_fft(self, sha: str, tensor: Any) -> str:
        cached = self._fft_cache.get(sha)
        if cached is None:
            cached = fft_spectrum_to_png_data_url(
                tensor,
                fft_device=self.session.device_profile.fft_op.device,
            )
            self._fft_cache[sha] = cached
        return cached

    def _cached_cam_png(self, sha: str, image: Any, cam: Any) -> str:
        key = f"cam:{sha}"
        cached = self._png_cache.get(key)
        if cached is None:
            cached = gradcam_overlay_to_png_data_url(image, cam)
            self._png_cache[key] = cached
        return cached

    def build_payload(self) -> WorkbenchPayload:
        with self.lock:
            ctx = self.session.fsm.context
            clean_png: str | None = None
            adv_png: str | None = None
            residual_png: str | None = None
            purified_png: str | None = None
            clean_cam_png: str | None = None
            adv_cam_png: str | None = None
            purified_cam_png: str | None = None
            fft_before_png: str | None = None
            fft_after_png: str | None = None

            if self.session.clean_tensor is not None and ctx.input_meta is not None:
                clean_png = self._cached_png(ctx.input_meta.sha256, self.session.clean_tensor)
                if self.session.clean_cam is not None:
                    clean_cam_png = self._cached_cam_png(
                        ctx.input_meta.sha256,
                        self.session.clean_tensor,
                        self.session.clean_cam,
                    )

            if self.session.adversarial_tensor is not None and ctx.adversarial_meta is not None:
                adv_sha = ctx.adversarial_meta.sha256
                adv_png = self._cached_png(adv_sha, self.session.adversarial_tensor)
                fft_before_png = self._cached_fft(adv_sha, self.session.adversarial_tensor)
                if self.session.adversarial_cam is not None:
                    adv_cam_png = self._cached_cam_png(
                        adv_sha,
                        self.session.adversarial_tensor,
                        self.session.adversarial_cam,
                    )
                if self.session.clean_tensor is not None and ctx.input_meta is not None:
                    res_key = f"res:{ctx.input_meta.sha256}:{adv_sha}"
                    residual_png = self._png_cache.get(res_key)
                    if residual_png is None:
                        residual_png = residual_to_png_data_url(
                            self.session.clean_tensor,
                            self.session.adversarial_tensor,
                        )
                        self._png_cache[res_key] = residual_png

            if self.session.purified_tensor is not None and ctx.purified_meta is not None:
                pur_sha = ctx.purified_meta.sha256
                purified_png = self._cached_png(pur_sha, self.session.purified_tensor)
                fft_after_png = self._cached_fft(pur_sha, self.session.purified_tensor)
                if self.session.purified_cam is not None:
                    purified_cam_png = self._cached_cam_png(
                        pur_sha,
                        self.session.purified_tensor,
                        self.session.purified_cam,
                    )

            return WorkbenchPayload(
                fsm=self.session.fsm.to_wire_state(history_limit=24),
                device_profile=self.session.device_profile,
                images=ImageBundle(
                    clean_png=clean_png,
                    adversarial_png=adv_png,
                    residual_png=residual_png,
                    purified_png=purified_png,
                    clean_cam_png=clean_cam_png,
                    adversarial_cam_png=adv_cam_png,
                    purified_cam_png=purified_cam_png,
                    fft_before_png=fft_before_png,
                    fft_after_png=fft_after_png,
                ),
                available_models=[
                    ModelOption(
                        model_id=spec.model_id,
                        display_name=spec.display_name,
                        architecture=spec.architecture,
                        input_size=spec.input_size,
                    )
                    for spec in MODEL_SPECS.values()
                ],
                available_presets=[
                    PresetOption(
                        preset_id=p.preset_id,
                        label=p.label,
                        description=p.description,
                    )
                    for p in SAMPLE_PRESETS.values()
                ],
                snapshots=[
                    SnapshotSummary(
                        snapshot_id=s.snapshot_id,
                        label=s.label,
                        state=s.state,
                        created_at=s.created_at,
                    )
                    for s in self.session.fsm.snapshots
                ],
            )


def _translate_domain_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (InvalidTransitionError, GuardRejectedError)):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, SnapshotNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


def create_app(*, prefer_mps: bool = True) -> FastAPI:
    """Create and configure the faro-cv FastAPI workbench application."""
    app = FastAPI(
        title="faro-cv Workbench API",
        version="0.1.0",
        description="Adversarial CV attack, SVD/2D-FFT purification, and transactional FSM backend.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    controller = WorkbenchController(prefer_mps=prefer_mps)
    app.state.controller = controller

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": "0.1.0",
            "fsm_state": controller.session.fsm.state.value,
            "preferred_accelerator": controller.session.device_profile.preferred_accelerator,
        }

    @app.get("/api/state", response_model=WorkbenchPayload)
    async def get_state() -> WorkbenchPayload:
        return controller.build_payload()

    @app.post("/api/model", response_model=WorkbenchPayload)
    async def load_model(req: LoadModelRequest) -> WorkbenchPayload:
        try:
            with controller.lock:
                await asyncio.to_thread(
                    controller.session.load_model,
                    req.model_id,
                    pretrained=req.pretrained,
                )
            payload = controller.build_payload()
            await controller.hub.broadcast_json(
                {"type": "state_update", "payload": payload.model_dump(mode="json")}
            )
            return payload
        except Exception as exc:
            raise _translate_domain_error(exc) from exc

    @app.post("/api/input/preset", response_model=WorkbenchPayload)
    async def load_preset(req: LoadPresetRequest) -> WorkbenchPayload:
        try:
            with controller.lock:
                tensor = generate_preset_tensor(req.preset_id, size=224)
                await asyncio.to_thread(controller.session.load_image, tensor)
            payload = controller.build_payload()
            await controller.hub.broadcast_json(
                {"type": "state_update", "payload": payload.model_dump(mode="json")}
            )
            return payload
        except Exception as exc:
            raise _translate_domain_error(exc) from exc

    @app.post("/api/input/upload", response_model=WorkbenchPayload)
    async def upload_image(file: Annotated[UploadFile, File(...)]) -> WorkbenchPayload:
        try:
            raw = await file.read()
            if not raw:
                raise ValueError("Uploaded file is empty.")
            tensor = bytes_to_chw_tensor(raw, size=224)
            with controller.lock:
                await asyncio.to_thread(controller.session.load_image, tensor)
            payload = controller.build_payload()
            await controller.hub.broadcast_json(
                {"type": "state_update", "payload": payload.model_dump(mode="json")}
            )
            return payload
        except Exception as exc:
            raise _translate_domain_error(exc) from exc

    @app.post("/api/attack", response_model=WorkbenchPayload)
    async def run_attack(req: RunAttackRequest) -> WorkbenchPayload:
        loop = asyncio.get_running_loop()

        def _execute_streaming() -> None:
            with controller.lock:
                for step_res in controller.session.iter_attack(req.config):
                    progress_msg = {
                        "type": "attack_progress",
                        "step": step_res.step,
                        "total_steps": step_res.total_steps,
                        "loss": round(step_res.loss, 5),
                    }
                    asyncio.run_coroutine_threadsafe(
                        controller.hub.broadcast_json(progress_msg),
                        loop,
                    )

        try:
            await asyncio.to_thread(_execute_streaming)
            payload = controller.build_payload()
            await controller.hub.broadcast_json(
                {"type": "state_update", "payload": payload.model_dump(mode="json")}
            )
            return payload
        except Exception as exc:
            raise _translate_domain_error(exc) from exc

    @app.post("/api/purify", response_model=WorkbenchPayload)
    async def run_purify(req: RunPurifyRequest) -> WorkbenchPayload:
        def _execute() -> None:
            with controller.lock:
                controller.session.purify(req.config)
                if req.auto_evaluate:
                    controller.session.evaluate()

        try:
            await asyncio.to_thread(_execute)
            payload = controller.build_payload()
            await controller.hub.broadcast_json(
                {"type": "state_update", "payload": payload.model_dump(mode="json")}
            )
            return payload
        except Exception as exc:
            raise _translate_domain_error(exc) from exc

    @app.post("/api/snapshot", response_model=WorkbenchPayload)
    async def create_snapshot(req: CreateSnapshotRequest) -> WorkbenchPayload:
        with controller.lock:
            controller.session.fsm.create_snapshot(req.label)
        payload = controller.build_payload()
        await controller.hub.broadcast_json(
            {"type": "state_update", "payload": payload.model_dump(mode="json")}
        )
        return payload

    @app.post("/api/snapshot/{snapshot_id}/restore", response_model=WorkbenchPayload)
    async def restore_snapshot(snapshot_id: str) -> WorkbenchPayload:
        try:
            with controller.lock:
                controller.session.restore_snapshot(snapshot_id)
            payload = controller.build_payload()
            await controller.hub.broadcast_json(
                {"type": "state_update", "payload": payload.model_dump(mode="json")}
            )
            return payload
        except Exception as exc:
            raise _translate_domain_error(exc) from exc

    @app.post("/api/reset", response_model=WorkbenchPayload)
    async def reset_session() -> WorkbenchPayload:
        with controller.lock:
            controller.session.reset()
            controller._png_cache.clear()
            controller._fft_cache.clear()
        payload = controller.build_payload()
        await controller.hub.broadcast_json(
            {"type": "state_update", "payload": payload.model_dump(mode="json")}
        )
        return payload

    @app.websocket("/ws/workbench")
    async def workbench_ws(ws: WebSocket) -> None:
        await controller.hub.connect(ws)
        try:
            initial = controller.build_payload()
            await ws.send_json({"type": "state_update", "payload": initial.model_dump(mode="json")})
            while True:
                # Keep connection alive; ignore client pings
                await ws.receive_text()
        except WebSocketDisconnect:
            controller.hub.disconnect(ws)

    return app


app = create_app()
