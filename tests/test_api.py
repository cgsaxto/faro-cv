"""End-to-end REST API tests for the faro-cv FastAPI workbench bridge."""

from __future__ import annotations

from fastapi.testclient import TestClient

from faro_cv.api.app import create_app


def test_api_workbench_full_lifecycle() -> None:
    app = create_app(prefer_mps=False)
    client = TestClient(app)

    # Health check
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["fsm_state"] == "IDLE"

    # Load model (offline weights for deterministic fast CI)
    res = client.post("/api/model", json={"model_id": "resnet18", "pretrained": False})
    assert res.status_code == 200
    data = res.json()
    assert data["fsm"]["state"] == "MODEL_READY"

    # Load preset calibration target
    res = client.post("/api/input/preset", json={"preset_id": "beacon_target"})
    assert res.status_code == 200
    data = res.json()
    assert data["fsm"]["state"] == "INPUT_LOADED"
    assert data["images"]["clean_png"].startswith("data:image/png;base64,")

    # Run PGD attack
    res = client.post(
        "/api/attack",
        json={
            "config": {
                "kind": "pgd",
                "norm": "linf",
                "epsilon": 8.0 / 255.0,
                "step_size": 2.0 / 255.0,
                "steps": 3,
            }
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["fsm"]["state"] == "ATTACKED"
    assert data["images"]["adversarial_png"].startswith("data:image/png;base64,")
    assert data["images"]["residual_png"].startswith("data:image/png;base64,")
    assert data["images"]["fft_before_png"].startswith("data:image/png;base64,")

    # Purify via SVD + auto-evaluate
    res = client.post(
        "/api/purify",
        json={
            "config": {
                "kind": "svd",
                "svd": {"rank_k": 24, "channel_mode": "per_channel"},
            },
            "auto_evaluate": True,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["fsm"]["state"] == "EVALUATED"
    assert data["images"]["purified_png"].startswith("data:image/png;base64,")
    assert data["images"]["fft_after_png"].startswith("data:image/png;base64,")
    assert data["fsm"]["context"]["spectral_summary"]["effective_rank"] == 24
    assert data["fsm"]["context"]["evaluation"] is not None

    # Create snapshot and reset
    snap_res = client.post("/api/snapshot", json={"label": "svd_k24"})
    assert snap_res.status_code == 200
    snap_id = snap_res.json()["snapshots"][0]["snapshot_id"]

    reset_res = client.post("/api/reset")
    assert reset_res.status_code == 200
    assert reset_res.json()["fsm"]["state"] == "IDLE"

    # Restore snapshot brings FSM straight back to EVALUATED
    restore_res = client.post(f"/api/snapshot/{snap_id}/restore")
    assert restore_res.status_code == 200
    assert restore_res.json()["fsm"]["state"] == "EVALUATED"
