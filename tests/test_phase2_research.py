"""Phase 2 tests: Adversarial Patch + EoT, Grad-CAM explainability, and Adaptive Attacks."""

from __future__ import annotations

import pytest
import torch
from fastapi.testclient import TestClient

from faro_cv.analysis.explainability import compute_attention_similarity, compute_gradcam
from faro_cv.api.app import create_app
from faro_cv.attacks.adaptive import AdaptiveDefendedModel
from faro_cv.attacks.patch import AdversarialPatchEoTAttack
from faro_cv.attacks.perturbation import PGDAttack, build_attack
from faro_cv.core.runner import LabSession
from faro_cv.core.types import AdaptiveMode, AttackConfig, DefenseConfig, FFTParams, SVDParams
from faro_cv.defenses.pipeline import build_defense
from faro_cv.models.registry import load_vision_model


def _test_image(size: int = 64) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, size).unsqueeze(1).expand(size, size)
    x = torch.linspace(0.0, 1.0, size).unsqueeze(0).expand(size, size)
    return torch.stack(
        [
            0.5 + 0.3 * torch.sin(4.0 * x),
            0.4 + 0.4 * y,
            0.3 + 0.3 * torch.cos(3.0 * (x + y)),
        ],
        dim=0,
    ).clamp(0.0, 1.0)


def test_adversarial_patch_eot_spatial_locality_and_determinism() -> None:
    loaded = load_vision_model("resnet18", pretrained=False, device="cpu")
    img = _test_image(64)
    cfg = AttackConfig(
        kind="patch_eot",
        steps=4,
        patch_size=20,
        patch_x=10,
        patch_y=12,
        eot_samples=3,
    )
    attack_a = build_attack(cfg, seed=17)
    attack_b = build_attack(cfg, seed=17)
    assert isinstance(attack_a, AdversarialPatchEoTAttack)

    res_a = attack_a.run(loaded.model, img, label_index=0)
    res_b = attack_b.run(loaded.model, img, label_index=0)

    assert torch.allclose(res_a.adversarial_tensor, res_b.adversarial_tensor)

    # Perturbation outside the 20x20 patch region at (10, 12) must be strictly zero
    delta = (res_a.adversarial_tensor - img).abs()
    mask = torch.ones_like(delta, dtype=torch.bool)
    mask[:, 12:32, 10:30] = False
    assert float(delta[mask].max().item()) == 0.0
    assert float(delta[:, 12:32, 10:30].sum().item()) > 0.0


@pytest.mark.parametrize("mode", ["exact", "bpda"])
def test_whitebox_adaptive_attack_penetrates_svd_and_fft(mode: AdaptiveMode) -> None:
    loaded = load_vision_model("resnet18", pretrained=False, device="cpu")
    img = _test_image(48)

    defense_cfg = (
        DefenseConfig(kind="svd", svd=SVDParams(rank_k=12))
        if mode == "exact"
        else DefenseConfig(
            kind="fft",
            fft=FFTParams(filter_type="ideal", cutoff_radius=0.35),
        )
    )
    defense = build_defense(defense_cfg)
    wrapped = AdaptiveDefendedModel(
        loaded.model,
        defense,
        mode=mode,
    )

    pgd_cfg = AttackConfig(
        kind="pgd",
        norm="linf",
        epsilon=12.0 / 255.0,
        step_size=4.0 / 255.0,
        steps=4,
        adaptive_mode=mode,
        adaptive_defense=defense_cfg,
    )
    steps = list(
        PGDAttack(pgd_cfg, seed=5, random_start=False).iterate(wrapped, img, label_index=0)
    )
    assert len(steps) == 4
    assert steps[-1].loss > steps[0].loss


def test_gradcam_attention_map_and_session_adaptive_patch_pipeline() -> None:
    session = LabSession(seed=42, prefer_mps=False)
    loaded = session.load_model("resnet18", pretrained=False)
    img = _test_image(64)

    cam = compute_gradcam(loaded, img, target_class=0)
    assert cam.shape == (64, 64)
    assert float(cam.min().item()) >= 0.0
    assert float(cam.max().item()) <= 1.0 + 1e-6
    assert compute_attention_similarity(cam, cam) == pytest.approx(1.0, abs=1e-4)

    session.load_image(img)
    assert session.clean_cam is not None

    defense_cfg = DefenseConfig(
        kind="fft",
        fft=FFTParams(filter_type="gaussian", cutoff_radius=0.35),
    )
    patch_cfg = AttackConfig(
        kind="patch_eot",
        steps=3,
        patch_size=20,
        patch_x=16,
        patch_y=16,
        eot_samples=2,
        adaptive_mode="exact",
        adaptive_defense=defense_cfg,
    )
    session.run_attack(patch_cfg)
    assert session.adversarial_cam is not None

    session.purify(defense_cfg)
    assert session.purified_cam is not None

    metrics = session.evaluate()
    assert metrics.attention_similarity_adv is not None
    assert metrics.attention_similarity_purified is not None
    assert 0.0 <= metrics.attention_similarity_adv <= 1.0
    assert 0.0 <= metrics.attention_similarity_purified <= 1.0


def test_api_exposes_gradcam_overlays_and_patch_attack() -> None:
    app = create_app(prefer_mps=False)
    client = TestClient(app)

    client.post("/api/model", json={"model_id": "resnet18", "pretrained": False})
    res_in = client.post("/api/input/preset", json={"preset_id": "beacon_target"})
    assert res_in.status_code == 200
    assert res_in.json()["images"]["clean_cam_png"].startswith("data:image/png;base64,")

    res_atk = client.post(
        "/api/attack",
        json={
            "config": {
                "kind": "patch_eot",
                "steps": 2,
                "patch_size": 32,
                "patch_x": 96,
                "patch_y": 96,
                "eot_samples": 2,
            }
        },
    )
    assert res_atk.status_code == 200
    assert res_atk.json()["images"]["adversarial_cam_png"].startswith("data:image/png;base64,")

    res_pur = client.post(
        "/api/purify",
        json={
            "config": {"kind": "svd", "svd": {"rank_k": 24, "channel_mode": "per_channel"}},
            "auto_evaluate": True,
        },
    )
    assert res_pur.status_code == 200
    payload = res_pur.json()
    assert payload["images"]["purified_cam_png"].startswith("data:image/png;base64,")
    assert payload["fsm"]["context"]["evaluation"]["attention_similarity_adv"] is not None
