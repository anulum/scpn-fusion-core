# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Pretrained Surrogates Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for pretrained surrogate bundle and evaluation helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest

import scpn_fusion.core.pretrained_surrogates as ps
from scpn_fusion.core.pretrained_surrogates import (
    bundle_pretrained_surrogates,
    evaluate_pretrained_fno,
    evaluate_pretrained_mlp,
    load_pretrained_mlp,
)


def test_bundle_pretrained_surrogates_creates_artifacts(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    mlp_path = tmp_path / "mlp_itpa.npz"
    fno_path = tmp_path / "fno_jet.npz"
    manifest = bundle_pretrained_surrogates(
        force_retrain=True,
        seed=9,
        weights_dir=tmp_path,
        manifest_path=manifest_path,
        mlp_path=mlp_path,
        fno_path=fno_path,
        mlp_hidden=16,
        mlp_epochs=220,
        fno_modes=4,
        fno_width=8,
        fno_epochs=2,
        fno_batch_size=4,
        fno_augment_per_file=2,
    )
    assert manifest_path.exists()
    assert mlp_path.exists()
    assert fno_path.exists()
    assert manifest["version"] == "task2-pretrained-v1"
    assert "mlp" in manifest["metrics"]
    assert "fno" in manifest["metrics"]
    assert "coverage" in manifest
    assert manifest["coverage"]["coverage_percent"] > 0.0


def test_pretrained_mlp_eval_and_load(tmp_path) -> None:
    mlp_path = tmp_path / "mlp_itpa_eval.npz"
    _ = bundle_pretrained_surrogates(
        force_retrain=True,
        seed=12,
        weights_dir=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        mlp_path=mlp_path,
        fno_path=tmp_path / "fno.npz",
        mlp_hidden=12,
        mlp_epochs=150,
        fno_modes=4,
        fno_width=6,
        fno_epochs=1,
        fno_batch_size=4,
        fno_augment_per_file=1,
    )
    eval_out = evaluate_pretrained_mlp(model_path=mlp_path)
    model = load_pretrained_mlp(path=mlp_path)
    sample = np.asarray([[15.0, 5.3, 10.1, 87.0, 6.2, 2.0, 1.70, 0.33, 2.5]], dtype=np.float64)
    pred = model.predict(sample)
    assert pred.shape == (1,)
    assert np.isfinite(pred[0])
    assert np.isfinite(eval_out["rmse_pct"])
    assert eval_out["samples"] > 0


def test_pretrained_fno_eval_returns_finite_metrics(tmp_path) -> None:
    fno_path = tmp_path / "fno_eval.npz"
    _ = bundle_pretrained_surrogates(
        force_retrain=True,
        seed=17,
        weights_dir=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        mlp_path=tmp_path / "mlp.npz",
        fno_path=fno_path,
        mlp_hidden=12,
        mlp_epochs=120,
        fno_modes=4,
        fno_width=8,
        fno_epochs=2,
        fno_batch_size=4,
        fno_augment_per_file=2,
    )
    out = evaluate_pretrained_fno(
        fno_path=fno_path,
        augment_per_file=1,
        max_samples=4,
    )
    assert out["eval_samples"] == 4.0
    assert np.isfinite(out["eval_relative_l2_mean"])
    assert np.isfinite(out["eval_relative_l2_p95"])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mlp_hidden": 0}, "mlp_hidden"),
        ({"mlp_epochs": 0}, "mlp_epochs"),
        ({"fno_modes": 0}, "fno_modes"),
        ({"fno_width": 0}, "fno_modes"),
        ({"fno_epochs": 0}, "fno_epochs"),
        ({"fno_batch_size": 0}, "fno_epochs"),
        ({"fno_augment_per_file": 0}, "fno_augment_per_file"),
    ],
)
def test_bundle_pretrained_surrogates_rejects_invalid_config(
    tmp_path, kwargs: dict[str, int], match: str
) -> None:
    params = {
        "force_retrain": True,
        "seed": 3,
        "weights_dir": tmp_path,
        "manifest_path": tmp_path / "manifest.json",
        "mlp_path": tmp_path / "mlp.npz",
        "fno_path": tmp_path / "fno.npz",
        "mlp_epochs": 32,
        "fno_epochs": 1,
        "fno_augment_per_file": 1,
    }
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        bundle_pretrained_surrogates(**params)


def test_bundle_pretrained_surrogates_reuses_valid_cached_manifest(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    mlp_path = tmp_path / "mlp.npz"
    fno_path = tmp_path / "fno.npz"
    mlp_path.write_bytes(b"cached")
    fno_path.write_bytes(b"cached")
    cached_manifest = {
        "version": "task2-pretrained-v1",
        "artifacts": {"mlp_itpa": "weights/mlp.npz", "fno_eurofusion_jet": "weights/fno.npz"},
        "datasets": {"itpa": "validation/itpa.csv", "eurofusion_proxy_jet": "validation/jet"},
        "config": {"seed": 7},
        "metrics": {"mlp": {"train_rmse_s": 1.0}, "fno": {"val_relative_l2": 0.1}},
        "coverage": {"coverage_percent": 50.0},
    }
    manifest_path.write_text(json.dumps(cached_manifest), encoding="utf-8")

    def _unexpected(*args, **kwargs):  # type: ignore[no-untyped-def]
        _ = (args, kwargs)
        raise AssertionError("training should not run when cached manifest is valid")

    monkeypatch.setattr(ps, "_train_itpa_mlp", _unexpected)
    monkeypatch.setattr(ps, "_train_fno_on_jet", _unexpected)

    out = bundle_pretrained_surrogates(
        force_retrain=False,
        weights_dir=tmp_path,
        manifest_path=manifest_path,
        mlp_path=mlp_path,
        fno_path=fno_path,
    )
    assert out == cached_manifest


def test_bundle_pretrained_surrogates_rebuilds_on_invalid_cached_manifest(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    mlp_path = tmp_path / "mlp.npz"
    fno_path = tmp_path / "fno.npz"
    mlp_path.write_bytes(b"cached")
    fno_path.write_bytes(b"cached")
    manifest_path.write_text('{"version":"task2-pretrained-v1"}', encoding="utf-8")

    calls = {"mlp": 0, "fno": 0}

    def _stub_train_itpa_mlp(**kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        calls["mlp"] += 1
        model = ps.PretrainedMLPSurrogate(
            feature_mean=np.array([0.0], dtype=np.float64),
            feature_std=np.array([1.0], dtype=np.float64),
            w1=np.array([[0.0]], dtype=np.float64),
            b1=np.array([0.0], dtype=np.float64),
            w2=np.array([0.0], dtype=np.float64),
            b2=np.array(0.0, dtype=np.float64),
            target_mean=0.0,
            target_std=1.0,
        )
        return model, {"train_rmse_s": 0.0, "train_rmse_pct": 0.0}

    def _stub_save_pretrained_mlp(model, path):  # type: ignore[no-untyped-def]
        _ = model
        np.savez(path, x=np.array([1.0], dtype=np.float64))

    def _stub_train_fno_on_jet(*, save_path, **kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        calls["fno"] += 1
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(b"fno")
        return {"train_relative_l2": 0.0, "val_relative_l2": 0.0, "dataset_samples": 1.0}

    monkeypatch.setattr(ps, "_train_itpa_mlp", _stub_train_itpa_mlp)
    monkeypatch.setattr(ps, "save_pretrained_mlp", _stub_save_pretrained_mlp)
    monkeypatch.setattr(ps, "_train_fno_on_jet", _stub_train_fno_on_jet)

    out = bundle_pretrained_surrogates(
        force_retrain=False,
        seed=3,
        weights_dir=tmp_path,
        manifest_path=manifest_path,
        mlp_path=mlp_path,
        fno_path=fno_path,
        mlp_hidden=4,
        mlp_epochs=1,
        fno_modes=2,
        fno_width=2,
        fno_epochs=1,
        fno_batch_size=1,
        fno_augment_per_file=1,
    )

    assert calls["mlp"] == 1
    assert calls["fno"] == 1
    assert out["version"] == "task2-pretrained-v1"
    assert "mlp" in out["metrics"]
    assert "fno" in out["metrics"]
