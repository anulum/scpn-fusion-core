# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Pretrained Surrogates Tests
"""Tests for pretrained surrogate bundle and evaluation helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scpn_fusion.core.pretrained_surrogates as ps
from scpn_fusion.core.pretrained_surrogates import (
    bundle_pretrained_surrogates,
    get_pretrained_surrogate_coverage,
)


def test_get_pretrained_surrogate_coverage_default_and_merge() -> None:
    base = get_pretrained_surrogate_coverage()
    assert base["coverage_percent"] > 0.0
    # A user manifest without a coverage section falls back to the default.
    assert get_pretrained_surrogate_coverage({"other": 1}) == base
    assert get_pretrained_surrogate_coverage({"coverage": "bad"}) == base
    # A valid coverage section is merged over the shipped baseline.
    merged = get_pretrained_surrogate_coverage({"coverage": {"coverage_percent": 12.5}})
    assert merged["coverage_percent"] == 12.5
    assert merged["pretrained_shipped"] == base["pretrained_shipped"]


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ("not json {", "invalid cached manifest"),
        ("[1, 2, 3]", "expected JSON object"),
        ('{"version": "v1"}', "missing keys"),
        (
            '{"version": 1, "artifacts": {}, "datasets": {}, "config": {}, '
            '"metrics": {}, "coverage": {}}',
            "version must be a string",
        ),
        (
            '{"version": "v1", "artifacts": "bad", "datasets": {}, "config": {}, '
            '"metrics": {}, "coverage": {}}',
            "artifacts must be an object",
        ),
    ],
)
def test_load_cached_manifest_rejects_malformed_payloads(
    tmp_path: Path, payload: str, match: str
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        ps._load_cached_manifest(manifest_path)


def test_bundle_pretrained_surrogates_creates_artifacts(tmp_path: Path) -> None:
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
    tmp_path: Path, kwargs: dict[str, int], match: str
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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

    def _unexpected(*args: Any, **kwargs: Any) -> None:
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    manifest_path = tmp_path / "manifest.json"
    mlp_path = tmp_path / "mlp.npz"
    fno_path = tmp_path / "fno.npz"
    mlp_path.write_bytes(b"cached")
    fno_path.write_bytes(b"cached")
    manifest_path.write_text('{"version":"task2-pretrained-v1"}', encoding="utf-8")

    calls = {"mlp": 0, "fno": 0}

    def _stub_train_itpa_mlp(**kwargs: Any) -> tuple[ps.PretrainedMLPSurrogate, dict[str, float]]:
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

    def _stub_save_pretrained_mlp(model: ps.PretrainedMLPSurrogate, path: Path) -> None:
        _ = model
        np.savez(path, x=np.array([1.0], dtype=np.float64))

    def _stub_train_fno_on_jet(*, save_path: Path, **kwargs: Any) -> dict[str, float]:
        _ = kwargs
        calls["fno"] += 1
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(b"fno")
        return {"train_relative_l2": 0.0, "val_relative_l2": 0.0, "dataset_samples": 1.0}

    monkeypatch.setattr(ps, "_train_itpa_mlp", _stub_train_itpa_mlp)
    monkeypatch.setattr(ps, "save_pretrained_mlp", _stub_save_pretrained_mlp)
    monkeypatch.setattr(ps, "_train_fno_on_jet", _stub_train_fno_on_jet)
    caplog.set_level(logging.WARNING, logger=ps.__name__)

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
    assert any(
        "Invalid cached pretrained-surrogates manifest" in rec.message for rec in caplog.records
    )
