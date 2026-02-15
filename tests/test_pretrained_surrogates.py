# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Pretrained Surrogates Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for pretrained surrogate bundle and evaluation helpers."""

from __future__ import annotations

import numpy as np
import pytest

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
