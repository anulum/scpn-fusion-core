# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Pretrained ITPA MLP Surrogate Tests
"""Tests for the pretrained ITPA MLP surrogate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core.pretrained_mlp_surrogate import (
    PretrainedMLPSurrogate,
    _load_itpa_training_data,
    _train_itpa_mlp,
    evaluate_pretrained_mlp,
    load_pretrained_mlp,
    save_pretrained_mlp,
)
from scpn_fusion.core.pretrained_surrogates import bundle_pretrained_surrogates


def test_pretrained_mlp_eval_and_load(tmp_path: Path) -> None:
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


def test_predict_reshapes_1d_input_and_rejects_wrong_width() -> None:
    model = PretrainedMLPSurrogate(
        feature_mean=np.zeros(3, dtype=np.float64),
        feature_std=np.ones(3, dtype=np.float64),
        w1=np.zeros((3, 2), dtype=np.float64),
        b1=np.zeros(2, dtype=np.float64),
        w2=np.zeros(2, dtype=np.float64),
        b2=np.array(0.0, dtype=np.float64),
        target_mean=1.0,
        target_std=1.0,
    )
    # A 1-D feature vector is treated as a single-sample batch.
    out = model.predict(np.zeros(3, dtype=np.float64))
    assert out.shape == (1,)
    # A mismatched feature width is rejected.
    with pytest.raises(ValueError, match="Expected feature width 3"):
        model.predict(np.zeros((1, 5), dtype=np.float64))


def test_load_itpa_training_data_rejects_short_dataset(tmp_path: Path) -> None:
    header = "Ip_MA,BT_T,ne19_1e19m3,Ploss_MW,R_m,a_m,kappa,delta,M_AMU,tau_E_s\n"
    row = "15,5.3,10,87,6.2,2.0,1.7,0.33,2.5,0.9\n"
    csv_path = tmp_path / "tiny_itpa.csv"
    csv_path.write_text(header + row * 3, encoding="utf-8")
    with pytest.raises(ValueError, match="at least 8 rows"):
        _load_itpa_training_data(csv_path=csv_path)


def test_evaluate_pretrained_mlp_honours_max_samples(tmp_path: Path) -> None:
    mlp_path = tmp_path / "mlp_cap.npz"
    _ = bundle_pretrained_surrogates(
        force_retrain=True,
        seed=4,
        weights_dir=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        mlp_path=mlp_path,
        fno_path=tmp_path / "fno.npz",
        mlp_hidden=8,
        mlp_epochs=40,
        fno_modes=4,
        fno_width=6,
        fno_epochs=1,
        fno_batch_size=4,
        fno_augment_per_file=1,
    )
    out = evaluate_pretrained_mlp(model_path=mlp_path, max_samples=5)
    assert out["samples"] == 5.0


def test_train_itpa_mlp_is_deterministic_and_persists(tmp_path: Path) -> None:
    # Direct-surface test: training twice with the same seed yields identical
    # metrics, and the saved model round-trips through the archive.
    model_a, metrics_a = _train_itpa_mlp(seed=5, hidden=8, epochs=40)
    model_b, metrics_b = _train_itpa_mlp(seed=5, hidden=8, epochs=40)
    assert metrics_a == metrics_b
    assert isinstance(model_a, PretrainedMLPSurrogate)

    path = tmp_path / "mlp_roundtrip.npz"
    save_pretrained_mlp(model_a, path=path)
    reloaded = load_pretrained_mlp(path=path)
    sample = np.asarray([[15.0, 5.3, 10.1, 87.0, 6.2, 2.0, 1.70, 0.33, 2.5]], dtype=np.float64)
    assert np.allclose(model_a.predict(sample), reloaded.predict(sample))
