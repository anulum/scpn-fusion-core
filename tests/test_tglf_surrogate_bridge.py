# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Direct tests for extracted TGLF surrogate bridge module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from scpn_fusion.core import tglf_interface as tglf
from scpn_fusion.core.tglf_surrogate_bridge import (
    DEFAULT_TGLF_FEATURES,
    DEFAULT_TGLF_TARGETS,
    TGLFDatasetGenerator,
    TGLFSurrogate,
    train_surrogate_from_tglf,
)


def _synthetic_dataset(n_samples: int = 200, *, seed: int = 7) -> list[dict[str, dict[str, float]]]:
    """Return a deterministic dataset whose targets lie in the surrogate's class.

    Each target is a constant + linear + per-feature-quadratic function of the
    drive features (no cross terms), i.e. exactly what :class:`TGLFSurrogate`
    represents, so a correct fit recovers it to floating-point precision.
    """
    rng = np.random.default_rng(seed)
    lo = {
        "R_LTi": 0.0,
        "R_LTe": 0.0,
        "R_Lne": 0.0,
        "q": 1.0,
        "s_hat": 0.0,
        "beta_e": 0.001,
        "Z_eff": 1.0,
    }
    hi = {
        "R_LTi": 12.0,
        "R_LTe": 12.0,
        "R_Lne": 5.0,
        "q": 5.0,
        "s_hat": 3.0,
        "beta_e": 0.05,
        "Z_eff": 3.0,
    }
    dataset: list[dict[str, dict[str, float]]] = []
    for _ in range(n_samples):
        feats = {name: float(rng.uniform(lo[name], hi[name])) for name in DEFAULT_TGLF_FEATURES}
        drive = feats["R_LTi"] + 0.5 * feats["R_LTe"] + 0.2 * feats["R_Lne"]
        out = {
            "chi_i": 0.3 + 0.4 * drive + 0.02 * feats["R_LTi"] ** 2,
            "chi_e": 0.2 + 0.25 * drive + 0.01 * feats["R_LTe"] ** 2,
            "gamma_max": 0.05 + 0.03 * feats["R_LTi"] - 0.01 * feats["s_hat"] ** 2,
            "q_i": 0.1 + 0.15 * drive,
            "q_e": 0.1 + 0.12 * drive,
        }
        dataset.append({"input": dict(feats), "output": dict(out)})
    return dataset


def test_dataset_generator_accepts_zero_samples() -> None:
    """A zero-sample request returns an empty dataset without invoking TGLF."""
    gen = TGLFDatasetGenerator("C:/fake/tglf")
    out = gen.generate_random_dataset(n_samples=0)
    assert out == []


def test_dataset_generator_records_successful_tglf_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful sampled TGLF calls are serialized into input/output records."""
    calls: list[tglf.TGLFInputDeck] = []

    def _run_tglf_binary(
        deck: tglf.TGLFInputDeck,
        binary: str | Path,
        *,
        timeout_s: float,
    ) -> tglf.TGLFOutput:
        calls.append(deck)
        assert Path(binary) == Path("/opt/tglf")
        assert timeout_s == 60.0
        return tglf.TGLFOutput(rho=deck.rho, chi_i=1.25, chi_e=2.5, gamma_max=0.75)

    monkeypatch.setattr(tglf, "run_tglf_binary", _run_tglf_binary)

    dataset = TGLFDatasetGenerator("/opt/tglf").generate_random_dataset(n_samples=2)

    assert len(calls) == 2
    assert len(dataset) == 2
    for record in dataset:
        input_payload = cast(dict[str, float], record["input"])
        output_payload = cast(dict[str, float], record["output"])
        assert 0.0 <= input_payload["R_LTi"] <= 12.0
        assert 0.0 <= input_payload["R_LTe"] <= 12.0
        assert 0.0 <= input_payload["R_Lne"] <= 5.0
        assert 1.0 <= input_payload["q"] <= 5.0
        assert 0.0 <= input_payload["s_hat"] <= 3.0
        assert 0.001 <= input_payload["beta_e"] <= 0.05
        assert 1.0 <= input_payload["Z_eff"] <= 3.0
        assert output_payload["chi_i"] == pytest.approx(1.25)
        assert output_payload["chi_e"] == pytest.approx(2.5)
        assert output_payload["gamma_max"] == pytest.approx(0.75)


def test_dataset_generator_logs_failed_tglf_runs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Failed sampled TGLF calls are logged and omitted from the dataset."""

    def _run_tglf_binary(
        deck: tglf.TGLFInputDeck,
        binary: str | Path,
        *,
        timeout_s: float,
    ) -> tglf.TGLFOutput:
        del deck, binary, timeout_s
        raise RuntimeError("synthetic TGLF failure")

    monkeypatch.setattr(tglf, "run_tglf_binary", _run_tglf_binary)

    with caplog.at_level(logging.WARNING, logger="scpn_fusion.core.tglf_surrogate_bridge"):
        dataset = TGLFDatasetGenerator("/opt/tglf").generate_random_dataset(n_samples=1)

    assert dataset == []
    assert "Sample 0 failed: synthetic TGLF failure" in caplog.text


def test_train_surrogate_writes_weights_and_report(tmp_path: Path) -> None:
    """Training fits the surrogate, persists weights, and reports per-target RMSE."""
    out_path = tmp_path / "weights.npz"
    report = train_surrogate_from_tglf(_synthetic_dataset(), out_path)
    assert out_path.is_file()
    assert report["n_samples"] == 200
    assert report["features"] == list(DEFAULT_TGLF_FEATURES)
    assert report["targets"] == list(DEFAULT_TGLF_TARGETS)
    assert set(report["rmse"]) == set(DEFAULT_TGLF_TARGETS)


def test_surrogate_recovers_in_class_function(tmp_path: Path) -> None:
    """A target inside the surrogate's hypothesis class is recovered near-exactly.

    The synthetic law is constant + linear + per-feature-quadratic — exactly the
    surrogate's design — so with a vanishing ridge the closed-form solve recovers it
    to floating-point precision (a finite ridge biases the fit; see the RMSE report).
    """
    out_path = tmp_path / "weights.npz"
    report = train_surrogate_from_tglf(_synthetic_dataset(), out_path, ridge=1e-9)
    assert max(report["rmse"].values()) < 1e-6


def test_surrogate_generalises_to_held_out_points() -> None:
    """The fitted surrogate predicts held-out in-class points to low error."""
    train = _synthetic_dataset(200, seed=1)
    test = _synthetic_dataset(60, seed=99)
    model = TGLFSurrogate()
    x_tr = np.array([[s["input"][f] for f in DEFAULT_TGLF_FEATURES] for s in train])
    y_tr = np.array([[s["output"][t] for t in DEFAULT_TGLF_TARGETS] for s in train])
    model.fit(x_tr, y_tr)
    x_te = np.array([[s["input"][f] for f in DEFAULT_TGLF_FEATURES] for s in test])
    y_te = np.array([[s["output"][t] for t in DEFAULT_TGLF_TARGETS] for s in test])
    pred = model.predict(x_te)
    assert float(np.sqrt(np.mean((pred - y_te) ** 2))) < 1e-4


def test_train_is_deterministic(tmp_path: Path) -> None:
    """The same dataset yields bit-identical persisted weights (no RNG in the fit)."""
    data = _synthetic_dataset()
    p1 = tmp_path / "a.npz"
    p2 = tmp_path / "b.npz"
    train_surrogate_from_tglf(data, p1)
    train_surrogate_from_tglf(data, p2)
    with np.load(p1) as a, np.load(p2) as b:
        np.testing.assert_array_equal(a["weights"], b["weights"])


def test_surrogate_save_load_roundtrip(tmp_path: Path) -> None:
    """A loaded surrogate predicts bit-identically to the in-memory model."""
    data = _synthetic_dataset()
    x = np.array([[s["input"][f] for f in DEFAULT_TGLF_FEATURES] for s in data])
    y = np.array([[s["output"][t] for t in DEFAULT_TGLF_TARGETS] for s in data])
    model = TGLFSurrogate().fit(x, y)
    out_path = tmp_path / "weights.npz"
    model.save(out_path)
    loaded = TGLFSurrogate.load(out_path)
    assert loaded.features == model.features
    assert loaded.targets == model.targets
    probe = np.array([[6.0, 5.0, 2.0, 2.0, 1.0, 0.02, 1.5], [3.0, 3.0, 1.0, 3.0, 0.5, 0.01, 2.0]])
    np.testing.assert_array_equal(loaded.predict(probe), model.predict(probe))


def test_train_rejects_empty_dataset(tmp_path: Path) -> None:
    """An empty/too-small dataset fails closed rather than emitting a stub."""
    with pytest.raises(ValueError, match="at least"):
        train_surrogate_from_tglf([], tmp_path / "w.npz")


def test_train_rejects_missing_key(tmp_path: Path) -> None:
    """A dataset sample missing a required feature/target key is rejected."""
    data = _synthetic_dataset()
    del data[0]["input"]["q"]
    with pytest.raises(ValueError, match="missing required key"):
        train_surrogate_from_tglf(data, tmp_path / "w.npz")


def test_predict_before_fit_raises() -> None:
    """Predicting before fitting is a runtime error."""
    with pytest.raises(RuntimeError, match="not fit"):
        TGLFSurrogate().predict(np.zeros((1, len(DEFAULT_TGLF_FEATURES))))
