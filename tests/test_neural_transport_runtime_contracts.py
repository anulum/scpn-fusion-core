# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Runtime contract tests for the neural transport model loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.core._neural_transport_runtime import NeuralTransportModel, _append_derived
from scpn_fusion.core._neural_transport_types import TransportInputs


FloatArray = NDArray[np.float64]


class ProfileArrays(TypedDict):
    """Keyword arrays accepted by ``NeuralTransportModel.predict_profile``."""

    rho: FloatArray
    te: FloatArray
    ti: FloatArray
    ne: FloatArray
    q_profile: FloatArray
    s_hat_profile: FloatArray


def _profile_arrays(n_points: int = 8) -> ProfileArrays:
    """Return a smooth, strictly increasing profile for neural-mode tests."""
    rho = np.linspace(0.05, 0.95, n_points, dtype=np.float64)
    te = 6.0 * (1.0 - 0.35 * rho**2) + 0.2
    ti = 5.5 * (1.0 - 0.30 * rho**2) + 0.2
    ne = 4.5 * (1.0 - 0.20 * rho**2) + 0.4
    q_profile = 1.1 + 1.7 * rho**2
    s_hat_profile = 0.2 + 3.5 * rho
    return {
        "rho": rho,
        "te": te,
        "ti": ti,
        "ne": ne,
        "q_profile": q_profile,
        "s_hat_profile": s_hat_profile,
    }


def _write_weights(
    path: Path,
    *,
    input_dim: int = 10,
    hidden_dim: int = 4,
    final_bias: FloatArray | None = None,
    output_scale: FloatArray | None = None,
    input_mean_dim: int | None = None,
    input_std_dim: int | None = None,
    gated: bool = False,
    gb_scale: bool = False,
    log_transform: bool = False,
) -> None:
    """Write a deterministic two-layer neural-transport weight bundle."""
    output_dim = 6 if gated else 3
    if final_bias is None:
        final_bias = np.zeros(output_dim, dtype=np.float64)
    if output_scale is None:
        output_scale = np.ones(3, dtype=np.float64)
    mean_dim = input_dim if input_mean_dim is None else input_mean_dim
    std_dim = input_dim if input_std_dim is None else input_std_dim
    payload: dict[str, Any] = {
        "version": np.array(1, dtype=np.int64),
        "w1": np.zeros((input_dim, hidden_dim), dtype=np.float64),
        "b1": np.zeros(hidden_dim, dtype=np.float64),
        "w2": np.zeros((hidden_dim, output_dim), dtype=np.float64),
        "b2": final_bias.astype(np.float64),
        "input_mean": np.zeros(mean_dim, dtype=np.float64),
        "input_std": np.ones(std_dim, dtype=np.float64),
        "output_scale": output_scale.astype(np.float64),
        "gated": np.array(int(gated), dtype=np.int64),
        "gb_scale": np.array(int(gb_scale), dtype=np.int64),
        "log_transform": np.array(int(log_transform), dtype=np.int64),
    }
    np.savez(path, **payload)


def _load_model(tmp_path: Path, **weight_options: Any) -> NeuralTransportModel:
    """Create a neural model from temporary deterministic weights."""
    weights_path = tmp_path / "weights.npz"
    _write_weights(weights_path, **weight_options)
    model = NeuralTransportModel(weights_path)
    assert model.is_neural
    return model


def _assert_profile_rejected(
    model: NeuralTransportModel, arrays: ProfileArrays, match: str
) -> None:
    """Assert that neural-mode profile validation rejects a bad profile."""
    with pytest.raises(ValueError, match=match):
        model.predict_profile(**arrays)


def test_append_derived_builds_full_qlknn_feature_vector() -> None:
    """Derived features extend the base ten inputs to the expected width."""
    base = np.arange(10, dtype=np.float64)
    inputs = TransportInputs(
        rho=0.4,
        te_kev=3.0,
        ti_kev=4.5,
        ne_19=5.0,
        grad_te=9.5,
        grad_ti=8.0,
        q=1.7,
    )

    extended = _append_derived(base, inputs, expected_dim=15)

    assert extended.shape == (15,)
    assert extended[10] == pytest.approx(1.5)
    assert extended[11] > 0.0
    assert extended[12] > 0.0
    assert extended[13] > 0.0
    assert np.isfinite(extended[14])


def test_weight_loader_rejects_non_npz_suffix(tmp_path: Path) -> None:
    """Weight files must be NPZ archives, not arbitrary local files."""
    weights_path = tmp_path / "weights.txt"
    weights_path.write_text("not an npz", encoding="utf-8")

    model = NeuralTransportModel(weights_path)

    assert model.is_neural is False


def test_weight_loader_handles_stat_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stat failure leaves the model in analytic fallback mode."""
    weights_path = tmp_path / "weights.npz"
    weights_path.write_bytes(b"placeholder")
    real_exists = Path.exists
    real_stat = Path.stat

    def fake_exists(path: Path) -> bool:
        if path == weights_path:
            return True
        return real_exists(path)

    def fake_stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if path == weights_path:
            raise OSError("stat unavailable")
        return real_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(Path, "stat", fake_stat)

    model = NeuralTransportModel(weights_path)

    assert model.is_neural is False


def test_weight_loader_rejects_empty_archive_path(tmp_path: Path) -> None:
    """Zero-byte weight paths are rejected before NPZ parsing."""
    weights_path = tmp_path / "empty.npz"
    weights_path.touch()

    model = NeuralTransportModel(weights_path)

    assert model.is_neural is False


def test_weight_loader_rejects_single_layer_payload(tmp_path: Path) -> None:
    """At least two affine layers are required for runtime inference."""
    weights_path = tmp_path / "one_layer.npz"
    payload: dict[str, Any] = {
        "version": np.array(1, dtype=np.int64),
        "w1": np.zeros((10, 3), dtype=np.float64),
        "b1": np.zeros(3, dtype=np.float64),
        "input_mean": np.zeros(10, dtype=np.float64),
        "input_std": np.ones(10, dtype=np.float64),
        "output_scale": np.ones(3, dtype=np.float64),
    }
    np.savez(weights_path, **payload)

    model = NeuralTransportModel(weights_path)

    assert model.is_neural is False


def test_weight_loader_rejects_missing_scale_key(tmp_path: Path) -> None:
    """The output-scale vector is part of the weight-file contract."""
    weights_path = tmp_path / "missing_output_scale.npz"
    payload: dict[str, Any] = {
        "version": np.array(1, dtype=np.int64),
        "w1": np.zeros((10, 3), dtype=np.float64),
        "b1": np.zeros(3, dtype=np.float64),
        "w2": np.zeros((3, 3), dtype=np.float64),
        "b2": np.zeros(3, dtype=np.float64),
        "input_mean": np.zeros(10, dtype=np.float64),
        "input_std": np.ones(10, dtype=np.float64),
    }
    np.savez(weights_path, **payload)

    model = NeuralTransportModel(weights_path)

    assert model.is_neural is False


def test_neural_predict_classifies_ion_dominant_channel(tmp_path: Path) -> None:
    """Ion-dominant MLP outputs are reported as ITG transport."""
    model = _load_model(tmp_path, final_bias=np.array([0.0, 4.0, 0.0], dtype=np.float64))

    fluxes = model.predict(TransportInputs())

    assert fluxes.channel == "ITG"
    assert fluxes.chi_i > fluxes.chi_e


def test_neural_predict_classifies_zero_transport_as_stable(tmp_path: Path) -> None:
    """Zero-scaled MLP outputs are reported as a stable channel."""
    model = _load_model(tmp_path, output_scale=np.zeros(3, dtype=np.float64))

    fluxes = model.predict(TransportInputs())

    assert fluxes.channel == "stable"
    assert fluxes.chi_e == 0.0
    assert fluxes.chi_i == 0.0


def test_neural_profile_rejects_rank_two_rho(tmp_path: Path) -> None:
    """Neural profile mode accepts only one-dimensional profile arrays."""
    model = _load_model(tmp_path)
    arrays = _profile_arrays()
    arrays["rho"] = np.zeros((2, 4), dtype=np.float64)

    _assert_profile_rejected(model, arrays, "must all be 1D")


def test_neural_profile_rejects_too_few_points(tmp_path: Path) -> None:
    """A neural profile needs at least three radial samples."""
    model = _load_model(tmp_path)
    arrays = _profile_arrays(n_points=2)

    _assert_profile_rejected(model, arrays, "at least 3 points")


def test_neural_profile_rejects_length_mismatch(tmp_path: Path) -> None:
    """All profile arrays must share the rho length."""
    model = _load_model(tmp_path)
    arrays = _profile_arrays()
    arrays["te"] = arrays["te"][:-1]

    _assert_profile_rejected(model, arrays, "identical length")


def test_neural_profile_rejects_nonfinite_rho(tmp_path: Path) -> None:
    """The radial coordinate must be finite."""
    model = _load_model(tmp_path)
    arrays = _profile_arrays()
    arrays["rho"] = arrays["rho"].copy()
    arrays["rho"][0] = np.nan

    _assert_profile_rejected(model, arrays, "rho must contain finite values")


def test_neural_profile_rejects_nonfinite_te(tmp_path: Path) -> None:
    """Electron temperature must be finite."""
    model = _load_model(tmp_path)
    arrays = _profile_arrays()
    arrays["te"] = arrays["te"].copy()
    arrays["te"][0] = np.inf

    _assert_profile_rejected(model, arrays, "te must contain finite values")


def test_neural_profile_rejects_nonfinite_ti(tmp_path: Path) -> None:
    """Ion temperature must be finite."""
    model = _load_model(tmp_path)
    arrays = _profile_arrays()
    arrays["ti"] = arrays["ti"].copy()
    arrays["ti"][0] = -np.inf

    _assert_profile_rejected(model, arrays, "ti must contain finite values")


def test_neural_profile_rejects_nonfinite_density(tmp_path: Path) -> None:
    """Electron density must be finite."""
    model = _load_model(tmp_path)
    arrays = _profile_arrays()
    arrays["ne"] = arrays["ne"].copy()
    arrays["ne"][0] = np.nan

    _assert_profile_rejected(model, arrays, "ne must contain finite values")


def test_neural_profile_rejects_nonfinite_q(tmp_path: Path) -> None:
    """Safety-factor profile values must be finite."""
    model = _load_model(tmp_path)
    arrays = _profile_arrays()
    arrays["q_profile"] = arrays["q_profile"].copy()
    arrays["q_profile"][0] = np.nan

    _assert_profile_rejected(model, arrays, "q_profile must contain finite values")


def test_neural_profile_rejects_nonfinite_shear(tmp_path: Path) -> None:
    """Magnetic-shear profile values must be finite."""
    model = _load_model(tmp_path)
    arrays = _profile_arrays()
    arrays["s_hat_profile"] = arrays["s_hat_profile"].copy()
    arrays["s_hat_profile"][0] = np.nan

    _assert_profile_rejected(model, arrays, "s_hat_profile must contain finite values")


def test_neural_profile_rejects_nonmonotonic_rho(tmp_path: Path) -> None:
    """The neural profile path requires strictly increasing rho."""
    model = _load_model(tmp_path)
    arrays = _profile_arrays()
    arrays["rho"] = arrays["rho"].copy()
    arrays["rho"][3] = arrays["rho"][2]

    _assert_profile_rejected(model, arrays, "rho must be strictly increasing")


def test_neural_profile_rejects_rho_outside_training_range(tmp_path: Path) -> None:
    """The neural profile path enforces the documented rho range."""
    model = _load_model(tmp_path)
    arrays = _profile_arrays()
    arrays["rho"] = np.linspace(0.0, 1.3, arrays["rho"].size, dtype=np.float64)

    _assert_profile_rejected(model, arrays, "rho must satisfy")


def test_neural_profile_rejects_invalid_major_radius(tmp_path: Path) -> None:
    """The gyro-Bohm contract requires a positive finite major radius."""
    model = _load_model(tmp_path)
    arrays = _profile_arrays()

    with pytest.raises(ValueError, match="r_major"):
        model.predict_profile(**arrays, r_major=0.0)


def test_neural_profile_rejects_invalid_minor_radius(tmp_path: Path) -> None:
    """The collisionality contract requires a positive finite minor radius."""
    model = _load_model(tmp_path)
    arrays = _profile_arrays()

    with pytest.raises(ValueError, match="a_minor"):
        model.predict_profile(**arrays, a_minor=np.nan)


def test_neural_profile_rejects_invalid_toroidal_field(tmp_path: Path) -> None:
    """The gyro-Bohm contract requires a positive finite toroidal field."""
    model = _load_model(tmp_path)
    arrays = _profile_arrays()

    with pytest.raises(ValueError, match="b_toroidal"):
        model.predict_profile(**arrays, b_toroidal=-1.0)


def test_neural_profile_rejects_input_mean_width_mismatch(tmp_path: Path) -> None:
    """Profile inference fails closed when input_mean has the wrong width."""
    model = _load_model(tmp_path, input_dim=12, input_mean_dim=11)
    arrays = _profile_arrays()

    _assert_profile_rejected(model, arrays, "input_mean dimension")


def test_neural_profile_rejects_input_std_width_mismatch(tmp_path: Path) -> None:
    """Profile inference fails closed when input_std has the wrong width."""
    model = _load_model(tmp_path, input_dim=12, input_std_dim=11)
    arrays = _profile_arrays()

    _assert_profile_rejected(model, arrays, "input_std dimension")


def test_neural_profile_records_full_width_stable_contract(tmp_path: Path) -> None:
    """Full-width gated profiles record OOD telemetry and stable channel metadata."""
    model = _load_model(
        tmp_path,
        input_dim=15,
        gated=True,
        gb_scale=True,
        log_transform=True,
        output_scale=np.zeros(3, dtype=np.float64),
    )
    arrays = _profile_arrays(n_points=9)

    chi_e, chi_i, d_e = model.predict_profile(**arrays, r_major=6.5, a_minor=2.1, b_toroidal=5.7)
    contract = model._last_surrogate_contract

    assert np.array_equal(chi_e, np.zeros_like(arrays["rho"]))
    assert np.array_equal(chi_i, np.zeros_like(arrays["rho"]))
    assert np.array_equal(d_e, np.zeros_like(arrays["rho"]))
    assert contract["dominant_channel"] == "stable"
    assert contract["channel_counts"]["stable"] == arrays["rho"].size
    assert contract["profile_contract"]["a_minor"] == pytest.approx(2.1)
    assert contract["profile_contract"]["b_toroidal"] == pytest.approx(5.7)
    assert contract["input_dim"] == 15
    assert contract["gated"] is True
    assert contract["gb_scale"] is True
    assert contract["log_transform"] is True
    assert model._last_max_abs_z_profile.shape == arrays["rho"].shape
