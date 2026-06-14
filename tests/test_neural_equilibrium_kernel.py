# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests for Neural Equilibrium Kernel Runtime

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scpn_fusion.core.neural_equilibrium import NeuralEquilibriumKernel


class _DummyAccel:
    def __init__(self) -> None:
        self.cfg = type("Cfg", (), {"grid_shape": (4, 5)})()
        self.loaded_weights: Path | None = None
        self.last_features: np.ndarray | None = None
        self._input_mean = np.array(
            [0.0, 9.9, 8.8, 7.7, 0.5, 0.6, -0.7, 2.4, 1.1, 0.2, 0.3, 4.4],
            dtype=float,
        )

    def load_weights(self, path: str | Path) -> None:
        self.loaded_weights = Path(path)

    def predict(self, features: np.ndarray) -> np.ndarray:
        self.last_features = np.array(features, dtype=float)
        base = np.arange(20, dtype=float).reshape(4, 5)
        return base + float(features[0])


def _write_config(path: Path) -> None:
    payload = {
        "dimensions": {"R_min": 5.0, "R_max": 7.0, "Z_min": -2.0, "Z_max": 2.0},
        "coils": [{"current": 1e6}] * 5,
        "physics": {
            "B_T": 5.3,
            "plasma_current_target": 15.0e6,
            "pprime_scale": 1.0,
            "ffprime_scale": 1.0,
            "simag": -2.0,
            "sibry": 1.0,
            "kappa": 1.7,
            "delta_upper": 0.33,
            "delta_lower": 0.33,
            "q95": 3.0,
        },
        "target": {"R_axis": 6.2, "Z_axis": 0.0},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_rich_config(path: Path) -> None:
    payload = {
        "dimensions": {"R_min": 4.5, "R_max": 8.5, "Z_min": -3.0, "Z_max": 3.0},
        "coils": [
            {"current": 1.1e6},
            {"current": 2.2e6},
            {"current": 3.3e6},
            {"current": 4.4e6},
            {"current": 5.5e6},
        ],
        "physics": {
            "B_T": 5.8,
            "plasma_current_target": 15.0e6,
            "pprime_scale": 0.82,
            "ffprime_scale": 1.17,
            "simag": -3.5,
            "sibry": 1.25,
            "kappa": 1.93,
            "delta_upper": 0.41,
            "delta_lower": 0.27,
            "q95": 3.45,
        },
        "target": {
            "R_axis": 6.35,
            "Z_axis": 0.12,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_reference_config(path: Path) -> None:
    payload = {
        "dimensions": {"R_min": 4.5, "R_max": 8.5, "Z_min": -3.0, "Z_max": 3.0},
        "coils": [{"current": 1.0e6}],
        "physics": {
            "plasma_current_target": 15.0,
        },
        "_reference": {
            "B_T": 5.3,
            "R_major_m": 6.2,
            "kappa": 1.7,
            "delta": 0.33,
            "q95": 3.0,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_neural_equilibrium_kernel_reexport_path() -> None:
    # Import path should stay stable after runtime extraction.
    assert NeuralEquilibriumKernel.__module__.endswith("neural_equilibrium_kernel")


def test_neural_equilibrium_kernel_init_and_solve(monkeypatch, tmp_path: Path) -> None:
    import scpn_fusion.core.neural_equilibrium as ne_mod

    weights_path = tmp_path / "weights.npz"
    weights_path.write_bytes(b"stub")
    monkeypatch.setattr(ne_mod, "NeuralEquilibriumAccelerator", _DummyAccel)
    monkeypatch.setattr(ne_mod, "DEFAULT_WEIGHTS_PATH", weights_path)

    config_path = tmp_path / "kernel_config.json"
    _write_config(config_path)

    kernel = NeuralEquilibriumKernel(config_path)
    assert kernel.Psi.shape == (4, 5)
    assert np.isclose(kernel.R[0], 5.0)
    assert np.isclose(kernel.Z[-1], 2.0)

    out = kernel.solve_equilibrium()
    assert out["converged"] is True
    assert out["solver_method"] == "neural_surrogate"
    assert out["wall_time_s"] >= 0.0
    assert kernel.Psi.shape == (4, 5)


def test_neural_equilibrium_kernel_uses_complete_12_feature_config_vector(
    monkeypatch, tmp_path: Path
) -> None:
    import scpn_fusion.core.neural_equilibrium as ne_mod

    monkeypatch.setattr(ne_mod, "NeuralEquilibriumAccelerator", _DummyAccel)

    config_path = tmp_path / "rich_kernel_config.json"
    _write_rich_config(config_path)

    kernel = NeuralEquilibriumKernel(config_path, weights_path=tmp_path / "missing_weights.npz")
    kernel.solve_equilibrium()

    expected = np.array(
        [
            15.0,
            5.8,
            6.35,
            0.12,
            0.82,
            1.17,
            -3.5,
            1.25,
            1.93,
            0.41,
            0.27,
            3.45,
        ]
    )
    assert kernel.accel.last_features is not None
    np.testing.assert_allclose(kernel.accel.last_features, expected)


def test_neural_equilibrium_kernel_uses_reference_metadata_and_model_priors(
    monkeypatch, tmp_path: Path
) -> None:
    import scpn_fusion.core.neural_equilibrium as ne_mod

    monkeypatch.setattr(ne_mod, "NeuralEquilibriumAccelerator", _DummyAccel)

    config_path = tmp_path / "reference_kernel_config.json"
    _write_reference_config(config_path)

    kernel = NeuralEquilibriumKernel(config_path, weights_path=tmp_path / "missing_weights.npz")
    kernel.solve_equilibrium()

    expected = np.array(
        [
            15.0,
            5.3,
            6.2,
            0.0,
            1.0,
            1.0,
            -0.7,
            2.4,
            1.7,
            0.33,
            0.33,
            3.0,
        ]
    )
    assert kernel.accel.last_features is not None
    np.testing.assert_allclose(kernel.accel.last_features, expected)


def test_neural_equilibrium_kernel_find_x_point(monkeypatch, tmp_path: Path) -> None:
    import scpn_fusion.core.neural_equilibrium as ne_mod

    monkeypatch.setattr(ne_mod, "NeuralEquilibriumAccelerator", _DummyAccel)

    config_path = tmp_path / "kernel_config.json"
    _write_config(config_path)

    kernel = NeuralEquilibriumKernel(config_path, weights_path=tmp_path / "missing_weights.npz")
    psi = np.zeros((4, 5), dtype=float)
    (r_x, z_x), psi_x = kernel.find_x_point(psi)

    assert np.isfinite(r_x)
    assert np.isfinite(z_x)
    assert np.isfinite(psi_x)
