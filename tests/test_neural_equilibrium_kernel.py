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

    def load_weights(self, path: str | Path) -> None:
        self.loaded_weights = Path(path)

    def predict(self, features: np.ndarray) -> np.ndarray:
        base = np.arange(20, dtype=float).reshape(4, 5)
        return base + float(features[0])


def _write_config(path: Path) -> None:
    payload = {
        "dimensions": {"R_min": 5.0, "R_max": 7.0, "Z_min": -2.0, "Z_max": 2.0},
        "coils": [{"current": 1e6}] * 5,
        "physics": {"beta_scale": 1.0},
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
