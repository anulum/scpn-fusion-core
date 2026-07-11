# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Backend-selection guard tests for the transport-closure mixin.

Covers the external-backend precondition guards that the reduced-multichannel
integration tests never reach: the TGLF-live and neural-hybrid binary-path
requirements and the neural-weights validation. Each guard raises inside the
closure and is caught by the legacy Ti-threshold fallback, so the observable
contract is ``fallback_used`` with the originating exception recorded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_fusion.core._integrated_transport_solver_base import FloatArray
from scpn_fusion.core.integrated_transport_solver import TransportSolver

_BASE_CONFIG: dict[str, Any] = {
    "reactor_name": "TransportSolver-Backend-Test",
    "grid_resolution": [24, 24],
    "dimensions": {"R_min": 4.0, "R_max": 8.0, "Z_min": -4.0, "Z_max": 4.0},
    "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
    "coils": [{"name": "CS", "r": 1.7, "z": 0.0, "current": 0.15}],
    "solver": {"max_iterations": 10, "convergence_threshold": 1e-4, "relaxation_factor": 0.1},
}


class _NonNeuralModel:
    """Stub transport model that reports itself as a non-neural fallback."""

    is_neural = False

    def __init__(self, weights_path: str | None) -> None:
        self.weights_path = weights_path


def _solver(tmp_path: Path) -> TransportSolver:
    """Construct a single-ion solver from a temp JSON config."""
    path = tmp_path / "transport_config.json"
    path.write_text(json.dumps(_BASE_CONFIG), encoding="utf-8")
    return TransportSolver(str(path), multi_ion=False)


def _closure_args(solver: TransportSolver) -> dict[str, Any]:
    """Build the rho-shaped closure arguments the backend dispatcher expects."""
    rho: FloatArray = solver.rho
    return {
        "chi_base": np.full_like(rho, 0.5),
        "chi_base_source": "constant_fallback",
        "chi_gb_reference": None,
        "q_profile": np.full_like(rho, 2.0),
        "s_hat_profile": np.full_like(rho, 1.0),
        "r_major": 6.2,
        "a_minor": 2.0,
        "b_toroidal": 5.3,
        "q_profile_source": "test",
    }


def _closure_contract(solver: TransportSolver, backend: str) -> dict[str, Any]:
    """Invoke the backend closure and return the resulting closure contract."""
    solver._compute_transport_backend_closure(transport_backend=backend, **_closure_args(solver))
    return dict(solver._last_transport_closure_contract)


def test_tglf_live_without_binary_falls_back(tmp_path: Path) -> None:
    """Selecting ``tglf_live`` with no binary path falls back with a ValueError."""
    solver = _solver(tmp_path)
    contract = _closure_contract(solver, "tglf_live")
    assert contract["fallback_used"] is True
    assert str(contract["error"]).startswith("ValueError")


def test_neural_hybrid_without_binary_falls_back(tmp_path: Path) -> None:
    """The neural-hybrid backend also requires a TGLF binary path."""
    solver = _solver(tmp_path)
    contract = _closure_contract(solver, "neural_transport_hybrid")
    assert contract["fallback_used"] is True
    assert str(contract["error"]).startswith("ValueError")


def test_neural_hybrid_without_weights_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-neural hybrid model with no weights path raises RuntimeError."""
    solver = _solver(tmp_path)
    solver.tglf_binary_path = "/nonexistent/tglf"
    monkeypatch.setattr(solver, "_get_neural_transport_model", lambda: _NonNeuralModel(None))
    contract = _closure_contract(solver, "neural_transport_hybrid")
    assert contract["fallback_used"] is True
    assert str(contract["error"]).startswith("RuntimeError")


def test_neural_hybrid_with_missing_weights_file_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hybrid model pointing at an absent weights file raises FileNotFoundError."""
    solver = _solver(tmp_path)
    solver.tglf_binary_path = "/nonexistent/tglf"
    missing = str(tmp_path / "absent_weights.npz")
    monkeypatch.setattr(solver, "_get_neural_transport_model", lambda: _NonNeuralModel(missing))
    contract = _closure_contract(solver, "neural_transport_hybrid")
    assert contract["fallback_used"] is True
    assert str(contract["error"]).startswith("FileNotFoundError")


def test_neural_backend_without_weights_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plain neural backend with no weights path raises RuntimeError."""
    solver = _solver(tmp_path)
    monkeypatch.setattr(solver, "_get_neural_transport_model", lambda: _NonNeuralModel(None))
    contract = _closure_contract(solver, "neural_transport")
    assert contract["fallback_used"] is True
    assert str(contract["error"]).startswith("RuntimeError")
