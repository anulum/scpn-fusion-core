# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Pedestal-mixin branch tests for the integrated transport solver.

Covers the two edge branches the primary transport-solver tests never enter:
the cached-model reuse path (a second H-mode call with unchanged neoclassical
parameters) and the EPED fallback path (``predict`` raising one of the tolerated
exceptions), which downgrades edge suppression and records a fallback event.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_fusion.core._integrated_transport_solver_base import FloatArray
from scpn_fusion.core.eped_pedestal import EpedPedestalModel
from scpn_fusion.core.integrated_transport_solver import TransportSolver

_BASE_CONFIG: dict[str, Any] = {
    "reactor_name": "TransportSolver-Pedestal-Test",
    "grid_resolution": [24, 24],
    "dimensions": {"R_min": 4.0, "R_max": 8.0, "Z_min": -4.0, "Z_max": 4.0},
    "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
    "coils": [{"name": "CS", "r": 1.7, "z": 0.0, "current": 0.15}],
    "solver": {"max_iterations": 10, "convergence_threshold": 1e-4, "relaxation_factor": 0.1},
}

_NEO_PARAMS: dict[str, Any] = {"R0": 6.2, "a": 2.0, "B0": 5.3, "Ip_MA": 15.0}


def _h_mode_solver(tmp_path: Path) -> TransportSolver:
    """Construct a single-ion solver primed with H-mode neoclassical parameters."""
    path = tmp_path / "transport_config.json"
    path.write_text(json.dumps(_BASE_CONFIG), encoding="utf-8")
    solver = TransportSolver(str(path), multi_ion=False)
    solver.neoclassical_params = copy.deepcopy(_NEO_PARAMS)
    return solver


def _unit_profiles(solver: TransportSolver) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return three unit turbulent-transport profiles shaped like the radial grid."""
    shape = solver.rho.shape
    return (np.ones(shape), np.ones(shape), np.ones(shape))


def test_cached_pedestal_model_is_reused(tmp_path: Path) -> None:
    """A second H-mode call with unchanged neoclassical params reuses the model."""
    solver = _h_mode_solver(tmp_path)
    chi_e, chi_i, d = _unit_profiles(solver)
    solver._apply_transport_pedestal_modifier(
        P_aux=40.0, chi_turb_e=chi_e, chi_turb_i=chi_i, d_turb=d
    )
    first_model = solver.pedestal_model
    assert isinstance(first_model, EpedPedestalModel)

    chi_e, chi_i, d = _unit_profiles(solver)
    solver._apply_transport_pedestal_modifier(
        P_aux=40.0, chi_turb_e=chi_e, chi_turb_i=chi_i, d_turb=d
    )
    # Same neoclassical-params identity → the cached model instance is reused.
    assert solver.pedestal_model is first_model
    assert solver._last_pedestal_contract["used"] is True


def test_eped_predict_failure_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``predict`` failure downgrades edge suppression and records a fallback."""
    solver = _h_mode_solver(tmp_path)
    chi_e, chi_i, d = _unit_profiles(solver)
    solver._apply_transport_pedestal_modifier(
        P_aux=40.0, chi_turb_e=chi_e, chi_turb_i=chi_i, d_turb=d
    )
    assert solver.pedestal_model is not None

    def _raise(*_args: object, **_kwargs: object) -> object:
        raise ValueError("synthetic eped failure")

    monkeypatch.setattr(solver.pedestal_model, "predict", _raise)
    chi_e, chi_i, d = _unit_profiles(solver)
    edge_mask = solver.rho > 0.9
    chi_e_out, _, _ = solver._apply_transport_pedestal_modifier(
        P_aux=40.0, chi_turb_e=chi_e, chi_turb_i=chi_i, d_turb=d
    )

    contract = solver._last_pedestal_contract
    assert contract["fallback_used"] is True
    assert contract["used"] is False
    assert any("eped_failure" in v for v in contract["domain_violations"])
    # Fallback applies the softer 0.1 edge factor rather than the 0.05 EPED factor.
    assert np.allclose(chi_e_out[edge_mask], 0.1)


def test_h_mode_without_neoclassical_params_falls_back(tmp_path: Path) -> None:
    """H-mode with no neoclassical params skips EPED and records the missing-param fallback."""
    solver = _h_mode_solver(tmp_path)
    solver.neoclassical_params = None
    chi_e, chi_i, d = _unit_profiles(solver)
    edge_mask = solver.rho > 0.9
    chi_e_out, _, _ = solver._apply_transport_pedestal_modifier(
        P_aux=40.0, chi_turb_e=chi_e, chi_turb_i=chi_i, d_turb=d
    )

    contract = solver._last_pedestal_contract
    assert contract["fallback_used"] is True
    assert contract["used"] is False
    assert contract["domain_violations"] == ["neoclassical_params_missing"]
    assert np.allclose(chi_e_out[edge_mask], 0.1)
