# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Confinement-time guard tests for the GS-coupling mixin.

Covers the edge branches of ``compute_confinement_time`` that the steady-state
integration tests never enter — non-positive power loss, profile/geometry shape
mismatch, and a non-physical stored-energy result — plus the lazily-resolved
``PhysicsError`` helper used to break the import cycle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_fusion.core.integrated_transport_solver import PhysicsError, TransportSolver
from scpn_fusion.core.integrated_transport_solver_coupling import _physics_error_type

_BASE_CONFIG: dict[str, Any] = {
    "reactor_name": "TransportSolver-Coupling-Test",
    "grid_resolution": [24, 24],
    "dimensions": {"R_min": 4.0, "R_max": 8.0, "Z_min": -4.0, "Z_max": 4.0},
    "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
    "coils": [{"name": "CS", "r": 1.7, "z": 0.0, "current": 0.15}],
    "solver": {"max_iterations": 10, "convergence_threshold": 1e-4, "relaxation_factor": 0.1},
}


def _solver(tmp_path: Path) -> TransportSolver:
    """Construct a single-ion solver from a temp JSON config."""
    path = tmp_path / "transport_config.json"
    path.write_text(json.dumps(_BASE_CONFIG), encoding="utf-8")
    return TransportSolver(str(path), multi_ion=False)


def test_physics_error_type_resolves_lazily() -> None:
    """The lazy helper returns the concrete ``PhysicsError`` type."""
    assert _physics_error_type() is PhysicsError


def test_non_positive_power_loss_returns_infinite_tau(tmp_path: Path) -> None:
    """A non-positive power loss yields an infinite confinement time."""
    solver = _solver(tmp_path)
    assert solver.compute_confinement_time(0.0) == float("inf")


def test_profile_geometry_shape_mismatch_raises(tmp_path: Path) -> None:
    """A geometry cell count that disagrees with the profiles raises PhysicsError."""
    solver = _solver(tmp_path)
    # Inject a volume element with one extra cell via the module's own cache.
    solver._dV_cache = np.ones(solver.rho.size + 1, dtype=np.float64)
    with pytest.raises(PhysicsError, match="shape mismatch"):
        solver.compute_confinement_time(10.0)


def test_non_physical_stored_energy_returns_infinite_tau(tmp_path: Path) -> None:
    """A negative stored-energy result (bad geometry) yields an infinite tau."""
    solver = _solver(tmp_path)
    # A negative volume element drives the positive energy density to a
    # non-physical negative stored energy, which the guard maps to infinity.
    solver._dV_cache = -np.ones(solver.rho.size, dtype=np.float64)
    assert solver.compute_confinement_time(10.0) == float("inf")
