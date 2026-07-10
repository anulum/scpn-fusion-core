# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Initialisation-branch tests for the transport-solver init mixin.

Covers the optional EPED-pedestal construction path and the
``max_numerical_recoveries_per_step`` cap parsing/validation that the primary
transport-solver tests skip by leaving both config options unset.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scpn_fusion.core.integrated_transport_solver import TransportSolver

_BASE_CONFIG: dict[str, Any] = {
    "reactor_name": "TransportSolver-Init-Test",
    "grid_resolution": [20, 20],
    "dimensions": {"R_min": 4.0, "R_max": 8.0, "Z_min": -4.0, "Z_max": 4.0},
    "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
    "coils": [{"name": "CS", "r": 1.7, "z": 0.0, "current": 0.15}],
    "solver": {"max_iterations": 10, "convergence_threshold": 1e-4, "relaxation_factor": 0.1},
}


def _solver_from(config: dict[str, Any], tmp_path: Path) -> TransportSolver:
    """Write ``config`` to a temp JSON file and construct a single-ion solver."""
    path = tmp_path / "transport_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return TransportSolver(str(path), multi_ion=False)


def test_eped_pedestal_mode_builds_pedestal_model(tmp_path: Path) -> None:
    """Selecting ``pedestal_mode: eped`` constructs an EPED pedestal model."""
    config = copy.deepcopy(_BASE_CONFIG)
    config["physics"]["pedestal_mode"] = "eped"
    config["physics"]["B0"] = 5.3
    solver = _solver_from(config, tmp_path)
    assert solver.pedestal_model is not None


def test_default_pedestal_mode_leaves_model_unset(tmp_path: Path) -> None:
    """Without EPED mode the pedestal model stays unset."""
    solver = _solver_from(copy.deepcopy(_BASE_CONFIG), tmp_path)
    assert solver.pedestal_model is None


def test_valid_recovery_cap_is_parsed(tmp_path: Path) -> None:
    """A non-negative recovery cap is parsed into an integer limit."""
    config = copy.deepcopy(_BASE_CONFIG)
    config["solver"]["max_numerical_recoveries_per_step"] = 3
    solver = _solver_from(config, tmp_path)
    assert solver.max_numerical_recoveries_per_step == 3


def test_absent_recovery_cap_defaults_to_none(tmp_path: Path) -> None:
    """An unset recovery cap leaves the limit unbounded (``None``)."""
    solver = _solver_from(copy.deepcopy(_BASE_CONFIG), tmp_path)
    assert solver.max_numerical_recoveries_per_step is None


@pytest.mark.parametrize("cap", [-1, True])
def test_invalid_recovery_cap_raises(cap: object, tmp_path: Path) -> None:
    """A negative or boolean recovery cap is rejected."""
    config = copy.deepcopy(_BASE_CONFIG)
    config["solver"]["max_numerical_recoveries_per_step"] = cap
    with pytest.raises(ValueError, match="max_numerical_recoveries_per_step"):
        _solver_from(config, tmp_path)
