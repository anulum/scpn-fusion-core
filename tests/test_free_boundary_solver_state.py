# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Equilibrium State Lifecycle Tests
"""Exercise owned transactions around actual public equilibrium executions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scpn_fusion.core.free_boundary_solver_state import FreeBoundaryState
from scpn_fusion.core.fusion_kernel import CoilSet, FusionKernel


def test_restore_evicted_magnetic_field_and_continue_equilibrium(tmp_path: Path) -> None:
    """Restore an evicted public field from owned state and resume real solving.

    Parameters
    ----------
    tmp_path : Path
        Isolated actual configuration and runtime directory.
    """
    configuration = {
        "reactor_name": "Public transaction lifecycle",
        "grid_resolution": [5, 5],
        "dimensions": {"R_min": 4, "R_max": 8, "Z_min": -4, "Z_max": 4},
        "physics": {"plasma_current_target": 0.0, "vacuum_permeability": 4e-7 * np.pi},
        "coils": [{"name": "CS", "r": 3, "z": 0, "current": 100000}],
        "solver": {
            "max_iterations": 1000,
            "convergence_threshold": 1e-12,
            "relaxation_factor": 1.0,
            "solver_method": "sor",
            "require_gs_residual": True,
            "gs_residual_threshold": 1e-12,
            "fail_on_diverge": True,
            "sor_omega": 1.0,
        },
    }
    path = tmp_path / "equilibrium.json"
    path.write_text(json.dumps(configuration), encoding="utf-8")
    kernel = FusionKernel(path)
    coils = CoilSet(positions=[(3.0, 0.0)], currents=np.array([1e5]), turns=[1])
    result = kernel.solve_free_boundary(coils, max_outer_iter=1)
    assert result["inner_status"] == "converged"
    checkpoint = FreeBoundaryState.capture(kernel, coils, result["inner_result"])
    expected = {name: getattr(kernel, name).copy() for name in ("Psi", "J_phi", "B_R", "B_Z")}
    # Explicit cache eviction and public-array edits model an interrupted caller.
    del kernel.B_R
    kernel.Psi[:] = -7.0
    kernel.J_phi[:] = 11.0
    coils.currents[:] = 0.0
    result["inner_result"]["residual_history"].append(123.0)
    checkpoint.restore(kernel, coils)
    for name, field in expected.items():
        np.testing.assert_array_equal(getattr(kernel, name), field)
        assert not np.shares_memory(getattr(kernel, name), checkpoint.fields[name])
    np.testing.assert_array_equal(coils.currents, [1e5])
    assert checkpoint.inner_result is not None
    assert 123.0 not in checkpoint.inner_result["residual_history"]
    continued = kernel.solve_free_boundary(coils, max_outer_iter=1)
    assert continued["inner_status"] == "converged"
    assert continued["canonical_admission"] == "not_evaluated"
    assert continued["vacuum_boundary_abs_error"] < 1e-15
