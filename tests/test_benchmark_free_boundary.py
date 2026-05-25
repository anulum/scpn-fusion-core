# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Behavioural contracts for validation/benchmark_free_boundary.py."""

from __future__ import annotations

from validation.benchmark_free_boundary import run_free_boundary_benchmark


def test_free_boundary_benchmark_reports_explicit_solver_modes() -> None:
    report = run_free_boundary_benchmark()

    assert report["benchmark_id"] == "free_boundary_coil_vacuum_reconstruction"
    assert report["benchmark_scope"] == "free_boundary_reconstruction"
    assert report["single_coil"]["physics_scope"] == "external_coil_vacuum_flux"
    assert (
        report["boundary_flux_reconstruction"]["solver_mode"]
        == "coil_green_boundary_reconstruction"
    )
    assert (
        report["solve_free_boundary_vacuum_reconstruction"]["solver_mode"]
        == "free_boundary_solver_with_coil_vacuum_boundary"
    )
    assert (
        report["jax_free_boundary_wall_flux"]["solver_mode"]
        == "jax_free_boundary_wall_flux_contract"
    )
