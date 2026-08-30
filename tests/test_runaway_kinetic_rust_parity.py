# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Full Runaway Kinetic Rust Parity
"""Compiled-backend parity for every unprojected kinetic output."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.core.runaway_kinetic_coefficients import (
    RunawayKineticCoefficients,
)
from scpn_fusion.core.runaway_kinetic_grid import RunawayKineticGrid
from scpn_fusion.core.runaway_kinetic_operator import RunawayKineticOperator
from scpn_fusion.core.runaway_kinetic_solver import RunawayKineticSolver
from validation.benchmark_runaway_kinetic_rust import (
    _checksum,
    _source_provenance,
    scientific_projection,
)


ROOT = Path(__file__).resolve().parents[1]


def _solver() -> tuple[
    RunawayKineticSolver,
    NDArray[np.float64],
    NDArray[np.float64],
]:
    grid = RunawayKineticGrid(
        radius_faces_m=np.array([0.0, 0.03, 0.11, 0.2]),
        pitch_faces=np.array([-1.0, -0.6, 0.1, 0.55, 1.0]),
        momentum_faces_mc=np.array([0.02, 0.08, 0.3, 0.9, 2.2, 5.0]),
    )
    radial_shape = (grid.nr + 1, grid.nxi, grid.np)
    momentum_shape = (grid.nr, grid.nxi, grid.np + 1)
    pitch_shape = (grid.nr, grid.nxi + 1, grid.np)
    radial_index = np.arange(np.prod(radial_shape), dtype=np.float64).reshape(radial_shape)
    momentum_index = np.arange(np.prod(momentum_shape), dtype=np.float64).reshape(momentum_shape)
    pitch_index = np.arange(np.prod(pitch_shape), dtype=np.float64).reshape(pitch_shape)
    cell_index = np.arange(np.prod(grid.shape), dtype=np.float64).reshape(grid.shape)
    coefficients = RunawayKineticCoefficients.checked(
        grid,
        radial_advection=1.0e-4 * (1.0 + radial_index / radial_index.size),
        momentum_electric_advection=2.0e-3 * (1.0 + momentum_index / momentum_index.size),
        momentum_collision_advection=-7.0e-4 * (1.0 + momentum_index / momentum_index.size),
        momentum_synchrotron_advection=-2.0e-4 * (1.0 + momentum_index / momentum_index.size),
        momentum_bremsstrahlung_advection=-1.0e-4 * (1.0 + momentum_index / momentum_index.size),
        pitch_electric_advection=8.0e-4 * (1.0 + pitch_index / pitch_index.size),
        pitch_synchrotron_advection=-3.0e-4 * (1.0 + pitch_index / pitch_index.size),
        radial_diffusion=2.0e-6 * (1.0 + radial_index / radial_index.size),
        momentum_diffusion=3.0e-6 * (1.0 + momentum_index / momentum_index.size),
        pitch_diffusion=4.0e-6 * (1.0 + pitch_index / pitch_index.size),
        momentum_pitch_diffusion=5.0e-7 * (1.0 + momentum_index / momentum_index.size),
        pitch_momentum_diffusion=6.0e-7 * (1.0 + pitch_index / pitch_index.size),
        avalanche_source_kernel=2.0e-30 * (1.0 + cell_index / cell_index.size),
        total_electron_density_m3=np.array([1.0e19, 1.5e19, 2.0e19]),
        total_density_avalanche_rate_s_inv=np.array([2.0, 3.0, 4.0]),
        total_density_external_source_m3_s=np.array([2.0e7, 3.0e7, 4.0e7]),
        external_source=3.0e5 * (1.0 + cell_index / cell_index.size),
    )
    initial = 1.0e10 * (1.0 + 0.2 * cell_index / cell_index.size)
    initial_density = np.array([1.0e12, 1.5e12, 2.0e12])
    return (
        RunawayKineticSolver(
            RunawayKineticOperator(grid, coefficients),
            maximum_step_s=2.5e-8,
            negativity_tolerance=1.0e-10,
        ),
        initial,
        initial_density,
    )


def test_rust_matches_numpy_for_every_full_kinetic_output() -> None:
    extension = pytest.importorskip("scpn_fusion_rs")
    assert hasattr(extension, "runaway_kinetic_solve_rust")
    solver, initial, density = _solver()
    times = np.array([0.0, 5.0e-8, 1.0e-7])

    numpy_result = solver.solve(
        initial,
        times,
        initial_runaway_density_m3=density,
        backend="numpy",
    )
    rust_result = solver.solve(
        initial,
        times,
        initial_runaway_density_m3=density,
        backend="rust",
    )

    assert rust_result.internal_steps == numpy_result.internal_steps == 4
    assert rust_result.distribution.shape == (3, *solver.operator.grid.shape)
    for name in (
        "times_s",
        "distribution",
        "radial_transport",
        "electric_acceleration",
        "collisional_drag_diffusion",
        "pitch_scattering",
        "cross_diffusion",
        "synchrotron_loss",
        "bremsstrahlung_loss",
        "avalanche_generation",
        "external_source",
        "total_tendency",
        "runaway_density_m3",
        "runaway_density_radial_transport_m3_s",
        "runaway_density_avalanche_generation_m3_s",
        "runaway_density_external_source_m3_s",
        "runaway_density_tendency_m3_s",
    ):
        np.testing.assert_allclose(
            getattr(rust_result, name),
            getattr(numpy_result, name),
            rtol=2.0e-12,
            atol=1.0e-8,
            err_msg=name,
        )
    for name in (
        "density_m3",
        "current_density_a_m2",
        "kinetic_energy_density_j_m3",
    ):
        np.testing.assert_allclose(
            getattr(rust_result.moments, name),
            getattr(numpy_result.moments, name),
            rtol=2.0e-12,
            atol=1.0e-8,
            err_msg=name,
        )
    assert rust_result.minimum_distribution == pytest.approx(
        numpy_result.minimum_distribution, rel=2.0e-12
    )
    assert np.any(rust_result.radial_transport != 0.0)
    assert np.any(rust_result.avalanche_generation != 0.0)


def test_explicit_rust_backend_fails_closed_for_stale_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver, initial, density = _solver()
    stale_extension = object()
    monkeypatch.setattr(
        "scpn_fusion.core.runaway_kinetic_solver.importlib.import_module",
        lambda _name: stale_extension,
    )
    with pytest.raises(RuntimeError, match="lacks runaway_kinetic_solve_rust"):
        solver.solve(
            initial,
            np.array([0.0, 1.0e-7]),
            initial_runaway_density_m3=density,
            backend="rust",
        )


def test_tracked_rust_benchmark_preserves_parity_and_language_boundaries() -> None:
    report = json.loads(
        (ROOT / "validation/reports/runaway_kinetic_rust_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["schema"] == "runaway-kinetic-rust-benchmark.v1"
    assert report["status"] == "accepted_full_kinetic_rust_backend_parity"
    assert report["all_pass"] is True
    assert all(report["gates"].values())
    assert report["problem"]["grid_shape_radius_pitch_momentum"] == [8, 16, 32]
    assert report["timing_seconds"]["rust_speedup_over_numpy"] > 1.0
    assert max(values["relative_l2_error"] for values in report["parity"].values()) <= 2.0e-12
    assert {
        "times_s",
        "moments.density_m3",
        "moments.current_density_a_m2",
        "moments.kinetic_energy_density_j_m3",
        "minimum_distribution",
    }.issubset(report["parity"])
    assert report["gates"]["internal_ssprk3_steps_exact"] is True
    assert report["source_revision"] == "ec8f6e24dd8104df7b43efa62d32dc39bce64253"
    assert report["source_provenance"] == _source_provenance()
    assert report["scientific_projection_sha256"] == _checksum(scientific_projection(report))
    extension = report["runtime_provenance"]["extension"]
    assert extension["artifact_name"].endswith(".so")
    assert len(extension["artifact_sha256"]) == 64
    int(extension["artifact_sha256"], 16)
    selection = report["language_selection"]
    assert selection["production_explicit_kernel"] == "rust"
    assert "stiff implicit" in selection["julia_gate"]
    assert "not this tensor kernel" in selection["go_role"]
