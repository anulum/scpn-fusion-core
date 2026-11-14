# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — tests of the FreeGSNKE multi-machine same-case parity benchmark
"""Contract and live tests for the FreeGSNKE parity benchmark on three public machines."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("freegsnke")

import validation.benchmark_freegsnke_inverse as base
import validation.benchmark_freegsnke_multimachine as benchmark


def test_case_registry_states_the_three_upstream_problems() -> None:
    assert set(benchmark.CASES) == {"mastu_like", "sparc_like", "iter_like"}
    for case in benchmark.CASES.values():
        nx, ny = case.grid
        # FreeGSNKE requires 2**n + 1 points per direction.
        assert (nx - 1) & (nx - 2) == 0 and (ny - 1) & (ny - 2) == 0
        assert len(case.isoflux[0]) == len(case.isoflux[1]) >= 2
        assert len(case.null_points[0]) == len(case.null_points[1]) >= 1
        nulls = set(zip(case.null_points[0], case.null_points[1], strict=True))
        assert all(target in nulls for target in case.xpoint_targets)
        assert case.ip_a > 0.0
        assert case.profile_class in {"ConstrainPaxisIp", "Fiesta_Topeol"}


def test_machine_files_exist_in_the_pinned_checkout() -> None:
    source = Path(benchmark.DEFAULT_SOURCE)
    base.validate_source(source)
    for case in benchmark.CASES.values():
        for path in benchmark.machine_paths(source, case).values():
            assert path.is_file(), path
            assert source.resolve() in path.resolve().parents


def test_case_selection_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown or empty case selection"):
        benchmark.build_report(case_names=("not_a_machine",))
    with pytest.raises(ValueError, match="unknown or empty case selection"):
        benchmark.build_report(case_names=())


def test_active_coil_file_outside_the_checkout_is_refused(tmp_path: Path) -> None:
    foreign = tmp_path / "coils.pickle"
    foreign.write_bytes(pickle.dumps({"PF1": {"R": [1.0], "Z": [0.0]}}))
    with pytest.raises(ValueError, match="must belong to the pinned FreeGSNKE checkout"):
        base._active_filaments(Path(benchmark.DEFAULT_SOURCE), np.zeros(1), foreign)


def test_active_coil_count_mismatch_names_the_count() -> None:
    source = Path(benchmark.DEFAULT_SOURCE)
    sparc = benchmark.machine_paths(source, benchmark.CASES["sparc_like"])["active_coils"]
    with pytest.raises(ValueError, match="does not match the 3 circuit currents"):
        base._active_filaments(source, np.zeros(3), sparc)


def test_host_identity_reads_the_processor_model(tmp_path: Path) -> None:
    table = tmp_path / "cpuinfo"
    table.write_text("processor\t: 0\nmodel name\t: Example CPU 1.0GHz\nflags\t: fpu\n")
    host = benchmark.host_identity(table)
    assert host["cpu_model"] == "Example CPU 1.0GHz"
    assert host["logical_cpus"] >= 1
    assert host["python"].count(".") == 2
    assert set(host) == {"cpu_model", "logical_cpus", "machine", "python", "system"}


def test_host_identity_falls_back_without_a_model_line(tmp_path: Path) -> None:
    table = tmp_path / "cpuinfo"
    table.write_text("processor\t: 0\nflags\t: fpu\n")
    without_line = benchmark.host_identity(table)
    missing = benchmark.host_identity(tmp_path / "absent")
    assert without_line["cpu_model"] == missing["cpu_model"]
    assert without_line["cpu_model"]


def _flux_object(psi_rz: np.ndarray, r: np.ndarray, z: np.ndarray) -> SimpleNamespace:
    grid_r, grid_z = np.meshgrid(r, z, indexing="ij")
    return SimpleNamespace(
        R=grid_r,
        Z=grid_z,
        psi=lambda: psi_rz,
        mask_inside_limiter=np.ones_like(grid_r, dtype=bool),
    )


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = np.linspace(0.5, 1.5, 33)
    z = np.linspace(-0.5, 0.5, 33)
    grid_r, grid_z = np.meshgrid(r, z, indexing="ij")
    return r, z, grid_r, grid_z


def test_magnetic_axis_found_on_a_peaked_flux_and_not_on_a_linear_one() -> None:
    r, z, grid_r, grid_z = _grid()
    peaked = -((grid_r - 1.0) ** 2 + grid_z**2)
    linear = grid_r.copy()
    assert benchmark.magnetic_axis_found(_flux_object(peaked, r, z), peaked) is True
    assert benchmark.magnetic_axis_found(_flux_object(linear, r, z), linear) is False


def test_magnetic_axis_check_propagates_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import freegs4e.critical

    def broken(*_args: object) -> None:
        raise ValueError("grid is not monotonic")

    monkeypatch.setattr(freegs4e.critical, "find_critical", broken)
    r, z, grid_r, _grid_z = _grid()
    with pytest.raises(ValueError, match="grid is not monotonic"):
        benchmark.magnetic_axis_found(_flux_object(grid_r, r, z), grid_r)


def test_reference_xpoints_refuses_a_flux_without_an_x_point() -> None:
    r, z, grid_r, grid_z = _grid()
    peaked = -((grid_r - 1.0) ** 2 + grid_z**2)
    with pytest.raises(ValueError, match="has no X-point"):
        benchmark.reference_xpoints(_flux_object(peaked, r, z), benchmark.CASES["sparc_like"])


def test_missing_machine_files_are_refused_before_any_solve(tmp_path: Path) -> None:
    absent = dataclasses.replace(benchmark.CASES["sparc_like"], machine_dir="ABSENT")
    with pytest.raises(FileNotFoundError, match="missing pinned machine files"):
        benchmark.solve_freegsnke_case(Path(benchmark.DEFAULT_SOURCE).resolve(), absent)


def test_script_entry_point_runs_from_the_repository_root() -> None:
    """Run the script directly: it must put the repository and ``src`` on its own path."""
    root = Path(benchmark.__file__).resolve().parents[1]
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    completed = subprocess.run(
        [sys.executable, "validation/benchmark_freegsnke_multimachine.py", "--help"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--case {iter_like,mastu_like,sparc_like}" in completed.stdout


def _synthetic_report() -> dict[str, Any]:
    case = {
        "cold_start": {"iteration_count": 7, "outcome": "no_magnetic_axis_inside_limiter"},
        "grid_shape_r_z": [129, 129],
        "profile_source_bridge": {"source_parity": {"relative_l2_error": 3.5e-8}},
        "total_psi_comparison": {
            "frozen_topology_control": {"psi_n_rmse_inside_limiter": 1.5e-4},
            "production_smooth": {"psi_n_rmse_inside_limiter": 4.6e-4},
        },
        "vacuum_bridge": {"max_abs_error_wb": 1.9e-14},
    }
    return {"cases": {"sparc_like": case}, "payload_sha256": "ab" * 32, "status": "pass"}


def test_markdown_renders_each_case_and_the_claim_boundary() -> None:
    rendered = benchmark.render_markdown(_synthetic_report())
    assert rendered.splitlines()[:7] == list(benchmark.MARKDOWN_HEADER)
    assert rendered.splitlines()[0] == "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->"
    assert "Status: **PASS**" in rendered
    assert "| sparc_like | 129×129 |" in rendered
    assert "no_magnetic_axis_inside_limiter (7 iterations)" in rendered
    assert "No facility, control, safety or speed claim; no comparison with measured" in rendered
    assert "ab" * 32 in rendered


# FreeGSNKE 3.0.1 builds the ITER-like coil resistances from one-element arrays
# (freegsnke/machine_config.py, line 118). NumPy 1.25+ deprecates that conversion, and this
# repository turns DeprecationWarning into an error, which NumPy then reports as a ValueError.
# The filter names that one upstream message only; every other warning stays an error.
UPSTREAM_ITER_COIL_DEPRECATION = (
    "ignore:Conversion of an array with ndim > 0 to a scalar is deprecated:DeprecationWarning"
)


@pytest.mark.slow
@pytest.mark.parametrize(
    "name",
    [
        "mastu_like",
        pytest.param("iter_like", marks=pytest.mark.filterwarnings(UPSTREAM_ITER_COIL_DEPRECATION)),
    ],
)
def test_live_freegsnke_problems_converge_and_refine(name: str) -> None:
    """Run the upstream inverse problem and its forward refinement only (about a minute)."""
    source = Path(benchmark.DEFAULT_SOURCE).resolve()
    eq, profiles, run = benchmark.solve_freegsnke_case(source, benchmark.CASES[name])
    assert run["forward_refinement_relative_change"] <= benchmark.REFINEMENT_TOLERANCE
    assert run["forward_refinement_tolerance"] == benchmark.REFINEMENT_TOLERANCE
    assert np.isfinite(run["inverse_relative_change"])
    assert float(profiles.Ip) == pytest.approx(benchmark.CASES[name].ip_a)
    assert benchmark.reference_xpoints(eq, benchmark.CASES[name]).shape == (
        len(benchmark.CASES[name].xpoint_targets),
        2,
    )

    # Cold-start classification on the real geometry: FreeGSNKE's own flux, then a flux
    # without an O-point, then a non-converged solve of the same flux.
    reference = np.asarray(eq.psi(), dtype=np.float64).T
    first = benchmark.classify_cold_start(eq, profiles, reference, True, float("inf"))
    assert first["outcome"] == "different_state"
    rmse = first["psi_n_rmse_inside_limiter"]
    assert 0.0 <= rmse < 1.0e-2
    matched = benchmark.classify_cold_start(eq, profiles, reference, True, rmse)
    assert matched["outcome"] == "same_fixed_point_as_warm_start"
    unconverged = benchmark.classify_cold_start(eq, profiles, reference, False, rmse)
    assert unconverged["outcome"] == "different_state"
    flat = np.broadcast_to(np.asarray(eq.R[:, 0], dtype=np.float64), reference.shape).copy()
    assert benchmark.classify_cold_start(eq, profiles, flat, True, rmse) == {
        "outcome": "no_magnetic_axis_inside_limiter"
    }


@pytest.mark.slow
def test_live_sparc_report_through_the_command_line(tmp_path: Path) -> None:
    """Run the SPARC-like case end to end through ``main`` (about ten minutes on one CPU host)."""
    output = tmp_path / "parity.json"
    markdown = tmp_path / "parity.md"
    code = benchmark.main(
        [
            "--case",
            "sparc_like",
            "--strict",
            "--output",
            str(output),
            "--markdown",
            str(markdown),
        ]
    )
    assert code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["schema_version"] == benchmark.SCHEMA_VERSION
    assert set(report["cases"]) == {"sparc_like"}
    unsigned = dict(report)
    unsigned["payload_sha256"] = ""
    assert report["payload_sha256"] == hashlib.sha256(base._canonical_json(unsigned)).hexdigest()
    assert report["provenance"]["host"]["cpu_model"]
    assert report["provenance"]["versions"]["freegsnke"] == base.UPSTREAM_VERSION
    result = report["cases"]["sparc_like"]
    assert all(result["checks"].values())
    assert result["vacuum_bridge"]["active_filament_count"] > 1000
    total = result["total_psi_comparison"]
    assert (
        total["frozen_topology_control"]["psi_n_rmse_inside_limiter"]
        <= total["production_smooth"]["psi_n_rmse_inside_limiter"]
    )
    assert result["cold_start"]["outcome"] in {
        "same_fixed_point_as_warm_start",
        "different_state",
        "no_magnetic_axis_inside_limiter",
    }
    source = Path(benchmark.DEFAULT_SOURCE).resolve()
    for entry in result["machine_files"].values():
        digest = hashlib.sha256((source / entry["path"]).read_bytes()).hexdigest()
        assert digest == entry["sha256"]
    rendered = markdown.read_text(encoding="utf-8")
    assert rendered.startswith(benchmark.MARKDOWN_HEADER[0])
    assert "| sparc_like | 129×129 |" in rendered
