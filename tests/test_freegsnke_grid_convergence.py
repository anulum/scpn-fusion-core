# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — tests of the FreeGSNKE grid-convergence study
"""Contract tests on hand-built grid records, and one live check of the base-grid build.

The contract tests replace the two codes with fixed records, so they check how the study
assembles, gates and renders its report. The live test solves the MAST-U-like case with
FreeGSNKE on the base grid through both builders and requires identical currents.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("freegsnke")

import validation.benchmark_freegsnke_inverse as base
import validation.freegsnke_grid_convergence as study


def _record(
    grid: tuple[int, int], rmse: float, currents: list[float], **flags: bool
) -> dict[str, Any]:
    return {
        "grid_shape_r_z": list(grid),
        "freegsnke_run": {
            "forward_refinement_relative_change": 1e-10 if flags.get("refined", True) else 1e-6
        },
        "freegsnke_currents_a": currents,
        "vacuum_max_abs_error_wb": 1e-16,
        "source_relative_l2": 3e-8,
        "warm": {
            "converged": flags.get("warm", True),
            "psi_n_rmse_inside_limiter": rmse,
            "axis_position_error_m": 1e-3,
            "worst_xpoint_error_m": 2e-3,
            "support_current_relative_l2": 7e-3,
        },
        "frozen": {"converged": True, "psi_n_rmse_inside_limiter": rmse / 10.0},
    }


def _equilibrium(currents: list[float]) -> SimpleNamespace:
    """Stand in for a FreeGSNKE equilibrium: active currents plus one passive structure."""
    vector = np.array([*currents, 0.0])
    return SimpleNamespace(
        tokamak=SimpleNamespace(n_active_coils=len(currents), getCurrentsVec=lambda: vector)
    )


def _patch_codes(
    monkeypatch: pytest.MonkeyPatch,
    records: dict[tuple[int, int], dict[str, Any]],
    base_currents: list[float],
) -> None:
    monkeypatch.setattr(base, "validate_source", lambda _source: {"commit": "c" * 40})
    monkeypatch.setattr(
        base,
        "solve_inverse_case_with_profiles",
        lambda _source: (_equilibrium(base_currents), None, None, None),
    )
    monkeypatch.setattr(study, "grid_record", lambda _source, grid: records[grid])


def test_changes_compare_each_grid_with_the_base_grid() -> None:
    records = [_record((65, 129), 4e-3, [3.0, 4.0]), _record((129, 257), 2e-3, [3.0, 4.5])]

    (change,) = study.grid_changes(records)

    assert change["freegsnke_current_relative_l2"] == pytest.approx(0.1)
    assert change["warm_rmse_ratio"] == pytest.approx(0.5)
    assert change["frozen_rmse_ratio"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("flags", "reproduced", "failed"),
    [
        ({}, True, set()),
        ({}, False, {"base_grid_reproduces_base_benchmark_currents"}),
        ({"refined": False}, True, {"freegsnke_refinements_converged"}),
        ({"warm": False}, True, {"scpn_solves_converged"}),
    ],
)
def test_gates_concern_completeness(
    flags: dict[str, bool], reproduced: bool, failed: set[str]
) -> None:
    records = [_record((65, 129), 4e-3, [1.0]), _record((129, 257), 2e-3, [1.0], **flags)]

    checks = study.grid_checks(records, reproduced)

    assert {name for name, ok in checks.items() if not ok} == failed


def test_report_runs_every_grid_and_checks_the_base_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    records = {
        (65, 129): _record((65, 129), 4e-3, [3.0, 4.0]),
        (129, 257): _record((129, 257), 2e-3, [3.0, 4.5]),
    }
    _patch_codes(monkeypatch, records, base_currents=[3.0, 4.0])

    report = study.build_report(Path("unused"))

    assert report["status"] == "pass"
    assert report["checks"]["base_grid_reproduces_base_benchmark_currents"] is True
    assert [record["grid_shape_r_z"] for record in report["grids"]] == [[65, 129], [129, 257]]
    assert report["provenance"]["freegsnke_source_commit"] == "c" * 40
    assert set(report["provenance"]["scpn_source_sha256"]) == set(study.SOURCE_FILES)
    assert len(report["payload_sha256"]) == 64


def test_report_fails_when_the_base_grid_does_not_reproduce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = {
        (65, 129): _record((65, 129), 4e-3, [3.0, 4.0]),
        (129, 257): _record((129, 257), 2e-3, [3.0, 4.5]),
    }
    _patch_codes(monkeypatch, records, base_currents=[3.0, 4.000001])

    assert study.build_report(Path("unused"))["status"] == "fail"


def test_report_refuses_a_study_without_the_base_grid_first() -> None:
    with pytest.raises(ValueError, match="base grid"):
        study.build_report(Path("unused"), ((129, 257), (65, 129)))
    with pytest.raises(ValueError, match="base grid"):
        study.build_report(Path("unused"), ((65, 129),))


@pytest.mark.parametrize("text", ["129x257", "129X257"])
def test_grid_text_parses(text: str) -> None:
    assert study.parse_grid(text) == (129, 257)


@pytest.mark.parametrize("text", ["129", "129x", "a x b", "128x257", "31x65"])
def test_grid_text_refuses_malformed_or_even_or_small_grids(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        study.parse_grid(text)


def test_command_line_writes_both_reports_and_honours_strict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    records = {
        (65, 129): _record((65, 129), 4e-3, [3.0, 4.0]),
        (97, 193): _record((97, 193), 3e-3, [3.0, 4.2], refined=False),
    }
    _patch_codes(monkeypatch, records, base_currents=[3.0, 4.0])
    output, markdown = tmp_path / "grid.json", tmp_path / "grid.md"
    arguments = ["--source", "unused", "--grid", "97x193", "--output", str(output)]

    assert study.main([*arguments, "--markdown", str(markdown)]) == 0
    assert study.main([*arguments, "--markdown", str(markdown), "--strict"]) == 1

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "fail"
    text = markdown.read_text(encoding="utf-8")
    assert text.splitlines()[:7] == list(study.MARKDOWN_HEADER)
    assert "| 97 × 193 | 0.003 | 0.0003 |" in text
    assert "97 × 193: FreeGSNKE currents change by 0.04 (relative L2)" in text


def test_script_entry_point_runs_from_the_repository_root() -> None:
    """Run the script directly: it must put the repository on its own path."""
    root = Path(study.__file__).resolve().parents[1]
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    completed = subprocess.run(
        [sys.executable, "validation/freegsnke_grid_convergence.py", "--help"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--grid" in completed.stdout


def test_live_base_grid_build_reproduces_the_base_benchmark_currents() -> None:
    """Both builders solve the same MAST-U-like problem on the base grid to identical currents."""
    if not base.DEFAULT_SOURCE.is_dir():
        pytest.skip("the pinned FreeGSNKE checkout is not present")
    reference, _profiles, _solver, _limits = base.solve_inverse_case_with_profiles(
        base.DEFAULT_SOURCE
    )
    eq, _profiles2, run = study.solve_mastu_case(base.DEFAULT_SOURCE, study.GRIDS[0])

    assert np.array_equal(study.active_currents(eq), study.active_currents(reference))
    assert run["forward_refinement_relative_change"] <= 1e-9


def test_build_refuses_a_source_without_the_pinned_machine_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing pinned case artifacts"):
        study.build_mastu_case(tmp_path, study.GRIDS[0])


def test_live_base_grid_record_matches_the_tracked_forward_report() -> None:
    """On the base grid the study reproduces the multi-machine report's MAST-U-like values."""
    if not base.DEFAULT_SOURCE.is_dir():
        pytest.skip("the pinned FreeGSNKE checkout is not present")
    tracked = json.loads(
        (Path(study.ROOT) / "validation/reports/freegsnke_multimachine_parity.json").read_text(
            encoding="utf-8"
        )
    )["cases"]["mastu_like"]
    record = study.grid_record(base.DEFAULT_SOURCE, study.GRIDS[0])

    smooth = tracked["total_psi_comparison"]["production_smooth"]
    frozen = tracked["total_psi_comparison"]["frozen_topology_control"]
    assert record["grid_shape_r_z"] == [65, 129]
    assert record["warm"]["converged"] and record["frozen"]["converged"]
    assert record["warm"]["psi_n_rmse_inside_limiter"] == pytest.approx(
        smooth["psi_n_rmse_inside_limiter"], rel=1e-3
    )
    assert record["frozen"]["psi_n_rmse_inside_limiter"] == pytest.approx(
        frozen["psi_n_rmse_inside_limiter"], rel=1e-3
    )
    assert record["source_relative_l2"] < 1e-7
    assert record["vacuum_max_abs_error_wb"] < 1e-12
