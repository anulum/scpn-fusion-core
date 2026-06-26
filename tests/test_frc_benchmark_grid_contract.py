# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Regression tests for the FRC rigid-rotor benchmark grid contract."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest


EXPECTED_FRC_GRID_POINTS = [65, 129, 257, 513]


def _repo_root() -> Path:
    """Return the repository root for benchmark contract fixtures."""
    return Path(__file__).resolve().parents[1]


def _python_benchmark_grids() -> list[int]:
    """Extract the Python FRC benchmark grid list without importing the script."""
    tree = ast.parse((_repo_root() / "benchmarks/bench_frc_rigid_rotor.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_GRIDS":
                    value = ast.literal_eval(node.value)
                    if isinstance(value, list):
                        return [int(item) for item in value]
    raise AssertionError("benchmarks/bench_frc_rigid_rotor.py is missing _GRIDS")


def _rust_criterion_grids() -> list[int]:
    """Extract the Rust Criterion FRC benchmark grid list."""
    text = (
        _repo_root() / "scpn-fusion-rs/crates/fusion-physics/benches/frc_rigid_rotor_bench.rs"
    ).read_text()
    match = re.search(r"for size in \[([^\]]+)\]\.iter\(\)", text)
    if match is None:
        raise AssertionError("Rust FRC Criterion benchmark grid list was not found")
    return [int(token.strip().removesuffix("_usize")) for token in match.group(1).split(",")]


def _tracked_report_grids() -> list[int]:
    """Read the tracked FRC benchmark report grid ladder."""
    report = json.loads(
        (_repo_root() / "validation/reports/frc_rigid_rotor_benchmark.json").read_text()
    )
    metrics = report["python_numpy"]["metrics"]
    return [int(row["grid_points"]) for row in metrics]


def test_frc_rigid_rotor_benchmark_grid_ladder_is_cross_surface_consistent() -> None:
    """Keep Python, Rust Criterion, the tracked report, and docs on one grid ladder."""
    assert _python_benchmark_grids() == EXPECTED_FRC_GRID_POINTS
    assert _rust_criterion_grids() == EXPECTED_FRC_GRID_POINTS
    assert _tracked_report_grids() == EXPECTED_FRC_GRID_POINTS

    benchmarks = (_repo_root() / "docs/BENCHMARKS.md").read_text()
    assert "surfaces on `65`, `129`, `257`, and `513` point radial grids" in benchmarks
    assert "surfaces on `64`, `256`, and `1024` point radial grids" not in benchmarks


def test_frc_grid_extractors_reject_missing_grid_declarations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify malformed benchmark fixtures fail with explicit assertion errors."""
    monkeypatch.setattr(
        "test_frc_benchmark_grid_contract._repo_root",
        lambda: tmp_path,
    )
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks/bench_frc_rigid_rotor.py").write_text("VALUE = 1\n")
    rust_bench = tmp_path / "scpn-fusion-rs/crates/fusion-physics/benches"
    rust_bench.mkdir(parents=True)
    (rust_bench / "frc_rigid_rotor_bench.rs").write_text("fn main() {}\n")

    with pytest.raises(AssertionError, match="missing _GRIDS"):
        _python_benchmark_grids()
    with pytest.raises(AssertionError, match="grid list was not found"):
        _rust_criterion_grids()
