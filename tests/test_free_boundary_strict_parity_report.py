# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Strict parity presentation tests
"""Render actual evaluator output and preserve its scientific boundaries."""

import json
from pathlib import Path
import subprocess
import sys

from validation.benchmark_free_boundary_strict_parity import (
    DEFAULT_FREEGS_REPORT,
    DEFAULT_MACHINE_METADATA_REPORT,
    ROOT,
    evaluate_strict_parity,
    run_benchmark,
)
from validation.free_boundary_strict_parity_report import render_markdown


def test_render_real_evidence_exposes_unverified_custody() -> None:
    """Matching historical source hashes do not imply positive physics admission."""
    report = run_benchmark(write=False)
    before = json.dumps(report, sort_keys=True)
    rendered = render_markdown(report)
    assert "No positive scientific evidence contract is admitted" in rendered
    assert "## Input byte binding" in rendered
    assert "## Source example files" in rendered
    assert "## Case identity errors" in rendered
    assert "q_profile_sanity_status" in rendered
    assert json.dumps(report, sort_keys=True) == before


def test_render_reports_actual_identity_and_threshold_errors() -> None:
    """Show the concrete errors from a corrupted historical evidence input."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text())
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text())
    metadata["same_case_public_reference_output"]["cases"][0]["machine_class"] = "other"
    freegs["strict_free_boundary_parity_evidence"]["cases"][0]["threshold_checks"][0]["value"] = 1e9
    report = evaluate_strict_parity(freegs, metadata)
    rendered = render_markdown(report)
    assert "machine_class_mismatch" in rendered
    assert "psi_n_rmse" in rendered
    assert "1000000000.0" in rendered


def test_direct_cli_stays_fail_closed_from_another_directory(tmp_path: Path) -> None:
    """The non-writing production CLI remains runnable after extraction."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "validation/benchmark_free_boundary_strict_parity.py"),
            "--check",
            "--strict",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    assert json.loads(result.stdout)["accepted_full_fidelity"] is False


def test_benchmark_writer_uses_extracted_renderer(tmp_path: Path) -> None:
    """The production writer emits the same Markdown as the presentation module."""
    json_path, md_path = tmp_path / "report.json", tmp_path / "report.md"
    report = run_benchmark(json_report_path=json_path, md_report_path=md_path)
    assert json.loads(json_path.read_text()) == report
    assert md_path.read_text() == render_markdown(report)


def test_render_includes_per_resolution_discrepancies() -> None:
    """Errors at an individual resolution remain visible in the human report."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text())
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text())
    row = freegs["strict_free_boundary_parity_evidence"]["grid_convergence_evidence"]["cases"][0][
        "resolution_rows"
    ][0]
    row["threshold_checks"][0]["value"] = 1e9
    rendered = render_markdown(evaluate_strict_parity(freegs, metadata))
    assert "Grid `33` inconsistent metric: `psi_n_rmse`" in rendered
    assert "Grid `33` failed `psi_n_rmse`: `1000000000.0`" in rendered
    assert "Grid `33` reported failure count disagrees" in rendered
