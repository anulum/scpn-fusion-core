# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — tests of the near-zero spectrum diagnostic of the linear reduced-MHD pencil
"""Contract tests on small pencils with known spectra, and one coarse run of the real pencil."""

from __future__ import annotations

import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from validation import linear_mhd_near_zero_spectrum as study

COARSE = 12


def _diagonal(values: list[complex]) -> tuple[np.ndarray, np.ndarray]:
    a = np.diag(np.asarray(values, dtype=np.complex128))
    return a, np.eye(len(values), dtype=np.complex128)


def test_near_zero_records_list_the_smallest_with_condition_and_bound() -> None:
    a, b = _diagonal([1e-6, -2e-6, 5e-5 + 1e-6j, 1.0])

    count, records = study.near_zero_records(a, b, limit=1e-4, listed=2)

    assert count == 3
    assert [record["real"] for record in records] == pytest.approx([1e-6, -2e-6])
    assert all(record["condition"] == pytest.approx(1.0) for record in records)
    eps = float(np.finfo(np.float64).eps)
    assert records[0]["error_bound"] == pytest.approx(eps * (1.0 + 1e-6))
    assert all(record["relative_residual"] <= 1e-15 for record in records)


def test_largest_real_part_counts_roots_above_the_floor() -> None:
    a, b = _diagonal([2e-12, 5e-13, -1.0, 3e-3])

    largest, unstable = study.largest_real_part(a, b)

    assert largest == pytest.approx(3e-3)
    assert unstable == 2


def test_perturbation_trials_are_seeded_and_vanish_at_zero_scale() -> None:
    a, b = _diagonal([1e-3, -1.0, 0.5j])

    first = study.perturbation_trials(a, b, trials=2, scale=1e-6, seed=7)
    again = study.perturbation_trials(a, b, trials=2, scale=1e-6, seed=7)
    exact = study.perturbation_trials(a, b, trials=1, scale=0.0)

    assert first == again and len(first) == 2
    assert exact == [{"largest_real_part": pytest.approx(1e-3), "unstable_count": 1}]


def test_harris_pencil_is_the_acceptance_problem_at_the_requested_size() -> None:
    a, b = study.harris_pencil(COARSE)

    assert a.shape == b.shape == (8 * COARSE, 8 * COARSE)
    assert np.allclose(b, b.conj().T)


def test_report_on_a_coarse_pencil_records_every_measurement() -> None:
    report = study.build_report(COARSE)

    assert report["case"]["elements"] == COARSE and report["unknowns"] == 8 * COARSE
    assert report["singular_values_a"]["largest"] >= report["singular_values_a"]["smallest"]
    assert len(report["smallest_eigenvalues"]) <= study.LISTED
    assert len(report["perturbation"]["trials"]) == study.TRIALS
    assert report["checks"]["every_value_finite"]
    assert report["status"] == ("pass" if all(report["checks"].values()) else "fail")
    assert set(report["provenance"]["source_sha256"]) == set(study.SOURCE_PATHS)
    assert isinstance(report["positive_listed_inside_error_bound"], bool)


def test_report_fails_when_a_measurement_is_not_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(study, "largest_real_part", lambda a, b: (math.nan, 0))
    report = study.build_report(COARSE)

    assert report["status"] == "fail"
    assert report["checks"]["every_value_finite"] is False


def test_command_line_writes_both_records(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original = study.build_report
    monkeypatch.setattr(study, "build_report", lambda: original(COARSE))
    output, markdown = tmp_path / "s.json", tmp_path / "s.md"

    code = study.main(["--output", str(output), "--markdown", str(markdown)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == study.SCHEMA
    assert code == (0 if report["status"] == "pass" else 1)
    text = markdown.read_text(encoding="utf-8")
    assert text.splitlines()[:7] == list(study.MARKDOWN_HEADER)
    assert "| Re λ | Im λ | Condition | Error bound | Residual |" in text
    assert "After relative perturbations of 1e-15 to A" in text


def test_script_entry_point_runs_from_the_repository_root() -> None:
    root = Path(study.__file__).resolve().parents[1]
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    completed = subprocess.run(
        [sys.executable, "validation/linear_mhd_near_zero_spectrum.py", "--help"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--markdown" in completed.stdout


def test_import_puts_the_source_tree_on_the_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Imported from outside the repository, the module adds ``src`` to the path itself."""
    source = str(study.ROOT / "src")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != source])
    importlib.reload(study)
    assert sys.path[0] == source
