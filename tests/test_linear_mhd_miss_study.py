# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — tests of the linear reduced-MHD miss study
"""Contract tests with a stand-in growth rate, and one live call of the real solver."""

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

from scpn_fusion.core.linear_mhd_1d import Geometry1D, harris_slab
from validation import linear_mhd_miss_study as study


def test_local_slopes_are_log_derivatives_between_neighbours() -> None:
    slopes = study.local_slopes((1e-6, 1e-7, 1e-8), [1e-3, 1e-3 * 10**-0.6, 1e-3 * 10**-1.1])
    assert slopes == pytest.approx([0.6, 0.5])


def test_slope_scan_reports_every_resistivity_and_the_refinement() -> None:
    scan = study.slope_scan(lambda geom, eta, elements: eta**0.6 * (1.0 + 1e-3 * (elements == 400)))

    assert scan["etas"] == list(study.SLOPE_ETAS)
    assert len(scan["growth_rates"]) == len(study.SLOPE_ETAS)
    assert scan["local_slopes"] == pytest.approx([0.6] * 4)
    assert scan["smallest_eta_doubling_change"] == pytest.approx(1e-3)
    assert scan["delta_prime"] == pytest.approx(9.6947, rel=1e-4)
    ratios = scan["ratio_to_asymptotic"]
    assert max(ratios) / min(ratios) == pytest.approx(1.0)


def test_root_scan_reports_roots_by_element_count_against_the_tearing_rate() -> None:
    calls: list[tuple[float, float, int]] = []

    def grow(geom: Geometry1D, eta: float, elements: int) -> float:
        calls.append((float(geom.kappa_sq(np.array([0.0]))[0]), eta, elements))
        return eta if elements == 200 else eta / 2.0

    scan = study.root_scan(grow)

    assert scan["roots_by_elements"]["200"] == list(study.ROOT_ETAS)
    assert scan["roots_by_elements"]["400"] == [eta / 2.0 for eta in study.ROOT_ETAS]
    assert scan["root_over_tearing"] == [1.0, 1.0, 1.0]
    assert len(calls) == 3 * len(study.ROOT_ETAS)
    assert [call[0] for call in calls] == pytest.approx([1.44] * 6 + [0.25] * 3)


def test_report_gates_on_finite_solves_only() -> None:
    report = study.build_report(lambda geom, eta, elements: eta**0.6)
    assert report["status"] == "pass"
    assert len(report["provenance"]["source_revision"]) == 40
    assert set(report["provenance"]["source_sha256"]) == set(study.SOURCE_PATHS)

    failed = study.build_report(lambda geom, eta, elements: math.nan)
    assert failed["status"] == "fail"
    assert failed["checks"] == {"every_solve_finite": False}


def test_command_line_writes_both_records(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output, markdown = tmp_path / "study.json", tmp_path / "study.md"
    original = study.build_report
    monkeypatch.setattr(study, "build_report", lambda: original(lambda g, e, n: e**0.6))

    assert study.main(["--output", str(output), "--markdown", str(markdown)]) == 0

    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == study.SCHEMA
    text = markdown.read_text(encoding="utf-8")
    assert text.splitlines()[:7] == list(study.MARKDOWN_HEADER)
    assert "| 1e-10 |" in text and "| 1e-06 |" in text
    assert "Doubling the elements at η = 1e-10" in text


def test_command_line_exits_nonzero_on_a_failed_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = study.build_report
    monkeypatch.setattr(study, "build_report", lambda: original(lambda g, e, n: math.inf))
    arguments = ["--output", str(tmp_path / "s.json"), "--markdown", str(tmp_path / "s.md")]
    assert study.main(arguments) == 1


def test_live_growth_rate_wiring_on_a_coarse_unstable_sheet() -> None:
    """The real solver path returns a positive tearing rate for the unstable sheet."""
    rate = study.solver_growth_rate(harris_slab(0.5, 20.0), 1e-4, 40)
    assert math.isfinite(rate) and rate > 0.0


def test_script_entry_point_runs_from_the_repository_root() -> None:
    root = Path(study.__file__).resolve().parents[1]
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    completed = subprocess.run(
        [sys.executable, "validation/linear_mhd_miss_study.py", "--help"],
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
