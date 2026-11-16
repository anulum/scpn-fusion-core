# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — tests of the per-regime coverage report of the QLKNN certificate
"""Per-regime coverage of a split-conformal certificate, on small synthetic splits.

A zero-weight network predicts zero for every row, so each residual is the target itself
and every coverage figure can be counted by hand.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools import generate_surrogate_conformal_certificates as certificates
from validation import qlknn_conformal_regime_coverage as coverage

ROWS = 40


def _weights(path: Path) -> None:
    np.savez(
        path,
        input_mean=np.zeros(10),
        input_std=np.ones(10),
        output_scale=np.ones(3),
        version=np.asarray(1),
        log_transform=np.asarray(0),
        gb_scale=np.asarray(0),
        gated=np.asarray(0),
        w1=np.zeros((10, 4)),
        b1=np.zeros(4),
        w2=np.zeros((4, 3)),
        b2=np.zeros(3),
    )


def _split(path: Path, seed: int, labels: np.ndarray | None = None) -> None:
    rng = np.random.default_rng(seed)
    x = np.ones((ROWS, 10))
    x[:, 0] = np.linspace(0.1, 0.9, ROWS)
    x[:, 1], x[:, 2], x[:, 7] = 2.0, 3.0, 2.0
    y = rng.uniform(0.1, 4.0, (ROWS, 3))
    y[: ROWS // 2] = 0.0
    regimes = np.tile([0, 1], ROWS // 2) if labels is None else labels
    np.savez(path, X=x, Y=y, regimes=regimes.astype(np.int32))


def _fixture(tmp_path: Path) -> Path:
    weights, calibration, holdout = (tmp_path / f"{n}.npz" for n in ("w", "cal", "test"))
    _weights(weights)
    _split(calibration, 1)
    _split(holdout, 2)
    payload = certificates.build_certificate(
        weights_path=weights, calibration_path=calibration, holdout_path=holdout, alpha=0.2
    )
    path = tmp_path / "certificate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_wilson_interval_matches_the_closed_form_and_refuses_bad_counts() -> None:
    low, high = coverage.wilson_interval(9, 10)
    assert (low, high) == pytest.approx((0.5958, 0.9821), abs=1e-4)
    assert coverage.wilson_interval(0, 5)[0] == 0.0
    assert coverage.wilson_interval(5, 5)[1] == pytest.approx(1.0)
    for covered, total in ((1, 0), (-1, 3), (4, 3)):
        with pytest.raises(ValueError, match="covered <= total"):
            coverage.wilson_interval(covered, total)


def test_target_regimes_partition_every_row() -> None:
    groups = coverage.target_regimes(np.array([0.0, 0.5, 1.0, 3.0]), threshold=1.0)
    assert {name: mask.tolist() for name, mask in groups.items()} == {
        "stable": [True, False, False, False],
        "near_threshold": [False, True, True, False],
        "unstable": [False, False, False, True],
    }


def test_group_coverage_counts_and_withholds_empty_groups() -> None:
    covered = np.array([True, False, True, True])
    widths = np.array([np.nan, 2.0, 4.0, np.nan])
    group = coverage.group_coverage(covered, np.array([True, True, True, False]), widths)
    assert group["n"] == 3 and group["covered"] == 2
    assert group["coverage"] == pytest.approx(2 / 3)
    assert group["median_relative_width"] == 3.0
    stable = coverage.group_coverage(covered, np.array([True, False, False, True]), widths)
    assert stable["median_relative_width"] is None
    empty = coverage.group_coverage(covered, np.zeros(4, dtype=bool), widths)
    assert empty == {
        "n": 0,
        "covered": 0,
        "coverage": None,
        "wilson95": None,
        "median_relative_width": None,
    }


def test_report_reproduces_the_certificate_and_counts_each_group(tmp_path: Path) -> None:
    report = coverage.build_report(_fixture(tmp_path))

    assert report["status"] == "pass", report["checks"]
    assert report["holdout_rows"] == ROWS
    assert len(report["source_revision"]) == 40
    for channel in report["channels"].values():
        labels = channel["input_labels"]
        assert sum(group["n"] for group in labels.values()) == ROWS
        assert labels["tem_label"]["n"] == 0
        regimes = channel["target_regimes"]
        assert sum(group["n"] for group in regimes.values()) == ROWS
        # Zero predictions cover every zero target.
        assert regimes["stable"]["n"] == ROWS // 2 and regimes["stable"]["coverage"] == 1.0
        assert regimes["stable"]["median_relative_width"] is None
        assert math.isfinite(channel["near_threshold_upper_value"])


def test_report_refuses_inputs_that_differ_from_the_certificate(tmp_path: Path) -> None:
    certificate = _fixture(tmp_path)
    _split(tmp_path / "test.npz", 3)
    with pytest.raises(ValueError, match="differ from the certificate's digests"):
        coverage.build_report(certificate)


def test_report_refuses_unknown_input_labels(tmp_path: Path) -> None:
    certificate = _fixture(tmp_path)
    payload = json.loads(certificate.read_text(encoding="utf-8"))
    holdout = tmp_path / "test.npz"
    _split(holdout, 2, labels=np.full(ROWS, 7))
    payload["certificates"][0]["holdout"]["split_sha256"] = hashlib.sha256(
        holdout.read_bytes()
    ).hexdigest()
    certificate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="one label in"):
        coverage.build_report(certificate)


def test_command_line_writes_both_reports_and_fails_on_an_unreproduced_coverage(
    tmp_path: Path,
) -> None:
    certificate = _fixture(tmp_path)
    output, markdown = tmp_path / "report.json", tmp_path / "report.md"
    arguments = ["--certificate", str(certificate), "--output", str(output)]

    assert coverage.main([*arguments, "--markdown", str(markdown)]) == 0
    text = markdown.read_text(encoding="utf-8")
    assert text.splitlines()[:7] == list(coverage.MARKDOWN_HEADER)
    assert "| chi_e | tem_label | — (0 rows) | — |" in text
    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == coverage.SCHEMA

    payload = json.loads(certificate.read_text(encoding="utf-8"))
    payload["certificates"][0]["channels"][0]["holdout_empirical_coverage"] = 0.0
    certificate.write_text(json.dumps(payload), encoding="utf-8")
    assert coverage.main([*arguments, "--markdown", str(markdown)]) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["checks"]["marginal_coverage_reproduces_certificate"] is False
    assert report["status"] == "fail"


def test_script_entry_point_runs_from_the_repository_root() -> None:
    """Run the script directly: it must put the repository on its own path to import tools."""
    root = Path(coverage.__file__).resolve().parents[1]
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    completed = subprocess.run(
        [sys.executable, "validation/qlknn_conformal_regime_coverage.py", "--help"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--certificate" in completed.stdout
