# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — tests of the split-conformal variant comparison for the QLKNN surrogate
"""Split-conformal variants, calibrated and scored on small synthetic splits.

A zero-weight network predicts zero for every row, so each residual equals its target and
the quantiles can be checked against sorted values by hand.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools import generate_surrogate_conformal_certificates as certificates
from validation import qlknn_conformal_variants as variants

ROWS = 60


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


def _split(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = np.ones((ROWS, 10))
    x[:, 0] = np.linspace(0.1, 0.9, ROWS)
    x[:, 1] = rng.uniform(1.0, 10.0, ROWS)
    x[:, 2], x[:, 7] = 3.0, 2.0
    y = rng.uniform(0.1, 4.0, (ROWS, 3))
    y[: ROWS // 3] = 0.0
    np.savez(path, X=x, Y=y, regimes=np.tile([0, 1, 2], ROWS // 3).astype(np.int32))


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


def test_rank_quantile_is_the_finite_sample_order_statistic() -> None:
    scores = np.arange(1.0, 11.0)
    # ceil(11 * 0.8) = 9: the ninth smallest of ten.
    assert variants.rank_quantile(scores, 0.2) == 9.0
    with pytest.raises(ValueError, match="too small"):
        variants.rank_quantile(np.arange(3.0), 0.1)


def test_group_quantiles_calibrate_each_group_and_refuse_a_missing_one() -> None:
    scores = np.concatenate([np.arange(1.0, 11.0), np.arange(101.0, 111.0)])
    groups = np.repeat([0, 1], 10)
    assert variants.group_quantiles(scores, groups, np.array([0, 1]), 0.2) == {0: 9.0, 1: 109.0}
    with pytest.raises(ValueError, match=r"no calibration rows for groups \[2\]"):
        variants.group_quantiles(scores, groups, np.array([0, 2]), 0.2)


def test_evaluate_counts_coverage_widths_and_groups() -> None:
    result = variants.evaluate(
        holdout_scores=np.array([0.5, 2.0, 0.1, 0.3]),
        quantiles=np.array([1.0, 1.0, 1.0, 1.0]),
        scale=np.array([2.0, 2.0, 1.0, 1.0]),
        targets=np.array([0.0, 4.0, 1.0, 0.0]),
        labels=np.array([0, 1, 1, 2]),
        threshold=2.0,
    )
    assert result["marginal"]["covered"] == 3
    assert result["median_half_width"] == 1.5
    regimes = result["target_regimes"]
    assert regimes["stable"]["coverage"] == 1.0
    assert regimes["near_threshold"]["median_relative_width"] == 2.0
    assert regimes["unstable"]["coverage"] == 0.0
    assert result["input_labels"]["itg_label"]["n"] == 2


def test_report_reproduces_the_certificate_and_balances_mondrian_groups(tmp_path: Path) -> None:
    report = variants.build_report(_fixture(tmp_path))

    assert report["status"] == "pass", report["checks"]
    assert report["variants"] == list(variants.VARIANTS)
    assert len(report["source_revision"]) == 40
    for channel in report["channels"].values():
        assert set(channel["variants"]) == set(variants.VARIANTS)
        assert set(channel["quantiles"]["mondrian_input_label"]) == {
            "neither_label",
            "itg_label",
            "tem_label",
        }
        for result in channel["variants"].values():
            assert result["marginal"]["n"] == ROWS
            assert sum(g["n"] for g in result["input_labels"].values()) == ROWS


def test_report_refuses_inputs_that_differ_from_the_certificate(tmp_path: Path) -> None:
    certificate = _fixture(tmp_path)
    _split(tmp_path / "cal.npz", 9)
    with pytest.raises(ValueError, match="differ from the certificate's digests"):
        variants.build_report(certificate)


def test_command_line_writes_reports_and_fails_when_the_marginal_is_not_reproduced(
    tmp_path: Path,
) -> None:
    certificate = _fixture(tmp_path)
    output, markdown = tmp_path / "variants.json", tmp_path / "variants.md"
    arguments = ["--certificate", str(certificate), "--output", str(output)]

    assert variants.main([*arguments, "--markdown", str(markdown)]) == 0
    text = markdown.read_text(encoding="utf-8")
    assert text.splitlines()[:7] == list(variants.MARKDOWN_HEADER)
    assert "| chi_e | normalised_prediction |" in text

    payload = json.loads(certificate.read_text(encoding="utf-8"))
    payload["certificates"][0]["channels"][1]["radius"] = 0.0
    certificate.write_text(json.dumps(payload), encoding="utf-8")
    assert variants.main([*arguments, "--markdown", str(markdown)]) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "fail"


def test_script_entry_point_runs_from_the_repository_root() -> None:
    """Run the script directly: it must put the repository on its own path to import tools."""
    root = Path(variants.__file__).resolve().parents[1]
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    completed = subprocess.run(
        [sys.executable, "validation/qlknn_conformal_variants.py", "--help"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--certificate" in completed.stdout
