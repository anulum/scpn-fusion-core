# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — tests of the class-conditional conformal stability sets
"""Class-conditional stability sets and their flux intervals, on hand-checkable inputs.

The synthetic certificate uses a zero-weight network, which predicts ln 2 (the softplus of
zero) for every row; with one prediction for all rows, both labels enter every set.
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
from validation import qlknn_conformal_stability_sets as sets

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


def _split(path: Path, seed: int, *, unstable: bool = True) -> None:
    rng = np.random.default_rng(seed)
    x = np.ones((ROWS, 10))
    x[:, 0] = np.linspace(0.1, 0.9, ROWS)
    x[:, 1], x[:, 2], x[:, 7] = 2.0, 3.0, 2.0
    y = rng.uniform(0.1, 4.0, (ROWS, 3)) if unstable else np.zeros((ROWS, 3))
    y[: ROWS // 2] = 0.0
    np.savez(path, X=x, Y=y, regimes=np.zeros(ROWS, dtype=np.int32))


def _fixture(tmp_path: Path, *, unstable: bool = True) -> Path:
    weights, calibration, holdout = (tmp_path / f"{n}.npz" for n in ("w", "cal", "test"))
    _weights(weights)
    _split(calibration, 1, unstable=unstable)
    _split(holdout, 2)
    payload = certificates.build_certificate(
        weights_path=weights, calibration_path=calibration, holdout_path=holdout, alpha=0.2
    )
    path = tmp_path / "certificate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_stability_sets_use_class_conditional_quantiles() -> None:
    calibration = np.array([0.0, 0.0, 0.1, 0.2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    unstable = calibration >= 1.0
    has_stable, has_unstable, quantiles = sets.stability_sets(
        np.concatenate([calibration] * 2),
        np.concatenate([unstable] * 2),
        np.array([0.05, 0.5, 3.0]),
        0.2,
    )
    # Stable scores y_hat: 8 rows (0, 0, 0.1, 0.2 twice), rank ceil(9 * 0.8) = 8 -> 0.2.
    # Unstable scores -y_hat: 12 rows, rank ceil(13 * 0.8) = 11 -> -1.0.
    assert quantiles == {"stable": 0.2, "unstable": -1.0}
    assert has_stable.tolist() == [True, False, False]
    assert has_unstable.tolist() == [False, False, True]


def test_flux_intervals_follow_the_set_and_abstain_on_an_empty_one() -> None:
    lower, upper = sets.flux_intervals(
        np.array([1.0, 1.0, 1.0, 1.0]),
        has_stable=np.array([True, False, True, False]),
        has_unstable=np.array([False, True, True, False]),
        q_flux=0.5,
    )
    assert lower[:3].tolist() == [0.0, 0.5, 0.0]
    assert upper[:3].tolist() == [0.0, 1.5, 1.5]
    assert lower[3] > upper[3]


def test_report_covers_both_classes_when_every_set_holds_both_labels(tmp_path: Path) -> None:
    report = sets.build_report(_fixture(tmp_path))

    assert report["holdout_rows"] == ROWS
    assert "after the variant comparison" in report["design_note"]
    assert len(report["source_revision"]) == 40
    for channel in report["channels"].values():
        assert channel["abstention_share"] == 0.0
        for label in ("stable", "unstable"):
            row = channel[label]
            assert row["n"] == ROWS // 2
            assert row["set_coverage"] == 1.0
            assert row["set_size_share"] == {"0": 0.0, "1": 0.0, "2": 1.0}
        assert channel["stable"]["flux_coverage"] == 1.0
        assert channel["stable"]["median_interval_width"] > 0.0
        resplits = channel["resplits"]
        assert resplits["unstable_set"]["mean"] == 1.0 and resplits["stable_set"]["std"] == 0.0
        flux = resplits["unstable_flux"]
        assert flux["min"] <= flux["mean"] <= flux["max"] <= 1.0


def test_report_refuses_a_class_without_calibration_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        sets.build_report(_fixture(tmp_path, unstable=False))


def test_report_refuses_inputs_that_differ_from_the_certificate(tmp_path: Path) -> None:
    certificate = _fixture(tmp_path)
    _split(tmp_path / "w.npz", 5)
    with pytest.raises(ValueError, match="differ from the certificate's digests"):
        sets.build_report(certificate)


def test_command_line_writes_both_reports(tmp_path: Path) -> None:
    certificate = _fixture(tmp_path)
    output, markdown = tmp_path / "sets.json", tmp_path / "sets.md"

    code = sets.main(
        ["--certificate", str(certificate), "--output", str(output), "--markdown", str(markdown)]
    )

    assert code == 0
    text = markdown.read_text(encoding="utf-8")
    assert text.splitlines()[:7] == list(sets.MARKDOWN_HEADER)
    assert "| chi_e | unstable | 30 | 1.000 | 1.000 |" in text
    assert f"Random halvings of the pooled rows ({len(sets.RESPLIT_SEEDS)} seeds)" in text
    assert "| chi_e | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |" in text
    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == sets.SCHEMA


def test_script_entry_point_runs_from_the_repository_root() -> None:
    """Run the script directly: it must put the repository on its own path to import tools."""
    root = Path(sets.__file__).resolve().parents[1]
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    completed = subprocess.run(
        [sys.executable, "validation/qlknn_conformal_stability_sets.py", "--help"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--certificate" in completed.stdout
