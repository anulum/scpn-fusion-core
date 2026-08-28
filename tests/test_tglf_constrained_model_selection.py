# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Constrained TGLF Selection Tests
"""Behavioral tests for charge-constrained targets and corrected science gates."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, cast

import numpy as np
import pytest
from typing_extensions import Self

from scpn_fusion.core.tglf_surrogate_candidates import QuadraticPolynomialCandidate
from scpn_fusion.io._tglf_constrained_targets import (
    constrained_coordinate_names,
    encode_constrained_targets,
    particle_closure_summary,
    reconstruct_constrained_prediction,
)
from scpn_fusion.io._tglf_model_selection_data import TGLFModelStudyData
from scpn_fusion.io import tglf_constrained_model_selection as selection_module
from scpn_fusion.io.tglf_constrained_model_selection import (
    constrained_candidate_rank,
    constrained_eligibility,
    run_tglf_constrained_model_selection,
    write_tglf_constrained_selection_report,
)

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "run_tglf_constrained_model_selection.py"


class _FailingCandidate:
    """Deterministic injected numerical failure for orchestration tests."""

    def fit(
        self,
        features: np.ndarray[Any, np.dtype[np.float64]],
        targets: np.ndarray[Any, np.dtype[np.float64]],
    ) -> Self:
        """Fail before producing candidate state."""
        del features, targets
        raise RuntimeError("injected numerical failure")

    def predict(
        self, features: np.ndarray[Any, np.dtype[np.float64]]
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Reject prediction because fitting never completed."""
        del features
        raise RuntimeError("injected numerical failure")

    def state_bytes(self) -> int:
        """Reject state measurement because fitting never completed."""
        raise RuntimeError("injected numerical failure")


def _study() -> TGLFModelStudyData:
    rows = 9
    features = np.zeros((rows, 44), dtype=np.float64)
    targets = np.zeros((rows, 13), dtype=np.float64)
    targets[:, 0] = np.linspace(0.2, 1.0, rows)
    targets[:, 5] = np.linspace(-0.4, 0.4, rows)
    targets[:, 9] = np.linspace(0.1, 0.3, rows)
    charges = np.tile(np.asarray([-1.0, 1.0, 6.0]), (rows, 1))
    targets[:, 1] = targets[:, 5] + 6.0 * targets[:, 9]
    for slot in range(3):
        offset = 1 + 4 * slot
        targets[:, offset + 1] = 0.1 * (slot + 1) + features[:, 0]
        targets[:, offset + 2] = -0.02 * (slot + 1)
        targets[:, offset + 3] = np.linspace(-1.0, 2.0, rows) * (slot + 1)
    names = ["gamma_max"]
    for slot in range(3):
        names.extend(
            (
                f"species_{slot}.particle_gb",
                f"species_{slot}.energy_gb",
                f"species_{slot}.momentum_gb",
                f"species_{slot}.exchange_gb",
            )
        )
    return TGLFModelStudyData(
        features=features,
        targets=targets,
        active_targets=np.ones_like(targets, dtype=np.bool_),
        charges=charges,
        feature_names=tuple(f"feature_{index}" for index in range(44)),
        target_names=tuple(names),
        splits=("train",) * 3 + ("calibration",) * 3 + ("test",) * 3,
        strata=("interior", "boundary", "threshold") * 3,
        compositions=(
            "electron-deuterium",
            "electron-deuterium-tritium",
            "electron-deuterium-carbon",
        )
        * 3,
        groups=tuple(f"group-{index // 3}" for index in range(rows)),
        sample_indices=tuple(range(rows)),
        verification={"status": "passed", "plan_replay": True},
    )


def test_constrained_coordinates_reconstruct_exact_ambipolarity() -> None:
    """Candidate ion coordinates determine electron particle flux exactly."""
    data = _study()
    coordinates = encode_constrained_targets(data)
    assert np.all(coordinates[:, 1] == 0.0)
    assert constrained_coordinate_names(data)[1] == ("charge_weighted_particle_residual.fixed_zero")
    rows = np.asarray([1, 4, 7], dtype=np.int64)
    reconstructed = reconstruct_constrained_prediction(data, rows, coordinates[rows])
    np.testing.assert_allclose(reconstructed, data.targets[rows])
    closure = particle_closure_summary(data, rows, reconstructed)
    assert closure["maximum"] <= 1.0e-16


def test_constrained_reconstruction_zeroes_absent_species_and_rejects_bad_contracts() -> None:
    """Padding is deterministic and malformed charge/shape metadata fails closed."""
    data = _study()
    active = data.active_targets.copy()
    active[0, 9:13] = False
    charges = data.charges.copy()
    charges[0, 2] = 0.0
    two_species = replace(data, active_targets=active, charges=charges)
    prediction = np.ones((1, 13), dtype=np.float64)
    reconstructed = reconstruct_constrained_prediction(
        two_species, np.asarray([0], dtype=np.int64), prediction
    )
    assert np.all(reconstructed[0, 9:13] == 0.0)
    assert reconstructed[0, 1] == pytest.approx(reconstructed[0, 5])

    with pytest.raises(ValueError, match="shape"):
        reconstruct_constrained_prediction(
            data, np.asarray([0], dtype=np.int64), np.zeros((1, 12), dtype=np.float64)
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        reconstruct_constrained_prediction(
            data, np.asarray([[0]], dtype=np.int64), np.zeros((1, 13), dtype=np.float64)
        )
    with pytest.raises(ValueError, match="out-of-range"):
        reconstruct_constrained_prediction(
            data, np.asarray([99], dtype=np.int64), np.zeros((1, 13), dtype=np.float64)
        )

    bad_charges = data.charges.copy()
    bad_charges[:, 0] = 1.0
    invalid = replace(data, charges=bad_charges)
    with pytest.raises(ValueError, match="electron"):
        reconstruct_constrained_prediction(
            invalid, np.asarray([0], dtype=np.int64), np.zeros((1, 13), dtype=np.float64)
        )

    bad_ion_charges = data.charges.copy()
    bad_ion_charges[0, 1] = 0.0
    invalid = replace(data, charges=bad_ion_charges)
    with pytest.raises(ValueError, match="ion charges"):
        reconstruct_constrained_prediction(
            invalid, np.asarray([0], dtype=np.int64), np.zeros((1, 13), dtype=np.float64)
        )

    with pytest.raises(ValueError, match="prediction shape"):
        particle_closure_summary(
            data, np.asarray([0], dtype=np.int64), np.zeros((1, 12), dtype=np.float64)
        )


def test_coordinate_encoding_rejects_matrix_contract_drift() -> None:
    """Target and charge widths remain fixed before candidate fitting."""
    data = _study()
    with pytest.raises(ValueError, match="targets and target names"):
        encode_constrained_targets(replace(data, targets=np.zeros((9, 12), dtype=np.float64)))
    with pytest.raises(ValueError, match="charge matrix"):
        encode_constrained_targets(replace(data, charges=np.zeros((9, 2), dtype=np.float64)))


def _perfect_evaluation() -> dict[str, Any]:
    channel = {
        "active_rows": 3,
        "normalised_rmse": 0.0,
        "normalised_bias": 0.0,
        "sign_agreement": 1.0,
    }
    return {
        "failed_rows": 0,
        "channels": {"species_0.exchange_gb": dict(channel)},
        "by_stratum": {"threshold": {"channels": {"exchange": dict(channel)}}},
        "constrained_particle_closure": {"median": 0.0, "p95": 0.0, "maximum": 0.0},
    }


def test_corrected_gate_does_not_impose_exchange_zero_sum() -> None:
    """Signed exchange accuracy is gated, but raw species summation is not."""
    evaluation = _perfect_evaluation()
    evaluation["closure"] = {"exchange_prediction": {"p95": 1.0}}
    assert constrained_eligibility(evaluation) == (True, [])

    evaluation["channels"]["species_0.exchange_gb"]["normalised_rmse"] = 1.01
    eligible, reasons = constrained_eligibility(evaluation)
    assert not eligible
    assert any("exchange_gb normalised RMSE" in reason for reason in reasons)

    failed = _perfect_evaluation()
    failed["failed_rows"] = 1
    assert constrained_eligibility(failed) == (False, ["non-finite prediction rows"])

    absent_threshold = _perfect_evaluation()
    absent_threshold["by_stratum"] = {}
    absent_threshold["constrained_particle_closure"]["p95"] = 2.0e-12
    eligible, reasons = constrained_eligibility(absent_threshold)
    assert not eligible
    assert reasons == [
        "threshold stratum is absent",
        "particle closure p95 exceeds exact reconstruction bound 1e-12",
    ]


def test_candidate_rank_and_report_writer_are_deterministic(tmp_path: Path) -> None:
    """Ranking excludes exchange closure and the report writer replaces atomically."""
    report = {
        "calibration": {
            "summary": {
                "median_normalised_rmse": 0.1,
                "p95_normalised_rmse": 0.2,
                "worst_normalised_rmse": 0.3,
            },
            "constrained_particle_closure": {"p95": 0.0},
        },
        "state_bytes": 123,
        "latency_orientation": {"median_microseconds_per_row": 4.0},
    }
    assert constrained_candidate_rank(report) == (0.1, 0.2, 0.3, 0.0, 123, 4.0)
    destination = tmp_path / "result.json"
    assert write_tglf_constrained_selection_report({"status": "passed"}, destination) == (
        destination
    )
    assert json.loads(destination.read_text(encoding="utf-8"))["status"] == "passed"
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        write_tglf_constrained_selection_report({}, directory)


def test_cli_fails_closed_before_loading_an_invalid_corpus(tmp_path: Path) -> None:
    """The executable boundary rejects a symlinked source and reports JSON failure."""
    source = tmp_path / "source"
    source.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    lock = tmp_path / "lock.md"
    lock.write_text("frozen\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--dataset-root",
            str(alias),
            "--selection-lock",
            str(lock),
            "--output",
            str(tmp_path / "result.json"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "must not be a symlink" in json.loads(completed.stdout)["failures"][0]


def test_study_rejects_missing_lock_and_invalid_latency_before_corpus_access(
    tmp_path: Path,
) -> None:
    """Pre-computation custody and timing arguments fail before any source read."""
    with pytest.raises(ValueError, match="at least one selection lock"):
        run_tglf_constrained_model_selection(tmp_path, selection_lock_paths=())
    lock = tmp_path / "lock.md"
    lock.write_text("frozen\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least three"):
        run_tglf_constrained_model_selection(
            tmp_path, selection_lock_paths=(lock,), latency_repeats=2
        )
    with pytest.raises(ValueError, match="regular non-symlink"):
        run_tglf_constrained_model_selection(
            tmp_path, selection_lock_paths=(tmp_path / "missing.md",)
        )


def test_nonfinite_prediction_and_candidate_failures_remain_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real evaluation and candidate orchestration retain explicit failure evidence."""
    data = _study()
    calibration = np.asarray([3, 4, 5], dtype=np.int64)
    prediction = data.targets[calibration].copy()
    prediction[0, 0] = np.nan
    evaluation = selection_module._evaluate_prediction(
        data, calibration, prediction, np.ones(13, dtype=np.float64)
    )
    assert evaluation["failed_rows"] == 1
    assert evaluation["constrained_particle_closure"] is None

    factories = {
        "failing": ({}, _FailingCandidate()),
        "working": ({"ridge": 1.0e-3}, QuadraticPolynomialCandidate(ridge=1.0e-3)),
    }
    monkeypatch.setattr(selection_module, "_candidate_factories", lambda: factories)
    coordinates = encode_constrained_targets(data)
    reports, fitted, leader, eligible = selection_module._fit_candidates(
        data,
        coordinate_targets=coordinates,
        coordinate_scales=np.ones(13, dtype=np.float64),
        evaluation_scales=np.ones(13, dtype=np.float64),
        latency_repeats=3,
    )
    assert reports["failing"]["fit_failure"] == "RuntimeError: injected numerical failure"
    assert set(fitted) == {"working"}
    assert leader == "working"
    assert isinstance(eligible, list)

    monkeypatch.setattr(
        selection_module,
        "_candidate_factories",
        lambda: {"failing": ({}, _FailingCandidate())},
    )
    with pytest.raises(RuntimeError, match="all constrained TGLF candidates failed"):
        selection_module._fit_candidates(
            data,
            coordinate_targets=coordinates,
            coordinate_scales=np.ones(13, dtype=np.float64),
            evaluation_scales=np.ones(13, dtype=np.float64),
            latency_repeats=3,
        )


def test_implementation_and_expanded_custody_failures_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing source identity, split drift, plan drift and revision drift are rejected."""
    private_module = cast(Any, selection_module)
    monkeypatch.setattr(private_module.target_module, "__file__", None)
    with pytest.raises(RuntimeError, match="implementation module has no source file"):
        selection_module._implementation_contract()
    monkeypatch.undo()

    lock = tmp_path / "lock.md"
    lock.write_text("frozen\n", encoding="utf-8")
    monkeypatch.setattr(selection_module, "load_tglf_model_study_data", lambda _: _study())
    with pytest.raises(ValueError, match="expanded split rows differ"):
        run_tglf_constrained_model_selection(tmp_path, selection_lock_paths=(lock,))

    monkeypatch.setattr(
        selection_module,
        "EXPECTED_SPLIT_ROWS",
        {"train": 3, "calibration": 3, "test": 3},
    )
    (tmp_path / "plan.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"source": {"revision": "wrong"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="plan differs"):
        run_tglf_constrained_model_selection(tmp_path, selection_lock_paths=(lock,))

    (tmp_path / "plan.json").write_text(
        json.dumps(
            {
                "profile": "expanded",
                "seed": 20260828,
                "plan_sha256": selection_module.TGLF_EXPANDED_PLAN_SHA256,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="revision differs"):
        run_tglf_constrained_model_selection(tmp_path, selection_lock_paths=(lock,))
