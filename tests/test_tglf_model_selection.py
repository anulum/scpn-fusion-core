# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TGLF Model Selection Tests
"""Frozen split, provenance, model-gate and CLI boundary tests."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from scpn_fusion.io import _tglf_model_selection_data as data_contract
from scpn_fusion.io import _tglf_model_selection_metrics as metric_contract
from scpn_fusion.io import tglf_model_selection as selection_module
from scpn_fusion.io.tglf_model_selection import (
    TGLFModelStudyData,
    load_tglf_model_study_data,
    run_tglf_model_selection,
    write_tglf_model_selection_report,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "validation" / "reference_data" / "tglf_gacode_development_v2_fixture"
CLI = ROOT / "tools" / "run_tglf_model_selection.py"


def test_private_data_contract_keeps_frozen_matrix_widths() -> None:
    """The decomposed loader retains the pre-fit 44-input/13-output contract."""
    assert len(data_contract.feature_names()) == 44
    assert len(data_contract.target_names()) == 13
    assert data_contract.feature_names()[:2] == ("rho", "s_hat")
    assert data_contract.target_names()[0] == "gamma_max"


def test_private_metric_contract_handles_exact_and_zero_closure() -> None:
    """The decomposed closure metric preserves exact cancellation semantics."""
    components = np.asarray([[1.0, -1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    np.testing.assert_array_equal(
        metric_contract.closure_ratio(components),
        np.asarray([0.0, 0.0], dtype=np.float64),
    )


def _closed_physics_study_data() -> TGLFModelStudyData:
    generator = np.random.default_rng(20260826)
    features = np.zeros((72, 44), dtype=np.float64)
    features[:, :11] = generator.uniform(-0.8, 0.8, size=(72, 11))
    particle_1 = 0.2 + 0.3 * features[:, 1]
    particle_2 = -0.1 + 0.2 * features[:, 2]
    particle_0 = particle_1 + particle_2
    exchange_0 = 0.15 * features[:, 3]
    exchange_1 = -0.05 * features[:, 4]
    exchange_2 = -(exchange_0 + exchange_1)
    targets = np.column_stack(
        (
            0.8 + 0.2 * features[:, 0],
            particle_0,
            0.4 + 0.1 * features[:, 5],
            0.05 * features[:, 6],
            exchange_0,
            particle_1,
            0.3 - 0.2 * features[:, 7],
            0.03 * features[:, 8],
            exchange_1,
            particle_2,
            0.2 + 0.1 * features[:, 9],
            -0.02 * features[:, 10],
            exchange_2,
        )
    )
    splits = ("train",) * 36 + ("calibration",) * 18 + ("test",) * 18
    strata = tuple(("interior", "boundary", "threshold")[index % 3] for index in range(72))
    compositions = tuple(
        (
            "electron-deuterium",
            "electron-deuterium-tritium",
            "electron-deuterium-carbon",
        )[index % 3]
        for index in range(72)
    )
    groups = tuple(f"study-{index // 3:03d}" for index in range(72))
    return TGLFModelStudyData(
        features=features,
        targets=targets,
        active_targets=np.ones_like(targets, dtype=np.bool_),
        charges=np.tile(np.asarray([-1.0, 1.0, 1.0]), (72, 1)),
        feature_names=tuple(f"feature_{index}" for index in range(44)),
        target_names=(
            "gamma_max",
            "species_0.particle_gb",
            "species_0.energy_gb",
            "species_0.momentum_gb",
            "species_0.exchange_gb",
            "species_1.particle_gb",
            "species_1.energy_gb",
            "species_1.momentum_gb",
            "species_1.exchange_gb",
            "species_2.particle_gb",
            "species_2.energy_gb",
            "species_2.momentum_gb",
            "species_2.exchange_gb",
        ),
        splits=splits,
        strata=strata,
        compositions=compositions,
        groups=groups,
        sample_indices=tuple(range(72)),
        verification={
            "status": "passed",
            "plan_replay": True,
            "dataset_id": "orchestration-matrix",
            "tree_sha256": "0" * 64,
        },
    )


def test_authentic_fixture_is_rejected_when_calibration_split_is_absent() -> None:
    """Verified official rows are still inadmissible without all frozen split roles."""
    with pytest.raises(ValueError, match="non-empty train/calibration/test"):
        load_tglf_model_study_data(FIXTURE)


def test_full_candidate_selection_uses_calibration_then_keeps_test_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full three-family evaluator selects before test and emits no promotion claim."""
    data = _closed_physics_study_data()
    monkeypatch.setattr(selection_module, "load_tglf_model_study_data", lambda _: data)
    lock = tmp_path / "selection-lock.md"
    lock.write_text("frozen before fit\n", encoding="utf-8")
    report = run_tglf_model_selection(
        FIXTURE,
        selection_lock_paths=(lock,),
        latency_repeats=3,
    )
    assert report["status"] == "passed"
    assert report["selection"]["calibration_leader"] == "quadratic_polynomial"
    assert report["selection"]["test_does_not_reselect"] is True
    assert set(report["candidates"]) == {
        "quadratic_polynomial",
        "randomised_tree_ensemble",
        "compact_neural_ensemble",
    }
    assert report["source"]["plan_design_sha256"] == (
        "102ba43f1f9a495d99e291bef36cdf6acde8e31809d0ba4a2d03392390080047"
    )
    assert len(report["implementation"]["candidate_module_sha256"]) == 64
    assert len(report["implementation"]["study_module_sha256"]) == 64
    assert all(value is False for value in report["admission"].values())
    repeated = run_tglf_model_selection(
        FIXTURE,
        selection_lock_paths=(lock,),
        latency_repeats=3,
    )
    assert repeated["scientific_projection_sha256"] == report["scientific_projection_sha256"]
    json.dumps(report, allow_nan=False)


def test_report_writer_is_atomic_strict_json_and_rejects_unsafe_targets(tmp_path: Path) -> None:
    """Report custody writes strict JSON and refuses symlink destinations."""
    report: dict[str, Any] = {"status": "failed", "reason": "scientific gate"}
    destination = tmp_path / "evidence" / "report.json"
    assert write_tglf_model_selection_report(report, destination) == destination
    assert json.loads(destination.read_text(encoding="utf-8")) == report
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    symlink = tmp_path / "linked.json"
    symlink.symlink_to(target)
    with pytest.raises(ValueError, match="non-symlink"):
        write_tglf_model_selection_report(report, symlink)
    with pytest.raises(ValueError, match="Out of range float values"):
        write_tglf_model_selection_report({"bad": float("nan")}, tmp_path / "bad.json")


def test_cli_fails_closed_without_calibration_and_does_not_write_report(tmp_path: Path) -> None:
    """The executable boundary reports corpus inadmissibility and leaves no artifact."""
    lock = tmp_path / "lock.md"
    lock.write_text("frozen\n", encoding="utf-8")
    output = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--dataset-root",
            str(FIXTURE),
            "--selection-lock",
            str(lock),
            "--output",
            str(output),
            "--latency-repeats",
            "3",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "non-empty train/calibration/test" in completed.stdout
    assert not output.exists()


def test_selection_lock_and_output_corpus_boundaries_fail_closed(tmp_path: Path) -> None:
    """Missing lock custody and attempts to mutate the source tree are rejected."""
    with pytest.raises(ValueError, match="at least one selection lock"):
        run_tglf_model_selection(FIXTURE, selection_lock_paths=())
    lock = tmp_path / "lock.md"
    lock.write_text("frozen\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least three"):
        run_tglf_model_selection(FIXTURE, selection_lock_paths=(lock,), latency_repeats=2)
    output = FIXTURE / "forbidden-model-report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--dataset-root",
            str(FIXTURE),
            "--selection-lock",
            str(tmp_path / "absent.md"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "outside the immutable source corpus" in completed.stdout
    assert not output.exists()
