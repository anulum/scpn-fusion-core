# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TGLF Model Selection Tests
"""Frozen split, provenance, model-gate and CLI boundary tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
from typing_extensions import Self

import scpn_fusion.core.tglf_surrogate_candidates as candidate_module
from scpn_fusion.core.tglf_surrogate_candidates import QuadraticPolynomialCandidate
from scpn_fusion.io import _tglf_model_selection_data as data_contract
from scpn_fusion.io import _tglf_model_selection_metrics as metric_contract
from scpn_fusion.io import tglf_model_selection as selection_module
from scpn_fusion.io.tglf_model_selection import (
    TGLFModelStudyData,
    load_tglf_model_study_data,
    run_tglf_model_selection,
    write_tglf_model_selection_report,
)
from scpn_fusion.io.tglf_species_dataset_contract import (
    build_tglf_species_dataset_manifest,
    write_tglf_species_dataset_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "validation" / "reference_data" / "tglf_gacode_development_v2_fixture"
CLI = ROOT / "tools" / "run_tglf_model_selection.py"


class _FailingCandidate:
    """Candidate double used only to exercise fail-closed orchestration."""

    def fit(
        self,
        features: np.ndarray[Any, np.dtype[np.float64]],
        targets: np.ndarray[Any, np.dtype[np.float64]],
    ) -> Self:
        """Fail as an injected candidate family would fail numerically."""
        raise RuntimeError("injected fit failure")

    def predict(
        self, features: np.ndarray[Any, np.dtype[np.float64]]
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Reject prediction because this candidate never fits."""
        raise RuntimeError("injected candidate is not fit")

    def state_bytes(self) -> int:
        """Reject state reporting because this candidate never fits."""
        raise RuntimeError("injected candidate is not fit")


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


def _authentic_record(index: int = 0) -> dict[str, Any]:
    """Return an independent authentic fixture record for row-contract tests."""
    records = json.loads((FIXTURE / "dataset.json").read_text(encoding="utf-8"))
    return deepcopy(records[index])


def _fast_candidate_result(
    data: TGLFModelStudyData,
    *,
    target_scales: np.ndarray[Any, np.dtype[np.float64]],
    latency_repeats: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, _FailingCandidate], str, list[str]]:
    """Return a frozen failed scientific report for orchestration-only tests."""
    del data, target_scales, latency_repeats
    report = {
        "injected_candidate": {
            "calibration_eligible": False,
            "test_gate_passed": False,
            "calibration_ineligibility_reasons": ["injected orchestration fixture"],
        }
    }
    return report, {}, "injected_candidate", []


def _write_orchestration_corpus(
    root: Path,
    *,
    plan_digest: str,
    manifest_digest: str,
    plan_revision: str,
    manifest_revision: str,
) -> None:
    """Write bounded custody files used after an injected verified data boundary."""
    root.mkdir()
    (root / "plan.json").write_text(
        json.dumps({"plan_sha256": plan_digest, "gacode_revision": plan_revision}),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "development": {"plan_sha256": manifest_digest},
                "source": {"revision": manifest_revision},
            }
        ),
        encoding="utf-8",
    )
    (root / "dataset.json").write_text("[]\n", encoding="utf-8")


def test_authentic_fixture_is_rejected_when_calibration_split_is_absent() -> None:
    """Verified official rows are still inadmissible without all frozen split roles."""
    with pytest.raises(ValueError, match="non-empty train/calibration/test"):
        load_tglf_model_study_data(FIXTURE)


def test_authentic_rows_cover_fixed_width_species_and_scalar_contracts() -> None:
    """Real two- and three-species records exercise the complete row conversion."""
    two_species = _authentic_record(0)
    row_x, row_y, active, charges = data_contract.record_row(two_species, row_index=0)
    assert len(row_x) == 44
    assert len(row_y) == len(active) == 13
    assert len(charges) == 3
    assert active[-4:] == [False, False, False, False]

    three_species = _authentic_record(3)
    row_x, row_y, active, charges = data_contract.record_row(three_species, row_index=3)
    assert len(row_x) == 44
    assert len(row_y) == len(active) == 13
    assert len(charges) == 3
    assert all(active)

    for invalid in (True, "1.0", float("nan")):
        record = _authentic_record()
        record["input"]["rho"] = invalid
        with pytest.raises(ValueError, match="numeric|finite"):
            data_contract.record_row(record, row_index=0)
    with pytest.raises(ValueError, match="object"):
        data_contract.object_value([], "payload")


def test_authentic_row_conversion_rejects_metadata_and_species_drift() -> None:
    """Mutated authentic rows fail at each model-specific conversion boundary."""
    record = _authentic_record()
    record["composition"] = "invalid"
    with pytest.raises(ValueError, match="composition"):
        data_contract.record_row(record, row_index=0)

    record = _authentic_record()
    record["input"]["use_bper"] = 1
    with pytest.raises(ValueError, match="boolean"):
        data_contract.record_row(record, row_index=0)

    for species in (
        [_authentic_record()["input"]["species"][0]],
        [*(_authentic_record()["input"]["species"] * 2)],
    ):
        record = _authentic_record()
        record["input"]["species"] = species
        with pytest.raises(ValueError, match="two or three species"):
            data_contract.record_row(record, row_index=0)

    record = _authentic_record()
    record["output"]["species_fluxes"] = record["output"]["species_fluxes"][:1]
    with pytest.raises(ValueError, match="flux count"):
        data_contract.record_row(record, row_index=0)

    record = _authentic_record()
    record["output"]["species_fluxes"][0]["species_index"] = 1
    with pytest.raises(ValueError, match="ordering"):
        data_contract.record_row(record, row_index=0)


def test_public_loader_rejects_a_real_checksum_failure(tmp_path: Path) -> None:
    """The loader surfaces a failure from the canonical complete-tree verifier."""
    corpus = tmp_path / "corpus"
    shutil.copytree(FIXTURE, corpus)
    dataset = corpus / "dataset.json"
    dataset.write_text(dataset.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="verification failed"):
        load_tglf_model_study_data(corpus)


def test_canonical_manifest_variant_loads_all_three_split_roles(tmp_path: Path) -> None:
    """A checksum-valid authentic fixture reaches the public loader success boundary."""
    corpus = tmp_path / "corpus"
    shutil.copytree(FIXTURE, corpus)
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    records = json.loads((corpus / "dataset.json").read_text(encoding="utf-8"))
    rebuilt = build_tglf_species_dataset_manifest(
        corpus,
        records,
        dataset_id=manifest["dataset_id"],
        gacode_revision=manifest["source"]["revision"],
        seed=10,
        development=manifest["development"],
        plan_file="plan.json",
        rejections_file="rejections.json",
    )
    write_tglf_species_dataset_manifest(corpus, rebuilt)
    data = load_tglf_model_study_data(corpus)
    assert data.features.shape == (9, 44)
    assert data.targets.shape == data.active_targets.shape == (9, 13)
    assert data.charges.shape == (9, 3)
    assert {split: data.splits.count(split) for split in data_contract.SPLITS} == {
        "train": 3,
        "calibration": 3,
        "test": 3,
    }
    assert data.verification["status"] == "passed"


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

    replacement = tmp_path / "replacement.json"
    replacement.write_text('{"old": true}\n', encoding="utf-8")
    write_tglf_model_selection_report({"new": True}, replacement)
    assert json.loads(replacement.read_text(encoding="utf-8")) == {"new": True}

    unsafe_temporary = tmp_path / ".unsafe.json.tmp"
    unsafe_temporary.mkdir()
    with pytest.raises(ValueError, match="temporary output path is unsafe"):
        write_tglf_model_selection_report(report, tmp_path / "unsafe.json")


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


def test_metric_contract_covers_inactive_near_zero_and_missing_categories() -> None:
    """Metric aggregation keeps inactive and legitimately absent slices explicit."""
    data = _closed_physics_study_data()
    active = data.active_targets.copy()
    active[:, -1] = False
    data = replace(data, active_targets=active)
    row_indices = np.asarray([0, 3, 6], dtype=np.int64)
    prediction = data.targets[row_indices].copy()
    evaluation = metric_contract.evaluate_rows(
        data,
        row_indices,
        prediction,
        np.ones(data.targets.shape[1], dtype=np.float64),
    )
    assert evaluation["channels"][data.target_names[-1]]["active_rows"] == 0
    assert evaluation["channels"][data.target_names[-1]]["sign_agreement"] is None
    assert set(evaluation["by_stratum"]) == {"interior"}
    assert set(evaluation["by_composition"]) == {"electron-deuterium"}

    zeros = np.zeros((2, 1), dtype=np.float64)
    near_zero = metric_contract.channel_metrics(
        zeros,
        zeros,
        np.ones_like(zeros, dtype=np.bool_),
        np.ones(1, dtype=np.float64),
        ("zero",),
    )
    assert near_zero["zero"]["sign_agreement"] is None
    with pytest.raises(ValueError, match="no active target"):
        metric_contract.channel_summary({"inactive": {"normalised_rmse": None}})


def test_metric_evaluation_rejects_shape_and_counts_nonfinite_rows() -> None:
    """Malformed or non-finite predictions cannot acquire aggregate metrics."""
    data = _closed_physics_study_data()
    row_indices = np.asarray([0, 1], dtype=np.int64)
    scales = np.ones(data.targets.shape[1], dtype=np.float64)
    with pytest.raises(ValueError, match="prediction shape"):
        metric_contract.evaluate_rows(data, row_indices, np.zeros((2, 1)), scales)
    prediction = data.targets[row_indices].copy()
    prediction[0, 0] = np.nan
    evaluation = metric_contract.evaluate_rows(data, row_indices, prediction, scales)
    assert evaluation["failed_rows"] == 1
    assert evaluation["summary"] is None
    assert metric_contract.eligibility(evaluation) == (False, ["non-finite prediction rows"])


def test_eligibility_reports_every_gate_and_accepts_exact_boundaries() -> None:
    """Every frozen eligibility reason is independent and equality remains admitted."""
    evaluation: dict[str, Any] = {
        "failed_rows": 0,
        "channels": {
            "flux": {
                "normalised_rmse": metric_contract.CHANNEL_NRMSE_MAX + 0.01,
                "normalised_bias": metric_contract.ABS_NORMALISED_BIAS_MAX + 0.01,
                "sign_agreement": metric_contract.SIGN_AGREEMENT_MIN - 0.01,
            }
        },
        "by_stratum": {
            "threshold": {
                "channels": {
                    "flux": {"normalised_rmse": metric_contract.THRESHOLD_NRMSE_MAX + 0.01}
                }
            }
        },
        "closure": {
            "charge_weighted_particle_prediction": {"p95": metric_contract.CLOSURE_P95_MAX + 0.01},
            "exchange_prediction": {"p95": metric_contract.CLOSURE_P95_MAX + 0.01},
        },
    }
    passed, reasons = metric_contract.eligibility(evaluation)
    assert passed is False
    assert len(reasons) == 6
    assert any("RMSE" in reason for reason in reasons)
    assert any("bias" in reason for reason in reasons)
    assert any("sign agreement" in reason for reason in reasons)
    assert any("threshold" in reason for reason in reasons)
    assert any("particle closure" in reason for reason in reasons)
    assert any("exchange closure" in reason for reason in reasons)

    evaluation["channels"]["flux"] = {
        "normalised_rmse": metric_contract.CHANNEL_NRMSE_MAX,
        "normalised_bias": -metric_contract.ABS_NORMALISED_BIAS_MAX,
        "sign_agreement": metric_contract.SIGN_AGREEMENT_MIN,
    }
    evaluation["by_stratum"]["threshold"]["channels"]["flux"] = {
        "normalised_rmse": metric_contract.THRESHOLD_NRMSE_MAX
    }
    evaluation["closure"]["charge_weighted_particle_prediction"]["p95"] = (
        metric_contract.CLOSURE_P95_MAX
    )
    evaluation["closure"]["exchange_prediction"]["p95"] = metric_contract.CLOSURE_P95_MAX
    assert metric_contract.eligibility(evaluation) == (True, [])

    evaluation["by_stratum"] = {}
    assert metric_contract.eligibility(evaluation) == (False, ["threshold stratum is absent"])


def test_candidate_rank_and_latency_contract_are_frozen() -> None:
    """Lexicographic ordering and minimum timing cohort remain exact."""
    report: dict[str, Any] = {
        "calibration": {
            "summary": {
                "median_normalised_rmse": 1.0,
                "p95_normalised_rmse": 2.0,
                "worst_normalised_rmse": 3.0,
            },
            "closure": {
                "charge_weighted_particle_prediction": {"p95": 0.1},
                "exchange_prediction": {"p95": 0.2},
            },
        },
        "state_bytes": 64,
        "latency_orientation": {"median_microseconds_per_row": 4.0},
    }
    rank = metric_contract.candidate_rank(report)
    assert rank[:3] == (1.0, 2.0, 3.0)
    assert rank[3] == pytest.approx(0.3)
    assert rank[4:] == (64, 4.0)

    features = np.arange(12, dtype=np.float64).reshape(4, 3)
    targets = np.arange(8, dtype=np.float64).reshape(4, 2)
    candidate = QuadraticPolynomialCandidate().fit(features, targets)
    with pytest.raises(ValueError, match="at least three"):
        metric_contract.latency_orientation(candidate, features, repeats=2)
    measured = metric_contract.latency_orientation(candidate, features, repeats=3)
    assert measured["batch_rows"] == 4
    assert measured["repeats"] == 3
    assert measured["minimum_microseconds_per_row"] <= measured["maximum_microseconds_per_row"]


def test_candidate_failure_is_isolated_and_all_failures_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One injected family failure is retained while remaining real candidates run."""
    data = _closed_physics_study_data()
    scales = np.ones(data.targets.shape[1], dtype=np.float64)
    monkeypatch.setattr(
        selection_module,
        "_candidate_factories",
        lambda: {
            "broken": ({}, _FailingCandidate()),
            "quadratic": ({"ridge": 1.0e-3}, QuadraticPolynomialCandidate()),
        },
    )
    reports, fitted, leader, _ = selection_module._fit_candidates(
        data, target_scales=scales, latency_repeats=3
    )
    assert reports["broken"]["fit_failure"] == "RuntimeError: injected fit failure"
    assert "quadratic" in fitted
    assert leader == "quadratic"

    monkeypatch.setattr(
        selection_module,
        "_candidate_factories",
        lambda: {"broken": ({}, _FailingCandidate())},
    )
    with pytest.raises(RuntimeError, match="all TGLF model candidates failed"):
        selection_module._fit_candidates(data, target_scales=scales, latency_repeats=3)


def test_implementation_and_selection_custody_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing source custody, unsafe locks and plan/manifest drift are rejected."""
    monkeypatch.setattr(candidate_module, "__file__", None)
    with pytest.raises(RuntimeError, match="no source file"):
        selection_module._implementation_contract()
    monkeypatch.undo()

    directory_lock = tmp_path / "directory-lock"
    directory_lock.mkdir()
    with pytest.raises(ValueError, match="regular non-symlink"):
        run_tglf_model_selection(FIXTURE, selection_lock_paths=(directory_lock,))
    lock_target = tmp_path / "lock-target.md"
    lock_target.write_text("frozen\n", encoding="utf-8")
    lock_link = tmp_path / "lock-link.md"
    lock_link.symlink_to(lock_target)
    with pytest.raises(ValueError, match="regular non-symlink"):
        run_tglf_model_selection(FIXTURE, selection_lock_paths=(lock_link,))

    monkeypatch.setattr(
        selection_module, "load_tglf_model_study_data", lambda _: _closed_physics_study_data()
    )
    monkeypatch.setattr(selection_module, "_fit_candidates", _fast_candidate_result)
    plan_drift = tmp_path / "plan-drift"
    _write_orchestration_corpus(
        plan_drift,
        plan_digest="a",
        manifest_digest="b",
        plan_revision="r",
        manifest_revision="r",
    )
    with pytest.raises(ValueError, match="development digests differ"):
        run_tglf_model_selection(plan_drift, selection_lock_paths=(lock_target,), latency_repeats=3)

    revision_drift = tmp_path / "revision-drift"
    _write_orchestration_corpus(
        revision_drift,
        plan_digest="a",
        manifest_digest="a",
        plan_revision="r1",
        manifest_revision="r2",
    )
    with pytest.raises(ValueError, match="GACODE revisions differ"):
        run_tglf_model_selection(
            revision_drift, selection_lock_paths=(lock_target,), latency_repeats=3
        )


def test_runtime_context_portable_fallbacks_are_serialisable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public report remains finite when affinity and load APIs are absent."""
    corpus = tmp_path / "corpus"
    _write_orchestration_corpus(
        corpus,
        plan_digest="a",
        manifest_digest="a",
        plan_revision="r",
        manifest_revision="r",
    )
    lock = tmp_path / "lock.md"
    lock.write_text("frozen\n", encoding="utf-8")
    monkeypatch.setattr(
        selection_module, "load_tglf_model_study_data", lambda _: _closed_physics_study_data()
    )
    monkeypatch.setattr(selection_module, "_fit_candidates", _fast_candidate_result)
    monkeypatch.delattr(os, "sched_getaffinity")
    monkeypatch.delattr(os, "getloadavg")
    report = run_tglf_model_selection(corpus, selection_lock_paths=(lock,), latency_repeats=3)
    assert report["runtime_context"]["logical_affinity_cpus"] == os.cpu_count()
    assert report["runtime_context"]["load_average"] is None
    json.dumps(report, allow_nan=False)
