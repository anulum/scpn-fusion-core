# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — IDA same-case evidence tests
"""Tests for the digest-bound IDA same-case benchmark and report contract."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import validation.benchmark_ida_same_case as benchmark


def _digest(character: str = "a") -> str:
    return character * 64


def _case(role: str, *, passed: bool = True) -> dict[str, Any]:
    return {
        "admitted": False,
        "case_id": f"{role}-case",
        "digests": {
            "candidate_psi_sha256": _digest("a"),
            "coil_current_sha256": _digest("b"),
            "ffprime_coefficients_sha256": _digest("c"),
            "ffprime_values_sha256": _digest("d"),
            "pprime_coefficients_sha256": _digest("e"),
            "pprime_values_sha256": _digest("f"),
            "psin_knots_sha256": _digest("1"),
            "r_grid_sha256": _digest("2"),
            "reference_psi_sha256": _digest("3"),
            "z_grid_sha256": _digest("4"),
        },
        "freegs_version": "0.8.2",
        "gradient_audit": {
            "all_finite": True,
            "cotangent_sha256": _digest("5"),
            "rows": [],
        },
        "grid_shape": [65, 65],
        "input_contract": {
            "anderson_depth": 8,
            "coil_filament_count": 4,
            "cutoff_width": 0.03,
            "ip_ramp": 30,
            "ip_target_a": -1.5e6,
            "mixing": 0.5,
            "n_iter_cap": 120,
            "profile_coefficient_count": 12,
            "profile_degree": 3,
            "profile_sample_count": 129,
            "self_field_wall_boundary": True,
            "separatrix": "smooth_xpoint_flux",
        },
        "latency": {
            "admissible_isolated_evidence": False,
            "compile_and_first_ms": 25.0,
            "p50_ms": 10.0,
            "p95_ms": 12.0,
            "reference_freegs_ms": 30.0,
            "repeat_count": 3,
            "synchronised": True,
            "warm_ms": [9.0, 10.0, 12.0],
        },
        "machine_class": "DIIID" if role == "evaluation_candidate" else "TestTokamak",
        "metrics": {
            "candidate_axis_wb": -0.6,
            "candidate_boundary_wb": -0.1,
            "candidate_current_a": -1.5e6,
            "psi_max_abs_error_wb": 0.01,
            "psi_rmse_wb": 0.001,
            "psi_n_rmse": 0.01,
            "raw_psi_span_nrmse": 0.01,
            "reference_axis_wb": -0.61,
            "reference_boundary_wb": -0.1,
            "reference_current_a": -1.5e6,
            "relative_current_error": 0.01,
            "relative_nonlinear_residual_rms": 0.01,
        },
        "public_example": {
            "path": f"data/{role}.py",
            "sha256": _digest("6"),
        },
        "reference_mask_point_count": 100,
        "role": role,
        "threshold_results": {
            "gradient_audit": passed,
            "latency": passed,
            "psi_n_rmse": passed,
            "relative_current_error": passed,
            "relative_nonlinear_residual_rms": passed,
        },
    }


def _environment(*, isolated: bool = False, x64: bool = True) -> dict[str, Any]:
    return {
        "affinity_cpu_count": 8,
        "backend": "cpu",
        "devices": ["TFRT_CPU_0"],
        "freegs_version": "0.8.2",
        "host_load_1m_5m_15m": [0.1, 0.2, 0.3],
        "isolated_host": isolated,
        "jax_version": "0.7.1",
        "jaxlib_version": "0.7.1",
        "machine": "x86_64",
        "platform": "Linux",
        "python_version": "3.12.0",
        "x64_enabled": x64,
    }


def _sources() -> dict[str, dict[str, str]]:
    return {
        name: {"path": path, "sha256": _digest(str(index))}
        for index, (name, path) in enumerate(
            benchmark._SOURCE_PATHS.items(),
            start=1,
        )
    }


def _lock(*, valid: bool = False) -> dict[str, Any]:
    return {
        "case_id": "evaluation_candidate-case",
        "created_before_execution": valid,
        "path": "lock.json" if valid else None,
        "sha256": _digest("7") if valid else None,
        "valid": valid,
    }


def _report(
    *,
    passed: bool = True,
    isolated: bool = False,
    x64: bool = True,
    lock_valid: bool = False,
) -> dict[str, Any]:
    return benchmark.build_report(
        [_case("development", passed=passed), _case("evaluation_candidate", passed=passed)],
        generated_at="2026-07-23T00:00:00Z",
        environment=_environment(isolated=isolated, x64=x64),
        source_artifacts=_sources(),
        selection_lock=_lock(valid=lock_valid),
    )


def _reseal(report: dict[str, Any]) -> None:
    report["payload_sha256"] = benchmark._payload_sha256(report)


def test_build_report_is_self_digested_and_fail_closed() -> None:
    """Keep integration evidence blocked despite numerically passing fixtures."""
    report = _report()

    assert report["status"] == "blocked_same_case_evidence"
    assert report["payload_sha256"] == benchmark._payload_sha256(report)
    assert report["claim_boundary"] == {field: False for field in benchmark.CLAIM_FIELDS}
    assert report["case_role_contract"]["statistically_held_out"] is False
    assert "isolated_latency_evidence_missing" in report["blockers"]
    assert "execution_preceding_selection_lock_missing" in report["blockers"]
    benchmark.validate_report(report)


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        (_environment(x64=False), "jax_fp64_disabled"),
        (_environment(isolated=False), "isolated_latency_evidence_missing"),
    ],
)
def test_build_report_records_environment_blockers(
    environment: dict[str, Any], expected: str
) -> None:
    """Project runtime admission blockers into the report."""
    report = benchmark.build_report(
        [_case("development"), _case("evaluation_candidate")],
        generated_at="2026-07-23T00:00:00Z",
        environment=environment,
        source_artifacts=_sources(),
        selection_lock=_lock(),
    )
    assert expected in report["blockers"]


def test_build_report_records_each_failed_evaluation_threshold() -> None:
    """Name every failed numerical gate instead of collapsing the reason."""
    report = _report(passed=False)
    for name in _case("evaluation_candidate")["threshold_results"]:
        assert f"evaluation_threshold_failed:{name}" in report["blockers"]


def test_build_report_rejects_empty_timestamp_and_wrong_case_order() -> None:
    """Require a timestamp and the frozen case-role order."""
    with pytest.raises(ValueError, match="generated_at"):
        benchmark.build_report(
            [_case("development"), _case("evaluation_candidate")],
            generated_at=" ",
            environment=_environment(),
            source_artifacts=_sources(),
            selection_lock=_lock(),
        )
    with pytest.raises(ValueError, match="ordered"):
        benchmark.build_report(
            [_case("evaluation_candidate"), _case("development")],
            generated_at="2026-07-23T00:00:00Z",
            environment=_environment(),
            source_artifacts=_sources(),
            selection_lock=_lock(),
        )


def test_validate_report_rejects_tamper_and_overclaim() -> None:
    """Reject digest drift and any claim-boundary promotion."""
    report = _report()
    report["generated_at"] = "tampered"
    with pytest.raises(ValueError, match="payload_sha256"):
        benchmark.validate_report(report)

    report = _report()
    report["claim_boundary"]["facility_validation"] = True
    _reseal(report)
    with pytest.raises(ValueError, match="claim_boundary"):
        benchmark.validate_report(report)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda report: report.update(schema_version="bad"), "unsupported"),
        (lambda report: report.update(benchmark_id="bad"), "benchmark_id"),
        (lambda report: report.update(status="pass"), "status"),
        (lambda report: report.update(thresholds={}), "thresholds"),
        (lambda report: report.update(blockers=[]), "blockers"),
        (lambda report: report.update(cases=[]), "development and evaluation"),
    ],
)
def test_validate_report_rejects_schema_drift(mutate: Any, message: str) -> None:
    """Reject structural drift across the top-level contract."""
    report = _report()
    mutate(report)
    _reseal(report)
    with pytest.raises(ValueError, match=message):
        benchmark.validate_report(report)


def test_validate_report_rejects_case_digest_and_threshold_drift() -> None:
    """Reject malformed case digests, admission flags, and gate projections."""
    report = _report()
    report["cases"][0]["digests"]["r_grid_sha256"] = "bad"
    _reseal(report)
    with pytest.raises(ValueError, match="lowercase SHA"):
        benchmark.validate_report(report)

    report = _report()
    report["cases"][0]["admitted"] = True
    _reseal(report)
    with pytest.raises(ValueError, match="must not independently"):
        benchmark.validate_report(report)

    report = _report()
    report["cases"][0]["threshold_results"].pop("latency")
    _reseal(report)
    with pytest.raises(ValueError, match="threshold results"):
        benchmark.validate_report(report)


def test_validate_report_rejects_environment_source_and_solver_drift() -> None:
    """Bind the runtime, source-artifact, repository, and solver contracts."""
    report = _report()
    report["environment"].pop("backend")
    _reseal(report)
    with pytest.raises(ValueError, match="environment fields"):
        benchmark.validate_report(report)

    report = _report()
    report["source_artifacts"]["solver"]["path"] = "wrong.py"
    _reseal(report)
    with pytest.raises(ValueError, match=r"source_artifacts\.solver"):
        benchmark.validate_report(report)

    report = _report()
    report["source_artifacts"]["repository"] = {
        "git_commit": "a" * 40,
        "path": ".",
    }
    _reseal(report)
    benchmark.validate_report(report)
    report["source_artifacts"]["repository"]["git_commit"] = "bad"
    _reseal(report)
    with pytest.raises(ValueError, match="Git object ID"):
        benchmark.validate_report(report)

    report = _report()
    report["solver_contract"]["solver_id"] = "wrong"
    _reseal(report)
    with pytest.raises(ValueError, match="solver_contract"):
        benchmark.validate_report(report)


def test_validate_report_rejects_role_blocker_and_status_overclaim() -> None:
    """Keep measured case roles, mandatory blockers, and v1 status fail closed."""
    report = _report()
    report["case_role_contract"]["evaluation_case_id"] = "wrong"
    _reseal(report)
    with pytest.raises(ValueError, match="measured cases"):
        benchmark.validate_report(report)

    report = _report()
    report["blockers"].remove("statistically_held_out_case_missing")
    _reseal(report)
    with pytest.raises(ValueError, match="omit required"):
        benchmark.validate_report(report)

    report = _report()
    report["status"] = "accepted_bounded_same_case_evidence"
    _reseal(report)
    with pytest.raises(ValueError, match="cannot be accepted"):
        benchmark.validate_report(report)


def test_validate_report_rejects_nonfinite_and_false_holdout() -> None:
    """Reject JSON non-finites and a fabricated statistical-holdout claim."""
    report = _report()
    report["cases"][0]["metrics"]["psi_rmse_wb"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        benchmark.validate_report(report)

    report = _report()
    report["case_role_contract"]["statistically_held_out"] = True
    _reseal(report)
    with pytest.raises(ValueError, match="statistical holdout"):
        benchmark.validate_report(report)


def test_source_and_runtime_metadata_are_exact_and_digest_bound() -> None:
    """Collect the real source paths, digests, and runtime admission flags."""
    sources = benchmark._source_artifacts()
    assert set(sources) == set(benchmark._SOURCE_PATHS)
    for name, relative_path in benchmark._SOURCE_PATHS.items():
        assert sources[name]["path"] == relative_path
        assert sources[name]["sha256"] == benchmark._file_sha256(benchmark.ROOT / relative_path)

    environment = benchmark._runtime_environment()
    assert set(environment) == benchmark._ENVIRONMENT_FIELDS
    assert isinstance(environment["isolated_host"], bool)
    assert isinstance(environment["x64_enabled"], bool)


def test_git_metadata_probe_returns_value_or_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read the source commit while degrading safely when Git is unavailable."""
    assert benchmark._git_value("rev-parse", "HEAD") is not None

    def fail(*_: Any, **__: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, ["git"])

    monkeypatch.setattr(benchmark.subprocess, "run", fail)
    assert benchmark._git_value("rev-parse", "HEAD") is None


def test_load_report_rejects_duplicate_keys_and_non_object(tmp_path: Path) -> None:
    """Reject ambiguous JSON and non-object report roots."""
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        benchmark.load_report(duplicate)

    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be an object"):
        benchmark.load_report(array)


def test_write_report_round_trip_and_markdown(tmp_path: Path) -> None:
    """Persist stable JSON plus bounded human-readable evidence."""
    report = _report()
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    benchmark.write_report(report, json_path=json_path, markdown_path=markdown_path)

    assert benchmark.load_report(json_path) == report
    text = markdown_path.read_text(encoding="utf-8")
    assert "# IDA free-boundary same-case evidence" in text
    assert "Facility/control/PCS/safety claims: `false`" in text
    assert "evaluation_candidate-case" in text


def test_selection_lock_absent_invalid_and_valid(tmp_path: Path) -> None:
    """Admit only the exact frozen selection-lock vocabulary."""
    assert (
        benchmark._selection_lock(None, evaluation_case_id="evaluation_candidate-case")["valid"]
        is False
    )

    path = tmp_path / "lock.json"
    payload = {
        "benchmark_id": benchmark.BENCHMARK_ID,
        "case_id": "evaluation_candidate-case",
        "created_at": "2026-07-23T00:00:00Z",
        "schema_version": "scpn-fusion.ida-same-case-selection-lock.v2",
        "thresholds": benchmark.THRESHOLDS,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = benchmark._selection_lock(path, evaluation_case_id="evaluation_candidate-case")
    assert result["valid"] is True
    assert result["created_before_execution"] is True
    assert result["sha256"] == benchmark._file_sha256(path)

    payload["case_id"] = "wrong"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert (
        benchmark._selection_lock(path, evaluation_case_id="evaluation_candidate-case")["valid"]
        is False
    )


def test_array_digest_binds_dtype_shape_and_values() -> None:
    """Make numeric digests deterministic while detecting shape/value drift."""
    vector = np.array([1.0, 2.0], dtype=np.float64)
    assert benchmark._array_sha256(vector) == benchmark._array_sha256(vector.copy())
    assert benchmark._array_sha256(vector) != benchmark._array_sha256(vector.reshape(1, 2))
    assert benchmark._array_sha256(vector) != benchmark._array_sha256(np.array([1.0, 3.0]))


def test_mask_boundary_axis_and_profile_fit_helpers() -> None:
    """Exercise the same-case mask, sign-aware axis, and compact basis helpers."""
    mask = np.zeros((5, 5), dtype=np.bool_)
    mask[1:4, 1:4] = True
    boundary = benchmark._reference_mask_boundary(mask)
    assert np.count_nonzero(boundary) == 8

    psi = np.zeros((5, 5), dtype=np.float64)
    psi[2, 2] = -3.0
    assert benchmark._axis_value(psi, mask, 0.0) == -3.0
    psi[2, 2] = 4.0
    assert benchmark._axis_value(psi, mask, 0.0) == 4.0

    knots = np.linspace(0.0, 1.0, 9)
    values = (1.0 - knots) ** 2
    coefficients, reconstructed = benchmark._fit_compact_profile(
        values, knots, n_coefficients=6, degree=3
    )
    assert coefficients.shape == (6,)
    assert np.allclose(reconstructed, values, atol=1.0e-12)


def test_percentile_and_finite_difference_row() -> None:
    """Measure latency percentiles and a smooth exact scalar gradient."""
    assert benchmark._percentile([1.0, 2.0, 3.0], 50.0) == 2.0
    with pytest.raises(ValueError, match="must not be empty"):
        benchmark._percentile([], 95.0)

    vector = np.array([2.0], dtype=np.float64)
    row = benchmark._finite_difference_row(
        name="x",
        index=0,
        vector=vector,
        autodiff=np.array([4.0]),
        objective=lambda value: float(value[0] ** 2),
        relative_step=1.0e-5,
        relative_error_limit=1.0e-6,
    )
    assert row["passed"] is True
    assert row["relative_error"] < 1.0e-10
    assert row["smoothness_ratio"] < benchmark.THRESHOLDS["gradient_smoothness_ratio_max"]


def test_run_benchmark_guards_repeats_and_x64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject invalid repeat counts and non-FP64 evidence environments."""
    with pytest.raises(ValueError, match="latency_repeats"):
        benchmark.run_benchmark(generated_at="2026-07-23T00:00:00Z", latency_repeats=0)
    monkeypatch.setattr(benchmark, "_runtime_environment", lambda: _environment(x64=False))
    with pytest.raises(RuntimeError, match="requires JAX FP64"):
        benchmark.run_benchmark(generated_at="2026-07-23T00:00:00Z", latency_repeats=1)


def test_main_writes_blocked_report_and_returns_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expose a script entry point whose blocked evidence exits non-zero."""
    report = _report()
    monkeypatch.setattr(benchmark, "run_benchmark", lambda **_: report)
    json_path = tmp_path / "out.json"
    markdown_path = tmp_path / "out.md"
    exit_code = benchmark.main(
        [
            "--generated-at",
            "2026-07-23T00:00:00Z",
            "--json-out",
            str(json_path),
            "--markdown-out",
            str(markdown_path),
        ]
    )
    assert exit_code == 2
    assert json_path.is_file()
    assert markdown_path.is_file()
