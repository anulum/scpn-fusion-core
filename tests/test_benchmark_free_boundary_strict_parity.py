# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
"""Exercise strict parity classification against real historical report bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from validation.benchmark_free_boundary_strict_parity import (
    DEFAULT_FREEGS_REPORT,
    DEFAULT_MACHINE_METADATA_REPORT,
    ROOT,
    evaluate_strict_parity,
    main,
    render_markdown,
    run_benchmark,
)


def test_strict_vacuum_sidecar_must_match_reconstruction() -> None:
    """Changed coil currents cannot retain the original reconstruction identity."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text())
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text())
    freegs["strict_free_boundary_parity_evidence"]["cases"][0]["vacuum_green_function_comparison"][
        "coils"
    ][0]["current_a"] *= 2
    report = evaluate_strict_parity(freegs, metadata)
    assert any("vacuum_sidecar_mismatch" in error for error in report["case_identity_errors"])


@pytest.mark.parametrize("field", ["coils", "native_vacuum_psi_sample", "freegs_vacuum_psi_sample"])
def test_matching_sidecars_cannot_hide_missing_rows(field: str) -> None:
    """Removing the same row in both summaries cannot preserve count evidence."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text())
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text())
    for section in (freegs, freegs["strict_free_boundary_parity_evidence"]):
        section["cases"][0]["vacuum_green_function_comparison"][field].pop()
    report = evaluate_strict_parity(freegs, metadata)
    assert any("vacuum_sidecar_structure" in error for error in report["case_identity_errors"])


def test_reference_coil_count_matches_actual_sidecar_rows() -> None:
    """Reference metadata cannot advertise coils absent from the linked sidecar."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text())
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text())
    metadata["same_case_public_reference_output"]["cases"][0]["coil_count"] = 100
    report = evaluate_strict_parity(freegs, metadata)
    assert any("coil_count_mismatch" in error for error in report["case_identity_errors"])


@pytest.mark.parametrize("mutation", ["sample", "limit", "reported_rmse", "reported_pass"])
def test_vacuum_numerics_are_derived_from_samples(mutation: str) -> None:
    """Matching sidecars cannot substitute flags or limits for their actual residuals."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text())
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text())
    for section in (freegs, freegs["strict_free_boundary_parity_evidence"]):
        vacuum = section["cases"][0]["vacuum_green_function_comparison"]
        if mutation == "sample":
            vacuum["native_vacuum_psi_sample"][0] += 1.0
        elif mutation == "limit":
            vacuum["thresholds"]["nrmse"] = 1e9
        elif mutation == "reported_rmse":
            vacuum["rmse_wb"] = 0.1
        else:
            vacuum["pass"] = False
    report = evaluate_strict_parity(freegs, metadata)
    assert any("vacuum_numerics_inconsistent" in error for error in report["case_identity_errors"])
    assert report["accepted_full_fidelity"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", ""),
        ("name", "P1U"),
        ("current_a", True),
        ("current_a", float("inf")),
        ("turns", 0),
        ("turns", -1),
        ("effective_current_a_turns", 1),
        ("filament_count", True),
        ("filament_count", 0),
        ("filament_count", 2.5),
    ],
)
def test_coil_records_require_consistent_physical_quantities(field: str, value: object) -> None:
    """Matching copies cannot launder invalid coil identity, current or discretization."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text())
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text())
    for section in (freegs, freegs["strict_free_boundary_parity_evidence"]):
        section["cases"][0]["vacuum_green_function_comparison"]["coils"][0][field] = value
    report = evaluate_strict_parity(freegs, metadata)
    assert any("vacuum_sidecar_structure" in error for error in report["case_identity_errors"])


def test_total_filament_count_is_sum_of_coil_records() -> None:
    """Aggregate filament counts must agree with the producer's concatenated terms."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text())
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text())
    for section in (freegs, freegs["strict_free_boundary_parity_evidence"]):
        section["cases"][0]["vacuum_green_function_comparison"]["filament_count"] += 1
    report = evaluate_strict_parity(freegs, metadata)
    assert any("vacuum_sidecar_structure" in error for error in report["case_identity_errors"])


@pytest.mark.parametrize("field", ["machine_class", "example_path", "example_sha256"])
def test_reference_identity_must_match_reconstruction(field: str) -> None:
    """Reference metadata cannot silently point to a different machine or source."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text())
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text())
    metadata["same_case_public_reference_output"]["cases"][0][field] = "different"
    report = evaluate_strict_parity(freegs, metadata)
    assert "case_identity_inconsistent" in report["blockers"]
    assert any(field in error for error in report["case_identity_errors"])


def test_strict_source_configuration_must_match_reconstruction() -> None:
    """A changed plasma-current configuration cannot retain same-case provenance."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text())
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text())
    freegs["strict_free_boundary_parity_evidence"]["cases"][0]["source_contract"][
        "plasma_current_a"
    ] *= 2
    report = evaluate_strict_parity(freegs, metadata)
    assert any("source_contract" in error for error in report["case_identity_errors"])
    assert report["accepted_full_fidelity"] is False


def test_reference_case_membership_must_match_reconstruction() -> None:
    """Missing reference cases cannot inherit another case's evidence."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text())
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text())
    metadata["same_case_public_reference_output"]["cases"].pop()
    report = evaluate_strict_parity(freegs, metadata)
    assert "reference_case_membership_mismatch" in report["case_identity_errors"]


def test_real_case_identity_agreement_does_not_establish_field_custody() -> None:
    """Historical matching identities remain distinct from absent raw array proof."""
    report = run_benchmark(write=False)
    assert report["case_identity_errors"] == []
    assert report["accepted_full_fidelity"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("example_sha256", "0" * 64),
        (
            "example_path",
            "data/external/full_fidelity_public_sources/repos/freegs/missing-example.py",
        ),
        ("example_path", "data/external/full_fidelity_public_sources/repos/freegs"),
        ("example_path", "../outside.py"),
    ],
)
def test_agreeing_source_claims_require_real_matching_bytes(field: str, value: str) -> None:
    """Mutually consistent reports cannot replace missing or different source bytes."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text())
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text())
    for section in (
        freegs,
        freegs["strict_free_boundary_parity_evidence"],
        metadata["same_case_public_reference_output"],
    ):
        section["cases"][0][field] = value
    report = evaluate_strict_parity(freegs, metadata)
    assert report["case_identity_errors"] == []
    assert "source_example_bytes_unverified" in report["blockers"]
    assert report["source_example_files"][0]["declared_hash_matches_file"] is False
    assert report["accepted_full_fidelity"] is False


def test_example_byte_status_matches_actual_source_availability() -> None:
    """Available real bytes are checked; absent optional cache remains non-admitting."""
    report = run_benchmark(write=False)
    assert len(report["source_example_files"]) == 2
    for row in report["source_example_files"]:
        path = ROOT / row["path"]
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        assert row["file_sha256"] == actual_hash
        assert row["declared_hash_matches_file"] is (
            actual_hash is not None and actual_hash == row["declared_sha256"]
        )
    if any(row["declared_hash_matches_file"] is False for row in report["source_example_files"]):
        assert "source_example_bytes_unverified" in report["blockers"]
    assert report["accepted_full_fidelity"] is False


def test_isolated_cli_without_source_cache_refuses_admission(tmp_path: Path) -> None:
    """Run copied production files and real reports with no external source cache."""
    validation = tmp_path / "validation"
    reports = validation / "reports"
    reports.mkdir(parents=True)
    for name in (
        "benchmark_free_boundary_strict_parity.py",
        "free_boundary_strict_parity_report.py",
        "free_boundary_vacuum_evidence.py",
        "free_boundary_threshold_evidence.py",
    ):
        shutil.copy2(ROOT / "validation" / name, validation / name)
    for source in (DEFAULT_FREEGS_REPORT, DEFAULT_MACHINE_METADATA_REPORT):
        shutil.copy2(source, reports / source.name)
    result = subprocess.run(
        [
            sys.executable,
            str(validation / "benchmark_free_boundary_strict_parity.py"),
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
    report = json.loads(result.stdout)
    assert report["accepted_full_fidelity"] is False
    assert "source_example_bytes_unverified" in report["blockers"]
    assert all(row["file_sha256"] is None for row in report["source_example_files"])
    assert all(row["payload_matches_file"] is True for row in report["provenance"]["input_reports"])
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("section", ["grid_convergence_evidence", "geometry_containment_evidence"])
def test_geometry_and_ladder_require_the_same_case_cohort(section: str) -> None:
    """Removing a case from a supporting section must invalidate its identity linkage."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text())
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text())
    freegs["strict_free_boundary_parity_evidence"][section]["cases"].pop()
    report = evaluate_strict_parity(freegs, metadata)
    assert any("membership_mismatch" in error for error in report["case_identity_errors"])
    assert "case_identity_inconsistent" in report["blockers"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("external_axis", [1.0, 0.0]),
        ("native_axis", [1.0, 0.0]),
        ("boundary_containment_fraction", 0.5),
    ],
)
def test_geometry_observables_match_reconstruction(field: str, value: object) -> None:
    """Geometry summaries cannot replace the reconstruction's reported observables."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text())
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text())
    freegs["strict_free_boundary_parity_evidence"]["geometry_containment_evidence"]["cases"][0][
        field
    ] = value
    report = evaluate_strict_parity(freegs, metadata)
    assert any(field in error for error in report["case_identity_errors"])


def test_ladder_machine_identity_matches_reconstruction() -> None:
    """A ladder for another machine cannot share only the same case label."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text())
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text())
    freegs["strict_free_boundary_parity_evidence"]["grid_convergence_evidence"]["cases"][0][
        "machine_class"
    ] = "other"
    report = evaluate_strict_parity(freegs, metadata)
    assert any(
        "grid:" in error and "machine_class" in error for error in report["case_identity_errors"]
    )


def test_strict_parity_blocks_current_tracked_reports() -> None:
    """Keep summary-only historical evidence non-admitting without changing its bytes."""
    paths = (DEFAULT_FREEGS_REPORT, DEFAULT_MACHINE_METADATA_REPORT)
    before = {path: path.read_bytes() for path in paths}
    report = run_benchmark(write=False)
    assert report["schema"] == "free-boundary-strict-parity-benchmark.v2"
    assert report["accepted_full_fidelity"] is False
    assert report["evidence_classification"] == "legacy_non_admitting"
    assert "legacy_evidence_has_no_verified_field_custody" in report["blockers"]
    assert report["case_count"] == len(report["threshold_cases"]) == 2
    assert all(value is False for value in report["acceptance_matrix"].values())
    assert {path: path.read_bytes() for path in paths} == before
    for source, path in zip(report["provenance"]["input_reports"], paths):
        assert source["file_sha256"] == hashlib.sha256(before[path]).hexdigest()
        assert source["payload_matches_file"] is True


@pytest.mark.parametrize("source_index", [0, 1])
def test_provenance_detects_payload_not_present_in_named_file(source_index: int) -> None:
    """An in-memory report edit cannot borrow provenance from unchanged disk bytes."""
    payloads = [
        json.loads(path.read_text())
        for path in (DEFAULT_FREEGS_REPORT, DEFAULT_MACHINE_METADATA_REPORT)
    ]
    payloads[source_index]["status"] = "altered_report"
    report = evaluate_strict_parity(*payloads)
    source = report["provenance"]["input_reports"][source_index]
    assert source["payload_matches_file"] is False
    assert "input_report_payload_not_bound_to_file" in report["blockers"]
    assert report["accepted_full_fidelity"] is False


@pytest.mark.parametrize("contents", [None, b"not json", b"[]", b"\xff"])
def test_provenance_refuses_missing_or_invalid_report_bytes(
    tmp_path: Path, contents: bytes | None
) -> None:
    """Named but absent or undecodable report files cannot establish byte custody."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text())
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text())
    source_path = tmp_path / "report.json"
    if contents is not None:
        source_path.write_bytes(contents)
    report = evaluate_strict_parity(freegs, metadata, freegs_report_path=source_path)
    source = report["provenance"]["input_reports"][0]
    assert source["payload_matches_file"] is False
    assert source["file_sha256"] == (
        None if contents is None else hashlib.sha256(contents).hexdigest()
    )
    assert "input_report_payload_not_bound_to_file" in report["blockers"]


@pytest.mark.parametrize("empty", [True, False])
@pytest.mark.parametrize("stale_total", [True, False])
def test_strict_parity_rejects_audited_row_counterexamples(empty: bool, stale_total: bool) -> None:
    """Replay both audited corruptions with stale and corrected summary counts."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text(encoding="utf-8"))
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text(encoding="utf-8"))
    strict = freegs["strict_free_boundary_parity_evidence"]
    if empty:
        strict["cases"] = []
        strict["grid_convergence_evidence"]["cases"] = []
        if not stale_total:
            freegs["case_count"] = 0
    else:
        check = strict["cases"][0]["threshold_checks"][0]
        check["value"] = 1e9
        check["passed"] = False
        if not stale_total:
            strict["failed_threshold_check_count"] = 1
    report = evaluate_strict_parity(freegs, metadata)
    assert report["accepted_full_fidelity"] is False
    assert report["case_count"] == len(strict["cases"])
    assert report["failed_threshold_check_count"] == (0 if empty else 1)
    assert report["blockers"]


@pytest.mark.parametrize("schema", [None, "strict-free-boundary-parity-evidence.v2", "unknown"])
def test_strict_parity_does_not_admit_relabelled_legacy_payload(schema: str | None) -> None:
    """Changing a schema label cannot supply absent verified field evidence."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text(encoding="utf-8"))
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text(encoding="utf-8"))
    freegs["strict_free_boundary_parity_evidence"]["schema"] = schema
    report = evaluate_strict_parity(freegs, metadata)
    assert report["accepted_full_fidelity"] is False
    assert report["evidence_classification"] == "unsupported_evidence_non_admitting"
    assert "unsupported_evidence_contract" in report["blockers"]


def test_strict_parity_markdown_exposes_legacy_boundary() -> None:
    """Render a real non-admitting report with provenance and the scientific boundary."""
    report = run_benchmark(write=False)
    markdown = render_markdown(report)
    assert "legacy_non_admitting" in markdown
    assert "Accepted full fidelity: `False`" in markdown
    assert "## Acceptance matrix" in markdown
    assert "## Provenance and checksums" in markdown


def test_strict_parity_cli_refuses_legacy_without_writing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The production strict CLI exits nonzero for preserved summary-only input."""
    before = DEFAULT_FREEGS_REPORT.read_bytes()
    assert main(["--check", "--strict"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["accepted_full_fidelity"] is False
    assert DEFAULT_FREEGS_REPORT.read_bytes() == before


@pytest.mark.parametrize("value", [1e9, -1.0, True, None, float("nan"), float("inf"), 10**400])
def test_strict_parity_recomputes_numeric_failure_despite_reported_pass(value: object) -> None:
    """A changed historical measurement cannot retain its stale passing verdict."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text(encoding="utf-8"))
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text(encoding="utf-8"))
    freegs["strict_free_boundary_parity_evidence"]["cases"][0]["threshold_checks"][0]["value"] = (
        value
    )
    report = evaluate_strict_parity(freegs, metadata)
    case = report["threshold_cases"][0]
    assert report["accepted_full_fidelity"] is False
    assert case["failed_threshold_check_count"] == 1
    assert case["strict_threshold_acceptance_ready"] is False
    assert "psi_n_rmse" in case["contradictory_threshold_metrics"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("field,value", [("limit", 1e9), ("limit", True), ("comparator", ">=")])
def test_strict_parity_rejects_changed_numeric_contract(field: str, value: object) -> None:
    """Historical row metadata cannot redefine the acceptance threshold."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text(encoding="utf-8"))
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text(encoding="utf-8"))
    freegs["strict_free_boundary_parity_evidence"]["cases"][0]["threshold_checks"][0][field] = value
    report = evaluate_strict_parity(freegs, metadata)
    assert report["failed_threshold_check_count"] == 1
    assert report["accepted_full_fidelity"] is False


@pytest.mark.parametrize("rows", [None, {}, "invalid", [None]])
def test_strict_parity_rejects_malformed_case_collection(rows: object) -> None:
    """Reject corrupt collection shapes rather than silently dropping their contents."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text(encoding="utf-8"))
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text(encoding="utf-8"))
    freegs["strict_free_boundary_parity_evidence"]["cases"] = rows
    with pytest.raises(ValueError, match="cases"):
        evaluate_strict_parity(freegs, metadata)


def test_strict_parity_keeps_q_label_separate_from_verified_profile() -> None:
    """Real historical passing labels do not verify absent q coordinates and samples."""
    report = run_benchmark(write=False)
    for case in report["threshold_cases"]:
        assert case["unverified_threshold_metrics"] == ["q_profile_sanity_status"]
        assert case["strict_threshold_acceptance_ready"] is False
        assert case["reported_readiness"]["strict_threshold_acceptance_ready"] is True


@pytest.mark.parametrize(
    "metric,value",
    [
        ("axis_error_m", 0.02500001),
        ("current_closure_relative_error", 0.05000001),
        ("boundary_max_abs_error_wb", 1.000001e-10),
        ("xpoint_psi_n_error_max", 0.05000001),
        ("boundary_containment_fraction", 0.99999999),
        ("boundary_containment_fraction", 1.00000001),
        ("q_profile_sanity_status", "failed"),
    ],
)
def test_strict_parity_reports_each_out_of_contract_metric(metric: str, value: object) -> None:
    """Exercise every other historical metric without changing the reported verdict."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text(encoding="utf-8"))
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text(encoding="utf-8"))
    checks = freegs["strict_free_boundary_parity_evidence"]["cases"][0]["threshold_checks"]
    next(check for check in checks if check["metric"] == metric)["value"] = value
    report = evaluate_strict_parity(freegs, metadata)
    case = report["threshold_cases"][0]
    assert case["failed_threshold_check_count"] == 1
    assert case["failed_threshold_checks"][0]["metric"] == metric
    assert metric in case["contradictory_threshold_metrics"]
    assert report["accepted_full_fidelity"] is False


@pytest.mark.parametrize("mode", ["missing", "duplicate", "unknown", "malformed"])
def test_strict_parity_validates_threshold_membership(mode: str) -> None:
    """Do not ignore missing, repeated, unknown or malformed metric rows."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text(encoding="utf-8"))
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text(encoding="utf-8"))
    case = freegs["strict_free_boundary_parity_evidence"]["cases"][0]
    checks = case["threshold_checks"]
    if mode == "missing":
        checks.pop(0)
        report = evaluate_strict_parity(freegs, metadata)
        assert report["failed_threshold_check_count"] == 1
        assert report["accepted_full_fidelity"] is False
        return
    if mode == "duplicate":
        checks.append(checks[0].copy())
    elif mode == "unknown":
        checks[0]["metric"] = "unrecognised"
    else:
        case["threshold_checks"] = None
    with pytest.raises(ValueError, match="threshold"):
        evaluate_strict_parity(freegs, metadata)


@pytest.mark.parametrize("case_id", [None, "", " ", "freegs_16_diiid_public_example"])
def test_strict_parity_rejects_missing_or_duplicate_case_identity(case_id: object) -> None:
    """The actual two-case cohort cannot silently lose or duplicate a case identity."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text(encoding="utf-8"))
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text(encoding="utf-8"))
    freegs["strict_free_boundary_parity_evidence"]["cases"][0]["case_id"] = case_id
    with pytest.raises(ValueError, match="case_id"):
        evaluate_strict_parity(freegs, metadata)


def test_strict_parity_derives_grid_counts_from_actual_rows() -> None:
    """A missing historical grid cannot hide behind unchanged aggregate counts."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text(encoding="utf-8"))
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text(encoding="utf-8"))
    grid = freegs["strict_free_boundary_parity_evidence"]["grid_convergence_evidence"]
    grid["cases"][0]["resolution_rows"].pop()
    report = evaluate_strict_parity(freegs, metadata)
    case = report["grid_convergence"]["cases"][0]
    assert case["observed_resolution_count"] == 2
    assert case["required_resolution_count"] == 3
    assert case["missing_resolution_count"] == 1
    assert case["grid_convergence_case_ready"] is False
    assert "observed_resolution_count" in case["contradictory_grid_fields"]


@pytest.mark.parametrize(
    "bad_grid", [{"nx": 65, "ny": 65}, {"nx": 64, "ny": 64}, {"nx": True, "ny": 33}, None]
)
def test_strict_parity_refuses_duplicate_or_wrong_grid(bad_grid: object) -> None:
    """Each historical case requires distinct exact square grids of 33/65/129."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text(encoding="utf-8"))
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text(encoding="utf-8"))
    grid = freegs["strict_free_boundary_parity_evidence"]["grid_convergence_evidence"]
    grid["cases"][0]["resolution_rows"][0]["grid"] = bad_grid
    with pytest.raises(ValueError, match="grid"):
        evaluate_strict_parity(freegs, metadata)


@pytest.mark.parametrize(
    "metric",
    [
        "psi_n_rmse",
        "native_plasma_psi_rmse",
        "xpoint_psi_n_error_max",
        "current_closure_relative_error",
    ],
)
@pytest.mark.parametrize("value", [1e9, float("nan"), True, None, -1.0])
def test_strict_parity_recomputes_grid_monotonicity(metric: str, value: object) -> None:
    """Corrupt one real ladder measurement while preserving stale monotonicity flags."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text(encoding="utf-8"))
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text(encoding="utf-8"))
    grid = freegs["strict_free_boundary_parity_evidence"]["grid_convergence_evidence"]
    grid["cases"][0]["resolution_rows"][1]["metrics"][metric] = value
    report = evaluate_strict_parity(freegs, metadata)
    case = report["grid_convergence"]["cases"][0]
    assert case["monotone_nonincreasing_metrics"][metric] is False
    assert "monotone_nonincreasing_metrics" in case["contradictory_grid_fields"]
    assert case["grid_convergence_case_ready"] is False


def test_strict_parity_sorts_real_grid_rows_without_admitting_evidence() -> None:
    """Report order is not grid order and a valid summary still lacks field custody."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text(encoding="utf-8"))
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text(encoding="utf-8"))
    grid = freegs["strict_free_boundary_parity_evidence"]["grid_convergence_evidence"]
    grid["cases"][0]["resolution_rows"].reverse()
    report = evaluate_strict_parity(freegs, metadata)
    case = report["grid_convergence"]["cases"][0]
    assert case["observed_resolutions"] == [33, 65, 129]
    assert all(case["monotone_nonincreasing_metrics"].values())
    assert case["grid_convergence_case_ready"] is False
    assert case["reported_grid_convergence_case_ready"] is True


@pytest.mark.parametrize("rows", [None, {}, [None]])
def test_strict_parity_refuses_malformed_grid_rows(rows: object) -> None:
    """Malformed ladder collections must not silently disappear from the report."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text(encoding="utf-8"))
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text(encoding="utf-8"))
    grid = freegs["strict_free_boundary_parity_evidence"]["grid_convergence_evidence"]
    grid["cases"][0]["resolution_rows"] = rows
    with pytest.raises(ValueError, match="resolution_rows"):
        evaluate_strict_parity(freegs, metadata)


@pytest.mark.parametrize("relative_increase,expected", [(0.5e-9, True), (2e-9, False)])
def test_strict_parity_preserves_relative_monotonicity_tolerance(
    relative_increase: float, expected: bool
) -> None:
    """Perturb actual ladder values around the original multiplicative tolerance."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text(encoding="utf-8"))
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text(encoding="utf-8"))
    grid = freegs["strict_free_boundary_parity_evidence"]["grid_convergence_evidence"]
    rows = grid["cases"][0]["resolution_rows"]
    rows[1]["metrics"]["native_plasma_psi_rmse"] = rows[0]["metrics"]["native_plasma_psi_rmse"] * (
        1 + relative_increase
    )
    report = evaluate_strict_parity(freegs, metadata)
    case = report["grid_convergence"]["cases"][0]
    assert case["monotone_nonincreasing_metrics"]["native_plasma_psi_rmse"] is expected
    assert report["accepted_full_fidelity"] is False


def test_strict_parity_rejects_different_grid_case_membership() -> None:
    """A removed case cannot be hidden by unchanged aggregate readiness."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text(encoding="utf-8"))
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text(encoding="utf-8"))
    freegs["strict_free_boundary_parity_evidence"]["grid_convergence_evidence"]["cases"].pop()
    report = evaluate_strict_parity(freegs, metadata)
    assert "grid_case_membership_mismatch" in report["blockers"]
    assert report["accepted_full_fidelity"] is False


def test_strict_parity_rejects_numeric_monotonicity_flags() -> None:
    """Integer one is not the boolean contract for a reported monotonicity result."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text(encoding="utf-8"))
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text(encoding="utf-8"))
    case = freegs["strict_free_boundary_parity_evidence"]["grid_convergence_evidence"]["cases"][0]
    case["monotone_nonincreasing_metrics"]["psi_n_rmse"] = 1
    report = evaluate_strict_parity(freegs, metadata)
    assert (
        "monotone_nonincreasing_metrics"
        in report["grid_convergence"]["cases"][0]["contradictory_grid_fields"]
    )


@pytest.mark.parametrize("resolution_index", [0, 1, 2])
def test_grid_thresholds_are_recomputed_at_every_resolution(resolution_index: int) -> None:
    """A failed per-grid metric cannot hide behind passing case-level summaries."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text(encoding="utf-8"))
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text(encoding="utf-8"))
    case = freegs["strict_free_boundary_parity_evidence"]["grid_convergence_evidence"]["cases"][0]
    case["resolution_rows"][resolution_index]["threshold_checks"][0]["value"] = 1e9
    report = evaluate_strict_parity(freegs, metadata)
    row = report["grid_convergence"]["cases"][0]["resolution_diagnostics"][resolution_index]
    assert row["failed_threshold_check_count"] == 1
    assert row["contradictory_threshold_metrics"] == ["psi_n_rmse"]
    assert "psi_n_rmse" in row["inconsistent_metric_values"]
    assert row["reported_failed_count_matches"] is False
    assert report["accepted_full_fidelity"] is False


def test_grid_metric_summary_cannot_disagree_with_threshold_row() -> None:
    """Two representations of the same historical measurement must agree exactly."""
    freegs = json.loads(DEFAULT_FREEGS_REPORT.read_text(encoding="utf-8"))
    metadata = json.loads(DEFAULT_MACHINE_METADATA_REPORT.read_text(encoding="utf-8"))
    case = freegs["strict_free_boundary_parity_evidence"]["grid_convergence_evidence"]["cases"][0]
    case["resolution_rows"][0]["metrics"]["psi_n_rmse"] *= 2
    report = evaluate_strict_parity(freegs, metadata)
    row = report["grid_convergence"]["cases"][0]["resolution_diagnostics"][0]
    assert row["inconsistent_metric_values"] == ["psi_n_rmse"]
    assert row["failed_threshold_check_count"] == 0
    assert row["unverified_threshold_metrics"] == ["q_profile_sanity_status"]
