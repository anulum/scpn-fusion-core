# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Threshold evidence tests
"""Exercise the owning public threshold/grid diagnostic API on real reports."""

import json

import pytest

from validation.benchmark_free_boundary_strict_parity import DEFAULT_FREEGS_REPORT
from validation.free_boundary_threshold_evidence import evaluate_threshold_evidence


def test_real_threshold_and_grid_diagnostics_preserve_input() -> None:
    """Historical numerical agreement is not scientific admission."""
    strict = json.loads(DEFAULT_FREEGS_REPORT.read_text())["strict_free_boundary_parity_evidence"]
    before = json.dumps(strict, sort_keys=True)
    result = evaluate_threshold_evidence(strict)
    assert result["required_resolution_count"] == 3
    assert len(result["threshold_cases"]) == len(result["grid_cases"]) == 2
    assert all(
        case["strict_threshold_acceptance_ready"] is False for case in result["threshold_cases"]
    )
    assert all(case["observed_resolutions"] == [33, 65, 129] for case in result["grid_cases"])
    assert json.dumps(strict, sort_keys=True) == before


@pytest.mark.parametrize("value", [1e9, True, None, float("nan")])
def test_numeric_summary_mutations_do_not_retain_passing_flags(value: object) -> None:
    """Real source-derived case rows must be recomputed by the extracted owner."""
    strict = json.loads(DEFAULT_FREEGS_REPORT.read_text())["strict_free_boundary_parity_evidence"]
    strict["cases"][0]["threshold_checks"][0]["value"] = value
    result = evaluate_threshold_evidence(strict)
    assert result["threshold_cases"][0]["failed_threshold_check_count"] == 1


def test_missing_real_ladder_resolution_is_visible() -> None:
    """Missing rows cannot inherit complete-ladder diagnostics after extraction."""
    strict = json.loads(DEFAULT_FREEGS_REPORT.read_text())["strict_free_boundary_parity_evidence"]
    strict["grid_convergence_evidence"]["cases"][0]["resolution_rows"].pop()
    result = evaluate_threshold_evidence(strict)
    assert result["grid_cases"][0]["missing_resolution_count"] == 1
    assert not any(result["grid_cases"][0]["monotone_nonincreasing_metrics"].values())


def test_duplicate_real_case_ids_are_rejected() -> None:
    """The owner rejects ambiguous identities rather than dropping duplicate rows."""
    strict = json.loads(DEFAULT_FREEGS_REPORT.read_text())["strict_free_boundary_parity_evidence"]
    strict["cases"].append(strict["cases"][0].copy())
    with pytest.raises(ValueError, match="case_id"):
        evaluate_threshold_evidence(strict)
