# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Vacuum evidence tests
"""Exercise stored real vacuum comparisons through their owning public validator."""

import json

import pytest

from validation.benchmark_free_boundary_strict_parity import DEFAULT_FREEGS_REPORT
from validation.free_boundary_vacuum_evidence import evaluate_vacuum_evidence


def test_real_vacuum_records_are_self_consistent_without_mutation() -> None:
    """Actual historical sidecars replay exactly; no physical admission is returned."""
    report = json.loads(DEFAULT_FREEGS_REPORT.read_text())
    for case in report["cases"]:
        sidecar = case["vacuum_green_function_comparison"]
        before = json.dumps(sidecar, sort_keys=True)
        assert evaluate_vacuum_evidence(sidecar) == {
            "structure_consistent": True,
            "metrics_consistent": True,
        }
        assert json.dumps(sidecar, sort_keys=True) == before


@pytest.mark.parametrize("mutation", ["coil", "sample", "limit", "flag"])
def test_corrupted_real_vacuum_records_are_detected(mutation: str) -> None:
    """Mutated real metadata cannot retain its numerical consistency verdict."""
    sidecar = json.loads(DEFAULT_FREEGS_REPORT.read_text())["cases"][0][
        "vacuum_green_function_comparison"
    ]
    if mutation == "coil":
        sidecar["coils"][0]["turns"] = 0
    elif mutation == "sample":
        sidecar["native_vacuum_psi_sample"][0] += 1
    elif mutation == "limit":
        sidecar["thresholds"]["nrmse"] = 1
    else:
        sidecar["pass"] = False
    result = evaluate_vacuum_evidence(sidecar)
    assert result["metrics_consistent"] is False
    assert result["structure_consistent"] is (mutation != "coil")


@pytest.mark.parametrize("sidecar", [None, [], "invalid", {}])
def test_non_sidecar_inputs_are_nonadmitting(sidecar: object) -> None:
    """Unsupported input shapes cannot produce a consistency pass."""
    assert evaluate_vacuum_evidence(sidecar) == {
        "structure_consistent": False,
        "metrics_consistent": False,
    }
