# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
"""Grad-Shafranov operator-current closure benchmark contract tests."""

from __future__ import annotations

import json

import pytest

from validation.benchmark_gs_operator_current_closure import _radial_convergence_order
from validation.benchmark_gs_operator_current_closure import build_gate_summary
from validation.benchmark_gs_operator_current_closure import REPORT_JSON
from validation.benchmark_gs_operator_current_closure import main as run_benchmark


def test_radial_quartic_convergence_order_reports_second_order_decay() -> None:
    """Radial-quartic analytic truncation must decay at second order in dR."""
    cases = [
        {"case": "radial_quartic", "dr": 0.125, "analytic_delta_star_max_abs_error": 9.765625e-4},
        {
            "case": "radial_quartic",
            "dr": 0.0625,
            "analytic_delta_star_max_abs_error": 2.44140625e-4,
        },
    ]

    order = _radial_convergence_order(cases)

    assert order == 2.0


def test_radial_quartic_convergence_order_rejects_invalid_radial_rows() -> None:
    """Radial-quartic convergence gates must reject invalid rows, not filter them."""
    with pytest.raises(ValueError, match="strictly decreasing positive dr"):
        _radial_convergence_order(
            [
                {
                    "case": "radial_quartic_17",
                    "dr": 0.0625,
                    "analytic_delta_star_max_abs_error": 2.44140625e-4,
                },
                {
                    "case": "radial_quartic_33",
                    "dr": 0.0625,
                    "analytic_delta_star_max_abs_error": 9.765625e-4,
                },
            ]
        )

    with pytest.raises(ValueError, match="positive finite analytic Delta"):
        _radial_convergence_order(
            [
                {
                    "case": "radial_quartic_17",
                    "dr": 0.125,
                    "analytic_delta_star_max_abs_error": 9.765625e-4,
                },
                {
                    "case": "radial_quartic_33",
                    "dr": 0.0625,
                    "analytic_delta_star_max_abs_error": 0.0,
                },
            ]
        )


def test_benchmark_report_includes_mixed_solovev_operator_case() -> None:
    """Benchmark report must include an R^2 Z^2 mixed Solov'ev-style operator case."""
    assert run_benchmark() == 0
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))

    assert "mixed_solovev" in {case["case"] for case in report["cases"]}


def test_benchmark_report_gates_radial_current_closure_stability() -> None:
    """Radial grid refinement should expose a total-current closure stability gate."""
    assert run_benchmark() == 0
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))

    assert (
        report["radial_total_current_relative_error_max"]
        <= report["thresholds"]["radial_total_current_relative_error_max"]
    )
    assert report["radial_current_closure_stability_pass"] is True


def test_benchmark_report_exposes_fail_closed_gate_summary() -> None:
    """Machine-readable benchmark gates must distinguish schema, scope, and failures."""
    assert run_benchmark() == 0
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))

    assert report["schema_version"] == "gs-operator-current-closure.v2"
    assert report["benchmark_scope"] == "native_grad_shafranov_operator_current_closure"
    assert (
        report["physics_scope"]
        == "manufactured_full_order_grad_shafranov_operator_current_relation"
    )
    assert report["solver_mode"] == "manufactured_flux_operator_current_closure"
    assert report["passes"] is True
    assert report["passed"] is True
    assert report["gate_summary"]["gates"] == {
        "case_thresholds": True,
        "radial_convergence_order": True,
        "radial_total_current_closure_stability": True,
    }
    assert report["gate_summary"]["gate_pass_count"] == 3
    assert report["gate_summary"]["failed_gates"] == []


def test_operator_current_gate_summary_fails_closed_for_missing_rows() -> None:
    """Missing physics evidence must fail rather than silently pass the benchmark."""
    summary = build_gate_summary(
        {
            "thresholds": {
                "delta_star_max_abs_error": 1.0e-10,
                "current_density_max_relative_error": 1.0e-11,
                "total_current_relative_error": 1.0e-12,
            },
            "cases": [],
            "radial_convergence_order": None,
            "radial_current_closure_stability_pass": False,
        }
    )

    assert summary["passes"] is False
    assert summary["failed_gates"] == [
        "case_thresholds",
        "radial_convergence_order",
        "radial_total_current_closure_stability",
    ]
