# SPDX-License-Identifier: AGPL-3.0-or-later
"""Grad-Shafranov operator-current closure benchmark contract tests."""

from __future__ import annotations

from validation.benchmark_gs_operator_current_closure import _radial_convergence_order


def test_radial_quartic_convergence_order_reports_second_order_decay() -> None:
    """Radial-quartic analytic truncation must decay at second order in dR."""
    cases = [
        {"case": "radial_quartic", "dr": 0.125, "analytic_delta_star_max_abs_error": 9.765625e-4},
        {"case": "radial_quartic", "dr": 0.0625, "analytic_delta_star_max_abs_error": 2.44140625e-4},
    ]

    order = _radial_convergence_order(cases)

    assert order == 2.0
