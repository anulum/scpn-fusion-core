# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests for GS mesh convergence validation
"""Behavioural contracts for the Grad-Shafranov mesh convergence study."""

from __future__ import annotations

import pytest

from validation.mesh_convergence_study import (
    add_convergence_rates,
    summarise_convergence_contract,
)


def test_add_convergence_rates_recovers_second_order_solovev_scaling():
    """Halving grid spacing with quartered error must report order two."""
    rows = [
        {"nr": 17, "nz": 17, "h": 0.125, "nrmse": 4.0e-4},
        {"nr": 33, "nz": 33, "h": 0.0625, "nrmse": 1.0e-4},
        {"nr": 65, "nz": 65, "h": 0.03125, "nrmse": 2.5e-5},
    ]

    rated = add_convergence_rates(rows)

    assert "convergence_rate" not in rated[0]
    assert rated[1]["convergence_rate"] == pytest.approx(2.0)
    assert rated[2]["convergence_rate"] == pytest.approx(2.0)
    assert rows[1].get("convergence_rate") is None


def test_summarise_convergence_contract_rejects_subsecond_order_solver():
    """A first-order error trend is a solver-fidelity regression."""
    rows = add_convergence_rates(
        [
            {"nr": 17, "nz": 17, "h": 0.125, "nrmse": 4.0e-4},
            {"nr": 33, "nz": 33, "h": 0.0625, "nrmse": 2.0e-4},
            {"nr": 65, "nz": 65, "h": 0.03125, "nrmse": 1.0e-4},
        ]
    )

    summary = summarise_convergence_contract(rows, min_rate=1.8)

    assert summary["passed"] is False
    assert summary["min_convergence_rate"] == pytest.approx(1.0)
    assert summary["required_min_rate"] == 1.8
    assert summary["rated_grid_count"] == 2


def test_summarise_convergence_contract_passes_second_order_solver():
    """The manufactured Solov'ev benchmark contract is second-order or better."""
    rows = add_convergence_rates(
        [
            {"nr": 17, "nz": 17, "h": 0.125, "nrmse": 4.0e-4},
            {"nr": 33, "nz": 33, "h": 0.0625, "nrmse": 1.0e-4},
            {"nr": 65, "nz": 65, "h": 0.03125, "nrmse": 2.5e-5},
        ]
    )

    summary = summarise_convergence_contract(rows, min_rate=1.8)

    assert summary["passed"] is True
    assert summary["min_convergence_rate"] == pytest.approx(2.0)
    assert summary["required_min_rate"] == 1.8
    assert summary["rated_grid_count"] == 2
