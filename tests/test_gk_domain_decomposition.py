"""Tests for radial/toroidal nonlinear GK decomposition contracts."""

from __future__ import annotations

import pytest

from scpn_fusion.core.gk_domain_decomposition import build_radial_toroidal_decomposition
from validation.benchmark_production_decomposition_contract import run_benchmark, write_reports


def test_radial_toroidal_decomposition_covers_domain_once() -> None:
    plan = build_radial_toroidal_decomposition(
        n_radial=17,
        n_toroidal=9,
        n_theta=8,
        n_vpar=6,
        n_mu=4,
        radial_parts=4,
        toroidal_parts=3,
        halo=1,
    )

    assert plan.total_ranks == 12
    assert plan.total_owned_phase_cells == 17 * 9 * 8 * 6 * 4
    assert plan.owned_cell_imbalance <= 1.6
    assert plan.halo_overhead_ratio > 1.0
    plan.validate()


def test_decomposition_rejects_invalid_partition() -> None:
    with pytest.raises(ValueError, match="parts must not exceed axis size"):
        build_radial_toroidal_decomposition(
            n_radial=4,
            n_toroidal=4,
            n_theta=8,
            n_vpar=6,
            n_mu=4,
            radial_parts=5,
            toroidal_parts=1,
        )


def test_production_decomposition_contract_is_fail_closed() -> None:
    report = run_benchmark()
    write_reports(report)

    assert report["schema"] == "production-decomposition-contract.v1"
    assert report["contract_pass"] is True
    assert report["production_scale_ready"] is False
    assert report["status"].startswith("blocked_")
    assert report["missing_requirements"]
