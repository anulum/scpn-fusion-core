# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — pinned FreeGSNKE inverse comparison tests
"""Real-surface tests for the pinned FreeGSNKE inverse comparison."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("freegsnke")

import validation.benchmark_freegsnke_inverse as benchmark


@pytest.fixture(scope="session")
def live_report() -> dict[str, Any]:
    """Execute the public inverse case once for all assertions in this module."""
    return benchmark.build_report()


def test_live_inverse_comparison_passes_every_predeclared_gate(
    live_report: dict[str, Any],
) -> None:
    assert live_report["schema_version"] == benchmark.SCHEMA_VERSION
    assert live_report["benchmark_id"] == benchmark.BENCHMARK_ID
    assert live_report["status"] == "pass"
    assert live_report["checks"]
    assert all(live_report["checks"].values())

    unsigned = dict(live_report)
    unsigned["payload_sha256"] = ""
    assert live_report["payload_sha256"] == hashlib.sha256(
        benchmark._canonical_json(unsigned)
    ).hexdigest()


def test_live_case_exercises_the_same_machine_and_gradient_surfaces(
    live_report: dict[str, Any],
) -> None:
    machine = live_report["machine_contract"]
    assert machine["active_circuit_count"] == 12
    assert machine["active_filament_count"] == 876
    assert machine["passive_circuit_count"] > 0
    assert machine["limiter_comparison_point_count"] > 1000
    assert machine["current_limits"]["passed"] is True

    inverse = live_report["inverse_regression"]
    assert inverse["solver_relative_change"] <= 1.0e-6
    assert inverse["active_current_max_abs_error_a"] <= benchmark.CURRENT_ATOL_A
    assert inverse["total_psi_max_abs_error_wb"] <= inverse["total_psi_atol_wb"]

    vacuum = live_report["vacuum_bridge"]
    assert vacuum["max_abs_error_wb"] <= benchmark.VACUUM_LIMITER_MAX_ABS_WB
    gradient = live_report["gradient_audit"]
    assert gradient["all_finite"] is True
    assert gradient["relative_l2_error"] <= benchmark.GRADIENT_RELATIVE_ERROR_MAX


def test_claim_boundary_remains_fail_closed(live_report: dict[str, Any]) -> None:
    assert live_report["claim_boundary"]
    assert not any(live_report["claim_boundary"].values())
    assert live_report["blockers"]
    assert "total-psi cross-solver parity" in live_report["blockers"][0]
    assert "full plasma-source or total-psi" in " ".join(
        live_report["comparison_scope"]["not_admitted"]
    )


def test_markdown_preserves_result_and_claim_boundary(live_report: dict[str, Any]) -> None:
    rendered = benchmark.render_markdown(live_report)
    assert "Status: **PASS**" in rendered
    assert "vacuum-psi max error inside limiter" in rendered
    assert "full total-psi cross-solver parity" in rendered
    assert live_report["payload_sha256"] in rendered


def test_pinned_source_is_the_expected_checkout() -> None:
    source = Path(benchmark.DEFAULT_SOURCE)
    versions = benchmark.validate_source(source)
    assert versions["commit"] == benchmark.UPSTREAM_COMMIT
    assert versions["freegsnke"] == benchmark.UPSTREAM_VERSION
    assert versions["freegs4e"] == benchmark.FREEGS4E_VERSION
