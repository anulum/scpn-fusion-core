# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Fixed-Physical Coil-Vacuum Test Fixtures
"""Reusable valid CVGC2 report fixture for entry-point tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from validation import ida_coil_vacuum_fixed_physical_contract as contract

ROOT = Path(__file__).resolve().parents[1]
_DIGEST = "1" * 64
FROZEN_SOURCE_COMMIT = "fee2a81432cccec9ad422979beb56777646e0a18"


def _field_metric(scale: float) -> dict[str, Any]:
    """Return one internally consistent field metric."""
    return {
        "area_weighted_l2": scale,
        "area_weighted_rms": 0.5 * scale,
        "field_sha256": _DIGEST,
        "l2": 2.0 * scale,
        "linf": scale,
    }


def _comparison_metric(scale: float) -> dict[str, float]:
    """Return one finite exact-restriction comparison metric."""
    return {
        "area_weighted_l2": scale,
        "area_weighted_rms": 0.5 * scale,
        "cosine": 1.0,
        "linf": scale,
        "projection": 0.25,
        "relative_l2": 0.75,
    }


def source_artifacts_fixture() -> dict[str, dict[str, Any]]:
    """Return test provenance from the authenticated commit stored in CVGC2 evidence.

    Test fixtures validate report structure, not live report generation. Binding this
    helper to ``HEAD`` makes every later unrelated repository commit fail the intentionally
    frozen CVGC2 source-bundle check.
    """
    artifacts = contract.source_artifacts_for_commit(ROOT, FROZEN_SOURCE_COMMIT)
    artifacts["repository"]["worktree_clean"] = True
    return artifacts


def report_fixture() -> dict[str, Any]:
    """Build a self-validating passing CVGC2 report."""
    grids: list[dict[str, Any]] = []
    for resolution, error in zip(
        (33, 65, 129, 257),
        (0.04, 0.01, 0.0025, 0.000625),
        strict=True,
    ):
        grids.append(
            {
                "current_recovery_weighted_error": error,
                "fixed_source_l2_fraction": 0.99,
                "forcing_closure_max_abs": 0.0,
                "forcing_partition": {
                    "source": _field_metric(1.0),
                    "source_free": _field_metric(0.01),
                    "total": _field_metric(1.01),
                },
                "inverse_timings_ms": {
                    "inverse_source": 1.0,
                    "inverse_source_free": 1.0,
                    "inverse_total": 1.0,
                },
                "resolution": resolution,
                "response_partition": {
                    "closure_max_abs_wb": 1.0e-14,
                    "source": _field_metric(1.0),
                    "source_free": _field_metric(0.01),
                    "total": _field_metric(1.0),
                },
                "source_free_response_fraction_of_total": 0.01,
                "total_response_reproduction_max_abs_wb": 1.0e-14,
            }
        )
    pairwise = {
        pair: {
            region: {
                "forcing": _comparison_metric(0.01),
                "response": _comparison_metric(0.001),
            }
            for region in (
                "fixed_physical_source_free",
                "full_interior",
                "plasma_support",
            )
        }
        for pair in ("33_65", "65_129", "129_257")
    }
    convergence = {
        "pairwise": pairwise,
        "source_free_forcing_order": {
            triple: {
                "coarse_to_medium_rms": 0.04,
                "medium_to_fine_rms": 0.01,
                "observed_order": 2.0,
            }
            for triple in contract.GRID_TRIPLES
        },
        "source_free_response_order": {
            triple: {
                "coarse_to_medium_rms": 0.004,
                "medium_to_fine_rms": 0.001,
                "observed_order": 2.0,
            }
            for triple in contract.GRID_TRIPLES
        },
    }
    artifacts = source_artifacts_fixture()
    upstream = contract.load_upstream_report(ROOT)
    return contract.build_report(
        generated_at="2026-07-26T03:30:00Z",
        environment={"backend": "gpu", "x64_enabled": True},
        execution_binding=contract.build_execution_binding(
            anchor=upstream["anchor"],
            coil_manifest=upstream["coil_manifest"],
            source_artifacts=artifacts,
        ),
        source_artifacts=artifacts,
        upstream=contract.upstream_binding(ROOT, upstream),
        grids=grids,
        convergence=convergence,
    )
