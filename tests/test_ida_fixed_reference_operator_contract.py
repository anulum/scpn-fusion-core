# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Contract tests for the fixed-reference GS operator decomposition."""

from __future__ import annotations

import copy
from typing import Any

import pytest

import validation.ida_fixed_reference_operator_contract as contract


def test_freegs_sources_are_bound_to_executed_package_resources() -> None:
    assert contract.SCHEMA_VERSION.endswith(".v2")
    assert contract.SOURCE_PATHS["freegs_boundary"] == ("python-package://freegs/boundary.py")
    assert contract.SOURCE_PATHS["freegs_operator"] == ("python-package://freegs/gradshafranov.py")


def _metric(digest: str, relative: float) -> dict[str, Any]:
    return {
        "field_sha256": digest * 64,
        "linf": relative * 2.0,
        "relative_l2_to_reference_scale": relative,
        "rms": relative,
    }


def _report() -> dict[str, Any]:
    interior = {
        name: _metric(digest, relative)
        for name, digest, relative in (
            ("exact_source_convention", "1", 0.04),
            ("freegs_fourth_order_baseline", "2", 1.0e-12),
            ("second_order_operator", "3", 0.20),
            ("vacuum_discretisation", "4", 0.03),
        )
    }
    wall = {
        name: _metric(digest, relative)
        for name, digest, relative in (
            ("coil_vacuum_convention", "5", 1.0e-12),
            ("exact_source_convention", "6", 0.02),
            ("plasma_response_quadrature", "7", 0.08),
        )
    }
    operator_residuals = {
        name: _metric(format(index, "x"), 0.2 + 0.01 * index)
        for index, name in enumerate(contract.OPERATOR_RESIDUALS, start=8)
    }
    wall_residuals = {
        name: _metric(digest, relative)
        for name, digest, relative in (
            ("compact_source_total_flux", "d", 0.10),
            ("exact_source_total_flux", "e", 0.10),
            ("reference_current_total_flux", "f", 0.08),
        )
    }
    return contract.build_report(
        generated_at="2026-07-23T13:00:00Z",
        environment={
            "affinity_cpu_count": 4,
            "backend": "cpu",
            "devices": ["TFRT_CPU_0"],
            "freegs_version": "0.8.2",
            "host_load_1m_5m_15m": [0.1, 0.2, 0.3],
            "isolated_host": False,
            "jax_version": "0.7.2",
            "jaxlib_version": "0.7.2",
            "machine": "x86_64",
            "platform": "test",
            "python_version": "3.12.0",
            "x64_enabled": True,
        },
        source_artifacts={
            **{
                name: {"path": path, "sha256": digest * 64}
                for (name, path), digest in zip(
                    sorted(contract.SOURCE_PATHS.items()),
                    ("1", "2", "3", "4", "5", "6"),
                    strict=True,
                )
            },
            "freegs_public_example": {
                "path": contract.FREEGS_EXAMPLE_PATH,
                "sha256": "7" * 64,
            },
        },
        source_same_case={
            "case_id": contract.EVALUATION_CASE_ID,
            "grid_shape": contract.GRID_SHAPE,
            "path": contract.SAME_CASE_PATH,
            "payload_sha256": "8" * 64,
            "source_commit": "9" * 40,
        },
        source_ablation={
            "path": contract.SOURCE_ABLATION_PATH,
            "payload_sha256": "a" * 64,
            "source_commit": "b" * 40,
            "source_same_case_payload_sha256": "8" * 64,
        },
        operator_residuals=operator_residuals,
        interior_components=interior,
        wall_residuals=wall_residuals,
        wall_components=wall,
        closure={
            "interior_compact_max_abs": 1.0e-12,
            "interior_exact_max_abs": 1.0e-12,
            "wall_compact_max_abs": 1.0e-14,
            "wall_exact_max_abs": 1.0e-14,
        },
        coil_region_diagnostic={
            "all_interior_vacuum_field": _metric("a", 4.0),
            "coil_filament_count": 216,
            "coil_filaments_inside_domain": 216,
            "outside_reference_support_l2_fraction": 0.99,
            "reference_plasma_support_fraction": 0.2,
            "reference_plasma_support_point_count": 100,
            "reference_plasma_support_vacuum_field": _metric("b", 0.03),
        },
        source_commit="c" * 40,
        source_worktree_clean=True,
    )


def test_build_validate_and_render_routes_measured_components() -> None:
    report = _report()
    contract.validate_report(report)
    assert report["routing"]["interior_dominant_component"] == "second_order_operator"
    assert report["routing"]["next_interior_target"] == "discrete_operator_order_and_stencil"
    assert report["routing"]["wall_dominant_component"] == "plasma_response_quadrature"
    assert report["claim_boundary"] == {field: False for field in contract.CLAIM_FIELDS}
    markdown = contract.render_markdown(report)
    assert "second_order_operator" in markdown
    assert "plasma_response_quadrature" in markdown
    assert "not a validation or admission result" in markdown


def test_validate_accepts_legacy_v1_source_paths() -> None:
    report = _report()
    report["schema_version"] = contract.LEGACY_SCHEMA_VERSION
    for name, path in contract.LEGACY_SOURCE_PATHS.items():
        report["source_artifacts"][name]["path"] = path
    report["payload_sha256"] = contract._payload_sha256(report)
    contract.validate_report(report)


def test_validate_rejects_tamper_overclaim_and_binding_drift() -> None:
    report = _report()
    tampered = copy.deepcopy(report)
    tampered["routing"]["interior_dominant_component"] = "vacuum_discretisation"
    tampered["payload_sha256"] = contract._payload_sha256(tampered)
    with pytest.raises(ValueError, match="routing is inconsistent"):
        contract.validate_report(tampered)

    overclaimed = copy.deepcopy(report)
    overclaimed["claim_boundary"]["scientific_validation"] = True
    overclaimed["payload_sha256"] = contract._payload_sha256(overclaimed)
    with pytest.raises(ValueError, match="claim_boundary"):
        contract.validate_report(overclaimed)

    unbound = copy.deepcopy(report)
    unbound["source_ablation"]["source_same_case_payload_sha256"] = "d" * 64
    unbound["payload_sha256"] = contract._payload_sha256(unbound)
    with pytest.raises(ValueError, match="bindings disagree"):
        contract.validate_report(unbound)


def test_validate_rejects_non_finite_or_malformed_metrics() -> None:
    report = _report()
    malformed = copy.deepcopy(report)
    del malformed["interior_components"]["second_order_operator"]["linf"]
    malformed["payload_sha256"] = contract._payload_sha256(malformed)
    with pytest.raises(ValueError, match="fields are invalid"):
        contract.validate_report(malformed)

    non_finite = copy.deepcopy(report)
    non_finite["closure"]["wall_exact_max_abs"] = float("inf")
    with pytest.raises(ValueError):
        contract.validate_report(non_finite)
