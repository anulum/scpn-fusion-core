# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Coil-Vacuum Grid Contract Tests
"""Fail-closed tests for coil-vacuum grid-convergence evidence."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from tests.ida_coil_vacuum_grid_fixtures import (
    _digest,
    _recovery_metric,
    _report,
    _resign,
    _resign_manifest,
)
from validation import ida_coil_vacuum_grid_contract as contract


def test_build_validate_render_and_keep_every_claim_false() -> None:
    """A passing numerical payload must remain blocked from admission claims."""
    report = _report()
    contract.validate_report(report)

    assert report["routing"] == {
        "solver_physics_changed": False,
        "source_numerics_pass": True,
        "state": "coil_source_footprint_resolved",
        "vacuum_numerics_pass": True,
    }
    assert report["status"] == "diagnostic_complete_claims_blocked"
    assert set(report["claim_boundary"].values()) == {False}
    markdown = contract.render_markdown(report)
    assert "Routing: `coil_source_footprint_resolved`" in markdown
    assert "| 257 | 0.99 | 0.01 | 1e-14 |" in markdown
    assert "production physics admission: `false`" in markdown


@pytest.mark.parametrize(
    ("report", "state"),
    [
        (
            _report(
                recovery_errors=(0.04, 0.06, 0.07, 0.08),
            ),
            "coil_source_discretisation_unresolved",
        ),
        (
            _report(
                observed_order=1.0,
                response_relative_l2=0.06,
            ),
            "vacuum_operator_discretisation_unresolved",
        ),
        (
            _report(
                recovery_errors=(0.04, 0.06, 0.07, 0.08),
                localisation=0.90,
                observed_order=1.0,
                response_relative_l2=0.06,
            ),
            "mixed_source_and_vacuum_error",
        ),
    ],
)
def test_routing_uses_only_predeclared_numerical_gates(
    report: dict[str, Any],
    state: str,
) -> None:
    """Source, vacuum, and mixed failures must route without threshold edits."""
    contract.validate_report(report)
    assert report["routing"]["state"] == state
    assert report["routing"]["solver_physics_changed"] is False


def test_validate_rejects_overclaim_binding_drift_and_dirty_source() -> None:
    """Claim, prerequisite, and clean-source provenance are fail-closed."""
    report = _report()

    overclaim = copy.deepcopy(report)
    overclaim["claim_boundary"]["scientific_validation"] = True
    _resign(overclaim)
    with pytest.raises(ValueError, match="claim boundary"):
        contract.validate_report(overclaim)

    binding = copy.deepcopy(report)
    binding["bindings"]["response"]["payload_sha256"] = _digest("changed")
    _resign(binding)
    with pytest.raises(ValueError, match="frozen payload"):
        contract.validate_report(binding)

    dirty = copy.deepcopy(report)
    dirty["source_artifacts"]["repository"]["worktree_clean"] = False
    _resign(dirty)
    with pytest.raises(ValueError, match="clean canonical repository"):
        contract.validate_report(dirty)


def test_validate_rejects_anchor_manifest_and_grid_tampering() -> None:
    """Anchor bytes, manifest lineage, and complete object rows are mandatory."""
    report = _report()

    anchor = copy.deepcopy(report)
    anchor["anchor"]["response_sha256"] = _digest("changed")
    _resign(anchor)
    with pytest.raises(ValueError, match="response anchor"):
        contract.validate_report(anchor)

    manifest = copy.deepcopy(report)
    manifest["coil_manifest"]["parents"][0]["filaments"][0]["filament_id"] = manifest[
        "coil_manifest"
    ]["parents"][0]["filaments"][1]["filament_id"]
    _resign(manifest)
    with pytest.raises(ValueError, match="identifiers"):
        contract.validate_report(manifest)

    rows = copy.deepcopy(report)
    rows["grids"][0] = "not-an-object"
    _resign(rows)
    with pytest.raises(ValueError, match="object rows"):
        contract.validate_report(rows)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("parent_index", "parent indices"),
        ("parent_name", "parent names"),
        ("coil_type", "coil_type"),
        ("zero_turns", "turns must be positive"),
        ("parent_effective", "parent effective current"),
        ("filament_lineage", "filament lineage"),
        ("filament_domain", "outside the fixed domain"),
        ("filament_effective", "filament effective current"),
        ("child_closure", "child currents do not close"),
        ("flattened_count", "flattened filament cardinality"),
    ],
)
def test_manifest_contract_rejects_lineage_current_and_domain_drift(
    case: str,
    message: str,
) -> None:
    """Every parent/filament identity and signed-current invariant is enforced."""
    report = _report()
    parent = report["coil_manifest"]["parents"][0]
    if case == "parent_index":
        parent["parent_index"] = -1
    elif case == "parent_name":
        parent["name"] = "FC2"
        for filament in parent["filaments"]:
            filament["parent_name"] = "FC2"
            filament["filament_id"] = f"FC2:{filament['filament_index']:03d}"
    elif case == "coil_type":
        parent["coil_type"] = ""
    elif case == "zero_turns":
        parent["turns"] = 0.0
    elif case == "parent_effective":
        parent["effective_current_a_turns"] = 2.0
    elif case == "filament_lineage":
        parent["filaments"][0]["parent_name"] = "wrong"
    elif case == "filament_domain":
        parent["filaments"][0]["r_m"] = 3.0
    elif case == "filament_effective":
        parent["filaments"][0]["effective_current_a_turns"] = 0.5
    elif case == "child_closure":
        for filament in parent["filaments"]:
            filament["weight"] = 1.0 / 13.0
            filament["effective_current_a_turns"] = 1.0 / 13.0
    elif case == "flattened_count":
        for filament in parent["filaments"]:
            filament["weight"] = 1.0 / 13.0
            filament["effective_current_a_turns"] = 1.0 / 13.0
        parent["filaments"].append(
            {
                "effective_current_a_turns": 1.0 / 13.0,
                "filament_id": "FC1:012",
                "filament_index": 12,
                "parent_index": 0,
                "parent_name": "FC1",
                "r_m": 1.12,
                "weight": 1.0 / 13.0,
                "z_m": -0.5,
            }
        )
        parent["filament_count"] = 13
    else:
        raise AssertionError(f"unhandled manifest case {case}")
    _resign_manifest(report)
    with pytest.raises(ValueError, match=message):
        contract.validate_report(report)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("field_metric", "fields are invalid"),
        ("comparison_metric", "fields are invalid"),
        ("mask_summary", "mask.*fields"),
        ("mask_count", "point_count"),
        ("mask_area", "area_m2 is inconsistent"),
        ("zero_mask_fraction", "point_fraction is inconsistent"),
        ("recovery_metric", "regions are invalid|fields are invalid"),
        ("recovery_absolute", "absolute_error"),
        ("recovery_signed", "signed_error"),
        ("recovery_relative", "relative_error"),
        ("grid_geometry", "grid geometry fields"),
        ("grid_domain", "grid physical domain"),
        ("grid_dr", "grid d_r_m"),
        ("grid_dz", "grid d_z_m"),
        ("grid_area", "grid cell_area_m2"),
        ("mask_fields", "grid mask fields"),
        ("fixed_radius", "fixed physical radius"),
        ("mask_nesting", "mask nesting"),
        ("mask_fraction", "point_fraction is inconsistent"),
        ("mask_overlap", "mask overlap"),
        ("empty_plasma", "plasma support mask"),
        ("forcing_fields", "forcing partition fields"),
        ("sensitivity_fields", "forcing sensitivity fields"),
        ("sensitivity_nesting", "forcing sensitivity nesting"),
        ("response_fields", "response partition fields"),
        ("vacuum_fields", "vacuum field fields"),
        ("parity_fields", "source-free parity fields"),
        ("parity_count", "parity point_count"),
        ("recovery_fields", "current recovery fields"),
        ("recovery_parents", "current recovery parent rows"),
        ("recovery_parent_row", "parent row fields"),
        ("recovery_regions", "parent regions"),
        ("recovery_expected", "expected current disagrees"),
        ("recovery_aggregate", "aggregate fields"),
        ("recovery_target", "aggregate target disagrees"),
        ("recovery_weighted", "weighted_primary_error is inconsistent"),
        ("timings", "timing fields"),
        ("pairwise", "pairwise fields"),
        ("pairwise_regions", "regions are invalid"),
        ("pairwise_surface", "fields are invalid"),
        ("order_row", "order 33_65_129 fields"),
        ("zero_order_difference", "differences must be non-zero"),
    ],
)
def test_measurement_contract_rejects_malformed_measured_surfaces(
    case: str,
    message: str,
) -> None:
    """Every emitted grid, recovery, and convergence sub-surface is fail-closed."""
    report = _report()
    grid = report["grids"][0]
    if case == "field_metric":
        del grid["forcing_partition"]["source"]["linf"]
    elif case == "comparison_metric":
        del report["convergence"]["pairwise"]["33_65"]["full"]["forcing"]["linf"]
    elif case == "mask_summary":
        del grid["masks"]["rho_h_le_1"]["area_m2"]
    elif case == "mask_count":
        grid["masks"]["rho_h_le_1"]["point_count"] = True
    elif case == "mask_area":
        grid["masks"]["rho_h_le_1"]["area_m2"] = 99.0
    elif case == "zero_mask_fraction":
        grid["masks"]["rho_h_le_1"] = {
            "area_m2": 0.0,
            "point_count": 0,
            "point_fraction": 0.1,
        }
    elif case == "recovery_metric":
        del grid["current_recovery"]["parents"][0]["regions"]["primary"]["relative_error"]
    elif case == "recovery_absolute":
        grid["current_recovery"]["parents"][0]["regions"]["primary"]["absolute_error_a_turns"] = (
            99.0
        )
    elif case == "recovery_signed":
        grid["current_recovery"]["parents"][0]["regions"]["primary"]["signed_error_a_turns"] = 99.0
    elif case == "recovery_relative":
        grid["current_recovery"]["parents"][0]["regions"]["primary"]["relative_error"] = 99.0
    elif case == "grid_geometry":
        del grid["grid"]["d_r_m"]
    elif case == "grid_domain":
        grid["grid"]["r_bounds_m"] = [0.0, 1.0]
    elif case == "grid_dr":
        grid["grid"]["d_r_m"] = 1.0
    elif case == "grid_dz":
        grid["grid"]["d_z_m"] = 1.0
    elif case == "grid_area":
        grid["grid"]["cell_area_m2"] = 1.0
    elif case == "mask_fields":
        del grid["masks"]["fixed_physical"]
    elif case == "fixed_radius":
        grid["masks"]["fixed_physical_radius_m"] = 1.0
    elif case == "mask_nesting":
        grid["masks"]["rho_h_le_1"]["point_count"] = 21
        grid["masks"]["rho_h_le_1"]["area_m2"] = 21 * grid["grid"]["cell_area_m2"]
        grid["masks"]["rho_h_le_1"]["point_fraction"] = 21 / (33 * 33)
    elif case == "mask_fraction":
        grid["masks"]["rho_h_le_1"]["point_fraction"] = 0.5
    elif case == "mask_overlap":
        grid["masks"]["primary_fixed_overlap_point_count"] = 21
    elif case == "empty_plasma":
        grid["masks"]["plasma_support"] = {
            "area_m2": 0.0,
            "point_count": 0,
            "point_fraction": 0.0,
        }
    elif case == "forcing_fields":
        del grid["forcing_partition"]["source"]
    elif case == "sensitivity_fields":
        del grid["forcing_partition"]["sensitivity_l2_fraction"]["rho_h_le_1"]
    elif case == "sensitivity_nesting":
        grid["forcing_partition"]["sensitivity_l2_fraction"]["rho_h_le_1"] = 1.0
    elif case == "response_fields":
        del grid["response_partition"]["source"]
    elif case == "vacuum_fields":
        del grid["vacuum_fields"]["freegs"]
    elif case == "parity_fields":
        del grid["vacuum_fields"]["source_free_parity"]["point_count"]
    elif case == "parity_count":
        grid["vacuum_fields"]["source_free_parity"]["point_count"] = 0
    elif case == "recovery_fields":
        del grid["current_recovery"]["aggregate"]
    elif case == "recovery_parents":
        grid["current_recovery"]["parents"] = []
    elif case == "recovery_parent_row":
        grid["current_recovery"]["parents"][0]["extra"] = True
    elif case == "recovery_regions":
        del grid["current_recovery"]["parents"][0]["regions"]["source_free"]
    elif case == "recovery_expected":
        for name, metric in grid["current_recovery"]["parents"][0]["regions"].items():
            grid["current_recovery"]["parents"][0]["regions"][name] = _recovery_metric(
                expected=2.0,
                recovered=float(metric["recovered_a_turns"]),
            )
    elif case == "recovery_aggregate":
        del grid["current_recovery"]["aggregate"]["target_net_a_turns"]
    elif case == "recovery_target":
        grid["current_recovery"]["aggregate"]["target_net_a_turns"] = 0.0
    elif case == "recovery_weighted":
        grid["current_recovery"]["weighted_primary_error"] = 0.5
    elif case == "timings":
        del grid["timings_ms"]["native_field"]
    elif case == "pairwise":
        report["convergence"]["pairwise"] = {}
    elif case == "pairwise_regions":
        report["convergence"]["pairwise"]["33_65"] = {}
    elif case == "pairwise_surface":
        report["convergence"]["pairwise"]["33_65"]["full"] = {}
    elif case == "order_row":
        report["convergence"]["source_free_forcing_order"]["33_65_129"] = {}
    elif case == "zero_order_difference":
        report["convergence"]["source_free_forcing_order"]["33_65_129"]["coarse_to_medium_rms"] = (
            0.0
        )
    else:
        raise AssertionError(f"unhandled measurement case {case}")
    _resign(report)
    with pytest.raises(ValueError, match=message):
        contract.validate_report(report)


def test_validate_rejects_open_closure_routing_forgery_and_nonfinite_data() -> None:
    """Open closure, forged routing, and non-finite metrics must never validate."""
    report = _report()

    closure = copy.deepcopy(report)
    closure["grids"][0]["response_partition"]["closure_max_abs_wb"] = 1.0e-3
    _resign(closure)
    with pytest.raises(ValueError, match="closure is open"):
        contract.validate_report(closure)

    routing = copy.deepcopy(report)
    routing["routing"]["state"] = "coil_source_footprint_resolved"
    routing["routing"]["source_numerics_pass"] = False
    _resign(routing)
    with pytest.raises(ValueError, match="routing is inconsistent"):
        contract.validate_report(routing)

    nonfinite = copy.deepcopy(report)
    nonfinite["environment"]["load"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        contract.validate_report(nonfinite)


def test_validate_rejects_signature_and_top_level_shape_drift() -> None:
    """Unsigned mutation and unknown top-level fields must fail independently."""
    report = _report()
    report["generated_at"] = "2026-07-25T19:31:00Z"
    with pytest.raises(ValueError, match="payload_sha256"):
        contract.validate_report(report)

    shape = _report()
    shape["unknown"] = True
    with pytest.raises(ValueError, match="top-level fields"):
        contract.validate_report(shape)


def test_validate_rejects_invalid_timestamp_and_empty_environment() -> None:
    """Evidence identity requires an explicit UTC timestamp and runtime environment."""
    timestamp = _report()
    timestamp["generated_at"] = "not-a-utc-timestamp"
    _resign(timestamp)
    with pytest.raises(ValueError, match="generated_at"):
        contract.validate_report(timestamp)

    environment = _report()
    environment["environment"] = {}
    _resign(environment)
    with pytest.raises(ValueError, match="environment"):
        contract.validate_report(environment)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("bindings_shape", "bindings fields"),
        ("binding_row", "bindings.response fields"),
        ("binding_path", "bindings.response.path"),
        ("source_artifacts_shape", "source_artifacts fields"),
        ("source_artifact_row", "source_artifacts.contract fields"),
        ("source_artifact_path", "source_artifacts.contract.path"),
        ("runtime_artifact_path", "source_artifacts.freegs_boundary.path"),
        ("source_artifact_digest", "lowercase SHA-256"),
        ("repository_shape", "repository fields"),
        ("repository_oid", "repository.git_commit"),
        ("anchor_shape", "anchor fields"),
        ("forcing_anchor", "forcing anchor"),
        ("anchor_closure", "anchor closure"),
        ("manifest_shape", "coil_manifest fields"),
        ("manifest_cardinality", "cardinality"),
        ("manifest_parents", "parents are invalid"),
        ("manifest_parent_row", "parent row"),
        ("manifest_filaments", "filament rows"),
        ("manifest_filament_row", "filament row"),
        ("manifest_digest", "manifest_sha256"),
        ("grid_ladder", "exact required ladder"),
        ("grid_resolution_type", "resolution must be an integer"),
        ("grid_fields", "grid row fields"),
        ("parity_nrmse", "Green-function parity"),
        ("parity_nrmse_type", "parity nrmse must be numeric"),
        ("parity_max_abs", "Green-function parity"),
        ("parity_flag", "Green-function parity"),
        ("forcing_closure", "partition closure"),
        ("identity", "report identity"),
        ("frozen_design", "frozen design"),
        ("blockers", "blockers"),
        ("convergence_shape", "convergence fields"),
        ("order_shape", "forcing order fields"),
        ("gates", "gates are inconsistent"),
        ("status", "status is inconsistent"),
        ("unsupported_value", "unsupported value type"),
    ],
)
def test_validate_rejects_fail_closed_contract_mutation_matrix(
    case: str,
    message: str,
) -> None:
    """Every evidence lineage, shape, numerical, and routing boundary is closed."""
    report = _report()
    if case == "bindings_shape":
        report["bindings"] = {}
    elif case == "binding_row":
        report["bindings"]["response"] = {}
    elif case == "binding_path":
        report["bindings"]["response"]["path"] = "wrong.json"
    elif case == "source_artifacts_shape":
        report["source_artifacts"] = {}
    elif case == "source_artifact_row":
        report["source_artifacts"]["contract"] = {}
    elif case == "source_artifact_path":
        report["source_artifacts"]["contract"]["path"] = "wrong.py"
    elif case == "runtime_artifact_path":
        report["source_artifacts"]["freegs_boundary"]["path"] = ""
    elif case == "source_artifact_digest":
        report["source_artifacts"]["contract"]["sha256"] = "BAD"
    elif case == "repository_shape":
        del report["source_artifacts"]["repository"]["path"]
    elif case == "repository_oid":
        report["source_artifacts"]["repository"]["git_commit"] = "bad"
    elif case == "anchor_shape":
        del report["anchor"]["response_sha256"]
    elif case == "forcing_anchor":
        report["anchor"]["forcing_sha256"] = _digest("changed")
    elif case == "anchor_closure":
        report["anchor"]["response_closure_max_abs_wb"] = 1.0e-3
    elif case == "manifest_shape":
        del report["coil_manifest"]["parents"]
    elif case == "manifest_cardinality":
        report["coil_manifest"]["parent_count"] = 17
    elif case == "manifest_parents":
        report["coil_manifest"]["parents"] = []
    elif case == "manifest_parent_row":
        report["coil_manifest"]["parents"][0] = "bad"
    elif case == "manifest_filaments":
        report["coil_manifest"]["parents"][0]["filaments"] = []
    elif case == "manifest_filament_row":
        report["coil_manifest"]["parents"][0]["filaments"][0] = "bad"
    elif case == "manifest_digest":
        report["coil_manifest"]["manifest_sha256"] = _digest("changed")
    elif case == "grid_ladder":
        report["grids"][0]["resolution"] = 17
    elif case == "grid_resolution_type":
        report["grids"][0]["resolution"] = 33.0
    elif case == "grid_fields":
        report["grids"][0]["unexpected"] = True
    elif case == "parity_nrmse":
        report["grids"][0]["vacuum_fields"]["source_free_parity"]["nrmse"] = 1.0e-3
    elif case == "parity_nrmse_type":
        report["grids"][0]["vacuum_fields"]["source_free_parity"]["nrmse"] = "nan"
    elif case == "parity_max_abs":
        report["grids"][0]["vacuum_fields"]["source_free_parity"]["max_abs_wb"] = 1.0e-3
    elif case == "parity_flag":
        report["grids"][0]["vacuum_fields"]["source_free_parity"]["passes"] = False
    elif case == "forcing_closure":
        report["grids"][0]["forcing_partition"]["closure_max_abs"] = 1.0e-3
    elif case == "identity":
        report["schema_version"] = "wrong"
    elif case == "frozen_design":
        report["input_contract"]["required_resolutions"] = [33]
    elif case == "blockers":
        report["blockers"] = []
    elif case == "convergence_shape":
        report["convergence"] = {}
    elif case == "order_shape":
        report["convergence"]["source_free_forcing_order"] = {}
    elif case == "gates":
        report["gates"]["all_structural_pass"] = False
    elif case == "status":
        report["status"] = "blocked_incomplete_required_ladder"
    elif case == "unsupported_value":
        report["environment"]["unsupported"] = (1, 2)
    else:
        raise AssertionError(f"unhandled mutation case {case}")
    _resign(report)
    with pytest.raises(ValueError, match=message):
        contract.validate_report(report)


@pytest.mark.parametrize(
    ("value", "minimum", "maximum", "message"),
    [
        (True, None, None, "numeric"),
        (float("nan"), None, None, "finite"),
        (-1.0, 0.0, None, "must be >="),
        (2.0, None, 1.0, "must be <="),
    ],
)
def test_require_number_rejects_non_numeric_nonfinite_and_out_of_range(
    value: object,
    minimum: float | None,
    maximum: float | None,
    message: str,
) -> None:
    """Numerical gates must reject type, finiteness, and range violations."""
    with pytest.raises(ValueError, match=message):
        contract._require_number(
            value,
            field="measured",
            minimum=minimum,
            maximum=maximum,
        )


def test_blocked_routing_is_explicit_for_structural_failure() -> None:
    """A structural failure must map only to the incomplete-ladder state."""
    routing = contract._routing(
        {"four_grid_ladder_complete": False},
        {
            "current_recovery_fine": True,
            "current_recovery_non_increasing": True,
            "finest_response_stability": True,
            "source_free_observed_order": True,
            "source_localisation": True,
        },
    )
    assert routing["state"] == "blocked_incomplete_required_ladder"
    assert routing["solver_physics_changed"] is False
