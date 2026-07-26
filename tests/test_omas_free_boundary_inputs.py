# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — OMAS Free-Boundary Input Tests
"""Tests for strict OMAS PF-active and magnetics input extraction."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from scpn_fusion.io import (
    OMAS_FREE_BOUNDARY_INPUT_SCHEMA,
    OmasSourceProvenance,
    extract_omas_free_boundary_inputs,
)
from scpn_fusion.io import imas_connector
from scpn_fusion.io import omas_free_boundary_inputs as omas_inputs


class DottedODS(dict[str, Any]):
    """Dictionary-backed dotted-path ODS test double."""

    cocos: int
    imas_version: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a dotted mapping with explicit schema metadata."""
        super().__init__(*args, **kwargs)
        self.cocos = 11
        self.imas_version = "3.41.0"


def _provenance() -> OmasSourceProvenance:
    return OmasSourceProvenance(
        machine="DIII-D",
        shot_id=145419,
        run_id=2,
        source_uri="omas://d3d/145419/2",
        source_sha256="a" * 64,
        license_id="access-approved-test-binding",
    )


def _full_ods() -> DottedODS:
    time = np.asarray([1.0, 1.1], dtype=np.float64)
    payload: dict[str, Any] = {}
    for coil_index in (0, 1):
        coil = f"pf_active.coil.{coil_index}"
        payload[f"{coil}.identifier"] = f"PF{coil_index + 1}"
        payload[f"{coil}.current.time"] = time
        payload[f"{coil}.current.data"] = np.asarray([1000.0 + coil_index, 1100.0 + coil_index])
        payload[f"{coil}.current.data_error_lower"] = [1.0, 1.0]
        payload[f"{coil}.current.data_error_upper"] = [1.5, 1.5]
        payload[f"{coil}.current.validity"] = 1
        element = f"{coil}.element.0"
        payload[f"{element}.identifier"] = f"PF{coil_index + 1}E1"
        payload[f"{element}.turns_with_sign"] = -12.0 if coil_index else 10.0
        payload[f"{element}.geometry.geometry_type"] = 2 if coil_index == 0 else 1
        if coil_index == 0:
            payload[f"{element}.geometry.rectangle.r"] = 1.1
            payload[f"{element}.geometry.rectangle.z"] = -0.2
            payload[f"{element}.geometry.rectangle.width"] = 0.1
            payload[f"{element}.geometry.rectangle.height"] = 0.2
        else:
            payload[f"{element}.geometry.outline.r"] = [1.2, 1.3, 1.3, 1.2]
            payload[f"{element}.geometry.outline.z"] = [-0.3, -0.3, -0.1, -0.1]

    probe = "magnetics.b_field_pol_probe.0"
    payload[f"{probe}.identifier"] = "BP1"
    payload[f"{probe}.position.r"] = 1.5
    payload[f"{probe}.position.z"] = 0.2
    payload[f"{probe}.poloidal_angle"] = 0.5
    payload[f"{probe}.length"] = 0.03
    payload[f"{probe}.field.time"] = time
    payload[f"{probe}.field.data"] = [0.01, 0.02]
    payload[f"{probe}.field.data_error_lower"] = [0.001, 0.001]
    payload[f"{probe}.field.data_error_upper"] = [0.002, 0.002]
    payload[f"{probe}.field.validity"] = 0

    loop = "magnetics.flux_loop.0"
    payload[f"{loop}.identifier"] = "FL1"
    payload[f"{loop}.position.0.r"] = 1.4
    payload[f"{loop}.position.0.z"] = -0.4
    payload[f"{loop}.position.1.r"] = 1.6
    payload[f"{loop}.position.1.z"] = -0.4
    payload[f"{loop}.flux.time"] = time
    payload[f"{loop}.flux.data"] = [0.2, 0.3]
    payload[f"{loop}.flux.data_error_lower"] = [0.01, 0.01]
    payload[f"{loop}.flux.data_error_upper"] = [0.02, 0.02]
    payload[f"{loop}.flux.validity"] = 1
    return DottedODS(payload)


def test_extracts_ingestion_ready_si_contract_and_stable_digest() -> None:
    """Complete same-axis evidence passes and preserves signs, geometry, and SI channels."""
    ods = _full_ods()

    result = extract_omas_free_boundary_inputs(
        ods,
        provenance=_provenance(),
        time_alignment="exact_common_axis",
    )
    repeated = extract_omas_free_boundary_inputs(
        ods,
        provenance=_provenance(),
        time_alignment="exact_common_axis",
    )

    assert result.schema == OMAS_FREE_BOUNDARY_INPUT_SCHEMA
    assert result.cocos == 11
    assert result.imas_version == "3.41.0"
    assert result.ingestion_ready is True
    assert result.ingestion_blockers == ()
    assert result.tier0_claim_admission_ready is False
    assert result.tier0_claim_blockers == (
        "equilibrium_and_control_targets_out_of_scope",
        "shot_split_and_heldout_evaluation_out_of_scope",
    )
    assert result.payload_sha256 == repeated.payload_sha256
    assert len(result.payload_sha256) == 64
    assert result.pf_coils[0].current_a.values == (1000.0, 1100.0)
    assert result.pf_coils[0].elements[0].width_m == pytest.approx(0.1)
    assert result.pf_coils[1].elements[0].turns_with_sign == pytest.approx(-12.0)
    assert result.pf_coils[1].elements[0].geometry_type == 1
    assert result.bpol_probes[0].field_t.values == (0.01, 0.02)
    assert result.flux_loops[0].r_m == (1.4, 1.6)
    assert imas_connector.extract_omas_free_boundary_inputs is extract_omas_free_boundary_inputs


def test_development_mode_reports_every_missing_admission_class() -> None:
    """Inspection mode returns blockers instead of fabricating missing evidence."""
    ods = _full_ods()
    for key in list(ods):
        if "data_error_" in key or key.endswith(".validity"):
            del ods[key]

    result = extract_omas_free_boundary_inputs(ods, require_ingestion_ready=False)

    assert result.ingestion_ready is False
    assert "missing_source_provenance" in result.ingestion_blockers
    assert "time_alignment_not_exact_common_axis" in result.ingestion_blockers
    assert "pf_current_0_missing_uncertainty" in result.ingestion_blockers
    assert "bpol_probe_0_missing_validity" in result.ingestion_blockers
    with pytest.raises(ValueError, match="not ingestion-ready"):
        extract_omas_free_boundary_inputs(ods)


def test_missing_channel_classes_remain_explicit_blockers() -> None:
    """An empty but structurally readable ODS cannot pass any channel gate."""
    result = extract_omas_free_boundary_inputs(DottedODS(), cocos=12, require_ingestion_ready=False)

    assert result.cocos == 12
    assert result.ingestion_blockers == (
        "missing_source_provenance",
        "missing_pf_active_coils",
        "missing_bpol_probes",
        "missing_flux_loops",
        "time_alignment_not_exact_common_axis",
    )


def test_declared_common_axis_is_verified_across_every_channel() -> None:
    """An exact-axis declaration fails closed when even one diagnostic differs."""
    ods = _full_ods()
    ods["magnetics.flux_loop.0.flux.time"] = [1.0, 1.2]

    result = extract_omas_free_boundary_inputs(
        ods,
        provenance=_provenance(),
        time_alignment="exact_common_axis",
        require_ingestion_ready=False,
    )

    assert result.ingestion_blockers == ("declared_common_axis_mismatch",)


def test_negative_diagnostic_validity_is_an_admission_blocker() -> None:
    """Problem or invalid diagnostic data remain readable but never admissible."""
    ods = _full_ods()
    ods["magnetics.b_field_pol_probe.0.field.validity"] = -1

    result = extract_omas_free_boundary_inputs(
        ods,
        provenance=_provenance(),
        time_alignment="exact_common_axis",
        require_ingestion_ready=False,
    )

    assert result.ingestion_blockers == ("bpol_probe_0_invalid_or_uncertified",)


def test_imas_version_is_required_for_admission_and_cannot_be_blank() -> None:
    """The data-dictionary version is part of the unit and path interpretation."""
    ods = _full_ods()
    del ods.imas_version
    result = extract_omas_free_boundary_inputs(
        ods,
        provenance=_provenance(),
        time_alignment="exact_common_axis",
        require_ingestion_ready=False,
    )
    assert result.ingestion_blockers == ("missing_imas_version",)

    ods.imas_version = " "
    with pytest.raises(ValueError, match="imas_version"):
        extract_omas_free_boundary_inputs(ods, require_ingestion_ready=False)


@pytest.mark.parametrize("version", ["", "3.41", "v3.41.0", True, 3.41, object()])
def test_rejects_malformed_imas_versions(version: Any) -> None:
    """Schema interpretation requires a genuine semantic-version string."""
    ods = _full_ods()
    ods.imas_version = cast(Any, version)
    with pytest.raises(ValueError, match="imas_version"):
        extract_omas_free_boundary_inputs(ods, require_ingestion_ready=False)


def test_rejects_unknown_time_alignment_modes() -> None:
    """Interpolation or resampling cannot be smuggled in through a free-form label."""
    with pytest.raises(ValueError, match="time_alignment"):
        extract_omas_free_boundary_inputs(
            _full_ods(),
            time_alignment=cast(Any, "resampled"),
            require_ingestion_ready=False,
        )


@pytest.mark.parametrize("cocos", [0, 9, 10, 11.0, 19, True])
def test_rejects_noncanonical_cocos(cocos: Any) -> None:
    """Only canonical positive COCOS indices enter the contract."""
    with pytest.raises(ValueError, match="COCOS"):
        extract_omas_free_boundary_inputs(DottedODS(), cocos=cocos)


def test_requires_cocos_when_the_container_has_no_cocos_attribute() -> None:
    """Plain dotted mappings cannot silently inherit an unknown convention."""
    with pytest.raises(ValueError, match="COCOS"):
        extract_omas_free_boundary_inputs({}, require_ingestion_ready=False)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"machine": " "}, "machine"),
        ({"source_uri": ""}, "source_uri"),
        ({"license_id": ""}, "license_id"),
        ({"machine": 3}, "machine"),
        ({"source_sha256": "ABC"}, "source_sha256"),
        ({"source_sha256": 3}, "source_sha256"),
        ({"shot_id": "1"}, "integers"),
        ({"shot_id": True}, "integers"),
        ({"shot_id": -1}, "non-negative"),
        ({"run_id": -1}, "non-negative"),
    ],
)
def test_provenance_rejects_incomplete_bindings(changes: dict[str, Any], message: str) -> None:
    """Source evidence is validated before it can affect readiness."""
    fields: dict[str, Any] = {
        "machine": "DIII-D",
        "shot_id": 1,
        "run_id": 0,
        "source_uri": "omas://source",
        "source_sha256": "A" * 64,
        "license_id": "internal",
    }
    fields.update(changes)
    with pytest.raises(ValueError, match=message):
        OmasSourceProvenance(**fields)


def test_provenance_normalizes_an_uppercase_digest() -> None:
    """A valid hexadecimal digest is stored in one canonical lowercase form."""
    assert _provenance().source_sha256 == "a" * 64


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("pf_active.coil.0.current.time", [1.0], "lengths differ"),
        ("pf_active.coil.0.current.time", [1.0, 1.0], "strictly increasing"),
        ("pf_active.coil.0.current.data", [1.0, np.nan], "finite"),
        ("pf_active.coil.0.current.data_error_lower", [1.0], "length differs"),
        ("pf_active.coil.0.current.data_error_upper", [1.0, -1.0], "non-negative"),
        ("pf_active.coil.0.current.validity", 0.5, "integer"),
        ("pf_active.coil.0.current.validity", 2, "no greater than 1"),
        ("pf_active.coil.0.element.0.turns_with_sign", 0.0, "non-zero"),
        ("pf_active.coil.0.element.0.geometry.geometry_type", 3, "must be 1"),
        ("pf_active.coil.0.element.0.geometry.rectangle.width", 0.0, "positive"),
        ("magnetics.b_field_pol_probe.0.length", 0.0, "positive"),
        ("magnetics.b_field_pol_probe.0.position.r", -1.0, "positive"),
        ("magnetics.flux_loop.0.position.0.r", 0.0, "positive"),
    ],
)
def test_rejects_malformed_channel_values(path: str, value: Any, message: str) -> None:
    """Malformed numerical or semantic fields fail before readiness is evaluated."""
    ods = _full_ods()
    ods[path] = value
    with pytest.raises(ValueError, match=message):
        extract_omas_free_boundary_inputs(ods, require_ingestion_ready=False)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("magnetics.b_field_pol_probe.0.poloidal_angle", object(), "real scalar"),
        ("magnetics.b_field_pol_probe.0.poloidal_angle", True, "not boolean"),
        ("magnetics.b_field_pol_probe.0.poloidal_angle", np.inf, "finite"),
        ("pf_active.coil.0.current.data", [object(), object()], "one-dimensional real"),
        ("pf_active.coil.0.current.data", [True, False], "not boolean"),
        ("pf_active.coil.0.current.data", [[1.0, 2.0]], "one-dimensional array"),
        ("pf_active.coil.0.current.data", [], "non-empty"),
        ("magnetics.flux_loop.0.identifier", " ", "non-empty identifier"),
        ("magnetics.flux_loop.0.identifier", 3, "string identifier"),
    ],
)
def test_rejects_noncoercible_or_empty_fields(path: str, value: Any, message: str) -> None:
    """Type, dimensionality, finiteness, and identifier checks are all fail-closed."""
    ods = _full_ods()
    ods[path] = value
    with pytest.raises(ValueError, match=message):
        extract_omas_free_boundary_inputs(ods, require_ingestion_ready=False)


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("pf_active.coil.0.current.data", "missing required"),
        ("pf_active.coil.0.identifier", "missing required"),
        ("pf_active.coil.0.element.0.identifier", "missing required"),
        ("magnetics.b_field_pol_probe.0.poloidal_angle", "missing required"),
        ("magnetics.flux_loop.0.position.0.z", "missing required"),
    ],
)
def test_rejects_missing_required_paths(path: str, message: str) -> None:
    """Present channel objects cannot silently default required fields."""
    ods = _full_ods()
    del ods[path]
    with pytest.raises(ValueError, match=message):
        extract_omas_free_boundary_inputs(ods, require_ingestion_ready=False)


def test_rejects_pf_coils_without_elements() -> None:
    """A current history without its matching physical geometry is unusable."""
    ods = _full_ods()
    for key in [key for key in ods if key.startswith("pf_active.coil.0.element")]:
        del ods[key]
    with pytest.raises(ValueError, match="at least one geometry element"):
        extract_omas_free_boundary_inputs(ods, require_ingestion_ready=False)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("pf_active.coil.1.element.0.geometry.outline.r", [1.0, 1.1], "length >= 3"),
        ("pf_active.coil.1.element.0.geometry.outline.z", [0.0, 0.1, 0.2], "matching"),
        ("pf_active.coil.1.element.0.geometry.outline.r", [1.0, 0.0, 1.2, 1.3], "positive"),
    ],
)
def test_rejects_malformed_outline_geometry(path: str, value: Any, message: str) -> None:
    """Outline elements require a physical closed-path point representation."""
    ods = _full_ods()
    ods[path] = value
    with pytest.raises(ValueError, match=message):
        extract_omas_free_boundary_inputs(ods, require_ingestion_ready=False)


def test_rejects_flux_loops_without_positions() -> None:
    """A flux value without loop geometry cannot enter the forward model."""
    ods = _full_ods()
    for key in [key for key in ods if ".position." in key and "flux_loop" in key]:
        del ods[key]
    with pytest.raises(ValueError, match="at least one position"):
        extract_omas_free_boundary_inputs(ods, require_ingestion_ready=False)


def test_rejects_duplicate_physical_identifiers() -> None:
    """Identifiers used for machine mapping cannot alias two physical objects."""
    ods = _full_ods()
    ods["pf_active.coil.1.identifier"] = "PF1"
    with pytest.raises(ValueError, match="pf_active.coil identifiers must be unique"):
        extract_omas_free_boundary_inputs(ods, require_ingestion_ready=False)


def test_rejects_duplicate_element_identifiers_within_a_coil() -> None:
    """Element identities are unique within each parent-coil mapping."""
    ods = _full_ods()
    prefix = "pf_active.coil.0.element"
    ods[f"{prefix}.1.identifier"] = "PF1E1"
    ods[f"{prefix}.1.turns_with_sign"] = 2.0
    ods[f"{prefix}.1.geometry.geometry_type"] = 2
    ods[f"{prefix}.1.geometry.rectangle.r"] = 1.2
    ods[f"{prefix}.1.geometry.rectangle.z"] = -0.1
    ods[f"{prefix}.1.geometry.rectangle.width"] = 0.1
    ods[f"{prefix}.1.geometry.rectangle.height"] = 0.1
    with pytest.raises(ValueError, match="coil PF1 elements identifiers must be unique"):
        extract_omas_free_boundary_inputs(ods, require_ingestion_ready=False)


def test_rejects_sparse_dotted_top_level_arrays() -> None:
    """A present index after a gap is rejected instead of silently truncated."""
    ods = _full_ods()
    for key in [key for key in ods if key.startswith("pf_active.coil.1.")]:
        ods[key.replace("pf_active.coil.1.", "pf_active.coil.2.")] = ods.pop(key)

    with pytest.raises(ValueError, match="pf_active.coil contains sparse or gapped"):
        extract_omas_free_boundary_inputs(ods, require_ingestion_ready=False)


def test_rejects_sparse_dotted_nested_arrays() -> None:
    """Nested physical positions receive the same no-gap guarantee."""
    ods = _full_ods()
    for key in [key for key in ods if "flux_loop.0.position.1." in key]:
        ods[key.replace("position.1.", "position.2.")] = ods.pop(key)

    with pytest.raises(ValueError, match="flux_loop.0.position contains sparse or gapped"):
        extract_omas_free_boundary_inputs(ods, require_ingestion_ready=False)


def test_collection_enumeration_rejects_opaque_or_malformed_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enumeration itself is fail-closed for opaque, malformed, or oversized arrays."""

    class GetOnly:
        def get(self, key: str, default: Any = None) -> Any:
            return default

    with pytest.raises(ValueError, match="cannot be enumerated"):
        omas_inputs._collection_indices(GetOnly(), "pf_active.coil")
    with pytest.raises(ValueError, match="enumerable OMAS array"):
        omas_inputs._collection_indices(DottedODS({"pf_active.coil": []}), "pf_active.coil")
    with pytest.raises(ValueError, match="non-integer array index"):
        omas_inputs._collection_indices(
            DottedODS({"pf_active.coil": {"bad": object()}}), "pf_active.coil"
        )
    with pytest.raises(ValueError, match="negative array index"):
        omas_inputs._collection_indices(
            DottedODS({"pf_active.coil": {-1: object()}}), "pf_active.coil"
        )
    with pytest.raises(ValueError, match="non-integer dotted array index"):
        omas_inputs._collection_indices(
            DottedODS({"pf_active.coil.bad.identifier": "bad"}), "pf_active.coil"
        )

    monkeypatch.setattr(omas_inputs, "_MAX_COLLECTION_ITEMS", 1)
    with pytest.raises(ValueError, match="safety limit"):
        omas_inputs._collection_indices(
            DottedODS(
                {
                    "pf_active.coil.0.identifier": "PF1",
                    "pf_active.coil.1.identifier": "PF2",
                }
            ),
            "pf_active.coil",
        )


def test_real_omas_ods_paths_are_compatible() -> None:
    """The strict dotted-path reader consumes a schema-valid installed OMAS ODS."""
    omas = pytest.importorskip("omas")
    ods = omas.ODS(cocos=11)
    for path, value in _full_ods().items():
        if path.startswith("pf_active") and path.endswith(".validity"):
            continue
        ods[path] = value

    result = extract_omas_free_boundary_inputs(
        ods,
        provenance=_provenance(),
        time_alignment="exact_common_axis",
    )

    assert result.ingestion_ready
    assert len(result.pf_coils) == 2


def test_real_omas_sparse_array_is_rejected() -> None:
    """A real OMAS dynamic array with an empty middle member cannot truncate silently."""
    omas = pytest.importorskip("omas")
    ods = omas.ODS(cocos=11)
    with omas.omas_environment(ods, dynamic_path_creation="dynamic_array_structures"):
        for path, value in _full_ods().items():
            if path.startswith("pf_active.coil.1."):
                path = path.replace("pf_active.coil.1.", "pf_active.coil.2.")
            if path.startswith("pf_active") and path.endswith(".validity"):
                continue
            ods[path] = value

    with pytest.raises(ValueError, match="pf_active.coil"):
        extract_omas_free_boundary_inputs(ods, require_ingestion_ready=False)
