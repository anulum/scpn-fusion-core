# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FAIR-MAST magnetic diagnostic qualification tests
"""Public-surface tests over the complete real-shot qualification witness."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from scpn_fusion.io import decode_mast_magnetic_diagnostic_qualification
from scpn_fusion.io.mast_magnetic_archive_codec import JsonObject

_REFERENCE = Path(
    "validation/reference_data/mast/disruption_shots/MAGNETIC_DIAGNOSTIC_QUALIFICATION.json"
)


def test_public_qualification_covers_every_archive_surface() -> None:
    """The real-shot witness classifies all arrays, clocks, measurements and channels."""
    payload = decode_mast_magnetic_diagnostic_qualification(_REFERENCE.read_bytes()).payload
    completeness = cast(JsonObject, payload["completeness"])
    arrays = cast(list[JsonObject], payload["array_inventory"])
    measurements = cast(list[JsonObject], payload["measurement_evidence"])
    mappings = cast(list[JsonObject], payload["channel_geometry_evidence"])

    assert completeness == {
        "archive_array_count": 72,
        "archive_arrays_classified": True,
        "channel_record_count": 132,
        "clock_count": 4,
        "measurement_count": 11,
        "measurements_analysed": True,
    }
    assert len(arrays) == 72
    assert len(measurements) == 11
    assert len(mappings) == 132
    assert all(mapping["physical_mapping_claimed"] is False for mapping in mappings)
    assert {cast(str, item["array_name"]) for item in measurements} == {
        "b_field_pol_probe_cc_field",
        "b_field_pol_probe_ccbv_field",
        "b_field_pol_probe_obr_field",
        "b_field_pol_probe_obv_field",
        "b_field_pol_probe_omv_voltage",
        "b_field_tor_probe_cc_field",
        "b_field_tor_probe_omaha_voltage",
        "b_field_tor_probe_saddle_field",
        "b_field_tor_probe_saddle_voltage",
        "flux_loop_flux",
        "ip",
    }


def test_public_qualification_remains_review_only_and_fail_closed() -> None:
    """Measured diagnostic evidence does not acquire phase or actuation authority."""
    payload = decode_mast_magnetic_diagnostic_qualification(_REFERENCE.read_bytes()).payload
    assert payload["authority"] == {
        "actionable": False,
        "classification_performed": False,
        "direct_actuation": False,
        "execution_permitted": False,
        "phase_inference_performed": False,
        "review_only": True,
    }
    assert payload["qualification_summary"] == {
        "calibration_state": "applied_transforms_recorded_lineage_unavailable",
        "channel_geometry_mapping_state": "identifier_correspondence_only",
        "event_identity_state": "shot_only_event_unresolved",
        "observation_operator_state": "quantity_paths_only_transfer_functions_unavailable",
        "provider_quality_state": "not_supplied",
        "source_clock_relationship_state": ("derived_archive_grids_no_instrument_clock_relation"),
        "uncertainty_state": "not_supplied",
        "validity_state": "source_shot_ranges_only",
    }
