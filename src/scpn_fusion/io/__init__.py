# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — IO Package Init
"""Data-interop adapters (IMAS/IDS and related exchange helpers)."""

from importlib import import_module as _import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .imas_connector import (
        CANONICAL_COCOS,
        HAS_OMAS,
        IMAS_DD_CORE_PROFILES_KEYS,
        IMAS_DD_EQUILIBRIUM_KEYS,
        IMAS_DD_SUMMARY_KEYS,
        OMAS_FREE_BOUNDARY_INPUT_SCHEMA,
        REQUIRED_DIGITAL_TWIN_STATE_KEYS,
        REQUIRED_DIGITAL_TWIN_SUMMARY_KEYS,
        REQUIRED_IDS_KEYS,
        REQUIRED_PROFILE_1D_KEYS,
        TIER0_OUT_OF_SCOPE_BLOCKERS,
        FluxLoopInput,
        OmasFreeBoundaryInputs,
        OmasSourceProvenance,
        PfCoilInput,
        PfElementGeometry,
        PoloidalFieldProbeInput,
        TimeSeriesSI,
        digital_twin_history_to_ids,
        digital_twin_history_to_ids_pulse,
        digital_twin_state_to_ids,
        digital_twin_summary_to_ids,
        extract_omas_free_boundary_inputs,
        geqdsk_to_imas_equilibrium,
        ids_pulse_to_digital_twin_history,
        ids_to_digital_twin_history,
        ids_to_digital_twin_state,
        ids_to_digital_twin_summary,
        ids_to_omas_core_profiles,
        ids_to_omas_equilibrium,
        imas_core_transport_to_state,
        imas_equilibrium_to_geqdsk,
        omas_core_profiles_to_ids,
        omas_equilibrium_to_ids,
        read_ids,
        state_to_imas_core_profiles,
        state_to_imas_core_transport,
        state_to_imas_summary,
        validate_ids_payload,
        validate_ids_payload_sequence,
        validate_ids_pulse_payload,
        write_ids,
    )

from .logging_config import FusionJSONFormatter, setup_fusion_logging
from .mast_magnetic_archive import (
    MastMagneticArchiveDependencyError,
    build_mast_complete_magnetic_archive_envelope,
    verify_mast_complete_magnetic_archive_source,
)
from .mast_magnetic_archive_acquisition import (
    MastMagneticArchiveAcquisitionError,
    acquire_mast_complete_magnetic_archive,
)
from .mast_magnetic_archive_codec import (
    MAST_COMPLETE_MAGNETIC_ARCHIVE_SCHEMA,
    MAST_COMPLETE_MAGNETIC_ARCHIVE_SCHEMA_VERSION,
    MastCompleteMagneticArchiveEnvelope,
    MastMagneticArchiveValidationError,
    decode_mast_complete_magnetic_archive_envelope,
    encode_mast_complete_magnetic_archive_envelope,
)
from .mast_magnetic_qualification import (
    build_mast_magnetic_diagnostic_qualification,
    verify_mast_magnetic_diagnostic_qualification,
)
from .mast_magnetic_qualification_codec import (
    MAST_MAGNETIC_DIAGNOSTIC_QUALIFICATION_SCHEMA,
    MAST_MAGNETIC_DIAGNOSTIC_QUALIFICATION_SCHEMA_VERSION,
    MastMagneticDiagnosticQualification,
    MastMagneticDiagnosticQualificationError,
    decode_mast_magnetic_diagnostic_qualification,
    encode_mast_magnetic_diagnostic_qualification,
)
from .tokamak_archive import (
    DEFAULT_MDSPLUS_NODE_MAP,
    TokamakProfile,
    fetch_mdsplus_profiles,
    generate_synthetic_shot_database,
    list_disruption_shots,
    list_synthetic_shots,
    load_cmod_reference_profiles,
    load_diiid_reference_profiles,
    load_disruption_shot,
    load_machine_profiles,
    load_synthetic_shot,
    poll_mdsplus_feed,
)

_IMAS_EXPORTS = frozenset(
    [
        "CANONICAL_COCOS",
        "FluxLoopInput",
        "HAS_OMAS",
        "IMAS_DD_CORE_PROFILES_KEYS",
        "IMAS_DD_EQUILIBRIUM_KEYS",
        "IMAS_DD_SUMMARY_KEYS",
        "OMAS_FREE_BOUNDARY_INPUT_SCHEMA",
        "OmasFreeBoundaryInputs",
        "OmasSourceProvenance",
        "PfCoilInput",
        "PfElementGeometry",
        "PoloidalFieldProbeInput",
        "REQUIRED_DIGITAL_TWIN_STATE_KEYS",
        "REQUIRED_DIGITAL_TWIN_SUMMARY_KEYS",
        "REQUIRED_IDS_KEYS",
        "REQUIRED_PROFILE_1D_KEYS",
        "TIER0_OUT_OF_SCOPE_BLOCKERS",
        "TimeSeriesSI",
        "digital_twin_history_to_ids",
        "digital_twin_history_to_ids_pulse",
        "digital_twin_state_to_ids",
        "digital_twin_summary_to_ids",
        "extract_omas_free_boundary_inputs",
        "geqdsk_to_imas_equilibrium",
        "ids_pulse_to_digital_twin_history",
        "ids_to_digital_twin_history",
        "ids_to_digital_twin_state",
        "ids_to_digital_twin_summary",
        "ids_to_omas_core_profiles",
        "ids_to_omas_equilibrium",
        "imas_core_transport_to_state",
        "imas_equilibrium_to_geqdsk",
        "omas_core_profiles_to_ids",
        "omas_equilibrium_to_ids",
        "read_ids",
        "state_to_imas_core_profiles",
        "state_to_imas_core_transport",
        "state_to_imas_summary",
        "validate_ids_payload",
        "validate_ids_payload_sequence",
        "validate_ids_pulse_payload",
        "write_ids",
    ]
)
_IMAS_MODULES = frozenset(
    [
        "imas_connector",
        "imas_connector_common",
        "imas_connector_digital_twin",
        "imas_connector_equilibrium",
        "imas_connector_omas",
        "imas_connector_storage",
        "imas_connector_transport",
        "imas_history_payloads",
        "omas_free_boundary_inputs",
        "safe_loaders",
    ]
)

__all__ = [
    "CANONICAL_COCOS",
    "DEFAULT_MDSPLUS_NODE_MAP",
    "FluxLoopInput",
    "FusionJSONFormatter",
    "HAS_OMAS",
    "IMAS_DD_CORE_PROFILES_KEYS",
    "IMAS_DD_EQUILIBRIUM_KEYS",
    "IMAS_DD_SUMMARY_KEYS",
    "MAST_COMPLETE_MAGNETIC_ARCHIVE_SCHEMA",
    "MAST_COMPLETE_MAGNETIC_ARCHIVE_SCHEMA_VERSION",
    "MAST_MAGNETIC_DIAGNOSTIC_QUALIFICATION_SCHEMA",
    "MAST_MAGNETIC_DIAGNOSTIC_QUALIFICATION_SCHEMA_VERSION",
    "MastCompleteMagneticArchiveEnvelope",
    "MastMagneticArchiveAcquisitionError",
    "MastMagneticArchiveDependencyError",
    "MastMagneticArchiveValidationError",
    "MastMagneticDiagnosticQualification",
    "MastMagneticDiagnosticQualificationError",
    "OMAS_FREE_BOUNDARY_INPUT_SCHEMA",
    "OmasFreeBoundaryInputs",
    "OmasSourceProvenance",
    "PfCoilInput",
    "PfElementGeometry",
    "PoloidalFieldProbeInput",
    "REQUIRED_DIGITAL_TWIN_STATE_KEYS",
    "REQUIRED_DIGITAL_TWIN_SUMMARY_KEYS",
    "REQUIRED_IDS_KEYS",
    "REQUIRED_PROFILE_1D_KEYS",
    "TIER0_OUT_OF_SCOPE_BLOCKERS",
    "TimeSeriesSI",
    "TokamakProfile",
    "acquire_mast_complete_magnetic_archive",
    "build_mast_complete_magnetic_archive_envelope",
    "build_mast_magnetic_diagnostic_qualification",
    "decode_mast_complete_magnetic_archive_envelope",
    "decode_mast_magnetic_diagnostic_qualification",
    "digital_twin_history_to_ids",
    "digital_twin_history_to_ids_pulse",
    "digital_twin_state_to_ids",
    "digital_twin_summary_to_ids",
    "encode_mast_complete_magnetic_archive_envelope",
    "encode_mast_magnetic_diagnostic_qualification",
    "extract_omas_free_boundary_inputs",
    "fetch_mdsplus_profiles",
    "generate_synthetic_shot_database",
    "geqdsk_to_imas_equilibrium",
    "ids_pulse_to_digital_twin_history",
    "ids_to_digital_twin_history",
    "ids_to_digital_twin_state",
    "ids_to_digital_twin_summary",
    "ids_to_omas_core_profiles",
    "ids_to_omas_equilibrium",
    "imas_connector",
    "imas_connector_common",
    "imas_connector_digital_twin",
    "imas_connector_equilibrium",
    "imas_connector_omas",
    "imas_connector_storage",
    "imas_connector_transport",
    "imas_core_transport_to_state",
    "imas_equilibrium_to_geqdsk",
    "imas_history_payloads",
    "list_disruption_shots",
    "list_synthetic_shots",
    "load_cmod_reference_profiles",
    "load_diiid_reference_profiles",
    "load_disruption_shot",
    "load_machine_profiles",
    "load_synthetic_shot",
    "logging_config",
    "mast_magnetic_archive",
    "mast_magnetic_archive_acquisition",
    "mast_magnetic_archive_codec",
    "mast_magnetic_qualification",
    "mast_magnetic_qualification_codec",
    "omas_core_profiles_to_ids",
    "omas_equilibrium_to_ids",
    "omas_free_boundary_inputs",
    "poll_mdsplus_feed",
    "read_ids",
    "safe_loaders",
    "setup_fusion_logging",
    "state_to_imas_core_profiles",
    "state_to_imas_core_transport",
    "state_to_imas_summary",
    "tokamak_archive",
    "tokamak_archive_profiles",
    "tokamak_disruption_archive",
    "tokamak_live_payload",
    "tokamak_synthetic_archive",
    "validate_ids_payload",
    "validate_ids_payload_sequence",
    "validate_ids_pulse_payload",
    "verify_mast_complete_magnetic_archive_source",
    "verify_mast_magnetic_diagnostic_qualification",
    "write_ids",
]


def __getattr__(name: str) -> object:
    """Load the requested IMAS export or original IO submodule on first use.

    Parameters
    ----------
    name : str
        Public package export or module name.

    Returns
    -------
    object
        Original object from its defining adapter, cached on this package.

    Raises
    ------
    AttributeError
        The requested name is outside the package export surface.

    """
    if name in _IMAS_EXPORTS:
        value: object = getattr(_import_module(".imas_connector", __name__), name)
    elif name in _IMAS_MODULES:
        value = _import_module(f".{name}", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List public adapter names without importing their optional dependencies.

    Returns
    -------
    list of str
        Sorted loaded names and declared deferred adapter exports.

    """
    return sorted(set(globals()) | _IMAS_EXPORTS | _IMAS_MODULES)
