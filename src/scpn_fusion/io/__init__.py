# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — IO Package Init
"""Data-interop adapters (IMAS/IDS and related exchange helpers)."""

from .imas_connector import (
    IMAS_DD_CORE_PROFILES_KEYS,
    IMAS_DD_EQUILIBRIUM_KEYS,
    IMAS_DD_SUMMARY_KEYS,
    HAS_OMAS,
    CANONICAL_COCOS as CANONICAL_COCOS,
    OMAS_FREE_BOUNDARY_INPUT_SCHEMA as OMAS_FREE_BOUNDARY_INPUT_SCHEMA,
    TIER0_OUT_OF_SCOPE_BLOCKERS as TIER0_OUT_OF_SCOPE_BLOCKERS,
    FluxLoopInput as FluxLoopInput,
    OmasFreeBoundaryInputs as OmasFreeBoundaryInputs,
    OmasSourceProvenance as OmasSourceProvenance,
    PfCoilInput as PfCoilInput,
    PfElementGeometry as PfElementGeometry,
    PoloidalFieldProbeInput as PoloidalFieldProbeInput,
    REQUIRED_DIGITAL_TWIN_STATE_KEYS,
    REQUIRED_DIGITAL_TWIN_SUMMARY_KEYS,
    REQUIRED_IDS_KEYS,
    REQUIRED_PROFILE_1D_KEYS,
    digital_twin_history_to_ids,
    digital_twin_history_to_ids_pulse,
    digital_twin_state_to_ids,
    digital_twin_summary_to_ids,
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
    TimeSeriesSI as TimeSeriesSI,
    validate_ids_payload,
    validate_ids_payload_sequence,
    validate_ids_pulse_payload,
    write_ids,
    extract_omas_free_boundary_inputs as extract_omas_free_boundary_inputs,
)
from .logging_config import FusionJSONFormatter, setup_fusion_logging
from .mast_magnetic_archive import (
    MastMagneticArchiveDependencyError as MastMagneticArchiveDependencyError,
    build_mast_complete_magnetic_archive_envelope as build_mast_complete_magnetic_archive_envelope,
    verify_mast_complete_magnetic_archive_source as verify_mast_complete_magnetic_archive_source,
)
from .mast_magnetic_archive_acquisition import (
    MastMagneticArchiveAcquisitionError as MastMagneticArchiveAcquisitionError,
    acquire_mast_complete_magnetic_archive as acquire_mast_complete_magnetic_archive,
)
from .mast_magnetic_archive_codec import (
    MAST_COMPLETE_MAGNETIC_ARCHIVE_SCHEMA as MAST_COMPLETE_MAGNETIC_ARCHIVE_SCHEMA,
    MAST_COMPLETE_MAGNETIC_ARCHIVE_SCHEMA_VERSION as MAST_COMPLETE_MAGNETIC_ARCHIVE_SCHEMA_VERSION,
    MastCompleteMagneticArchiveEnvelope as MastCompleteMagneticArchiveEnvelope,
    MastMagneticArchiveValidationError as MastMagneticArchiveValidationError,
    decode_mast_complete_magnetic_archive_envelope as decode_mast_complete_magnetic_archive_envelope,
    encode_mast_complete_magnetic_archive_envelope as encode_mast_complete_magnetic_archive_envelope,
)
from .mast_magnetic_qualification import (
    build_mast_magnetic_diagnostic_qualification as build_mast_magnetic_diagnostic_qualification,
    verify_mast_magnetic_diagnostic_qualification as verify_mast_magnetic_diagnostic_qualification,
)
from .mast_magnetic_qualification_codec import (
    MAST_MAGNETIC_DIAGNOSTIC_QUALIFICATION_SCHEMA as MAST_MAGNETIC_DIAGNOSTIC_QUALIFICATION_SCHEMA,
    MAST_MAGNETIC_DIAGNOSTIC_QUALIFICATION_SCHEMA_VERSION as MAST_MAGNETIC_DIAGNOSTIC_QUALIFICATION_SCHEMA_VERSION,
    MastMagneticDiagnosticQualification as MastMagneticDiagnosticQualification,
    MastMagneticDiagnosticQualificationError as MastMagneticDiagnosticQualificationError,
    decode_mast_magnetic_diagnostic_qualification as decode_mast_magnetic_diagnostic_qualification,
    encode_mast_magnetic_diagnostic_qualification as encode_mast_magnetic_diagnostic_qualification,
)
from .tokamak_archive import (
    TokamakProfile,
    DEFAULT_MDSPLUS_NODE_MAP,
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
