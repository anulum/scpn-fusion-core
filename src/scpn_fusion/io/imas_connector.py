# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — IMAS Connector
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Facade API for IMAS/IDS adapter modules.

This module intentionally stays as a stable import surface while implementation
is decomposed into focused submodules:

- ``imas_connector_common``: validation/coercion primitives
- ``imas_connector_digital_twin``: summary/state IDS mappings
- ``imas_connector_equilibrium``: GEQDSK <-> IMAS equilibrium
- ``imas_connector_transport``: core_profiles/summary/core_transport
- ``imas_connector_storage``: JSON I/O helpers
- ``imas_connector_omas``: OMAS bridge
"""

from __future__ import annotations

from scpn_fusion.io.imas_connector_common import (
    REQUIRED_DIGITAL_TWIN_STATE_KEYS,
    REQUIRED_DIGITAL_TWIN_SUMMARY_KEYS,
    REQUIRED_IDS_KEYS,
    REQUIRED_PROFILE_1D_KEYS,
    _coerce_finite_real,
    _coerce_int,
    _missing_required_keys,
)
from scpn_fusion.io.imas_connector_digital_twin import (
    digital_twin_state_to_ids,
    digital_twin_summary_to_ids,
    ids_to_digital_twin_state,
    ids_to_digital_twin_summary,
    validate_ids_payload,
)
from scpn_fusion.io.imas_connector_equilibrium import (
    IMAS_DD_EQUILIBRIUM_KEYS,
    geqdsk_to_imas_equilibrium,
    imas_equilibrium_to_geqdsk,
)
from scpn_fusion.io.imas_connector_omas import (
    HAS_OMAS,
    ids_to_omas_equilibrium,
    omas_equilibrium_to_ids,
)
from scpn_fusion.io.imas_connector_storage import (
    _VALID_IDS_TYPES,
    read_ids,
    write_ids,
)
from scpn_fusion.io.imas_connector_transport import (
    IMAS_DD_CORE_PROFILES_KEYS,
    IMAS_DD_SUMMARY_KEYS,
    imas_core_transport_to_state,
    state_to_imas_core_profiles,
    state_to_imas_core_transport,
    state_to_imas_summary,
)
from scpn_fusion.io.imas_history_payloads import (
    digital_twin_history_to_ids,
    digital_twin_history_to_ids_pulse,
    ids_pulse_to_digital_twin_history,
    ids_to_digital_twin_history,
    validate_ids_payload_sequence,
    validate_ids_pulse_payload,
)

__all__ = [
    # Common constants
    "REQUIRED_IDS_KEYS",
    "REQUIRED_DIGITAL_TWIN_SUMMARY_KEYS",
    "REQUIRED_PROFILE_1D_KEYS",
    "REQUIRED_DIGITAL_TWIN_STATE_KEYS",
    "_coerce_int",
    "_coerce_finite_real",
    "_missing_required_keys",
    # Digital twin IDS adapters
    "validate_ids_payload",
    "digital_twin_summary_to_ids",
    "digital_twin_state_to_ids",
    "ids_to_digital_twin_summary",
    "ids_to_digital_twin_state",
    # History/pulse adapters
    "digital_twin_history_to_ids",
    "digital_twin_history_to_ids_pulse",
    "ids_to_digital_twin_history",
    "ids_pulse_to_digital_twin_history",
    "validate_ids_payload_sequence",
    "validate_ids_pulse_payload",
    # Equilibrium adapters
    "IMAS_DD_EQUILIBRIUM_KEYS",
    "geqdsk_to_imas_equilibrium",
    "imas_equilibrium_to_geqdsk",
    # Transport/summary adapters
    "IMAS_DD_CORE_PROFILES_KEYS",
    "IMAS_DD_SUMMARY_KEYS",
    "state_to_imas_core_profiles",
    "state_to_imas_summary",
    "state_to_imas_core_transport",
    "imas_core_transport_to_state",
    # Storage helpers
    "_VALID_IDS_TYPES",
    "write_ids",
    "read_ids",
    # OMAS bridge
    "HAS_OMAS",
    "ids_to_omas_equilibrium",
    "omas_equilibrium_to_ids",
]
