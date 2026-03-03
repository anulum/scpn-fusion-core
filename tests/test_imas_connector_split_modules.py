# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — IMAS Connector Split Module Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Direct linkage tests for split IMAS connector modules."""

from __future__ import annotations

from scpn_fusion.io import imas_connector_common as common
from scpn_fusion.io import imas_connector_digital_twin as digital_twin
from scpn_fusion.io import imas_connector_equilibrium as equilibrium
from scpn_fusion.io import imas_connector_omas as omas_bridge
from scpn_fusion.io import imas_connector_storage as storage
from scpn_fusion.io import imas_connector_transport as transport


def test_common_module_exports_required_key_sets() -> None:
    assert "schema" in common.REQUIRED_IDS_KEYS
    assert "rho_norm" in common.REQUIRED_PROFILE_1D_KEYS


def test_digital_twin_module_exports_core_api() -> None:
    assert callable(digital_twin.validate_ids_payload)
    assert callable(digital_twin.digital_twin_summary_to_ids)
    assert callable(digital_twin.ids_to_digital_twin_state)


def test_equilibrium_module_exports_core_api() -> None:
    assert callable(equilibrium.geqdsk_to_imas_equilibrium)
    assert callable(equilibrium.imas_equilibrium_to_geqdsk)


def test_transport_module_exports_core_api() -> None:
    assert callable(transport.state_to_imas_core_profiles)
    assert callable(transport.state_to_imas_summary)
    assert callable(transport.state_to_imas_core_transport)
    assert callable(transport.imas_core_transport_to_state)


def test_storage_module_exports_core_api() -> None:
    assert callable(storage.write_ids)
    assert callable(storage.read_ids)
    assert "equilibrium" in storage._VALID_IDS_TYPES


def test_omas_module_exposes_bridge_contract() -> None:
    assert isinstance(omas_bridge.HAS_OMAS, bool)
    assert callable(omas_bridge.ids_to_omas_equilibrium)
    assert callable(omas_bridge.omas_equilibrium_to_ids)

