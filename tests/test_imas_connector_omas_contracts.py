# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — IMAS Connector OMAS Contract Tests
"""Contract tests for the optional OMAS IMAS bridge."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_fusion.io import imas_connector_omas as omas_bridge


class FakeODS(dict[str, Any]):
    """Dictionary-backed stand-in for ``omas.ODS`` dotted-key storage."""


class FakeOmasModule:
    """Minimal module stand-in exposing the ``ODS`` constructor."""

    @staticmethod
    def ODS() -> FakeODS:
        """Return a fresh fake ODS container."""
        return FakeODS()


@pytest.fixture
def fake_omas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable the OMAS-dependent public functions with a local test double."""
    monkeypatch.setattr(omas_bridge, "HAS_OMAS", True)
    monkeypatch.setattr(omas_bridge, "omas", FakeOmasModule)


def _equilibrium_ids() -> dict[str, Any]:
    """Build a representative equilibrium IDS payload."""
    return {
        "ids_properties": {"homogeneous_time": 0, "comment": "contract"},
        "time": [0.0, 1.0],
        "time_slice": [
            {
                "time": 0.0,
                "global_quantities": {
                    "ip": 15.0,
                    "magnetic_axis": {"r": 6.2, "z": 0.1},
                    "psi_axis": -1.0,
                    "psi_boundary": 0.0,
                    "vacuum_toroidal_field": {"r0": 6.2, "b0": 5.3},
                },
                "profiles_1d": {
                    "psi": [0.0, 0.5, 1.0],
                    "q": [1.0, 2.0, 3.0],
                    "pressure": [10.0, 5.0, 1.0],
                    "f": [5.0, 4.0, 3.0],
                },
                "boundary": {"outline": {"r": [5.0, 6.0], "z": [-1.0, 1.0]}},
            }
        ],
    }


def _core_profiles_ids() -> dict[str, Any]:
    """Build a representative core_profiles IDS payload."""
    return {
        "ids_properties": {"homogeneous_time": 1, "comment": "core"},
        "time": [0.0],
        "profiles_1d": [
            {
                "time": 0.0,
                "grid": {"rho_tor_norm": [0.0, 0.5, 1.0]},
                "electrons": {
                    "temperature": [1000.0, 800.0, 100.0],
                    "density": [1.0e20, 8.0e19, 3.0e19],
                },
            }
        ],
    }


def test_equilibrium_bridge_roundtrips_full_payload(fake_omas: None) -> None:
    """Equilibrium conversion preserves scalar fields, arrays, and boundary outlines."""
    ids = _equilibrium_ids()

    ods = omas_bridge.ids_to_omas_equilibrium(ids)
    roundtrip = omas_bridge.omas_equilibrium_to_ids(ods)

    assert roundtrip["ids_properties"] == ids["ids_properties"]
    assert roundtrip["time"] == [0.0, 1.0]
    time_slice = roundtrip["time_slice"][0]
    assert time_slice["global_quantities"]["ip"] == pytest.approx(15.0)
    assert time_slice["global_quantities"]["vacuum_toroidal_field"]["b0"] == pytest.approx(5.3)
    assert time_slice["profiles_1d"]["pressure"] == [10.0, 5.0, 1.0]
    assert time_slice["boundary"]["outline"]["z"] == [-1.0, 1.0]


def test_equilibrium_to_ids_accepts_scalar_and_plain_list_values(fake_omas: None) -> None:
    """Reverse equilibrium conversion handles scalar time and list-valued ODS entries."""
    ods = FakeODS(
        {
            "equilibrium.ids_properties.homogeneous_time": 1,
            "equilibrium.ids_properties.comment": "manual",
            "equilibrium.time": 2.5,
            "equilibrium.time_slice.0.time": 2.5,
            "equilibrium.time_slice.0.profiles_1d.psi": [0.0, 1.0],
            "equilibrium.time_slice.0.boundary.outline.r": [5.5, 6.5],
            "equilibrium.time_slice.0.boundary.outline.z": [-0.5, 0.5],
        }
    )

    ids = omas_bridge.omas_equilibrium_to_ids(ods)

    assert ids["time"] == [2.5]
    assert ids["time_slice"][0]["profiles_1d"]["psi"] == [0.0, 1.0]
    assert ids["time_slice"][0]["boundary"]["outline"]["r"] == [5.5, 6.5]


def test_core_profiles_bridge_roundtrips_full_payload(fake_omas: None) -> None:
    """core_profiles conversion preserves grid and electron profile vectors."""
    ids = _core_profiles_ids()

    ods = omas_bridge.ids_to_omas_core_profiles(ids)
    roundtrip = omas_bridge.omas_core_profiles_to_ids(ods)

    assert roundtrip["ids_properties"] == ids["ids_properties"]
    assert roundtrip["time"] == [0.0]
    profile = roundtrip["profiles_1d"][0]
    assert profile["grid"]["rho_tor_norm"] == [0.0, 0.5, 1.0]
    assert profile["electrons"]["temperature"] == [1000.0, 800.0, 100.0]
    assert profile["electrons"]["density"] == [1.0e20, 8.0e19, 3.0e19]


def test_core_profiles_population_uses_defaults_for_non_mapping_blocks() -> None:
    """Internal population defaults non-mapping optional blocks without crashing."""
    ods = FakeODS()

    populated = omas_bridge._populate_omas_core_profiles(
        ods,
        {
            "ids_properties": "not-a-mapping",
            "time": [1.0],
            "profiles_1d": [{"time": 1.0, "grid": "bad", "electrons": "bad"}],
        },
    )

    assert populated["core_profiles.ids_properties.homogeneous_time"] == 1
    assert populated["core_profiles.ids_properties.comment"] == ""
    assert populated["core_profiles.profiles_1d.0.time"] == pytest.approx(1.0)
    assert "core_profiles.profiles_1d.0.grid.rho_tor_norm" not in populated


def test_core_profiles_reverse_defaults_missing_optional_vectors(fake_omas: None) -> None:
    """Reverse core_profiles conversion defaults missing electron vectors to empty lists."""
    ods = FakeODS(
        {
            "core_profiles.ids_properties.homogeneous_time": 0,
            "core_profiles.ids_properties.comment": "manual",
            "core_profiles.time": 4.0,
            "core_profiles.profiles_1d.0.time": 4.0,
            "core_profiles.profiles_1d.0.grid.rho_tor_norm": [0.0, 1.0],
        }
    )

    ids = omas_bridge.omas_core_profiles_to_ids(ods)

    assert ids["time"] == [4.0]
    assert ids["profiles_1d"][0]["electrons"]["temperature"] == []
    assert ids["profiles_1d"][0]["electrons"]["density"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {"profiles_1d": {}},
        {"profiles_1d": [object()]},
    ],
)
def test_core_profiles_population_rejects_malformed_profiles(payload: dict[str, Any]) -> None:
    """core_profiles population rejects malformed profile containers."""
    with pytest.raises(ValueError, match="profiles_1d"):
        omas_bridge._populate_omas_core_profiles(FakeODS(), payload)


def test_public_converters_fail_closed_without_omas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Public OMAS converters keep the optional dependency boundary fail-closed."""
    monkeypatch.setattr(omas_bridge, "HAS_OMAS", False)
    monkeypatch.setattr(omas_bridge, "omas", None)

    with pytest.raises(ImportError, match="omas"):
        omas_bridge.ids_to_omas_equilibrium(_equilibrium_ids())
    with pytest.raises(ImportError, match="omas"):
        omas_bridge.omas_equilibrium_to_ids(FakeODS())
    with pytest.raises(ImportError, match="omas"):
        omas_bridge.ids_to_omas_core_profiles(_core_profiles_ids())
    with pytest.raises(ImportError, match="omas"):
        omas_bridge.omas_core_profiles_to_ids(FakeODS())


def test_ods_values_are_numpy_arrays_after_forward_conversion(fake_omas: None) -> None:
    """Forward conversion stores vector fields as finite float64 NumPy arrays."""
    equilibrium_ods = omas_bridge.ids_to_omas_equilibrium(_equilibrium_ids())
    core_ods = omas_bridge.ids_to_omas_core_profiles(_core_profiles_ids())

    assert np.asarray(equilibrium_ods["equilibrium.time"]).dtype == np.float64
    assert np.asarray(equilibrium_ods["equilibrium.time_slice.0.profiles_1d.q"]).dtype == np.float64
    assert np.asarray(core_ods["core_profiles.time"]).dtype == np.float64
    assert np.asarray(core_ods["core_profiles.profiles_1d.0.electrons.density"]).dtype == np.float64
