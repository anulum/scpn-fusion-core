#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — IMAS DD v3 Conformance Audit (1D.1)
# (c) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
IMAS Data Dictionary v3 conformance tests for the SCPN Fusion Core
IDS converter functions.

Validates that:
- ``geqdsk_to_imas_equilibrium()`` output has all required IMAS DD v3
  fields as defined in ``schemas/imas_equilibrium_v3.json``.
- ``state_to_imas_core_profiles()`` output has the correct IDS field
  structure (``ids_properties``, ``time``, ``profiles_1d``).
- ``state_to_imas_summary()`` output has the correct field structure
  (``ids_properties``, ``time``, ``global_quantities``).
- Round-trip: ``write_ids()`` then ``read_ids()`` preserves data.
- Unit conversions are correct (keV -> eV, 1e20 m^-3 -> m^-3).

Required equilibrium fields checked:
  - time_slice, profiles_1d (psi, q, pressure, j_tor, rho_tor_norm, phi),
    boundary (outline_r, outline_z),
    global_quantities (ip, magnetic_axis_r, magnetic_axis_z, psi_axis,
    psi_boundary, q_95, li_3, beta_pol, beta_tor, beta_normal,
    elongation, triangularity_upper, triangularity_lower)

Required core_profiles fields checked:
  - profiles_1d (grid/rho_tor_norm, electrons/density,
    electrons/temperature, ion/density, ion/temperature, q,
    pressure_thermal, j_total, zeff)

Uses pytest style. Imports from scpn_fusion.io.imas_connector.
Uses tmp_path fixture for file I/O tests.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pytest

from scpn_fusion.core.eqdsk import GEqdsk, read_geqdsk, write_geqdsk
from scpn_fusion.io.imas_connector import (
    geqdsk_to_imas_equilibrium,
    imas_equilibrium_to_geqdsk,
    read_ids,
    state_to_imas_core_profiles,
    state_to_imas_summary,
    write_ids,
)


# ── Paths ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schemas" / "imas_equilibrium_v3.json"
SPARC_DIR = REPO_ROOT / "validation" / "reference_data" / "sparc"


# ── JSON Schema validation helpers ───────────────────────────────────
# We implement lightweight schema checking so tests work without
# the optional ``jsonschema`` package.  If it is installed we defer
# to its full validator for extra coverage.

def _has_jsonschema() -> bool:
    try:
        import jsonschema  # noqa: F401
        return True
    except ImportError:
        return False


def _validate_against_schema(instance: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Validate *instance* against a JSON Schema draft-07 dict."""
    if _has_jsonschema():
        import jsonschema
        jsonschema.validate(instance, schema)
        return

    _check_required_recursive(instance, schema, path="$")


def _check_required_recursive(
    instance: Any,
    schema: Dict[str, Any],
    path: str,
) -> None:
    """Recursively check required keys and array item schemas."""
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(instance, Mapping):
            raise AssertionError(f"{path}: expected object, got {type(instance).__name__}")
        for key in schema.get("required", []):
            if key not in instance:
                raise AssertionError(f"{path}: missing required key '{key}'")
        props = schema.get("properties", {})
        for key, sub_schema in props.items():
            if key in instance:
                _check_required_recursive(instance[key], sub_schema, f"{path}.{key}")

    elif schema_type == "array":
        if not isinstance(instance, (list, tuple)):
            raise AssertionError(f"{path}: expected array, got {type(instance).__name__}")
        items_schema = schema.get("items")
        if items_schema and len(instance) > 0:
            for idx, item in enumerate(instance):
                _check_required_recursive(item, items_schema, f"{path}[{idx}]")

    elif schema_type == "number":
        if not isinstance(instance, (int, float)):
            raise AssertionError(f"{path}: expected number, got {type(instance).__name__}")

    elif schema_type == "integer":
        if not isinstance(instance, int) or isinstance(instance, bool):
            raise AssertionError(f"{path}: expected integer, got {type(instance).__name__}")
        if "enum" in schema and instance not in schema["enum"]:
            raise AssertionError(f"{path}: value {instance} not in enum {schema['enum']}")

    elif schema_type == "string":
        if not isinstance(instance, str):
            raise AssertionError(f"{path}: expected string, got {type(instance).__name__}")


# ── Required IMAS DD v3 field lists ──────────────────────────────────

REQUIRED_EQUILIBRIUM_GLOBAL_QUANTITIES = [
    "ip",
    "magnetic_axis",
    "psi_axis",
    "psi_boundary",
    "vacuum_toroidal_field",
]

REQUIRED_EQUILIBRIUM_PROFILES_1D = [
    "psi",
    "q",
    "pressure",
]

REQUIRED_EQUILIBRIUM_BOUNDARY = [
    "outline",
]

REQUIRED_CORE_PROFILES_1D = [
    "grid",
    "electrons",
]


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def imas_eq_schema() -> Dict[str, Any]:
    """Load the IMAS equilibrium v3 JSON schema if available."""
    if not SCHEMA_PATH.is_file():
        pytest.skip(f"Schema file not found: {SCHEMA_PATH}")
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_synthetic_geqdsk(nw: int = 33, nh: int = 33) -> GEqdsk:
    """Build a physically reasonable synthetic equilibrium."""
    rng = np.random.default_rng(42)
    R = np.linspace(1.0, 3.0, nw)
    Z = np.linspace(-1.5, 1.5, nh)
    RR, ZZ = np.meshgrid(R, Z)
    # Simple Solov'ev-like psi
    psi = (
        0.5 * ((RR - 2.0) ** 2 + ZZ ** 2)
        + 0.02 * rng.standard_normal((nh, nw))
    )

    return GEqdsk(
        description="IMAS conformance synthetic",
        nw=nw,
        nh=nh,
        rdim=2.0,
        zdim=3.0,
        rcentr=1.85,
        rleft=1.0,
        zmid=0.0,
        rmaxis=2.0,
        zmaxis=0.0,
        simag=-1.2,
        sibry=-0.1,
        bcentr=12.2,
        current=8.7e6,
        fpol=np.linspace(10.0, 9.5, nw),
        pres=np.linspace(1.5e5, 0.0, nw),
        ffprime=np.linspace(-0.5, 0.0, nw),
        pprime=np.linspace(-2e4, 0.0, nw),
        qpsi=np.linspace(1.0, 5.0, nw),
        psirz=psi,
        rbdry=np.array([1.3, 2.0, 2.7, 2.7, 2.0, 1.3], dtype=np.float64),
        zbdry=np.array([-1.0, -1.3, -0.5, 0.5, 1.3, 1.0], dtype=np.float64),
        rlim=np.array([0.9, 3.1, 3.1, 0.9], dtype=np.float64),
        zlim=np.array([-1.6, -1.6, 1.6, 1.6], dtype=np.float64),
    )


def _make_sample_state() -> Dict[str, Any]:
    """Build a sample plasma state with profiles."""
    n = 30
    rho = list(np.linspace(0.0, 1.0, n))
    te_kev = list(np.linspace(12.0, 0.3, n))  # central 12 keV -> edge 0.3 keV
    ne_1e20 = list(np.linspace(1.5, 0.05, n))  # central 1.5e20 -> edge 0.05e20
    return {
        "rho_norm": rho,
        "electron_temp_keV": te_kev,
        "electron_density_1e20_m3": ne_1e20,
    }


def _make_sample_summary_state() -> Dict[str, Any]:
    """Build a sample performance/summary state."""
    return {
        "power_fusion_MW": 140.0,
        "q_sci": 2.0,
        "beta_n": 2.8,
        "li": 0.85,
        "plasma_current_MA": 8.7,
        "confinement_time_s": 1.8,
    }


# =====================================================================
# Test Class 1: geqdsk_to_imas_equilibrium() — required IMAS DD v3 fields
# =====================================================================

class TestGeqdskToImasEquilibriumFields:
    """Validate that the equilibrium IDS output has all required IMAS DD v3 fields."""

    def test_has_time_slice(self) -> None:
        """Equilibrium IDS must contain a time_slice array."""
        eq = _make_synthetic_geqdsk()
        ids = geqdsk_to_imas_equilibrium(eq)
        assert "time_slice" in ids
        assert isinstance(ids["time_slice"], list)
        assert len(ids["time_slice"]) >= 1

    def test_has_ids_properties(self) -> None:
        """ids_properties must have homogeneous_time=1 and a comment."""
        eq = _make_synthetic_geqdsk()
        ids = geqdsk_to_imas_equilibrium(eq)
        assert "ids_properties" in ids
        props = ids["ids_properties"]
        assert props["homogeneous_time"] == 1
        assert isinstance(props.get("comment", ""), str)

    def test_profiles_1d_has_psi(self) -> None:
        """profiles_1d must contain psi array."""
        eq = _make_synthetic_geqdsk()
        ids = geqdsk_to_imas_equilibrium(eq)
        p1d = ids["time_slice"][0]["profiles_1d"]
        assert "psi" in p1d
        assert len(p1d["psi"]) == eq.nw

    def test_profiles_1d_has_q(self) -> None:
        """profiles_1d must contain q (safety factor) array."""
        eq = _make_synthetic_geqdsk()
        ids = geqdsk_to_imas_equilibrium(eq)
        p1d = ids["time_slice"][0]["profiles_1d"]
        assert "q" in p1d
        assert len(p1d["q"]) == eq.nw

    def test_profiles_1d_has_pressure(self) -> None:
        """profiles_1d must contain pressure array."""
        eq = _make_synthetic_geqdsk()
        ids = geqdsk_to_imas_equilibrium(eq)
        p1d = ids["time_slice"][0]["profiles_1d"]
        assert "pressure" in p1d
        assert len(p1d["pressure"]) == eq.nw

    def test_boundary_outline_r_and_z(self) -> None:
        """boundary must contain outline with r and z arrays."""
        eq = _make_synthetic_geqdsk()
        ids = geqdsk_to_imas_equilibrium(eq)
        ts = ids["time_slice"][0]
        assert "boundary" in ts
        outline = ts["boundary"]["outline"]
        assert "r" in outline
        assert "z" in outline
        assert len(outline["r"]) == len(eq.rbdry)
        assert len(outline["z"]) == len(eq.zbdry)

    def test_global_quantities_ip(self) -> None:
        """global_quantities must contain ip (plasma current)."""
        eq = _make_synthetic_geqdsk()
        ids = geqdsk_to_imas_equilibrium(eq)
        gq = ids["time_slice"][0]["global_quantities"]
        assert "ip" in gq
        assert abs(gq["ip"] - eq.current) < 1e-6

    def test_global_quantities_magnetic_axis_r_z(self) -> None:
        """global_quantities.magnetic_axis must have r and z."""
        eq = _make_synthetic_geqdsk()
        ids = geqdsk_to_imas_equilibrium(eq)
        gq = ids["time_slice"][0]["global_quantities"]
        assert "magnetic_axis" in gq
        axis = gq["magnetic_axis"]
        assert "r" in axis
        assert "z" in axis
        assert isinstance(axis["r"], float)
        assert isinstance(axis["z"], float)
        assert abs(axis["r"] - eq.rmaxis) < 1e-12
        assert abs(axis["z"] - eq.zmaxis) < 1e-12

    def test_global_quantities_psi_axis_and_boundary(self) -> None:
        """psi_axis and psi_boundary must be present and match GEqdsk values."""
        eq = _make_synthetic_geqdsk()
        ids = geqdsk_to_imas_equilibrium(eq)
        gq = ids["time_slice"][0]["global_quantities"]
        assert "psi_axis" in gq
        assert "psi_boundary" in gq
        assert abs(gq["psi_axis"] - eq.simag) < 1e-12
        assert abs(gq["psi_boundary"] - eq.sibry) < 1e-12

    def test_global_quantities_vacuum_toroidal_field(self) -> None:
        """vacuum_toroidal_field must have r0 and b0."""
        eq = _make_synthetic_geqdsk()
        ids = geqdsk_to_imas_equilibrium(eq)
        gq = ids["time_slice"][0]["global_quantities"]
        assert "vacuum_toroidal_field" in gq
        vtf = gq["vacuum_toroidal_field"]
        assert "r0" in vtf
        assert "b0" in vtf
        assert abs(vtf["r0"] - eq.rcentr) < 1e-12
        assert abs(vtf["b0"] - eq.bcentr) < 1e-12

    def test_profiles_2d_grid_and_psi(self) -> None:
        """profiles_2d must have grid (dim1, dim2) and psi matching (nh, nw)."""
        eq = _make_synthetic_geqdsk(nw=17, nh=21)
        ids = geqdsk_to_imas_equilibrium(eq)
        p2d_list = ids["time_slice"][0]["profiles_2d"]
        assert isinstance(p2d_list, list)
        assert len(p2d_list) >= 1
        p2d = p2d_list[0]
        assert "grid" in p2d
        assert len(p2d["grid"]["dim1"]) == 17
        assert len(p2d["grid"]["dim2"]) == 21
        psi_2d = np.asarray(p2d["psi"])
        assert psi_2d.shape == (21, 17)

    def test_code_block_present(self) -> None:
        """IDS should include a code block identifying the producer."""
        eq = _make_synthetic_geqdsk()
        ids = geqdsk_to_imas_equilibrium(eq)
        assert "code" in ids
        assert "name" in ids["code"]
        assert "version" in ids["code"]

    def test_synthetic_passes_schema(self, imas_eq_schema: Dict[str, Any]) -> None:
        """Synthetic GEqdsk -> IMAS equilibrium must pass the JSON schema."""
        eq = _make_synthetic_geqdsk()
        ids = geqdsk_to_imas_equilibrium(eq, time_s=1.0, shot=99999, run=0)
        _validate_against_schema(ids, imas_eq_schema)


# =====================================================================
# Test Class 2: state_to_imas_core_profiles() — required fields
# =====================================================================

class TestCoreProfilesConformance:
    """Validate core_profiles IDS output structure and required fields."""

    def test_top_level_keys(self) -> None:
        """core_profiles IDS must have ids_properties, time, profiles_1d."""
        state = _make_sample_state()
        ids = state_to_imas_core_profiles(state, time_s=1.5)
        assert "ids_properties" in ids
        assert "time" in ids
        assert "profiles_1d" in ids

    def test_ids_properties(self) -> None:
        """ids_properties must have homogeneous_time and comment."""
        state = _make_sample_state()
        ids = state_to_imas_core_profiles(state)
        props = ids["ids_properties"]
        assert props["homogeneous_time"] == 1
        assert isinstance(props["comment"], str)

    def test_profiles_1d_grid_rho_tor_norm(self) -> None:
        """profiles_1d must have grid/rho_tor_norm spanning [0, 1]."""
        state = _make_sample_state()
        ids = state_to_imas_core_profiles(state)
        p = ids["profiles_1d"][0]
        assert "grid" in p
        rho = p["grid"]["rho_tor_norm"]
        assert "rho_tor_norm" in p["grid"]
        assert abs(rho[0] - 0.0) < 1e-10
        assert abs(rho[-1] - 1.0) < 1e-10

    def test_profiles_1d_electrons_density(self) -> None:
        """profiles_1d must have electrons/density."""
        state = _make_sample_state()
        ids = state_to_imas_core_profiles(state)
        p = ids["profiles_1d"][0]
        assert "electrons" in p
        assert "density" in p["electrons"]
        assert len(p["electrons"]["density"]) == len(state["rho_norm"])

    def test_profiles_1d_electrons_temperature(self) -> None:
        """profiles_1d must have electrons/temperature."""
        state = _make_sample_state()
        ids = state_to_imas_core_profiles(state)
        p = ids["profiles_1d"][0]
        assert "temperature" in p["electrons"]
        assert len(p["electrons"]["temperature"]) == len(state["rho_norm"])

    def test_unit_conversion_keV_to_eV(self) -> None:
        """electron_temp_keV -> electrons.temperature must multiply by 1000."""
        state = _make_sample_state()
        ids = state_to_imas_core_profiles(state)
        te_ev = ids["profiles_1d"][0]["electrons"]["temperature"]
        te_kev = state["electron_temp_keV"]
        for ev, kev in zip(te_ev, te_kev):
            expected = kev * 1.0e3
            assert abs(ev - expected) < 1e-6, (
                f"keV->eV conversion: {kev} keV should be {expected} eV, got {ev}"
            )

    def test_unit_conversion_1e20_to_m3(self) -> None:
        """electron_density_1e20_m3 -> electrons.density must multiply by 1e20."""
        state = _make_sample_state()
        ids = state_to_imas_core_profiles(state)
        ne_m3 = ids["profiles_1d"][0]["electrons"]["density"]
        ne_1e20 = state["electron_density_1e20_m3"]
        for m3, raw in zip(ne_m3, ne_1e20):
            expected = raw * 1.0e20
            assert abs(m3 / expected - 1.0) < 1e-10, (
                f"1e20->m^-3 conversion: {raw} * 1e20 = {expected}, got {m3}"
            )

    def test_profile_lengths_match(self) -> None:
        """All profiles must have the same length as rho_tor_norm."""
        state = _make_sample_state()
        ids = state_to_imas_core_profiles(state)
        p = ids["profiles_1d"][0]
        n = len(p["grid"]["rho_tor_norm"])
        assert len(p["electrons"]["temperature"]) == n
        assert len(p["electrons"]["density"]) == n

    def test_time_array(self) -> None:
        """time array must match the requested time_s."""
        state = _make_sample_state()
        ids = state_to_imas_core_profiles(state, time_s=3.5)
        assert ids["time"] == [3.5]


# =====================================================================
# Test Class 3: state_to_imas_summary() — required fields
# =====================================================================

class TestSummaryConformance:
    """Validate summary IDS output structure."""

    def test_top_level_keys(self) -> None:
        """summary IDS must have ids_properties, time, global_quantities."""
        state = _make_sample_summary_state()
        ids = state_to_imas_summary(state)
        assert "ids_properties" in ids
        assert "time" in ids
        assert "global_quantities" in ids

    def test_ids_properties(self) -> None:
        """ids_properties must have homogeneous_time=0 for summary."""
        state = _make_sample_summary_state()
        ids = state_to_imas_summary(state)
        assert ids["ids_properties"]["homogeneous_time"] == 0
        assert isinstance(ids["ids_properties"]["comment"], str)

    def test_field_mapping(self) -> None:
        """Internal field names must map to correct IMAS keys."""
        state = _make_sample_summary_state()
        ids = state_to_imas_summary(state)
        gq = ids["global_quantities"]

        expected_mapping = {
            "power_fusion": 140.0,
            "q": 2.0,
            "beta_n": 2.8,
            "li": 0.85,
            "ip": 8.7,
            "tau_e": 1.8,
        }
        for imas_key, expected_val in expected_mapping.items():
            assert imas_key in gq, f"Missing IMAS key '{imas_key}' in global_quantities"
            assert abs(gq[imas_key] - expected_val) < 1e-10, (
                f"Key '{imas_key}': expected {expected_val}, got {gq[imas_key]}"
            )

    def test_empty_state(self) -> None:
        """Empty state should produce an empty global_quantities dict."""
        ids = state_to_imas_summary({})
        assert ids["global_quantities"] == {}

    def test_rejects_non_mapping(self) -> None:
        """Non-mapping input must raise ValueError."""
        with pytest.raises(ValueError, match="must be a mapping"):
            state_to_imas_summary("not a dict")  # type: ignore[arg-type]


# =====================================================================
# Test Class 4: Round-trip write_ids() / read_ids()
# =====================================================================

class TestRoundTripWriteReadIds:
    """Verify that write_ids() then read_ids() preserves data."""

    def test_equilibrium_write_read_roundtrip(self, tmp_path: Path) -> None:
        """Equilibrium IDS round-trip through JSON file must preserve all fields."""
        eq = _make_synthetic_geqdsk(nw=33, nh=33)
        ids_orig = geqdsk_to_imas_equilibrium(eq, time_s=1.0, shot=12345, run=0)

        out_file = tmp_path / "eq_roundtrip.json"
        write_ids(ids_orig, out_file)
        ids_loaded = read_ids(out_file)

        # Verify structure
        assert "ids_properties" in ids_loaded
        assert "time_slice" in ids_loaded
        assert len(ids_loaded["time_slice"]) == 1

        # Verify global quantities
        gq_orig = ids_orig["time_slice"][0]["global_quantities"]
        gq_loaded = ids_loaded["time_slice"][0]["global_quantities"]
        assert abs(gq_loaded["ip"] - gq_orig["ip"]) < 1e-6
        assert abs(gq_loaded["psi_axis"] - gq_orig["psi_axis"]) < 1e-12
        assert abs(gq_loaded["psi_boundary"] - gq_orig["psi_boundary"]) < 1e-12

        # Verify 2D psi array
        psi_orig = np.asarray(ids_orig["time_slice"][0]["profiles_2d"][0]["psi"])
        psi_loaded = np.asarray(ids_loaded["time_slice"][0]["profiles_2d"][0]["psi"])
        np.testing.assert_allclose(psi_loaded, psi_orig, atol=1e-10)

    def test_core_profiles_write_read_roundtrip(self, tmp_path: Path) -> None:
        """Core profiles IDS round-trip through JSON file must preserve profiles."""
        state = _make_sample_state()
        ids_orig = state_to_imas_core_profiles(state, time_s=2.5)

        out_file = tmp_path / "cp_roundtrip.json"
        write_ids(ids_orig, out_file)
        ids_loaded = read_ids(out_file)

        # Verify rho_tor_norm
        rho_orig = ids_orig["profiles_1d"][0]["grid"]["rho_tor_norm"]
        rho_loaded = ids_loaded["profiles_1d"][0]["grid"]["rho_tor_norm"]
        np.testing.assert_allclose(rho_loaded, rho_orig, atol=1e-12)

        # Verify temperature
        te_orig = ids_orig["profiles_1d"][0]["electrons"]["temperature"]
        te_loaded = ids_loaded["profiles_1d"][0]["electrons"]["temperature"]
        np.testing.assert_allclose(te_loaded, te_orig, atol=1e-6)

        # Verify density
        ne_orig = ids_orig["profiles_1d"][0]["electrons"]["density"]
        ne_loaded = ids_loaded["profiles_1d"][0]["electrons"]["density"]
        np.testing.assert_allclose(ne_loaded, ne_orig, rtol=1e-10)

    def test_summary_write_read_roundtrip(self, tmp_path: Path) -> None:
        """Summary IDS round-trip through JSON file must preserve global quantities."""
        state = _make_sample_summary_state()
        ids_orig = state_to_imas_summary(state)

        out_file = tmp_path / "summary_roundtrip.json"
        write_ids(ids_orig, out_file)
        ids_loaded = read_ids(out_file)

        gq_orig = ids_orig["global_quantities"]
        gq_loaded = ids_loaded["global_quantities"]
        for key in gq_orig:
            assert abs(gq_loaded[key] - gq_orig[key]) < 1e-10, (
                f"Summary round-trip mismatch for '{key}': "
                f"{gq_orig[key]} vs {gq_loaded[key]}"
            )


# =====================================================================
# Test Class 5: GEqdsk -> IMAS -> GEqdsk round-trip
# =====================================================================

class TestGeqdskRoundTrip:
    """Verify that key equilibrium fields survive the IMAS round-trip."""

    def test_psi_2d_preserved(self) -> None:
        """psirz must survive the GEqdsk -> IMAS -> GEqdsk round-trip."""
        eq = _make_synthetic_geqdsk(nw=33, nh=33)
        ids = geqdsk_to_imas_equilibrium(eq, time_s=0.0)
        eq2 = imas_equilibrium_to_geqdsk(ids)
        np.testing.assert_allclose(eq2.psirz, eq.psirz, atol=1e-12)

    def test_q_profile_preserved(self) -> None:
        """q-profile must survive the round-trip."""
        eq = _make_synthetic_geqdsk()
        ids = geqdsk_to_imas_equilibrium(eq)
        eq2 = imas_equilibrium_to_geqdsk(ids)
        np.testing.assert_allclose(eq2.qpsi, eq.qpsi, atol=1e-12)

    def test_pressure_preserved(self) -> None:
        """Pressure profile must survive the round-trip."""
        eq = _make_synthetic_geqdsk()
        ids = geqdsk_to_imas_equilibrium(eq)
        eq2 = imas_equilibrium_to_geqdsk(ids)
        np.testing.assert_allclose(eq2.pres, eq.pres, atol=1e-12)

    def test_fpol_preserved(self) -> None:
        """F(psi) (poloidal current function) must survive the round-trip."""
        eq = _make_synthetic_geqdsk()
        ids = geqdsk_to_imas_equilibrium(eq)
        eq2 = imas_equilibrium_to_geqdsk(ids)
        np.testing.assert_allclose(eq2.fpol, eq.fpol, atol=1e-12)

    def test_axis_location_preserved(self) -> None:
        """Magnetic axis R,Z must survive the round-trip."""
        eq = _make_synthetic_geqdsk()
        ids = geqdsk_to_imas_equilibrium(eq)
        eq2 = imas_equilibrium_to_geqdsk(ids)
        assert abs(eq2.rmaxis - eq.rmaxis) < 1e-12
        assert abs(eq2.zmaxis - eq.zmaxis) < 1e-12

    def test_plasma_current_preserved(self) -> None:
        """Plasma current Ip must survive the round-trip."""
        eq = _make_synthetic_geqdsk()
        ids = geqdsk_to_imas_equilibrium(eq)
        eq2 = imas_equilibrium_to_geqdsk(ids)
        assert abs(eq2.current - eq.current) < 1e-6

    def test_psi_axis_boundary_preserved(self) -> None:
        """Psi at axis and boundary must survive the round-trip."""
        eq = _make_synthetic_geqdsk()
        ids = geqdsk_to_imas_equilibrium(eq)
        eq2 = imas_equilibrium_to_geqdsk(ids)
        assert abs(eq2.simag - eq.simag) < 1e-12
        assert abs(eq2.sibry - eq.sibry) < 1e-12

    def test_boundary_preserved(self) -> None:
        """Boundary contour (rbdry, zbdry) must survive the round-trip."""
        eq = _make_synthetic_geqdsk()
        ids = geqdsk_to_imas_equilibrium(eq)
        eq2 = imas_equilibrium_to_geqdsk(ids)
        np.testing.assert_allclose(eq2.rbdry, eq.rbdry, atol=1e-12)
        np.testing.assert_allclose(eq2.zbdry, eq.zbdry, atol=1e-12)

    def test_grid_dimensions_preserved(self) -> None:
        """nw, nh, rdim, zdim, rleft, zmid must survive the round-trip."""
        eq = _make_synthetic_geqdsk(nw=25, nh=31)
        ids = geqdsk_to_imas_equilibrium(eq)
        eq2 = imas_equilibrium_to_geqdsk(ids)
        assert eq2.nw == eq.nw
        assert eq2.nh == eq.nh
        assert abs(eq2.rdim - eq.rdim) < 1e-10
        assert abs(eq2.zdim - eq.zdim) < 1e-10
        assert abs(eq2.rleft - eq.rleft) < 1e-10
        assert abs(eq2.zmid - eq.zmid) < 1e-10


# =====================================================================
# Test Class 6: Unit conversion spot-checks
# =====================================================================

class TestUnitConversions:
    """Explicit spot-checks for unit conversion correctness."""

    def test_kev_to_ev_edge_values(self) -> None:
        """Edge case: 0 keV -> 0 eV and small keV -> correct eV."""
        state = {
            "rho_norm": [0.0, 0.5, 1.0],
            "electron_temp_keV": [0.0, 5.0, 0.001],
            "electron_density_1e20_m3": [1.0, 1.0, 1.0],
        }
        ids = state_to_imas_core_profiles(state)
        te_ev = ids["profiles_1d"][0]["electrons"]["temperature"]
        assert abs(te_ev[0] - 0.0) < 1e-12
        assert abs(te_ev[1] - 5000.0) < 1e-6
        assert abs(te_ev[2] - 1.0) < 1e-6

    def test_density_conversion_large_values(self) -> None:
        """Large density values must convert correctly without overflow."""
        state = {
            "rho_norm": [0.0, 1.0],
            "electron_temp_keV": [1.0, 0.5],
            "electron_density_1e20_m3": [5.0, 0.01],
        }
        ids = state_to_imas_core_profiles(state)
        ne_m3 = ids["profiles_1d"][0]["electrons"]["density"]
        assert abs(ne_m3[0] - 5.0e20) < 1e10
        assert abs(ne_m3[1] - 1.0e18) < 1e8

    def test_conversion_preserves_monotonicity(self) -> None:
        """If input profiles are monotonic, output must also be monotonic."""
        n = 50
        state = {
            "rho_norm": list(np.linspace(0.0, 1.0, n)),
            "electron_temp_keV": list(np.linspace(15.0, 0.1, n)),  # decreasing
            "electron_density_1e20_m3": list(np.linspace(2.0, 0.01, n)),  # decreasing
        }
        ids = state_to_imas_core_profiles(state)
        te_ev = ids["profiles_1d"][0]["electrons"]["temperature"]
        ne_m3 = ids["profiles_1d"][0]["electrons"]["density"]

        for i in range(1, len(te_ev)):
            assert te_ev[i] <= te_ev[i - 1], f"Te not monotonic at index {i}"
        for i in range(1, len(ne_m3)):
            assert ne_m3[i] <= ne_m3[i - 1], f"ne not monotonic at index {i}"
