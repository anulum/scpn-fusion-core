# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — IMAS IDS Standard Compliance Tests
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core.eqdsk import GEqdsk
from scpn_fusion.io.imas_connector import (
    geqdsk_to_imas_equilibrium,
    imas_equilibrium_to_geqdsk,
    state_to_imas_core_profiles,
    state_to_imas_summary,
    write_ids,
    read_ids,
)


def _sample_geqdsk(nw: int = 33, nh: int = 33) -> GEqdsk:
    """Create a minimal valid GEqdsk for testing."""
    rdim = 2.0
    zdim = 3.0
    rleft = 1.0
    zmid = 0.0
    return GEqdsk(
        description="test equilibrium",
        nw=nw,
        nh=nh,
        rdim=rdim,
        zdim=zdim,
        rcentr=1.7,
        rleft=rleft,
        zmid=zmid,
        rmaxis=1.65,
        zmaxis=0.02,
        simag=-1.5,
        sibry=-0.2,
        bcentr=5.3,
        current=15e6,
        fpol=np.linspace(4.0, 5.0, nw),
        pres=np.linspace(3e4, 0.0, nw),
        ffprime=np.zeros(nw, dtype=np.float64),
        pprime=np.zeros(nw, dtype=np.float64),
        qpsi=np.linspace(1.0, 6.0, nw),
        psirz=np.random.default_rng(42).standard_normal((nh, nw)),
        rbdry=np.array([1.2, 1.8, 2.5, 2.8, 2.5, 1.8, 1.2], dtype=np.float64),
        zbdry=np.array([-1.0, -1.2, -0.5, 0.0, 0.5, 1.2, 1.0], dtype=np.float64),
        rlim=np.array([], dtype=np.float64),
        zlim=np.array([], dtype=np.float64),
    )


def _sample_state() -> dict[str, object]:
    rho = list(np.linspace(0.0, 1.0, 20))
    te = list(np.linspace(10.0, 0.5, 20))
    ne = list(np.linspace(1.2, 0.1, 20))
    return {
        "rho_norm": rho,
        "electron_temp_keV": te,
        "electron_density_1e20_m3": ne,
    }


# ── GEqdsk ↔ IMAS equilibrium round-trip ──────────────────────────────


def test_geqdsk_to_imas_roundtrip() -> None:
    eq = _sample_geqdsk()
    ids = geqdsk_to_imas_equilibrium(eq, time_s=1.5, shot=12345, run=1)
    eq2 = imas_equilibrium_to_geqdsk(ids)

    assert eq2.nw == eq.nw
    assert eq2.nh == eq.nh
    assert abs(eq2.current - eq.current) < 1e-6
    assert abs(eq2.rmaxis - eq.rmaxis) < 1e-6
    assert abs(eq2.zmaxis - eq.zmaxis) < 1e-6
    assert abs(eq2.simag - eq.simag) < 1e-6
    assert abs(eq2.sibry - eq.sibry) < 1e-6
    assert abs(eq2.bcentr - eq.bcentr) < 1e-6
    assert abs(eq2.rcentr - eq.rcentr) < 1e-6
    np.testing.assert_allclose(eq2.qpsi, eq.qpsi, atol=1e-10)
    np.testing.assert_allclose(eq2.pres, eq.pres, atol=1e-10)
    np.testing.assert_allclose(eq2.psirz, eq.psirz, atol=1e-10)
    np.testing.assert_allclose(eq2.rbdry, eq.rbdry, atol=1e-10)
    np.testing.assert_allclose(eq2.zbdry, eq.zbdry, atol=1e-10)


def test_imas_equilibrium_has_ids_properties() -> None:
    eq = _sample_geqdsk()
    ids = geqdsk_to_imas_equilibrium(eq, time_s=0.5)
    assert "ids_properties" in ids
    assert ids["ids_properties"]["homogeneous_time"] == 1
    assert "comment" in ids["ids_properties"]


def test_imas_equilibrium_profiles_2d_shape() -> None:
    eq = _sample_geqdsk(nw=17, nh=21)
    ids = geqdsk_to_imas_equilibrium(eq)
    ts = ids["time_slice"][0]
    p2d = ts["profiles_2d"][0]
    psi = np.asarray(p2d["psi"])
    assert psi.shape == (21, 17)
    assert len(p2d["grid"]["dim1"]) == 17
    assert len(p2d["grid"]["dim2"]) == 21


def test_imas_equilibrium_global_quantities() -> None:
    eq = _sample_geqdsk()
    ids = geqdsk_to_imas_equilibrium(eq)
    gq = ids["time_slice"][0]["global_quantities"]
    assert abs(gq["ip"] - eq.current) < 1e-6
    assert abs(gq["magnetic_axis"]["r"] - eq.rmaxis) < 1e-6
    assert abs(gq["magnetic_axis"]["z"] - eq.zmaxis) < 1e-6
    assert abs(gq["psi_axis"] - eq.simag) < 1e-6
    assert abs(gq["psi_boundary"] - eq.sibry) < 1e-6


# ── Core profiles ────────────────────────────────────────────────────


def test_imas_core_profiles_roundtrip() -> None:
    state = _sample_state()
    ids = state_to_imas_core_profiles(state, time_s=2.0)
    assert "profiles_1d" in ids
    p1d = ids["profiles_1d"][0]
    assert "grid" in p1d
    assert "electrons" in p1d
    assert len(p1d["grid"]["rho_tor_norm"]) == 20
    assert len(p1d["electrons"]["temperature"]) == 20
    assert len(p1d["electrons"]["density"]) == 20


def test_imas_core_profiles_unit_conversion() -> None:
    state = _sample_state()
    ids = state_to_imas_core_profiles(state)
    p1d = ids["profiles_1d"][0]

    # keV -> eV: multiply by 1000
    te_kev = list(state["electron_temp_keV"])  # type: ignore[arg-type]
    te_ev = p1d["electrons"]["temperature"]
    np.testing.assert_allclose(te_ev, [v * 1e3 for v in te_kev], atol=1e-6)

    # 1e20 m^-3 -> m^-3: multiply by 1e20
    ne_1e20 = list(state["electron_density_1e20_m3"])  # type: ignore[arg-type]
    ne_m3 = p1d["electrons"]["density"]
    np.testing.assert_allclose(ne_m3, [v * 1e20 for v in ne_1e20], rtol=1e-10)


def test_imas_core_profiles_rejects_mismatched_lengths() -> None:
    state = {
        "rho_norm": list(np.linspace(0.0, 1.0, 20)),
        "electron_temp_keV": list(np.linspace(10.0, 0.5, 15)),  # wrong length
        "electron_density_1e20_m3": list(np.linspace(1.2, 0.1, 20)),
    }
    with pytest.raises(ValueError, match="length must match"):
        state_to_imas_core_profiles(state)


# ── Summary ──────────────────────────────────────────────────────────


def test_imas_summary_has_required_fields() -> None:
    state = {
        "power_fusion_MW": 500.0,
        "q_sci": 10.0,
        "beta_n": 2.5,
        "plasma_current_MA": 15.0,
    }
    ids = state_to_imas_summary(state)
    assert "ids_properties" in ids
    assert "global_quantities" in ids
    gq = ids["global_quantities"]
    assert abs(gq["power_fusion"] - 500.0) < 1e-6
    assert abs(gq["q"] - 10.0) < 1e-6
    assert abs(gq["beta_n"] - 2.5) < 1e-6
    assert abs(gq["ip"] - 15.0) < 1e-6


# ── File I/O ─────────────────────────────────────────────────────────


def test_write_ids_read_ids_roundtrip(tmp_path: Path) -> None:
    eq = _sample_geqdsk()
    ids = geqdsk_to_imas_equilibrium(eq, time_s=1.0)
    out_path = tmp_path / "eq.ids.json"
    write_ids(ids, out_path)
    loaded = read_ids(out_path)
    assert loaded["ids_properties"]["homogeneous_time"] == 1
    ts = loaded["time_slice"][0]
    assert abs(ts["global_quantities"]["ip"] - eq.current) < 1e-6
    psi = np.asarray(ts["profiles_2d"][0]["psi"])
    np.testing.assert_allclose(psi, eq.psirz, atol=1e-10)


def test_write_ids_validates_schema(tmp_path: Path) -> None:
    out_path = tmp_path / "bad.ids.json"
    with pytest.raises(ValueError, match="ids_properties"):
        write_ids({"data": 42}, out_path)


def test_read_ids_rejects_corrupt(tmp_path: Path) -> None:
    bad_path = tmp_path / "corrupt.json"
    bad_path.write_text("{invalid json", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt"):
        read_ids(bad_path)


def test_geqdsk_to_imas_rejects_degenerate() -> None:
    eq = GEqdsk(nw=0, nh=0)
    with pytest.raises(ValueError, match="nw >= 2"):
        geqdsk_to_imas_equilibrium(eq)
