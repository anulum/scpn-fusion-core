# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — G-EQDSK Parser Tests
# © 1998–2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for the G-EQDSK reader/writer round-trip and SPARC data parsing."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core.eqdsk import GEqdsk, read_geqdsk, write_geqdsk

SPARC_DIR = Path(__file__).resolve().parent.parent / "validation" / "reference_data" / "sparc"


# ── Helpers ────────────────────────────────────────────────────────────

def _make_synthetic(nw: int = 17, nh: int = 17) -> GEqdsk:
    """Build a small synthetic equilibrium for round-trip testing."""
    R = np.linspace(1.0, 3.0, nw)
    Z = np.linspace(-1.5, 1.5, nh)
    RR, ZZ = np.meshgrid(R, Z)
    psi = np.sin(np.pi * (RR - 1.0) / 2.0) * np.cos(np.pi * ZZ / 3.0)

    return GEqdsk(
        description="synthetic test equilibrium",
        nw=nw,
        nh=nh,
        rdim=2.0,
        zdim=3.0,
        rcentr=2.0,
        rleft=1.0,
        zmid=0.0,
        rmaxis=2.0,
        zmaxis=0.0,
        simag=0.0,
        sibry=1.0,
        bcentr=5.3,
        current=15e6,
        fpol=np.linspace(10.0, 9.5, nw),
        pres=np.linspace(1e5, 0, nw),
        ffprime=np.linspace(-0.5, 0.0, nw),
        pprime=np.linspace(-1e4, 0, nw),
        qpsi=np.linspace(1.0, 5.0, nw),
        psirz=psi,
        rbdry=np.array([1.5, 2.5, 2.5, 1.5]),
        zbdry=np.array([0.0, 0.5, -0.5, 0.0]),
        rlim=np.array([1.0, 3.0, 3.0, 1.0]),
        zlim=np.array([-1.5, -1.5, 1.5, 1.5]),
    )


# ── Round-trip tests ──────────────────────────────────────────────────

class TestEqdskRoundTrip:
    """Write then read synthetic GEQDSK; values must survive the trip."""

    def test_scalars(self):
        eq = _make_synthetic()
        with tempfile.NamedTemporaryFile(suffix=".geqdsk", delete=False) as f:
            path = Path(f.name)
        write_geqdsk(eq, path)
        eq2 = read_geqdsk(path)
        path.unlink()

        assert eq2.nw == eq.nw
        assert eq2.nh == eq.nh
        assert abs(eq2.rdim - eq.rdim) < 1e-6
        assert abs(eq2.rmaxis - eq.rmaxis) < 1e-6
        assert abs(eq2.bcentr - eq.bcentr) < 1e-6
        assert abs(eq2.current - eq.current) < 1.0  # 1 A tolerance

    def test_profiles(self):
        eq = _make_synthetic()
        with tempfile.NamedTemporaryFile(suffix=".geqdsk", delete=False) as f:
            path = Path(f.name)
        write_geqdsk(eq, path)
        eq2 = read_geqdsk(path)
        path.unlink()

        np.testing.assert_allclose(eq2.fpol, eq.fpol, atol=1e-6)
        np.testing.assert_allclose(eq2.pres, eq.pres, rtol=1e-5)
        np.testing.assert_allclose(eq2.qpsi, eq.qpsi, atol=1e-6)

    def test_flux_map(self):
        eq = _make_synthetic()
        with tempfile.NamedTemporaryFile(suffix=".geqdsk", delete=False) as f:
            path = Path(f.name)
        write_geqdsk(eq, path)
        eq2 = read_geqdsk(path)
        path.unlink()

        np.testing.assert_allclose(eq2.psirz, eq.psirz, atol=1e-6)

    def test_boundary_limiter(self):
        eq = _make_synthetic()
        with tempfile.NamedTemporaryFile(suffix=".geqdsk", delete=False) as f:
            path = Path(f.name)
        write_geqdsk(eq, path)
        eq2 = read_geqdsk(path)
        path.unlink()

        np.testing.assert_allclose(eq2.rbdry, eq.rbdry, atol=1e-6)
        np.testing.assert_allclose(eq2.zbdry, eq.zbdry, atol=1e-6)
        np.testing.assert_allclose(eq2.rlim, eq.rlim, atol=1e-6)


# ── SPARC data tests ─────────────────────────────────────────────────

@pytest.mark.skipif(
    not SPARC_DIR.exists(),
    reason="SPARC reference data not available",
)
class TestSparcData:
    """Validate SPARC GEQDSK files parse with correct physical values."""

    def test_lmode_vv_dimensions(self):
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")
        assert eq.nw == 129
        assert eq.nh == 129
        assert eq.psirz.shape == (129, 129)

    def test_lmode_vv_physics(self):
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")
        assert abs(eq.bcentr) == pytest.approx(12.16, abs=0.5)
        assert abs(eq.current / 1e6) == pytest.approx(8.5, abs=0.5)
        assert eq.rmaxis == pytest.approx(1.85, abs=0.05)

    def test_sparc_1310_full_current(self):
        eq = read_geqdsk(SPARC_DIR / "sparc_1310.eqdsk")
        assert abs(eq.current / 1e6) == pytest.approx(8.7, abs=0.2)
        assert eq.nw == 61
        assert len(eq.fpol) == 61

    def test_all_files_parse(self):
        """Every GEQDSK/EQDSK file in the directory must parse."""
        files = list(SPARC_DIR.glob("*.geqdsk")) + list(SPARC_DIR.glob("*.eqdsk"))
        assert len(files) >= 5, "Expected at least 5 equilibrium files"
        for f in files:
            eq = read_geqdsk(f)
            assert eq.nw > 0
            assert eq.nh > 0
            assert eq.psirz.shape == (eq.nh, eq.nw)

    def test_q_profile_physical(self):
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")
        q = np.abs(eq.qpsi)
        # q should be order ~1 at axis, ~3-5 at edge for standard tokamak
        assert q[0] > 0.5
        assert q[-1] < 50

    def test_boundary_closed(self):
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")
        assert len(eq.rbdry) > 10
        # Boundary should roughly close (first ~ last point)
        assert abs(eq.rbdry[0] - eq.rbdry[-1]) < 0.1
        assert abs(eq.zbdry[0] - eq.zbdry[-1]) < 0.1


# ── Derived properties ────────────────────────────────────────────────

class TestGEqdskProperties:
    """Test derived grid and normalisation methods."""

    def test_r_grid(self):
        eq = _make_synthetic()
        r = eq.r
        assert len(r) == eq.nw
        assert r[0] == pytest.approx(eq.rleft)
        assert r[-1] == pytest.approx(eq.rleft + eq.rdim)

    def test_z_grid(self):
        eq = _make_synthetic()
        z = eq.z
        assert len(z) == eq.nh
        assert z[0] == pytest.approx(eq.zmid - eq.zdim / 2)

    def test_psi_norm(self):
        eq = _make_synthetic()
        pn = eq.psi_norm
        assert pn[0] == pytest.approx(0.0)
        assert pn[-1] == pytest.approx(1.0)

    def test_to_config(self):
        eq = _make_synthetic()
        cfg = eq.to_config("test-reactor")
        assert cfg["reactor_name"] == "test-reactor"
        assert cfg["grid_resolution"] == [eq.nw, eq.nh]
        assert cfg["physics"]["plasma_current_target"] == pytest.approx(15.0, abs=0.1)
