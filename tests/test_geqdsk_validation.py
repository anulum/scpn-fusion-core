# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GEQDSK Validation Smoke Tests
# (c) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Smoke tests for the GEQDSK real-data validation pipeline.

These tests load one SPARC GEQDSK file and verify basic sanity:
  - psirz shape matches (nh, nw)
  - magnetic axis is within the computational domain
  - profiles (pprime, ffprime, fpol, pres, qpsi) are finite
  - sub-grid axis interpolation returns sensible values
  - the GS source term is finite and non-zero in the plasma interior
  - a short Picard+SOR solve runs without crashing
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ and validation/ are importable
ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from scpn_fusion.core.eqdsk import read_geqdsk

# Import validation helpers via importlib (validation/ is not a package)
import importlib.util as _ilu

_GEQDSK_VAL_PATH = ROOT / "validation" / "run_geqdsk_validation.py"
_spec = _ilu.spec_from_file_location("run_geqdsk_validation", _GEQDSK_VAL_PATH)
_gval = _ilu.module_from_spec(_spec)
sys.modules.setdefault("run_geqdsk_validation", _gval)
_spec.loader.exec_module(_gval)

_find_axis_subgrid = _gval._find_axis_subgrid
_compute_gs_source_from_profiles = _gval._compute_gs_source_from_profiles
picard_sor_geqdsk = _gval.picard_sor_geqdsk
validate_one_geqdsk = _gval.validate_one_geqdsk

# Path to SPARC reference GEQDSK files
SPARC_DIR = ROOT / "validation" / "reference_data" / "sparc"
LMODE_VV = SPARC_DIR / "lmode_vv.geqdsk"
LMODE_HV = SPARC_DIR / "lmode_hv.geqdsk"


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def eq_vv():
    """Load the lmode_vv equilibrium once for the module."""
    pytest.importorskip("numpy")
    if not LMODE_VV.exists():
        pytest.skip("SPARC reference data not available")
    return read_geqdsk(LMODE_VV)


@pytest.fixture(scope="module")
def eq_hv():
    """Load the lmode_hv equilibrium once for the module."""
    if not LMODE_HV.exists():
        pytest.skip("SPARC reference data not available")
    return read_geqdsk(LMODE_HV)


# ── Shape and dimension checks ────────────────────────────────────────

class TestGeqdskSanity:
    """Basic sanity checks on loaded GEQDSK data."""

    def test_psirz_shape(self, eq_vv):
        assert eq_vv.psirz.shape == (eq_vv.nh, eq_vv.nw)
        assert eq_vv.nh > 0
        assert eq_vv.nw > 0

    def test_grid_dimensions(self, eq_vv):
        r = eq_vv.r
        z = eq_vv.z
        assert len(r) == eq_vv.nw
        assert len(z) == eq_vv.nh
        assert r[0] < r[-1], "R grid must be increasing"
        assert z[0] < z[-1], "Z grid must be increasing"

    def test_axis_within_domain(self, eq_vv):
        r = eq_vv.r
        z = eq_vv.z
        assert r[0] <= eq_vv.rmaxis <= r[-1], \
            f"rmaxis={eq_vv.rmaxis} outside R domain [{r[0]}, {r[-1]}]"
        assert z[0] <= eq_vv.zmaxis <= z[-1], \
            f"zmaxis={eq_vv.zmaxis} outside Z domain [{z[0]}, {z[-1]}]"

    def test_profiles_finite(self, eq_vv):
        assert np.all(np.isfinite(eq_vv.pprime)), "pprime contains non-finite values"
        assert np.all(np.isfinite(eq_vv.ffprime)), "ffprime contains non-finite values"
        assert np.all(np.isfinite(eq_vv.fpol)), "fpol contains non-finite values"
        assert np.all(np.isfinite(eq_vv.pres)), "pres contains non-finite values"
        assert np.all(np.isfinite(eq_vv.qpsi)), "qpsi contains non-finite values"

    def test_profile_lengths(self, eq_vv):
        assert len(eq_vv.pprime) == eq_vv.nw
        assert len(eq_vv.ffprime) == eq_vv.nw
        assert len(eq_vv.fpol) == eq_vv.nw
        assert len(eq_vv.pres) == eq_vv.nw
        assert len(eq_vv.qpsi) == eq_vv.nw

    def test_psirz_finite(self, eq_vv):
        assert np.all(np.isfinite(eq_vv.psirz)), "psirz contains non-finite values"

    def test_psi_axis_boundary_distinct(self, eq_vv):
        assert eq_vv.simag != eq_vv.sibry, \
            "simag and sibry must be different for a valid equilibrium"

    def test_boundary_points_exist(self, eq_vv):
        assert len(eq_vv.rbdry) > 3, "Expected boundary with > 3 points"
        assert len(eq_vv.zbdry) == len(eq_vv.rbdry)


# ── Sub-grid axis interpolation ───────────────────────────────────────

class TestSubgridAxis:
    """Tests for the quadratic sub-grid axis finder."""

    def test_axis_near_declared(self, eq_vv):
        """Sub-grid axis should be close to the GEQDSK declared axis."""
        r_ax, z_ax = _find_axis_subgrid(
            eq_vv.psirz, eq_vv.r, eq_vv.z,
            psi_axis_hint=eq_vv.simag,
        )
        # Should be within ~2 grid cells of the declared axis
        dr = eq_vv.r[1] - eq_vv.r[0]
        dz = eq_vv.z[1] - eq_vv.z[0]
        assert abs(r_ax - eq_vv.rmaxis) < 3 * dr, \
            f"R axis off by {abs(r_ax - eq_vv.rmaxis):.4f} m"
        assert abs(z_ax - eq_vv.zmaxis) < 3 * dz, \
            f"Z axis off by {abs(z_ax - eq_vv.zmaxis):.4f} m"

    def test_axis_within_domain(self, eq_vv):
        r_ax, z_ax = _find_axis_subgrid(
            eq_vv.psirz, eq_vv.r, eq_vv.z,
            psi_axis_hint=eq_vv.simag,
        )
        assert eq_vv.r[0] <= r_ax <= eq_vv.r[-1]
        assert eq_vv.z[0] <= z_ax <= eq_vv.z[-1]

    def test_synthetic_parabola(self):
        """Sub-grid finder should exactly locate a parabola minimum."""
        r_grid = np.linspace(1.0, 3.0, 21)
        z_grid = np.linspace(-1.0, 1.0, 21)
        RR, ZZ = np.meshgrid(r_grid, z_grid)  # (nz, nr)
        # Parabola with minimum at R=2.05, Z=0.03
        r0, z0 = 2.05, 0.03
        psi = (RR - r0) ** 2 + (ZZ - z0) ** 2
        r_ax, z_ax = _find_axis_subgrid(psi, r_grid, z_grid)
        assert abs(r_ax - r0) < 0.005, f"R axis error: {abs(r_ax - r0):.4f}"
        assert abs(z_ax - z0) < 0.005, f"Z axis error: {abs(z_ax - z0):.4f}"


# ── GS source term ───────────────────────────────────────────────────

class TestGSSource:
    """Tests for the Grad-Shafranov source computation."""

    def test_source_finite(self, eq_vv):
        RR = np.ones((eq_vv.nh, 1)) * eq_vv.r[np.newaxis, :]
        source = _compute_gs_source_from_profiles(
            eq_vv.psirz, RR, eq_vv.pprime, eq_vv.ffprime,
            eq_vv.simag, eq_vv.sibry,
        )
        assert np.all(np.isfinite(source)), "Source contains non-finite values"

    def test_source_nonzero_interior(self, eq_vv):
        RR = np.ones((eq_vv.nh, 1)) * eq_vv.r[np.newaxis, :]
        source = _compute_gs_source_from_profiles(
            eq_vv.psirz, RR, eq_vv.pprime, eq_vv.ffprime,
            eq_vv.simag, eq_vv.sibry,
        )
        # Interior should have non-zero source (plasma carries current)
        interior = source[eq_vv.nh // 4: 3 * eq_vv.nh // 4,
                          eq_vv.nw // 4: 3 * eq_vv.nw // 4]
        assert np.max(np.abs(interior)) > 0, "Source is identically zero in interior"

    def test_source_shape(self, eq_vv):
        RR = np.ones((eq_vv.nh, 1)) * eq_vv.r[np.newaxis, :]
        source = _compute_gs_source_from_profiles(
            eq_vv.psirz, RR, eq_vv.pprime, eq_vv.ffprime,
            eq_vv.simag, eq_vv.sibry,
        )
        assert source.shape == (eq_vv.nh, eq_vv.nw)


# ── Solver smoke test ────────────────────────────────────────────────

class TestSolverSmoke:
    """Quick solver runs (low iteration count) to verify no crashes."""

    def test_picard_sor_runs(self, eq_hv):
        """A short Picard+SOR run should produce a finite psi."""
        result = picard_sor_geqdsk(
            eq_hv,
            max_picard=2,
            max_sor=200,
            omega=1.5,
        )
        assert result["psi"].shape == (eq_hv.nh, eq_hv.nw)
        assert np.all(np.isfinite(result["psi"])), \
            "Solver produced non-finite psi values"

    def test_solver_axis_in_domain(self, eq_hv):
        result = picard_sor_geqdsk(
            eq_hv,
            max_picard=2,
            max_sor=200,
            omega=1.5,
        )
        assert eq_hv.r[0] <= result["r_axis"] <= eq_hv.r[-1]
        assert eq_hv.z[0] <= result["z_axis"] <= eq_hv.z[-1]

    def test_validate_one_returns_result(self, eq_hv):
        """validate_one_geqdsk should return a GeqdskResult dataclass."""
        res = validate_one_geqdsk(
            LMODE_HV,
            max_picard=2,
            max_sor=200,
            omega=1.5,
        )
        assert hasattr(res, "filename")
        assert hasattr(res, "psi_rmse")
        assert hasattr(res, "status")
        assert res.psi_rmse >= 0.0
        assert res.axis_dr_mm >= 0.0
        assert res.axis_dz_mm >= 0.0
        assert res.status in ("OK", "WARN", "FAIL")


# ── Cross-file consistency ───────────────────────────────────────────

class TestCrossFile:
    """Check that all three lmode files load consistently."""

    @pytest.mark.skipif(
        not all((SPARC_DIR / f).exists()
                for f in ["lmode_hv.geqdsk", "lmode_vh.geqdsk", "lmode_vv.geqdsk"]),
        reason="Not all SPARC lmode files available",
    )
    def test_all_lmode_files_parse(self):
        for fname in ["lmode_hv.geqdsk", "lmode_vh.geqdsk", "lmode_vv.geqdsk"]:
            eq = read_geqdsk(SPARC_DIR / fname)
            assert eq.nw > 0, f"{fname}: nw is 0"
            assert eq.nh > 0, f"{fname}: nh is 0"
            assert eq.psirz.shape == (eq.nh, eq.nw), f"{fname}: shape mismatch"
            assert np.all(np.isfinite(eq.psirz)), f"{fname}: non-finite psirz"
            assert len(eq.pprime) == eq.nw, f"{fname}: pprime length mismatch"
            assert len(eq.ffprime) == eq.nw, f"{fname}: ffprime length mismatch"
