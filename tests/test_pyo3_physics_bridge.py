# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — PyO3 Physics Bridge Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Tests for PyO3 bindings: fusion-physics crate → Python.

Covers: HallMHD, FnoController, DriftWavePhysics, DesignScanner.
"""

import numpy as np
import pytest

try:
    import scpn_fusion_rs

    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="scpn_fusion_rs not compiled")


# ── WP-PY1: Hall-MHD ────────────────────────────────────────────────


class TestPyHallMHD:
    """Tests for PyHallMHD binding (fusion-physics/hall_mhd.rs)."""

    def test_construction_default(self):
        mhd = scpn_fusion_rs.PyHallMHD()
        assert mhd.grid_size == 64

    def test_construction_custom(self):
        mhd = scpn_fusion_rs.PyHallMHD(32)
        assert mhd.grid_size == 32

    def test_step_returns_tuple(self):
        mhd = scpn_fusion_rs.PyHallMHD(32)
        result = mhd.step()
        assert isinstance(result, tuple) and len(result) == 2
        e_total, e_zonal = result
        assert isinstance(e_total, float) and e_total >= 0
        assert isinstance(e_zonal, float) and e_zonal >= 0

    def test_run_100_steps(self):
        mhd = scpn_fusion_rs.PyHallMHD(32)
        results = mhd.run(100)
        assert len(results) == 100
        for e_total, e_zonal in results:
            assert np.isfinite(e_total) and np.isfinite(e_zonal)

    def test_energy_history_grows(self):
        mhd = scpn_fusion_rs.PyHallMHD(32)
        mhd.run(50)
        hist = mhd.energy_history()
        assert isinstance(hist, np.ndarray)
        assert len(hist) == 50

    def test_zonal_history_grows(self):
        mhd = scpn_fusion_rs.PyHallMHD(32)
        mhd.run(50)
        hist = mhd.zonal_history()
        assert isinstance(hist, np.ndarray)
        assert len(hist) == 50


# ── WP-PY2: FNO Controller ──────────────────────────────────────────


class TestPyFnoController:
    """Tests for PyFnoController binding (fusion-physics/fno.rs)."""

    def test_construction_random(self):
        fno = scpn_fusion_rs.PyFnoController()
        assert fno is not None

    def test_predict_shape(self):
        fno = scpn_fusion_rs.PyFnoController()
        field = np.random.randn(64, 64)
        result = fno.predict(field)
        assert result.shape == (64, 64)
        assert np.all(np.isfinite(result))

    def test_predict_and_suppress(self):
        fno = scpn_fusion_rs.PyFnoController()
        field = np.random.randn(64, 64)
        energy_reduction, suppressed = fno.predict_and_suppress(field)
        assert isinstance(energy_reduction, float)
        assert suppressed.shape == (64, 64)


# ── WP-PY7: Design Scanner ──────────────────────────────────────────


class TestDesignScanner:
    """Tests for design scanner binding (fusion-physics/design_scanner.rs)."""

    def test_evaluate_iter_like(self):
        # ITER: R=6.2m, B=5.3T, Ip=15MA
        result = scpn_fusion_rs.py_evaluate_design(6.2, 5.3, 15.0)
        assert isinstance(result, tuple)
        assert len(result) >= 4
        assert all(np.isfinite(v) for v in result)

    def test_scan_100(self):
        results = scpn_fusion_rs.py_run_design_scan(100)
        assert len(results) == 100
        for row in results:
            assert all(np.isfinite(v) for v in row)

    def test_scan_monotonic_count(self):
        r50 = scpn_fusion_rs.py_run_design_scan(50)
        r100 = scpn_fusion_rs.py_run_design_scan(100)
        assert len(r100) == 100
        assert len(r50) == 50


# ── WP-PY8: Drift-Wave Turbulence ───────────────────────────────────


class TestPyDriftWave:
    """Tests for DriftWavePhysics binding (fusion-physics/turbulence.rs)."""

    def test_construction(self):
        dw = scpn_fusion_rs.PyDriftWave()
        assert dw is not None

    def test_step_returns_array(self):
        dw = scpn_fusion_rs.PyDriftWave(32)
        result = dw.step()
        assert isinstance(result, np.ndarray)
        assert np.all(np.isfinite(result))

    def test_100_steps(self):
        dw = scpn_fusion_rs.PyDriftWave(32)
        for _ in range(100):
            result = dw.step()
            assert np.all(np.isfinite(result))
