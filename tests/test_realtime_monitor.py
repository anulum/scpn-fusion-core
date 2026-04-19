# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Realtime Monitor Tests
"""
Tests for RealtimeMonitor, TrajectoryRecorder, and NPZ/HDF5 export.
test_phase_kuramoto.py covers basic from_paper27 and tick smoke tests.
This file covers from_plasma, adaptive engine wiring, recorder mechanics,
save_npz, save_hdf5 with mock, reset edge cases, and recorder properties.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from scpn_fusion.phase.adaptive_knm import (
    AdaptiveKnmEngine,
)
from scpn_fusion.phase.knm import build_knm_paper27
from scpn_fusion.phase.realtime_monitor import RealtimeMonitor, TrajectoryRecorder


# ── TrajectoryRecorder ───────────────────────────────────────────────


class TestTrajectoryRecorder:
    def test_initially_empty(self):
        rec = TrajectoryRecorder()
        assert rec.n_ticks == 0
        assert len(rec.R_global) == 0

    def test_record_increments(self):
        rec = TrajectoryRecorder()
        snap = {
            "R_global": 0.5,
            "R_layer": [0.3, 0.7],
            "V_global": 0.8,
            "V_layer": [0.9, 0.7],
            "lambda_exp": -0.1,
            "guard_approved": True,
            "latency_us": 12.5,
            "Psi_global": 0.3,
        }
        rec.record(snap)
        assert rec.n_ticks == 1
        assert rec.R_global[0] == 0.5
        assert rec.Psi_global[0] == 0.3

    def test_clear(self):
        rec = TrajectoryRecorder()
        snap = {
            "R_global": 0.5,
            "R_layer": [0.3],
            "V_global": 0.8,
            "V_layer": [0.9],
            "lambda_exp": -0.1,
            "guard_approved": True,
            "latency_us": 12.5,
            "Psi_global": 0.3,
        }
        rec.record(snap)
        rec.record(snap)
        assert rec.n_ticks == 2
        rec.clear()
        assert rec.n_ticks == 0
        assert len(rec.R_global) == 0
        assert len(rec.guard_approved) == 0

    def test_multiple_records(self):
        rec = TrajectoryRecorder()
        for i in range(10):
            rec.record(
                {
                    "R_global": float(i) / 10,
                    "R_layer": [0.5],
                    "V_global": 1.0,
                    "V_layer": [1.0],
                    "lambda_exp": 0.0,
                    "guard_approved": True,
                    "latency_us": 10.0,
                    "Psi_global": 0.0,
                }
            )
        assert rec.n_ticks == 10
        assert rec.R_global[-1] == pytest.approx(0.9)


# ── RealtimeMonitor.from_paper27 ────────────────────────────────────


class TestFromPaper27:
    def test_default_params(self):
        mon = RealtimeMonitor.from_paper27()
        assert len(mon.theta_layers) == 16
        assert mon.theta_layers[0].shape == (50,)

    def test_custom_params(self):
        mon = RealtimeMonitor.from_paper27(L=4, N_per=20, dt=0.01, seed=7)
        assert len(mon.theta_layers) == 4
        assert mon.theta_layers[0].shape == (20,)

    def test_deterministic_seed(self):
        a = RealtimeMonitor.from_paper27(L=3, N_per=10, seed=42)
        b = RealtimeMonitor.from_paper27(L=3, N_per=10, seed=42)
        for i in range(3):
            np.testing.assert_array_equal(a.theta_layers[i], b.theta_layers[i])


# ── RealtimeMonitor.from_plasma ─────────────────────────────────────


class TestFromPlasma:
    def test_default_plasma(self):
        mon = RealtimeMonitor.from_plasma()
        assert len(mon.theta_layers) == 8
        assert mon.theta_layers[0].shape == (50,)

    def test_custom_plasma(self):
        mon = RealtimeMonitor.from_plasma(L=4, N_per=30, mode="elm", seed=11)
        assert len(mon.theta_layers) == 4
        assert mon.theta_layers[0].shape == (30,)

    def test_plasma_modes(self):
        for mode in ("baseline", "elm", "ntm", "sawtooth", "hybrid"):
            mon = RealtimeMonitor.from_plasma(L=8, N_per=10, mode=mode)
            assert len(mon.theta_layers) == 8


# ── tick() ───────────────────────────────────────────────────────────


class TestTick:
    def test_tick_snapshot_keys(self):
        mon = RealtimeMonitor.from_paper27(L=3, N_per=10)
        snap = mon.tick()
        expected_keys = {
            "tick",
            "R_global",
            "R_layer",
            "Psi_global",
            "V_global",
            "V_layer",
            "lambda_exp",
            "guard_approved",
            "guard_score",
            "guard_violations",
            "latency_us",
            "director_ai",
        }
        assert expected_keys.issubset(snap.keys())

    def test_tick_records_by_default(self):
        mon = RealtimeMonitor.from_paper27(L=3, N_per=10)
        mon.tick()
        mon.tick()
        assert mon.recorder.n_ticks == 2

    def test_tick_no_record(self):
        mon = RealtimeMonitor.from_paper27(L=3, N_per=10)
        mon.tick(record=False)
        assert mon.recorder.n_ticks == 0

    def test_tick_count_increments(self):
        mon = RealtimeMonitor.from_paper27(L=3, N_per=10)
        for i in range(5):
            snap = mon.tick()
        assert snap["tick"] == 5

    def test_r_global_bounded(self):
        mon = RealtimeMonitor.from_paper27(L=4, N_per=20)
        for _ in range(10):
            snap = mon.tick()
        assert 0.0 <= snap["R_global"] <= 1.0

    def test_latency_positive(self):
        mon = RealtimeMonitor.from_paper27(L=3, N_per=10)
        snap = mon.tick()
        assert snap["latency_us"] >= 0.0


# ── tick() with adaptive engine ──────────────────────────────────────


class TestTickAdaptive:
    def test_adaptive_engine_wired(self):
        spec = build_knm_paper27(L=4, zeta_uniform=0.5)
        engine = AdaptiveKnmEngine(spec)
        mon = RealtimeMonitor.from_paper27(L=4, N_per=10, zeta_uniform=0.5)
        mon.adaptive_engine = engine
        snap = mon.tick()
        assert "adaptive" in snap
        assert "K_mean" in snap["adaptive"]

    def test_adaptive_multiple_ticks(self):
        spec = build_knm_paper27(L=4, zeta_uniform=0.5)
        engine = AdaptiveKnmEngine(spec)
        mon = RealtimeMonitor.from_paper27(L=4, N_per=10, zeta_uniform=0.5)
        mon.adaptive_engine = engine
        for _ in range(5):
            snap = mon.tick(beta_n=1.5, q95=3.0, disruption_risk=0.1)
        assert snap["tick"] == 5
        assert "adaptive" in snap

    def test_adaptive_with_diagnostics(self):
        spec = build_knm_paper27(L=4, zeta_uniform=0.5)
        engine = AdaptiveKnmEngine(spec)
        mon = RealtimeMonitor.from_paper27(L=4, N_per=10, zeta_uniform=0.5)
        mon.adaptive_engine = engine
        snap = mon.tick(beta_n=2.0, q95=2.5, disruption_risk=0.5, mirnov_rms=0.1)
        assert snap["adaptive"]["K_mean"] > 0


# ── reset() ──────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_tick_count(self):
        mon = RealtimeMonitor.from_paper27(L=3, N_per=10)
        for _ in range(5):
            mon.tick()
        mon.reset(seed=99)
        assert mon._tick_count == 0

    def test_reset_clears_recorder(self):
        mon = RealtimeMonitor.from_paper27(L=3, N_per=10)
        for _ in range(5):
            mon.tick()
        mon.reset()
        assert mon.recorder.n_ticks == 0

    def test_reset_new_seed(self):
        mon = RealtimeMonitor.from_paper27(L=3, N_per=10, seed=42)
        theta_before = [t.copy() for t in mon.theta_layers]
        mon.reset(seed=99)
        differs = any(not np.array_equal(a, b) for a, b in zip(theta_before, mon.theta_layers))
        assert differs

    def test_tick_after_reset(self):
        mon = RealtimeMonitor.from_paper27(L=3, N_per=10)
        mon.tick()
        mon.reset()
        snap = mon.tick()
        assert snap["tick"] == 1


# ── save_npz ─────────────────────────────────────────────────────────


class TestSaveNpz:
    def test_save_and_load(self, tmp_path):
        mon = RealtimeMonitor.from_paper27(L=3, N_per=10, seed=42)
        for _ in range(10):
            mon.tick()
        path = mon.save_npz(tmp_path / "traj.npz")
        assert path.exists()
        data = np.load(path)
        assert data["R_global"].shape == (10,)
        assert data["R_layer"].shape[0] == 10
        assert data["guard_approved"].dtype == bool
        assert data["Psi_global"].shape == (10,)

    def test_empty_recorder_saves(self, tmp_path):
        mon = RealtimeMonitor.from_paper27(L=3, N_per=10)
        path = mon.save_npz(tmp_path / "empty.npz")
        data = np.load(path)
        assert data["R_global"].shape == (0,)


# ── save_hdf5 ────────────────────────────────────────────────────────


class TestSaveHdf5:
    def test_h5py_import_error(self, tmp_path, monkeypatch):
        mon = RealtimeMonitor.from_paper27(L=3, N_per=10)
        mon.tick()
        monkeypatch.setitem(sys.modules, "h5py", None)
        with pytest.raises(ImportError, match="h5py"):
            mon.save_hdf5(tmp_path / "test.h5")

    def test_save_with_mock_h5py(self, tmp_path, monkeypatch):
        mon = RealtimeMonitor.from_paper27(L=3, N_per=10)
        for _ in range(5):
            mon.tick()

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.attrs = {}

        mock_h5py = MagicMock()
        mock_h5py.File.return_value = mock_file

        monkeypatch.setitem(sys.modules, "h5py", mock_h5py)
        path = mon.save_hdf5(tmp_path / "test.h5")
        assert path == tmp_path / "test.h5"
        mock_h5py.File.assert_called_once()
        assert mock_file.create_dataset.call_count == 8  # 8 datasets


# ── recorder property ────────────────────────────────────────────────


class TestRecorderProperty:
    def test_recorder_is_trajectory_recorder(self):
        mon = RealtimeMonitor.from_paper27(L=3, N_per=10)
        assert isinstance(mon.recorder, TrajectoryRecorder)
