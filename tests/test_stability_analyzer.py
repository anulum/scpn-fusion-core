# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# SCPN Fusion Core — Stability Analyzer Tests
import json
import numpy as np

from scpn_fusion.core.stability_analyzer import StabilityAnalyzer


def _write_config(tmp_path):
    cfg = {
        "reactor_name": "test-stability",
        "grid_resolution": [33, 33],
        "dimensions": {"R_min": 1.0, "R_max": 3.0, "Z_min": -1.0, "Z_max": 1.0},
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "coils": [
            {"r": 1.7, "z": 0.8, "current": 2.0},
            {"r": 1.7, "z": -0.8, "current": 2.0},
        ],
        "solver": {"max_iterations": 50, "convergence_threshold": 1e-4, "relaxation_factor": 0.5},
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return str(p)


class TestStabilityAnalyzer:
    def test_init(self, tmp_path):
        cfg = _write_config(tmp_path)
        sa = StabilityAnalyzer(cfg)
        assert sa.Psi_vac is not None
        assert sa.Psi_vac.shape[0] == 33

    def test_vacuum_field_returns_three_values(self, tmp_path):
        cfg = _write_config(tmp_path)
        sa = StabilityAnalyzer(cfg)
        Bz, Br, n_idx = sa.get_vacuum_field_at(1.7, 0.0)
        assert np.isfinite(Bz)
        assert np.isfinite(Br)
        assert np.isfinite(n_idx)

    def test_calculate_forces(self, tmp_path):
        cfg = _write_config(tmp_path)
        sa = StabilityAnalyzer(cfg)
        Fr, Fz, n_idx = sa.calculate_forces(1.7, 0.0, 1.0)
        assert np.isfinite(Fr)
        assert np.isfinite(Fz)
        assert np.isfinite(n_idx)

    def test_analyze_stability_runs(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path)
        sa = StabilityAnalyzer(cfg)
        monkeypatch.setattr("matplotlib.pyplot.savefig", lambda *a, **kw: None)
        monkeypatch.setattr(
            "matplotlib.pyplot.subplots",
            lambda **kw: (
                None,
                type(
                    "A",
                    (),
                    {
                        "contour": lambda *a, **kw: type("C", (), {"collections": []})(),
                        "clabel": lambda *a, **kw: None,
                        "set_title": lambda *a: None,
                        "set_xlabel": lambda *a: None,
                        "set_ylabel": lambda *a: None,
                        "axvline": lambda *a, **kw: None,
                        "axhline": lambda *a, **kw: None,
                        "scatter": lambda *a, **kw: None,
                    },
                )(),
            ),
        )
        sa.analyze_stability(R_target=1.7, Z_target=0.0)

    def test_analyze_mhd_stability_default_profiles(self, tmp_path):
        cfg = _write_config(tmp_path)
        sa = StabilityAnalyzer(cfg)
        result = sa.analyze_mhd_stability(R0=1.7, a=0.5, B0=2.0, Ip_MA=1.0)
        assert "q_profile" in result
        assert "mercier" in result
        assert "ballooning" in result
        assert hasattr(result["q_profile"], "q")
        assert len(result["q_profile"].q) > 0

    def test_analyze_mhd_stability_with_solver(self, tmp_path):
        cfg = _write_config(tmp_path)
        sa = StabilityAnalyzer(cfg)

        class FakeSolver:
            rho = np.linspace(0, 1, 30)
            ne = 10.0 * (1 - rho**2) ** 0.5
            Ti = 10.0 * (1 - rho**2) ** 1.5
            Te = Ti.copy()

        result = sa.analyze_mhd_stability(
            R0=1.7, a=0.5, B0=2.0, Ip_MA=1.0, transport_solver=FakeSolver()
        )
        assert "q_profile" in result

    def test_field_index_values(self, tmp_path):
        cfg = _write_config(tmp_path)
        sa = StabilityAnalyzer(cfg)
        _, _, n_idx = sa.get_vacuum_field_at(1.7, 0.0)
        # n_idx should be a real number (can be positive or negative)
        assert isinstance(n_idx, (float, np.floating))
