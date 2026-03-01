from __future__ import annotations

import json

import pytest

from scpn_fusion.core import force_balance


class _DummyKernel:
    def __init__(self) -> None:
        self.cfg = {
            "physics": {"plasma_current_target": 8.7},
            "coils": [
                {"current": 0.0},
                {"current": 0.0},
                {"current": 0.0},
                {"current": 0.0},
                {"current": 0.0},
            ],
        }
        self.vacuum_recompute_count = 0

    def calculate_vacuum_field(self) -> float:
        self.vacuum_recompute_count += 1
        return float(self.vacuum_recompute_count)


class _DummyAnalyzer:
    def __init__(self, _config_path: str) -> None:
        self.kernel = _DummyKernel()
        self.Psi_vac = 0.0

    def calculate_forces(self, _target_r: float, _target_z: float, _ip: float) -> tuple[float, float, int]:
        # Linear force model around I=1.0 MA so Newton update converges quickly.
        i_pf3 = float(self.kernel.cfg["coils"][2]["current"])
        fr = (i_pf3 - 1.0) * 1.0e6
        return fr, 0.0, 0


class _SingularJacobianAnalyzer:
    def __init__(self, _config_path: str) -> None:
        self.kernel = _DummyKernel()
        self.Psi_vac = 0.0

    def calculate_forces(self, _target_r: float, _target_z: float, _ip: float) -> tuple[float, float, int]:
        # Force independent of current -> numerical Jacobian exactly zero.
        return 1.0e6, 0.0, 0


def test_solve_for_equilibrium_converges_with_dummy_analyzer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(force_balance, "StabilityAnalyzer", _DummyAnalyzer)
    solver = force_balance.ForceBalanceSolver("dummy.json")

    saved: dict[str, bool] = {"called": False}

    def _mark_saved() -> None:
        saved["called"] = True

    monkeypatch.setattr(solver, "save_config", _mark_saved)
    solver.solve_for_equilibrium(target_R=6.2, target_Z=0.0)

    i_pf3 = float(solver.analyzer.kernel.cfg["coils"][2]["current"])
    i_pf4 = float(solver.analyzer.kernel.cfg["coils"][3]["current"])
    assert i_pf3 == pytest.approx(1.0, abs=1e-6)
    assert i_pf4 == pytest.approx(1.0, abs=1e-6)
    assert saved["called"] is True
    assert solver.analyzer.kernel.vacuum_recompute_count > 0


def test_solve_for_equilibrium_handles_singular_jacobian(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(force_balance, "StabilityAnalyzer", _SingularJacobianAnalyzer)
    solver = force_balance.ForceBalanceSolver("dummy.json")

    saved: dict[str, bool] = {"called": False}

    def _mark_saved() -> None:
        saved["called"] = True

    monkeypatch.setattr(solver, "save_config", _mark_saved)
    solver.solve_for_equilibrium(max_iterations=2)
    assert saved["called"] is True
    i_pf3 = float(solver.analyzer.kernel.cfg["coils"][2]["current"])
    assert i_pf3 < 0.0


def test_save_config_writes_json(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(force_balance, "StabilityAnalyzer", _DummyAnalyzer)
    solver = force_balance.ForceBalanceSolver("dummy.json")
    out_path = tmp_path / "force_balanced.json"
    solver.save_config(output_path=out_path)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "physics" in payload
    assert "coils" in payload
