# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the PF-coil force-balance solver contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from scpn_fusion.core import force_balance


class _DummyKernel:
    """Minimal kernel config and vacuum-field recompute surface."""

    def __init__(self) -> None:
        """Create a finite five-coil equilibrium configuration."""
        self.cfg: dict[str, Any] = {
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
        """Return a deterministic marker for vacuum-field recomputes."""
        self.vacuum_recompute_count += 1
        return float(self.vacuum_recompute_count)


class _DummyAnalyzer:
    """Linear force model used by convergence tests."""

    def __init__(self, _config_path: str) -> None:
        """Create a kernel and placeholder vacuum field."""
        self.kernel = _DummyKernel()
        self.Psi_vac = 0.0

    def calculate_forces(
        self, _target_r: float, _target_z: float, _ip: float
    ) -> tuple[float, float, int]:
        """Return radial force from a linear current residual."""
        coils = cast(list[dict[str, Any]], self.kernel.cfg["coils"])
        i_pf3 = float(coils[2]["current"])
        fr = (i_pf3 - 1.0) * 1.0e6
        return fr, 0.0, 0


class _SingularJacobianAnalyzer:
    """Force model whose radial force is independent of PF current."""

    def __init__(self, _config_path: str) -> None:
        """Create a kernel and placeholder vacuum field."""
        self.kernel = _DummyKernel()
        self.Psi_vac = 0.0

    def calculate_forces(
        self, _target_r: float, _target_z: float, _ip: float
    ) -> tuple[float, float, int]:
        """Return constant radial force to exercise singular-Jacobian handling."""
        return 1.0e6, 0.0, 0


def test_solve_for_equilibrium_converges_with_dummy_analyzer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Newton updates converge on a finite linear force model."""
    monkeypatch.setattr(force_balance, "StabilityAnalyzer", _DummyAnalyzer)
    solver = force_balance.ForceBalanceSolver("dummy.json")

    saved: dict[str, bool] = {"called": False}

    def _mark_saved() -> None:
        saved["called"] = True

    monkeypatch.setattr(solver, "save_config", _mark_saved)
    summary = solver.solve_for_equilibrium(target_R=6.2, target_Z=0.0)

    i_pf3 = float(solver.analyzer.kernel.cfg["coils"][2]["current"])
    i_pf4 = float(solver.analyzer.kernel.cfg["coils"][3]["current"])
    assert i_pf3 == pytest.approx(1.0, abs=1e-6)
    assert i_pf4 == pytest.approx(1.0, abs=1e-6)
    assert saved["called"] is True
    assert solver.analyzer.kernel.vacuum_recompute_count > 0
    assert summary["converged"] is True
    assert float(summary["final_radial_force_n"]) == pytest.approx(0.0, abs=1e-6)


def test_solve_for_equilibrium_handles_singular_jacobian(monkeypatch: pytest.MonkeyPatch) -> None:
    """Near-singular Jacobians use the conservative directional fallback."""
    monkeypatch.setattr(force_balance, "StabilityAnalyzer", _SingularJacobianAnalyzer)
    solver = force_balance.ForceBalanceSolver("dummy.json")

    saved: dict[str, bool] = {"called": False}

    def _mark_saved() -> None:
        saved["called"] = True

    monkeypatch.setattr(solver, "save_config", _mark_saved)
    summary = solver.solve_for_equilibrium(max_iterations=2)
    assert saved["called"] is True
    i_pf3 = float(solver.analyzer.kernel.cfg["coils"][2]["current"])
    assert i_pf3 < 0.0
    assert summary["converged"] is False


def test_save_config_writes_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Saving writes the current balanced kernel configuration as JSON."""
    monkeypatch.setattr(force_balance, "StabilityAnalyzer", _DummyAnalyzer)
    solver = force_balance.ForceBalanceSolver("dummy.json")
    out_path = tmp_path / "force_balanced.json"
    solver.save_config(output_path=out_path)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "physics" in payload
    assert "coils" in payload


def test_solve_for_equilibrium_rejects_invalid_runtime_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime argument validation rejects invalid iteration and target settings."""
    monkeypatch.setattr(force_balance, "StabilityAnalyzer", _DummyAnalyzer)
    solver = force_balance.ForceBalanceSolver("dummy.json")

    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        solver.solve_for_equilibrium(max_iterations=0)

    with pytest.raises(ValueError, match="jacobian_floor must be finite and > 0"):
        solver.solve_for_equilibrium(jacobian_floor=0.0)

    with pytest.raises(ValueError, match="target_R and target_Z"):
        solver.solve_for_equilibrium(target_R=float("nan"))


def test_solve_for_equilibrium_rejects_missing_plasma_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config validation requires the plasma-current target."""
    monkeypatch.setattr(force_balance, "StabilityAnalyzer", _DummyAnalyzer)
    solver = force_balance.ForceBalanceSolver("dummy.json")
    solver.analyzer.kernel.cfg["physics"] = {}

    with pytest.raises(ValueError, match="plasma_current_target"):
        solver.solve_for_equilibrium()


def test_solve_for_equilibrium_rejects_incomplete_coil_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config validation requires PF3/PF4 control coil entries."""
    monkeypatch.setattr(force_balance, "StabilityAnalyzer", _DummyAnalyzer)
    solver = force_balance.ForceBalanceSolver("dummy.json")
    solver.analyzer.kernel.cfg["coils"] = [{"current": 0.0}]

    with pytest.raises(ValueError, match="at least 4 coils"):
        solver.solve_for_equilibrium()


def test_solve_for_equilibrium_rejects_nonfinite_control_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config validation rejects non-finite PF control currents."""
    monkeypatch.setattr(force_balance, "StabilityAnalyzer", _DummyAnalyzer)
    solver = force_balance.ForceBalanceSolver("dummy.json")
    coils = cast(list[dict[str, Any]], solver.analyzer.kernel.cfg["coils"])
    coils[2]["current"] = float("nan")

    with pytest.raises(ValueError, match="must be finite"):
        solver.solve_for_equilibrium()
