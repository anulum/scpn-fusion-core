# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Director Interface Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Deterministic tests for DirectorInterface fallback runtime."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

from scpn_fusion.control import director_interface as director_mod
from scpn_fusion.control.director_interface import DirectorInterface


class _DummyBrain:
    def step(self, error: float) -> float:
        return 0.5 * float(error)


class _DummyKernel:
    def __init__(self) -> None:
        self.cfg = {
            "physics": {"plasma_current_target": 5.0},
            "coils": [{"current": 0.0} for _ in range(5)],
        }
        self.R = np.linspace(5.8, 6.4, 13)
        self.Z = np.linspace(-0.3, 0.3, 13)
        self.Psi = np.zeros((len(self.Z), len(self.R)), dtype=np.float64)
        self.solve_equilibrium()

    def solve_equilibrium(self) -> None:
        radial_drive = float(self.cfg["coils"][2]["current"])
        vertical_drive = float(self.cfg["coils"][4]["current"]) - float(
            self.cfg["coils"][0]["current"]
        )
        center_r = 6.2 + 0.03 * np.tanh(radial_drive / 5.0)
        center_z = 0.0 + 0.02 * np.tanh(vertical_drive / 5.0)
        ir = int(np.argmin(np.abs(self.R - center_r)))
        iz = int(np.argmin(np.abs(self.Z - center_z)))
        self.Psi.fill(-1.0)
        self.Psi[iz, ir] = 1.0


class _DummyNeuroController:
    def __init__(self, _config_path: str) -> None:
        self.kernel = _DummyKernel()
        self.brain_R: Optional[_DummyBrain] = None
        self.brain_Z: Optional[_DummyBrain] = None

    def initialize_brains(self, use_quantum: bool = False) -> None:
        del use_quantum
        self.brain_R = _DummyBrain()
        self.brain_Z = _DummyBrain()


def test_director_interface_uses_fallback_backend_when_director_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(director_mod, "DIRECTOR_AVAILABLE", False)
    monkeypatch.setattr(director_mod, "DirectorModule", None)
    di = DirectorInterface(
        "dummy.json",
        allow_fallback=True,
        controller_factory=_DummyNeuroController,
    )
    assert di.director_backend == "fallback_rule_based"
    summary = di.run_directed_mission(
        duration=12,
        glitch_start_step=4,
        glitch_std=0.0,
        rng_seed=11,
        save_plot=False,
        verbose=False,
    )
    assert summary["backend"] == "fallback_rule_based"
    assert summary["steps"] == 12
    assert summary["plot_saved"] is False
    assert summary["plot_error"] is None
    assert np.isfinite(summary["final_target_ip"])
    assert np.isfinite(summary["mean_abs_err_r"])
    assert int(summary["intervention_count"]) >= 0


def test_directed_mission_is_deterministic_with_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(director_mod, "DIRECTOR_AVAILABLE", False)
    monkeypatch.setattr(director_mod, "DirectorModule", None)
    kwargs = dict(
        duration=14,
        glitch_start_step=5,
        glitch_std=0.1,
        rng_seed=7,
        save_plot=False,
        verbose=False,
    )
    a = DirectorInterface(
        "dummy.json",
        allow_fallback=True,
        controller_factory=_DummyNeuroController,
    ).run_directed_mission(**kwargs)
    b = DirectorInterface(
        "dummy.json",
        allow_fallback=True,
        controller_factory=_DummyNeuroController,
    ).run_directed_mission(**kwargs)
    assert a["final_target_ip"] == b["final_target_ip"]
    assert a["mean_abs_err_r"] == b["mean_abs_err_r"]
    assert a["intervention_count"] == b["intervention_count"]


def test_director_interface_can_disable_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(director_mod, "DIRECTOR_AVAILABLE", False)
    monkeypatch.setattr(director_mod, "DirectorModule", None)
    with pytest.raises(ImportError, match="allow_fallback=False"):
        DirectorInterface(
            "dummy.json",
            allow_fallback=False,
            controller_factory=_DummyNeuroController,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"duration": 0}, "duration"),
        ({"glitch_start_step": -1}, "glitch_start_step"),
        ({"glitch_std": -0.1}, "glitch_std"),
        ({"glitch_std": float("nan")}, "glitch_std"),
    ],
)
def test_directed_mission_rejects_invalid_runtime_inputs(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, float | int],
    match: str,
) -> None:
    monkeypatch.setattr(director_mod, "DIRECTOR_AVAILABLE", False)
    monkeypatch.setattr(director_mod, "DirectorModule", None)
    di = DirectorInterface(
        "dummy.json",
        allow_fallback=True,
        controller_factory=_DummyNeuroController,
    )
    with pytest.raises(ValueError, match=match):
        di.run_directed_mission(
            rng_seed=7,
            save_plot=False,
            verbose=False,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"entropy_threshold": 0.0}, "entropy_threshold"),
        ({"entropy_threshold": float("nan")}, "entropy_threshold"),
        ({"history_window": 0}, "history_window"),
    ],
)
def test_director_fallback_rejects_invalid_constructor_inputs(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, float | int],
    match: str,
) -> None:
    monkeypatch.setattr(director_mod, "DIRECTOR_AVAILABLE", False)
    monkeypatch.setattr(director_mod, "DirectorModule", None)
    with pytest.raises(ValueError, match=match):
        DirectorInterface(
            "dummy.json",
            allow_fallback=True,
            controller_factory=_DummyNeuroController,
            **kwargs,
        )
