# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Director Interface Tests
"""Deterministic tests for DirectorInterface fallback runtime."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.control import director_interface as director_mod
from scpn_fusion.control.director_interface import DirectorInterface, _RuleBasedDirector

FloatArray = NDArray[np.float64]


class _DummyBrain:
    """Small deterministic brain stand-in for controller-loop tests."""

    def step(self, error: float) -> float:
        """Return a bounded proportional correction."""
        return 0.5 * float(error)


class _DummyKernel:
    """Small finite equilibrium kernel with movable flux peak."""

    def __init__(self) -> None:
        self.cfg: dict[str, Any] = {
            "physics": {"plasma_current_target": 5.0},
            "coils": [{"current": 0.0} for _ in range(5)],
        }
        self.R: FloatArray = np.linspace(5.8, 6.4, 13)
        self.Z: FloatArray = np.linspace(-0.3, 0.3, 13)
        self.Psi: FloatArray = np.zeros((len(self.Z), len(self.R)), dtype=np.float64)
        self.solve_equilibrium()

    def solve_equilibrium(self) -> None:
        """Move the peak location according to deterministic coil currents."""
        coils = cast(list[dict[str, Any]], self.cfg["coils"])
        radial_drive = float(coils[2]["current"])
        vertical_drive = float(coils[4]["current"]) - float(coils[0]["current"])
        center_r = 6.2 + 0.03 * np.tanh(radial_drive / 5.0)
        center_z = 0.0 + 0.02 * np.tanh(vertical_drive / 5.0)
        ir = int(np.argmin(np.abs(self.R - center_r)))
        iz = int(np.argmin(np.abs(self.Z - center_z)))
        self.Psi.fill(-1.0)
        self.Psi[iz, ir] = 1.0


class _DummyNeuroController:
    """Controller stand-in exposing the kernel and brain attributes."""

    def __init__(self, _config_path: str) -> None:
        self.kernel = _DummyKernel()
        self.brain_R: Optional[_DummyBrain] = None
        self.brain_Z: Optional[_DummyBrain] = None

    def initialize_brains(self, use_stochastic_entropy: bool = False) -> None:
        """Install deterministic brain stubs for both control axes."""
        del use_stochastic_entropy
        self.brain_R = _DummyBrain()
        self.brain_Z = _DummyBrain()


class _AlwaysApproveDirector:
    """Director stand-in that approves every proposed action."""

    def review_action(self, prompt: str, proposed_action: str) -> tuple[bool, float]:
        """Approve the action while asserting real prompt/action text was built."""
        assert "Time=" in prompt
        assert proposed_action.startswith("Increase Ip")
        return True, 0.1


class _AlwaysRejectDirector:
    """Director stand-in that rejects every proposed action."""

    def review_action(self, prompt: str, proposed_action: str) -> tuple[bool, float]:
        """Reject the action while asserting real prompt/action text was built."""
        assert "Time=" in prompt
        assert proposed_action.startswith("Increase Ip")
        return False, 2.0


class _ExternalDirectorModule:
    """External DirectorModule stand-in used by the optional import path."""

    init_args: tuple[float, int] | None = None

    def __init__(self, entropy_threshold: float, history_window: int) -> None:
        _ExternalDirectorModule.init_args = (float(entropy_threshold), int(history_window))

    def review_action(self, prompt: str, proposed_action: str) -> tuple[bool, float]:
        """Approve a formatted action using the external-module contract."""
        assert prompt
        assert proposed_action
        return True, 0.2


def test_director_interface_uses_fallback_backend_when_director_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback backend runs a deterministic mission when DirectorModule is absent."""
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
    """Mission summaries are reproducible when the same RNG seed is supplied."""
    monkeypatch.setattr(director_mod, "DIRECTOR_AVAILABLE", False)
    monkeypatch.setattr(director_mod, "DirectorModule", None)
    a = DirectorInterface(
        "dummy.json",
        allow_fallback=True,
        controller_factory=_DummyNeuroController,
    ).run_directed_mission(
        duration=14,
        glitch_start_step=5,
        glitch_std=0.1,
        rng_seed=7,
        save_plot=False,
        verbose=False,
    )
    b = DirectorInterface(
        "dummy.json",
        allow_fallback=True,
        controller_factory=_DummyNeuroController,
    ).run_directed_mission(
        duration=14,
        glitch_start_step=5,
        glitch_std=0.1,
        rng_seed=7,
        save_plot=False,
        verbose=False,
    )
    assert a["final_target_ip"] == b["final_target_ip"]
    assert a["mean_abs_err_r"] == b["mean_abs_err_r"]
    assert a["intervention_count"] == b["intervention_count"]


def test_director_interface_can_disable_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback-disabled construction fails when no external Director is available."""
    monkeypatch.setattr(director_mod, "DIRECTOR_AVAILABLE", False)
    monkeypatch.setattr(director_mod, "DirectorModule", None)
    with pytest.raises(ImportError, match="allow_fallback=False"):
        DirectorInterface(
            "dummy.json",
            allow_fallback=False,
            controller_factory=_DummyNeuroController,
        )


@pytest.mark.parametrize(
    ("duration", "glitch_start_step", "glitch_std", "match"),
    [
        (0, 50, 500.0, "duration"),
        (100, -1, 500.0, "glitch_start_step"),
        (100, 50, -0.1, "glitch_std"),
        (100, 50, float("nan"), "glitch_std"),
    ],
)
def test_directed_mission_rejects_invalid_runtime_inputs(
    monkeypatch: pytest.MonkeyPatch,
    duration: int,
    glitch_start_step: int,
    glitch_std: float,
    match: str,
) -> None:
    """Runtime envelope validation rejects invalid mission parameters."""
    monkeypatch.setattr(director_mod, "DIRECTOR_AVAILABLE", False)
    monkeypatch.setattr(director_mod, "DirectorModule", None)
    di = DirectorInterface(
        "dummy.json",
        allow_fallback=True,
        controller_factory=_DummyNeuroController,
    )
    with pytest.raises(ValueError, match=match):
        di.run_directed_mission(
            duration=duration,
            glitch_start_step=glitch_start_step,
            glitch_std=glitch_std,
            rng_seed=7,
            save_plot=False,
            verbose=False,
        )


@pytest.mark.parametrize(
    ("entropy_threshold", "history_window", "match"),
    [
        (0.0, 10, "entropy_threshold"),
        (float("nan"), 10, "entropy_threshold"),
        (0.3, 0, "history_window"),
    ],
)
def test_director_fallback_rejects_invalid_constructor_inputs(
    monkeypatch: pytest.MonkeyPatch,
    entropy_threshold: float,
    history_window: int,
    match: str,
) -> None:
    """Fallback Director construction validates entropy and history settings."""
    monkeypatch.setattr(director_mod, "DIRECTOR_AVAILABLE", False)
    monkeypatch.setattr(director_mod, "DirectorModule", None)
    with pytest.raises(ValueError, match=match):
        DirectorInterface(
            "dummy.json",
            allow_fallback=True,
            controller_factory=_DummyNeuroController,
            entropy_threshold=entropy_threshold,
            history_window=history_window,
        )


def test_rule_based_director_rolls_history_window() -> None:
    """Rule-based scoring retains only the configured rolling history window."""
    rb = _RuleBasedDirector(entropy_threshold=0.3, history_window=1)
    assert rb.review_action("Stability=Stable, BrainEntropy=0.10", "Increase Ip")[0] is True
    assert rb.review_action("Stability=Stable, BrainEntropy=0.90", "Increase Ip")[0] is False
    assert len(rb._scores) == 1


def test_optional_director_module_import_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reloading with a fake DirectorModule covers the optional dependency path."""
    fake_module = types.ModuleType("director_module")
    fake_module_any = cast(Any, fake_module)
    fake_module_any.DirectorModule = _ExternalDirectorModule
    monkeypatch.setitem(sys.modules, "director_module", fake_module)
    reloaded = importlib.reload(director_mod)
    try:
        di = reloaded.DirectorInterface(
            "dummy.json",
            allow_fallback=True,
            controller_factory=_DummyNeuroController,
            entropy_threshold=0.4,
            history_window=3,
        )
        assert reloaded.DIRECTOR_AVAILABLE is True
        assert di.director_backend == "director_module"
        assert _ExternalDirectorModule.init_args == (0.4, 3)
    finally:
        monkeypatch.delitem(sys.modules, "director_module", raising=False)
        importlib.reload(director_mod)


def test_injected_director_verbose_mission_saves_plot(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Injected approving Director logs the approved path and saves a plot."""
    output_path = tmp_path / "director.png"
    di = DirectorInterface(
        "dummy.json",
        director=_AlwaysApproveDirector(),
        controller_factory=_DummyNeuroController,
    )
    with caplog.at_level("INFO", logger=director_mod.__name__):
        summary = di.run_directed_mission(
            duration=1,
            glitch_start_step=1,
            glitch_std=0.0,
            rng_seed=9,
            save_plot=True,
            output_path=str(output_path),
            verbose=True,
        )
    assert summary["backend"] == "injected"
    assert summary["plot_saved"] is True
    assert summary["plot_error"] is None
    assert output_path.exists()
    assert "APPROVED" in caplog.text


def test_injected_director_denial_triggers_intervention_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Injected rejecting Director records an intervention and warning log."""
    di = DirectorInterface(
        "dummy.json",
        director=_AlwaysRejectDirector(),
        controller_factory=_DummyNeuroController,
    )
    with caplog.at_level("INFO", logger=director_mod.__name__):
        summary = di.run_directed_mission(
            duration=1,
            glitch_start_step=1,
            glitch_std=0.0,
            rng_seed=9,
            save_plot=False,
            verbose=True,
        )
    assert summary["backend"] == "injected"
    assert summary["intervention_count"] == 1
    assert "DENIED" in caplog.text
    assert "INTERVENTION" in caplog.text


def test_format_state_reports_unstable_critical_and_err_z_guard() -> None:
    """Prompt formatting reports stability classes and validates vertical error."""
    di = DirectorInterface(
        "dummy.json",
        director=_AlwaysApproveDirector(),
        controller_factory=_DummyNeuroController,
    )
    unstable = di.format_state_for_director(1, 5.0, 0.2, 0.0, [0.1, 0.2])
    critical = di.format_state_for_director(1, 5.0, 0.6, 0.0, [0.1, 0.2])
    assert "Stability=Unstable" in unstable
    assert "Stability=Critical" in critical
    with pytest.raises(ValueError, match="err_z must be finite"):
        di.format_state_for_director(1, 5.0, 0.0, float("nan"), [0.1])
