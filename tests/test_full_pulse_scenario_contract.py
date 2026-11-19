# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Outcome Contract of the Full-Pulse Optimal-Control Check
"""The optimal-control check in validation/ fails, errors or passes; it never turns into a skip."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = ROOT / "validation" / "test_full_pulse_scenario.py"


def _scenario_module() -> ModuleType:
    """Load the validation scenario file as a module without collecting it as tests here."""
    spec = importlib.util.spec_from_file_location("full_pulse_scenario_under_test", SCENARIO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_stability_check(module: ModuleType) -> unittest.TestResult:
    """Run only the optimal-control stability test and return its unittest result."""
    result = unittest.TestResult()
    module.TestFullPulseScenario("test_optimal_control_stability").run(result)
    return result


def _controller_returning(error_norm: float) -> type:
    """Build a controller stand-in whose shot reports the given mean position error."""

    class _Controller:
        """Optimal-controller stand-in with a fixed shot summary."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            """Accept the configuration path and options."""

        def run_optimal_shot(self, **_kwargs: Any) -> dict[str, float]:
            """Return the fixed summary."""
            return {"mean_error_norm": error_norm}

    return _Controller


def test_unexpected_solver_defect_is_an_error_not_a_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    """A solver or configuration exception surfaces as an error with its traceback."""
    module = _scenario_module()

    class _BrokenController:
        """Controller stand-in whose construction fails like a solver defect."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            """Fail the way a defective configuration would."""
            raise ValueError("coil table has no PF coils")

    monkeypatch.setattr(module, "OptimalController", _BrokenController)

    result = _run_stability_check(module)

    assert result.skipped == []
    assert result.failures == []
    assert len(result.errors) == 1
    assert "ValueError: coil table has no PF coils" in result.errors[0][1]


def test_tracking_error_above_the_bound_is_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mean position error at or above 1 m fails the assertion; it is not skipped."""
    module = _scenario_module()
    monkeypatch.setattr(module, "OptimalController", _controller_returning(5.4417))

    result = _run_stability_check(module)

    assert result.skipped == [] and result.errors == []
    assert len(result.failures) == 1
    assert "Controller tracking error too high" in result.failures[0][1]


def test_tracking_error_below_the_bound_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Below 1 m the check passes with no skip, error or failure."""
    module = _scenario_module()
    monkeypatch.setattr(module, "OptimalController", _controller_returning(0.25))

    result = _run_stability_check(module)

    assert result.wasSuccessful()
    assert result.skipped == [] and result.testsRun == 1
