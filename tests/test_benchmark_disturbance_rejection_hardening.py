# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Disturbance Benchmark Hardening Tests
"""Hardening regression tests for disturbance rejection benchmark contracts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

# Ensure validation/ and src/ are importable.
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))

import validation.benchmark_disturbance_rejection as mod  # noqa: E402


def test_build_hinf_controller_strict_rejects_lqr_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict mode refuses an infeasible H-infinity synthesis without building the LQR fallback."""
    monkeypatch.setattr(mod, "_hinf_available", True)

    def _raise_are(*_args: object, **_kwargs: object) -> None:
        """Fail the synthesis the way an infeasible Riccati problem would."""
        raise ValueError("unit-test-infeasible")

    class _ShouldNotConstruct:
        """Stand-in whose construction fails the test."""

        def __init__(self, *_args, **_kwargs) -> None:
            """Fail if strict mode ever builds the fallback."""
            raise AssertionError("LQR fallback must not run in strict mode")

    monkeypatch.setattr(mod, "get_radial_robust_controller", _raise_are)
    monkeypatch.setattr(mod, "LQRRobustController", _ShouldNotConstruct)

    with pytest.raises(RuntimeError, match="strict mode"):
        mod._build_hinf_controller(strict_hinf=True)


def test_build_hinf_controller_reports_fallback_metadata(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Non-strict mode falls back to LQR and records the backend and the reason."""
    monkeypatch.setattr(mod, "_hinf_available", True)

    def _raise_are(*_args: object, **_kwargs: object) -> None:
        """Fail the synthesis the way an infeasible Riccati problem would."""
        raise ValueError("unit-test-infeasible")

    class _DummyLQR:
        """Minimal LQR stand-in with the controller interface."""

        def __init__(self, gamma_growth: float = 100.0) -> None:
            """Record the growth rate and report a stable design."""
            self.gamma_growth = float(gamma_growth)
            self.is_stable = True

        def step(self, error: float, dt: float) -> float:
            """Return zero command."""
            return 0.0

        def reset(self) -> None:
            """Nothing to reset."""
            return None

    monkeypatch.setattr(mod, "get_radial_robust_controller", _raise_are)
    monkeypatch.setattr(mod, "LQRRobustController", _DummyLQR)

    with caplog.at_level(logging.INFO, logger=mod.logger.name):
        ctrl, build = mod._build_hinf_controller(strict_hinf=False)

    assert isinstance(ctrl, _DummyLQR)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["    [H-infinity] ARE failed (unit-test-infeasible); using LQR fallback"]
    assert build["backend"] == "lqr_fallback"
    assert build["fallback_used"] is True
    assert "ValueError" in str(build["fallback_reason"])


def test_build_controllers_strict_requires_hinf_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict mode refuses to build controllers when the H-infinity module is absent."""
    monkeypatch.setattr(mod, "_hinf_available", False)
    with pytest.raises(RuntimeError, match="strict mode requires"):
        mod.build_controllers(strict_hinf=True)


def test_generate_json_results_includes_controller_build_metadata() -> None:
    """The JSON results carry each controller's build metadata."""
    metrics = [
        mod.ScenarioMetrics(
            controller="H-infinity",
            scenario="VDE",
            ise=1.0,
            settling_time_s=0.2,
            peak_overshoot=0.1,
            control_effort=2.0,
            wall_clock_s=0.01,
            stable=True,
        )
    ]
    payload = mod.generate_json_results(
        metrics,
        controller_build={
            "H-infinity": {
                "backend": "lqr_fallback",
                "fallback_used": True,
                "fallback_reason": "ValueError: unit-test-infeasible",
            }
        },
        strict_hinf=True,
    )

    assert payload["strict_hinf"] is True
    assert payload["controller_build"]["H-infinity"]["backend"] == "lqr_fallback"
    assert payload["controller_build"]["H-infinity"]["fallback_used"] is True


_SNN_MODULE = "scpn_fusion.control.neuro_cybernetic_controller"


def _reload_with_snn_module(monkeypatch: pytest.MonkeyPatch, replacement: object) -> None:
    """Reload the benchmark with the SNN controller module replaced in ``sys.modules``."""
    import importlib

    monkeypatch.setitem(sys.modules, _SNN_MODULE, replacement)
    try:
        importlib.reload(mod)
    finally:
        monkeypatch.undo()
        importlib.reload(mod)


def test_missing_snn_module_is_skipped_with_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """An absent optional SNN controller is skipped, not an error."""
    with pytest.warns(UserWarning, match="SNN controller not available"):
        _reload_with_snn_module(monkeypatch, None)


def test_broken_snn_module_is_not_reported_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A defect inside the SNN module surfaces instead of becoming an 'unavailable' warning."""
    import types

    class BrokenModule(types.ModuleType):
        """Module whose attributes fail the way a defective import would."""

        def __getattr__(self, name: str) -> object:
            """Fail every public attribute lookup with a non-import error."""
            if name.startswith("__"):
                raise AttributeError(name)
            raise RuntimeError(f"defect while resolving {name}")

    with pytest.raises(RuntimeError, match="defect while resolving SpikingControllerPool"):
        _reload_with_snn_module(monkeypatch, BrokenModule(_SNN_MODULE))


def test_unavailable_controllers_are_logged_as_skipped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Without the optional H-infinity and SNN modules, the build logs both as skipped."""
    monkeypatch.setattr(mod, "_hinf_available", False)
    monkeypatch.setattr(mod, "_snn_available", False)

    with caplog.at_level(logging.INFO, logger=mod.logger.name):
        controllers, _ = mod.build_controllers(strict_hinf=False)

    assert "H-infinity" not in controllers and "SNN" not in controllers
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == [
        "    [H-infinity] SKIPPED (import failed)",
        "    [SNN] SKIPPED (import failed)",
    ]
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert "    [PID] Initialised: Kp=1.5e4, Ki=3.0e3, Kd=1.5e2" in infos


def test_available_controllers_are_logged_as_initialised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A successful H-infinity synthesis and SNN build are logged at INFO with the SNN backend."""

    class _Pool:
        """Spiking-pool stand-in naming its backend."""

        backend = "unit-test-backend"

    class _SNN:
        """SNN wrapper stand-in."""

        def __init__(self, **_kwargs: object) -> None:
            """Expose the pool stand-in."""
            self._pool = _Pool()

    monkeypatch.setattr(mod, "_hinf_available", True)
    monkeypatch.setattr(mod, "_snn_available", True)
    monkeypatch.setattr(mod, "get_radial_robust_controller", lambda **_kwargs: object())
    monkeypatch.setattr(mod, "SNNControllerWrapper", _SNN)

    with caplog.at_level(logging.INFO, logger=mod.logger.name):
        controllers, build = mod.build_controllers(strict_hinf=False)

    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert "    [H-infinity] Riccati synthesis: OK" in infos
    assert "    [SNN] Initialised: 50 neurons, backend=unit-test-backend" in infos
    assert build["SNN"]["backend"] == "unit-test-backend"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_script_entry_configures_stdout_logging_and_writes_its_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Run as a script, the benchmark sets INFO logging on stdout and writes its JSON report."""
    import runpy

    configured: list[dict[str, object]] = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: configured.append(kwargs))
    monkeypatch.setattr(
        sys, "argv", ["benchmark_disturbance_rejection.py", "--output-dir", str(tmp_path)]
    )

    runpy.run_path(
        str(repo_root / "validation" / "benchmark_disturbance_rejection.py"), run_name="__main__"
    )

    assert configured == [{"level": logging.INFO, "format": "%(message)s", "stream": sys.stdout}]
    assert (tmp_path / "disturbance_rejection_benchmark.json").is_file()
