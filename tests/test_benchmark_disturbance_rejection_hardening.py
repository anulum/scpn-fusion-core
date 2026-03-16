# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Disturbance Benchmark Hardening Tests
"""Hardening regression tests for disturbance rejection benchmark contracts."""

from __future__ import annotations

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
    monkeypatch.setattr(mod, "_hinf_available", True)

    def _raise_are(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("unit-test-infeasible")

    class _ShouldNotConstruct:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("LQR fallback must not run in strict mode")

    monkeypatch.setattr(mod, "get_radial_robust_controller", _raise_are)
    monkeypatch.setattr(mod, "LQRRobustController", _ShouldNotConstruct)

    with pytest.raises(RuntimeError, match="strict mode"):
        mod._build_hinf_controller(strict_hinf=True)


def test_build_hinf_controller_reports_fallback_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "_hinf_available", True)

    def _raise_are(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("unit-test-infeasible")

    class _DummyLQR:
        def __init__(self, gamma_growth: float = 100.0) -> None:
            self.gamma_growth = float(gamma_growth)
            self.is_stable = True

        def step(self, error: float, dt: float) -> float:
            return 0.0

        def reset(self) -> None:
            return None

    monkeypatch.setattr(mod, "get_radial_robust_controller", _raise_are)
    monkeypatch.setattr(mod, "LQRRobustController", _DummyLQR)

    ctrl, build = mod._build_hinf_controller(strict_hinf=False)

    assert isinstance(ctrl, _DummyLQR)
    assert build["backend"] == "lqr_fallback"
    assert build["fallback_used"] is True
    assert "ValueError" in str(build["fallback_reason"])


def test_build_controllers_strict_requires_hinf_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "_hinf_available", False)
    with pytest.raises(RuntimeError, match="strict mode requires"):
        mod.build_controllers(strict_hinf=True)


def test_generate_json_results_includes_controller_build_metadata() -> None:
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
