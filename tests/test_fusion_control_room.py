# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Fusion Control Room Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Deterministic smoke tests for fusion_control_room runtime entry point."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.fusion_control_room import TokamakPhysicsEngine, run_control_room


class _DummyKernel:
    """Small deterministic kernel stand-in exposing Psi/R/Z fields."""

    def __init__(self, _config_path: str) -> None:
        self.cfg = {"coils": [{"current": 0.0} for _ in range(5)]}
        self.R = np.linspace(1.0, 5.0, 40)
        self.Z = np.linspace(-3.0, 3.0, 40)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((40, 40), dtype=np.float64)
        self._ticks = 0
        self.solve_equilibrium()

    def solve_equilibrium(self) -> None:
        self._ticks += 1
        radial_drive = float(self.cfg["coils"][2]["current"])
        vertical_drive = float(self.cfg["coils"][4]["current"]) - float(
            self.cfg["coils"][0]["current"]
        )
        center_r = 3.0 + 0.2 * np.tanh(radial_drive / 25.0)
        center_z = 0.0 + 0.35 * np.tanh(vertical_drive / 25.0)
        self.Psi = (self.RR - center_r) ** 2 + ((self.ZZ - center_z) / 1.7) ** 2


class _FailingKernel:
    """Kernel stand-in that consistently fails update/solve paths."""

    def __init__(self, _config_path: str) -> None:
        self.cfg = None
        r = np.linspace(1.0, 5.0, 32)
        z = np.linspace(-3.0, 3.0, 32)
        self.Psi = np.zeros((z.size, r.size), dtype=np.float64)

    def solve_equilibrium(self) -> None:
        raise RuntimeError("solver diverged")


class _InitFailKernel:
    def __init__(self, _config_path: str) -> None:
        raise RuntimeError("kernel init failed")


def test_run_control_room_returns_finite_summary_without_outputs() -> None:
    summary = run_control_room(
        sim_duration=18,
        seed=123,
        save_animation=False,
        save_report=False,
        verbose=False,
    )
    for key in (
        "seed",
        "steps",
        "psi_source",
        "kernel_fallback_allowed",
        "kernel_fallback_used",
        "final_z",
        "mean_abs_z",
        "max_abs_z",
        "mean_top_action",
        "mean_bottom_action",
        "kernel_coil_update_failures",
        "kernel_solve_failures",
        "animation_saved",
        "report_saved",
    ):
        assert key in summary
    assert summary["steps"] == 18
    assert summary["psi_source"] == "analytic"
    assert summary["kernel_fallback_allowed"] is True
    assert summary["kernel_fallback_used"] is False
    assert summary["animation_saved"] is False
    assert summary["report_saved"] is False
    assert np.isfinite(summary["final_z"])
    assert np.isfinite(summary["mean_abs_z"])
    assert np.isfinite(summary["max_abs_z"])
    assert summary["kernel_coil_update_failures"] == 0
    assert summary["kernel_coil_update_error"] is None
    assert summary["kernel_solve_failures"] == 0
    assert summary["kernel_solve_error"] is None


def test_run_control_room_is_deterministic_for_fixed_seed() -> None:
    kwargs = dict(
        sim_duration=14,
        seed=77,
        save_animation=False,
        save_report=False,
        verbose=False,
    )
    a = run_control_room(**kwargs)
    b = run_control_room(**kwargs)
    assert a["final_z"] == b["final_z"]
    assert a["mean_abs_z"] == b["mean_abs_z"]
    assert a["max_abs_z"] == b["max_abs_z"]
    assert a["mean_top_action"] == b["mean_top_action"]
    assert a["mean_bottom_action"] == b["mean_bottom_action"]


def test_run_control_room_supports_kernel_backed_psi_source() -> None:
    summary = run_control_room(
        sim_duration=10,
        seed=9,
        save_animation=False,
        save_report=False,
        verbose=False,
        kernel_factory=_DummyKernel,
        config_file="dummy.json",
    )
    assert summary["psi_source"] == "kernel"
    assert summary["kernel_error"] is None


def test_run_control_room_tracks_kernel_failures_without_abort() -> None:
    summary = run_control_room(
        sim_duration=7,
        seed=5,
        save_animation=False,
        save_report=False,
        verbose=False,
        kernel_factory=_FailingKernel,
        config_file="dummy.json",
    )
    assert summary["psi_source"] == "kernel"
    assert summary["kernel_fallback_allowed"] is True
    assert summary["kernel_fallback_used"] is True
    assert summary["kernel_coil_update_failures"] == 7
    assert summary["kernel_solve_failures"] == 7
    assert summary["kernel_coil_update_error"] is not None
    assert summary["kernel_solve_error"] == "solver diverged"


def test_run_control_room_rejects_invalid_sim_duration() -> None:
    with pytest.raises(ValueError, match="sim_duration"):
        run_control_room(
            sim_duration=0,
            seed=1,
            save_animation=False,
            save_report=False,
            verbose=False,
        )


def test_tokamak_physics_engine_rejects_invalid_size() -> None:
    with pytest.raises(ValueError, match="size"):
        TokamakPhysicsEngine(size=8, seed=1)


def test_run_control_room_strict_mode_raises_on_kernel_init_failure() -> None:
    with pytest.raises(RuntimeError, match="allow_kernel_fallback=False"):
        run_control_room(
            sim_duration=4,
            seed=1,
            save_animation=False,
            save_report=False,
            verbose=False,
            kernel_factory=_InitFailKernel,
            config_file="dummy.json",
            allow_kernel_fallback=False,
        )


def test_run_control_room_strict_mode_raises_on_kernel_runtime_failure() -> None:
    with pytest.raises(RuntimeError, match="allow_kernel_fallback=False"):
        run_control_room(
            sim_duration=4,
            seed=2,
            save_animation=False,
            save_report=False,
            verbose=False,
            kernel_factory=_FailingKernel,
            config_file="dummy.json",
            allow_kernel_fallback=False,
        )
