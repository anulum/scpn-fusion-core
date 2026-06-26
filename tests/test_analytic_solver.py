# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Analytic Solver Tests
"""Deterministic tests for analytic_solver runtime and solve paths."""

from __future__ import annotations

import json
import logging
from typing import TypedDict
from pathlib import Path
import numpy as np
import pytest

from scpn_fusion.control import analytic_solver as analytic_solver_mod
from scpn_fusion.control.analytic_solver import (
    AnalyticEquilibriumSolver,
    run_analytic_solver,
    shafranov_bv,
    solve_coil_currents,
)


class _CoilConfig(TypedDict):
    name: str
    r: float
    z: float
    current: float


class _SolverConfig(TypedDict):
    coils: list[_CoilConfig]


class _DummyKernel:
    """Minimal kernel exposing config/grid/vacuum field for analytic solver tests."""

    def __init__(self, _config_path: str) -> None:
        self.cfg: _SolverConfig = {
            "coils": [
                {"name": "PF1", "r": 5.9, "z": -0.2, "current": 0.0},
                {"name": "PF2", "r": 6.0, "z": 0.2, "current": 0.0},
                {"name": "PF3", "r": 6.3, "z": -0.2, "current": 0.0},
                {"name": "PF4", "r": 6.4, "z": 0.2, "current": 0.0},
            ]
        }
        self.R = np.linspace(5.6, 6.7, 51)
        self.Z = np.linspace(-0.6, 0.6, 31)
        self.dR = float(self.R[1] - self.R[0])
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)

    def calculate_vacuum_field(self) -> np.ndarray:
        psi = np.zeros_like(self.RR, dtype=np.float64)
        for coil in self.cfg["coils"]:
            cur = float(coil["current"])
            r0 = float(coil["r"])
            z0 = float(coil["z"])
            psi += cur * np.exp(-((self.RR - r0) ** 2 + (self.ZZ - z0) ** 2) / 0.08)
        return psi


def test_calculate_required_bv_returns_finite_expected_sign() -> None:
    """The ITER-like Shafranov estimate is finite and directed downward."""
    solver = AnalyticEquilibriumSolver("dummy.json", kernel_factory=_DummyKernel, verbose=False)
    bv = solver.calculate_required_Bv(6.2, 2.0, 15.0, beta_p=0.5, li=0.8)
    assert np.isfinite(bv)
    assert bv < 0.0


def test_calculate_required_bv_delegates_to_free_function_bit_exact() -> None:
    """The solver method delegates to the free function with no value drift."""
    solver = AnalyticEquilibriumSolver("dummy.json", kernel_factory=_DummyKernel, verbose=False)
    for r_geo, a_min, ip_ma, beta_p, li in [
        (6.2, 2.0, 15.0, 0.5, 0.8),
        (1.7, 0.5, 1.0, 0.3, 1.1),
        (3.0, 1.0, 8.0, 0.9, 0.6),
    ]:
        method_bv = solver.calculate_required_Bv(r_geo, a_min, ip_ma, beta_p=beta_p, li=li)
        assert method_bv == shafranov_bv(r_geo, a_min, ip_ma, beta_p=beta_p, li=li)


def test_shafranov_bv_matches_force_balance_closed_form() -> None:
    """The free function reproduces the Shafranov radial-force-balance expression."""
    r_geo, a_min, ip_ma, beta_p, li = 6.2, 2.0, 15.0, 0.5, 0.8
    mu0 = 4.0 * np.pi * 1e-7
    expected = -((mu0 * ip_ma * 1e6) / (4.0 * np.pi * r_geo)) * (
        np.log(8.0 * r_geo / a_min) + beta_p + li / 2.0 - 1.5
    )
    assert shafranov_bv(r_geo, a_min, ip_ma, beta_p=beta_p, li=li) == pytest.approx(
        expected, rel=1e-15
    )


def test_shafranov_bv_shaping_parameters_increase_field_magnitude() -> None:
    """Larger beta_p or li raises (term_log + term_physics) and |B_v|."""
    base = shafranov_bv(6.2, 2.0, 15.0, beta_p=0.5, li=0.8)
    assert abs(shafranov_bv(6.2, 2.0, 15.0, beta_p=0.9, li=0.8)) > abs(base)
    assert abs(shafranov_bv(6.2, 2.0, 15.0, beta_p=0.5, li=1.2)) > abs(base)


@pytest.mark.parametrize(
    ("r_geo", "a_min", "ip_ma"),
    [
        (0.0, 2.0, 15.0),
        (-1.0, 2.0, 15.0),
        (6.2, 0.0, 15.0),
        (6.2, -2.0, 15.0),
        (6.2, 2.0, 0.0),
        (6.2, 2.0, -15.0),
    ],
)
def test_shafranov_bv_rejects_nonpositive_inputs(r_geo: float, a_min: float, ip_ma: float) -> None:
    """The canonical domain requires r_geo, a_min and ip_ma strictly positive."""
    with pytest.raises(ValueError, match="must be > 0"):
        shafranov_bv(r_geo, a_min, ip_ma)


def test_solve_coil_currents_hits_target_bv_projection() -> None:
    """The solver current vector reproduces the target vertical field."""
    solver = AnalyticEquilibriumSolver("dummy.json", kernel_factory=_DummyKernel, verbose=False)
    target_bv = -0.02
    currents = solver.solve_coil_currents(target_bv, 6.2, target_Z=0.0)
    efficiencies = solver.compute_coil_efficiencies(6.2, target_Z=0.0)
    projected_bv = float(np.dot(efficiencies, currents))
    assert projected_bv == pytest.approx(target_bv, rel=1e-7, abs=1e-9)

    solver.apply_currents(currents)
    applied = np.asarray(
        [float(c["current"]) for c in solver.kernel.cfg["coils"]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(applied, currents, rtol=0.0, atol=0.0)


def test_solve_coil_currents_free_function_minimum_norm() -> None:
    """A uniform Green's vector splits the target field evenly (minimum norm)."""
    np.testing.assert_allclose(
        solve_coil_currents([1.0, 1.0], 1.0), [0.5, 0.5], rtol=0.0, atol=1e-15
    )


def test_solve_coil_currents_free_function_projection_hits_target() -> None:
    """The recovered currents reproduce the target field under G·I."""
    green = np.array([0.01, 0.02, 0.015, 0.005, 0.01], dtype=np.float64)
    currents = solve_coil_currents(green, -0.05)
    assert float(np.dot(green, currents)) == pytest.approx(-0.05, rel=1e-12)


def test_solve_coil_currents_ridge_shrinks_norm_and_clamps_negative() -> None:
    """Positive ridge shrinks the current norm; negative ridge clamps to zero."""
    green = np.array([0.01, 0.02, 0.015, 0.005, 0.01], dtype=np.float64)
    plain = solve_coil_currents(green, -0.05, ridge_lambda=0.0)
    ridged = solve_coil_currents(green, -0.05, ridge_lambda=1e-3)
    assert float(np.linalg.norm(ridged)) < float(np.linalg.norm(plain))
    np.testing.assert_array_equal(solve_coil_currents(green, -0.05, ridge_lambda=-5.0), plain)


def test_solve_coil_currents_method_delegates_to_free_function() -> None:
    """The solver method routes the linear solve through the free function."""
    solver = AnalyticEquilibriumSolver("dummy.json", kernel_factory=_DummyKernel, verbose=False)
    eff = solver.compute_coil_efficiencies(6.2, target_Z=0.0)
    method_currents = solver.solve_coil_currents(-0.02, 6.2, target_Z=0.0)
    np.testing.assert_array_equal(method_currents, solve_coil_currents(eff, -0.02))


class _EmptyCoilKernel(_DummyKernel):
    """Kernel whose configuration declares no coils."""

    def __init__(self, _config_path: str) -> None:
        super().__init__(_config_path)
        self.cfg = {"coils": []}


class _ZeroSpacingKernel(_DummyKernel):
    """Kernel whose grid spacing collapses to zero (degenerate R axis)."""

    def __init__(self, _config_path: str) -> None:
        super().__init__(_config_path)
        self.R = np.full(51, 6.0)  # all-equal axis -> dR == 0
        self.dR = 0.0


def test_log_emits_through_logger_only_when_verbose(caplog) -> None:
    """The solver logs at INFO under verbose=True and stays silent otherwise."""
    verbose = AnalyticEquilibriumSolver("dummy.json", kernel_factory=_DummyKernel, verbose=True)
    with caplog.at_level(logging.INFO, logger="scpn_fusion.control.analytic_solver"):
        verbose.calculate_required_Bv(6.2, 2.0, 15.0)
    assert any("SHAFRANOV EQUILIBRIUM CHECK" in rec.message for rec in caplog.records)

    caplog.clear()
    quiet = AnalyticEquilibriumSolver("dummy.json", kernel_factory=_DummyKernel, verbose=False)
    with caplog.at_level(logging.INFO, logger="scpn_fusion.control.analytic_solver"):
        quiet.calculate_required_Bv(6.2, 2.0, 15.0)
    assert caplog.records == []


def test_compute_coil_efficiencies_rejects_empty_coil_config() -> None:
    """A kernel with no coils cannot form a control-efficiency vector."""
    solver = AnalyticEquilibriumSolver("dummy.json", kernel_factory=_EmptyCoilKernel, verbose=False)
    with pytest.raises(ValueError, match="no coils"):
        solver.compute_coil_efficiencies(6.2)


@pytest.mark.parametrize("target_r", [0.0, -3.0])
def test_compute_coil_efficiencies_rejects_nonpositive_target_r(target_r: float) -> None:
    """The efficiency calculation rejects non-positive target radii."""
    solver = AnalyticEquilibriumSolver("dummy.json", kernel_factory=_DummyKernel, verbose=False)
    with pytest.raises(ValueError, match="target_R must be > 0"):
        solver.compute_coil_efficiencies(target_r)


def test_compute_coil_efficiencies_rejects_nonpositive_grid_spacing() -> None:
    """Degenerate kernel grids fail before finite-difference evaluation."""
    solver = AnalyticEquilibriumSolver(
        "dummy.json", kernel_factory=_ZeroSpacingKernel, verbose=False
    )
    with pytest.raises(ValueError, match="dR must be > 0"):
        solver.compute_coil_efficiencies(6.0)


def test_apply_currents_rejects_length_mismatch() -> None:
    """Current vectors must match the configured coil count exactly."""
    solver = AnalyticEquilibriumSolver("dummy.json", kernel_factory=_DummyKernel, verbose=False)
    with pytest.raises(ValueError, match="length mismatch"):
        solver.apply_currents(np.zeros(3))  # kernel has 4 coils


def test_apply_and_save_writes_kernel_config(tmp_path) -> None:
    """Explicit output paths receive the updated kernel configuration."""
    solver = AnalyticEquilibriumSolver("dummy.json", kernel_factory=_DummyKernel, verbose=False)
    currents = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float64)
    out = tmp_path / "nested" / "iter_analytic_config.json"
    written = solver.apply_and_save(currents, output_path=str(out))

    assert written == str(out)
    assert out.exists()
    saved = json.loads(out.read_text(encoding="utf-8"))
    np.testing.assert_allclose([c["current"] for c in saved["coils"]], currents, rtol=0.0, atol=0.0)


def test_apply_and_save_defaults_to_artifact_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no output_path the config lands under the writable artifact root."""
    solver = AnalyticEquilibriumSolver("dummy.json", kernel_factory=_DummyKernel, verbose=False)
    monkeypatch.setenv("SCPN_ARTIFACT_DIR", str(tmp_path))
    written = solver.apply_and_save(np.zeros(4))
    assert Path(written) == tmp_path / "validation" / "iter_analytic_config.json"
    assert Path(written).exists()


def test_resolve_default_config_prefers_validation_fallback_when_calibration_missing(
    tmp_path,
) -> None:
    """When the preferred config is absent the validation fallback is used + flagged."""
    (tmp_path / "validation").mkdir()
    fallback = tmp_path / "validation" / "iter_validated_config.json"
    fallback.write_text("{}", encoding="utf-8")

    path, source, used = analytic_solver_mod._resolve_default_config_path(tmp_path)
    assert path == str(fallback)
    assert source == "validation_fallback_default"
    assert used is True


def test_resolve_default_config_raises_when_nothing_present(tmp_path) -> None:
    """The default resolver fails closed when neither candidate config exists."""
    with pytest.raises(FileNotFoundError, match="No default analytic config"):
        analytic_solver_mod._resolve_default_config_path(tmp_path)


def test_run_analytic_solver_resolves_default_config_and_saves(tmp_path, monkeypatch) -> None:
    """config_path=None resolves the default config; save_config writes the result."""
    cfg = tmp_path / "iter.json"
    cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        analytic_solver_mod,
        "_resolve_default_config_path",
        lambda repo_root, **_kw: (str(cfg), "preferred_default", False),
    )
    out = tmp_path / "out.json"
    summary = run_analytic_solver(
        config_path=None,
        kernel_factory=_DummyKernel,
        save_config=True,
        output_config_path=str(out),
        verbose=False,
    )
    assert summary["config_source"] == "preferred_default"
    assert summary["output_config_path"] == str(out)
    assert out.exists()


@pytest.mark.parametrize(
    ("green", "target", "ridge"),
    [
        ([], -0.05, 0.0),
        ([0.01, float("nan")], -0.05, 0.0),
        ([0.01, 0.02], float("inf"), 0.0),
        ([0.01, 0.02], -0.05, float("nan")),
        ([0.0, 0.0], -0.05, 0.0),
    ],
)
def test_solve_coil_currents_rejects_invalid_inputs(
    green: list[float], target: float, ridge: float
) -> None:
    """Invalid Green's vectors, targets, and ridge values are rejected.

    Empty/non-finite Green's vectors, non-finite targets/ridge, and a zero-norm
    unregularised solve are all fail-closed branches.
    """
    with pytest.raises(ValueError):
        solve_coil_currents(green, target, ridge_lambda=ridge)


def test_run_analytic_solver_returns_deterministic_summary_without_write() -> None:
    """Repeated no-write analytic solves return identical scalar summaries."""
    kwargs = dict(
        config_path="dummy.json",
        target_r=6.2,
        target_z=0.0,
        a_minor=2.0,
        ip_target_ma=15.0,
        ridge_lambda=0.01,
        save_config=False,
        verbose=False,
        kernel_factory=_DummyKernel,
    )
    a = run_analytic_solver(**kwargs)
    b = run_analytic_solver(**kwargs)
    for key in (
        "config_path",
        "config_source",
        "fallback_used",
        "target_r_m",
        "target_z_m",
        "a_minor_m",
        "ip_target_ma",
        "required_bv_t",
        "coil_current_l2_norm",
        "max_abs_coil_current_ma",
    ):
        assert key in a
    assert a["config_path"] == "dummy.json"
    assert a["config_source"] == "explicit"
    assert a["fallback_used"] is False
    assert a["output_config_path"] is None
    assert set(a["coil_currents_ma"].keys()) == {"PF1", "PF2", "PF3", "PF4"}
    for key in (
        "required_bv_t",
        "coil_current_l2_norm",
        "max_abs_coil_current_ma",
    ):
        assert a[key] == pytest.approx(b[key], rel=0.0, abs=0.0)
    for k, v in a["coil_currents_ma"].items():
        assert float(v) == pytest.approx(float(b["coil_currents_ma"][k]), rel=0.0, abs=0.0)


def test_resolve_default_config_prefers_calibration_config(tmp_path) -> None:
    """A calibration config wins over the validation fallback when present."""
    calibration = tmp_path / "calibration"
    validation = tmp_path / "validation"
    calibration.mkdir(parents=True, exist_ok=True)
    validation.mkdir(parents=True, exist_ok=True)
    (calibration / "iter_genetic_temp.json").write_text("{}", encoding="utf-8")
    (validation / "iter_validated_config.json").write_text("{}", encoding="utf-8")

    path, source, used_fallback = analytic_solver_mod._resolve_default_config_path(tmp_path)
    assert Path(path).as_posix().endswith("calibration/iter_genetic_temp.json")
    assert source == "preferred_default"
    assert used_fallback is False


def test_resolve_default_config_can_disable_validation_fallback(tmp_path) -> None:
    """Callers can disable the validation fallback for stricter deployments."""
    validation = tmp_path / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    (validation / "iter_validated_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="validation fallback is disabled"):
        analytic_solver_mod._resolve_default_config_path(
            tmp_path,
            allow_validation_fallback=False,
        )
