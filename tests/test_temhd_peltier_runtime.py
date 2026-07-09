# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TEMHD Peltier Runtime Tests
"""Runtime summary tests for TEMHD experiment entrypoint."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scpn_fusion.nuclear.temhd_peltier import TEMHD_Stabilizer, run_temhd_experiment


def test_run_temhd_experiment_returns_finite_summary_without_plot() -> None:
    summary = run_temhd_experiment(
        layer_thickness_mm=5.0,
        B_field=10.0,
        flux_min_MW_m2=0.0,
        flux_max_MW_m2=80.0,
        flux_points=14,
        settle_steps_per_flux=10,
        dt_s=0.25,
        save_plot=False,
        verbose=False,
    )
    for key in (
        "flux_points",
        "min_surface_temp_K",
        "max_surface_temp_K",
        "max_k_eff",
        "plot_saved",
    ):
        assert key in summary
    assert summary["flux_points"] == 14
    assert summary["plot_saved"] is False
    assert summary["plot_error"] is None
    assert np.isfinite(summary["max_surface_temp_K"])
    assert np.isfinite(summary["max_k_eff"])
    assert summary["max_surface_temp_K"] >= summary["min_surface_temp_K"] >= 0.0
    assert summary["max_k_eff"] > 0.0


def test_run_temhd_experiment_is_deterministic_for_fixed_inputs() -> None:
    a = run_temhd_experiment(
        layer_thickness_mm=4.5,
        B_field=9.0,
        flux_min_MW_m2=0.0,
        flux_max_MW_m2=70.0,
        flux_points=12,
        settle_steps_per_flux=8,
        dt_s=0.2,
        save_plot=False,
        verbose=False,
    )
    b = run_temhd_experiment(
        layer_thickness_mm=4.5,
        B_field=9.0,
        flux_min_MW_m2=0.0,
        flux_max_MW_m2=70.0,
        flux_points=12,
        settle_steps_per_flux=8,
        dt_s=0.2,
        save_plot=False,
        verbose=False,
    )
    for key in ("min_surface_temp_K", "max_surface_temp_K", "max_k_eff"):
        assert a[key] == pytest.approx(b[key], rel=0.0, abs=0.0)


def test_run_temhd_experiment_rejects_invalid_flux_range() -> None:
    with pytest.raises(ValueError, match="flux_min_MW_m2"):
        run_temhd_experiment(
            flux_min_MW_m2=10.0,
            flux_max_MW_m2=10.0,
            save_plot=False,
            verbose=False,
        )


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("flux_points", "flux_points"),
        ("settle_steps_per_flux", "settle_steps_per_flux"),
        ("dt_s", "dt_s"),
    ],
)
def test_run_temhd_experiment_rejects_invalid_runtime_parameters(field: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        if field == "flux_points":
            run_temhd_experiment(flux_points=1, save_plot=False, verbose=False)
        elif field == "settle_steps_per_flux":
            run_temhd_experiment(settle_steps_per_flux=0, save_plot=False, verbose=False)
        else:
            run_temhd_experiment(dt_s=0.0, save_plot=False, verbose=False)


def test_run_temhd_experiment_verbose_path_logs_progress(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verbose TEMHD flux ramps route progress through structured logging."""
    caplog.set_level(logging.INFO, logger="scpn_fusion.nuclear.temhd_peltier")

    summary = run_temhd_experiment(
        flux_min_MW_m2=0.0,
        flux_max_MW_m2=20.0,
        flux_points=4,
        settle_steps_per_flux=2,
        save_plot=False,
        verbose=True,
    )

    assert summary["flux_points"] == 4
    assert "Flux" in caplog.text
    assert "T_surf" in caplog.text


def test_temhd_tridiagonal_solver_covers_validation_branches() -> None:
    """The Thomas solver rejects malformed and singular systems."""
    sim = TEMHD_Stabilizer()
    with pytest.raises(ValueError, match="b length"):
        sim.solve_tridiagonal([], [1.0, 2.0], [], [1.0])
    assert sim.solve_tridiagonal([], [], [], []).size == 0
    with pytest.raises(ValueError, match="Invalid tridiagonal sizes"):
        sim.solve_tridiagonal([], [1.0, 2.0], [1.0], [1.0, 2.0])
    np.testing.assert_allclose(sim.solve_tridiagonal([], [2.0], [], [4.0]), np.array([2.0]))
    with pytest.raises(ValueError, match="Singular diagonal"):
        sim.solve_tridiagonal([], [0.0], [], [1.0])
    with pytest.raises(ValueError, match="Singular diagonal"):
        sim.solve_tridiagonal([1.0], [0.0, 1.0], [1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="Singular diagonal"):
        sim.solve_tridiagonal([1.0], [1.0, 1.0], [1.0], [1.0, 2.0])


def test_temhd_step_rejects_invalid_state() -> None:
    """The implicit step validates time step, heat flux, grid, and state."""
    sim = TEMHD_Stabilizer()
    with pytest.raises(ValueError, match="dt"):
        sim.step(10.0, dt=0.0)
    with pytest.raises(ValueError, match="heat_flux"):
        sim.step(-1.0, dt=0.1)

    sim_bad_grid = TEMHD_Stabilizer()
    sim_bad_grid.dz = 0.0
    with pytest.raises(ValueError, match="grid spacing"):
        sim_bad_grid.step(10.0, dt=0.1)

    sim_bad_temperature = TEMHD_Stabilizer()
    sim_bad_temperature.T[0] = np.nan
    with pytest.raises(ValueError, match="Temperature state"):
        sim_bad_temperature.step(10.0, dt=0.1)

    sim_bad_coefficients = TEMHD_Stabilizer()
    sim_bad_coefficients.k_thermal = float("inf")
    with pytest.raises(ValueError, match="Non-finite diffusion"):
        sim_bad_coefficients.step(10.0, dt=0.1)


def test_run_temhd_experiment_saves_plot_and_logs_path(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Saving a TEMHD plot updates the summary and logs the artifact path."""
    output_path = tmp_path / "result.png"
    caplog.set_level(logging.INFO, logger="scpn_fusion.nuclear.temhd_peltier")
    summary = run_temhd_experiment(
        flux_points=3,
        settle_steps_per_flux=2,
        output_path=str(output_path),
        save_plot=True,
        verbose=True,
    )
    assert summary["plot_saved"] is True
    assert output_path.exists()
    assert str(output_path) in caplog.text


def test_run_temhd_experiment_records_plot_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plot failures are recorded in the TEMHD summary."""

    def raise_subplots() -> tuple[object, object]:
        raise RuntimeError("forced plot failure")

    monkeypatch.setattr(plt, "subplots", raise_subplots)
    summary = run_temhd_experiment(flux_points=3, save_plot=True, verbose=False)
    assert summary["plot_saved"] is False
    assert summary["plot_error"] == "RuntimeError: forced plot failure"
