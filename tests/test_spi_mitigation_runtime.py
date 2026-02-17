# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — SPI Mitigation Runtime Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Runtime summary tests for SPI mitigation entrypoint."""

from __future__ import annotations

import pytest

from scpn_fusion.control.spi_mitigation import run_spi_mitigation


def test_run_spi_mitigation_returns_finite_summary_without_plot() -> None:
    summary = run_spi_mitigation(
        plasma_energy_mj=300.0,
        plasma_current_ma=15.0,
        neon_quantity_mol=0.1,
        argon_quantity_mol=0.01,
        xenon_quantity_mol=0.005,
        duration_s=0.05,
        dt_s=1e-5,
        save_plot=False,
        verbose=False,
    )
    for key in (
        "samples",
        "initial_energy_mj",
        "final_energy_mj",
        "initial_current_ma",
        "final_current_ma",
        "z_eff",
        "tau_cq_ms_mean",
        "argon_quantity_mol",
        "xenon_quantity_mol",
        "total_impurity_mol",
        "plot_saved",
    ):
        assert key in summary
    assert summary["samples"] > 100
    assert summary["plot_saved"] is False
    assert summary["plot_error"] is None
    assert summary["final_current_ma"] < summary["initial_current_ma"]
    assert summary["z_eff"] >= 1.0


def test_run_spi_mitigation_is_deterministic_for_fixed_inputs() -> None:
    kwargs = dict(
        plasma_energy_mj=280.0,
        plasma_current_ma=14.0,
        neon_quantity_mol=0.12,
        argon_quantity_mol=0.015,
        xenon_quantity_mol=0.003,
        duration_s=0.03,
        dt_s=1e-5,
        save_plot=False,
        verbose=False,
    )
    a = run_spi_mitigation(**kwargs)
    b = run_spi_mitigation(**kwargs)
    for key in (
        "final_energy_mj",
        "final_current_ma",
        "z_eff",
        "tau_cq_ms_mean",
        "tau_cq_ms_p95",
    ):
        assert a[key] == pytest.approx(b[key], rel=0.0, abs=0.0)


def test_run_spi_mitigation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="Plasma_Energy_MJ"):
        run_spi_mitigation(plasma_energy_mj=0.0, save_plot=False, verbose=False)
    with pytest.raises(ValueError, match="neon_quantity_mol"):
        run_spi_mitigation(neon_quantity_mol=-0.1, save_plot=False, verbose=False)
    with pytest.raises(ValueError, match="argon_quantity_mol"):
        run_spi_mitigation(argon_quantity_mol=-0.1, save_plot=False, verbose=False)
    with pytest.raises(ValueError, match="dt_s"):
        run_spi_mitigation(dt_s=0.0, save_plot=False, verbose=False)
