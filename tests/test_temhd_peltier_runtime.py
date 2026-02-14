# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — TEMHD Peltier Runtime Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Runtime summary tests for TEMHD experiment entrypoint."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.nuclear.temhd_peltier import run_temhd_experiment


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
    kwargs = dict(
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
    a = run_temhd_experiment(**kwargs)
    b = run_temhd_experiment(**kwargs)
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
