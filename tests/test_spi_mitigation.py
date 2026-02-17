# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — SPI Mitigation Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Deterministic regression tests for SPI mitigation reduced models."""

from __future__ import annotations

import numpy as np

from scpn_fusion.control.spi_mitigation import ShatteredPelletInjection


def test_estimate_z_eff_increases_with_neon_quantity() -> None:
    low = ShatteredPelletInjection.estimate_z_eff(0.02)
    high = ShatteredPelletInjection.estimate_z_eff(0.20)
    assert low >= 1.0
    assert high > low


def test_estimate_z_eff_cocktail_reflects_species_mix() -> None:
    neon_only = ShatteredPelletInjection.estimate_z_eff_cocktail(neon_quantity_mol=0.12)
    argon_only = ShatteredPelletInjection.estimate_z_eff_cocktail(
        argon_quantity_mol=0.12
    )
    xenon_only = ShatteredPelletInjection.estimate_z_eff_cocktail(
        xenon_quantity_mol=0.12
    )
    assert neon_only >= 1.0
    assert argon_only > neon_only
    assert xenon_only > argon_only


def test_estimate_tau_cq_is_iter_like_for_cold_impure_plasma() -> None:
    tau = ShatteredPelletInjection.estimate_tau_cq(te_keV=0.1, z_eff=2.0)
    assert 0.015 <= tau <= 0.025

    tau_hot = ShatteredPelletInjection.estimate_tau_cq(te_keV=0.5, z_eff=2.0)
    tau_impure = ShatteredPelletInjection.estimate_tau_cq(te_keV=0.1, z_eff=4.0)
    assert tau_hot > tau
    assert tau_impure < tau


def test_estimate_mitigation_cocktail_shifts_to_heavier_species_with_risk() -> None:
    low = ShatteredPelletInjection.estimate_mitigation_cocktail(
        risk_score=0.2,
        disturbance=0.2,
        action_bias=0.0,
    )
    high = ShatteredPelletInjection.estimate_mitigation_cocktail(
        risk_score=0.9,
        disturbance=0.9,
        action_bias=1.0,
    )
    assert np.isclose(
        low["total_quantity_mol"],
        low["neon_quantity_mol"]
        + low["argon_quantity_mol"]
        + low["xenon_quantity_mol"],
    )
    assert high["xenon_quantity_mol"] > low["xenon_quantity_mol"]
    assert high["total_quantity_mol"] >= low["total_quantity_mol"]


def test_trigger_mitigation_returns_finite_histories_and_diagnostics() -> None:
    spi = ShatteredPelletInjection(Plasma_Energy_MJ=300.0, Plasma_Current_MA=15.0)
    t, w, i, diag = spi.trigger_mitigation(
        neon_quantity_mol=0.1,
        argon_quantity_mol=0.02,
        xenon_quantity_mol=0.01,
        return_diagnostics=True,
    )
    assert len(t) == len(w) == len(i)
    assert len(t) > 100
    assert np.isfinite(np.asarray(w, dtype=float)).all()
    assert np.isfinite(np.asarray(i, dtype=float)).all()
    assert i[-1] < i[0]
    assert diag["z_eff"] >= 1.0
    assert diag["tau_cq_ms_mean"] > 0.0
    assert diag["tau_cq_ms_p95"] > 0.0
    assert np.isclose(diag["argon_quantity_mol"], 0.02)
    assert np.isclose(diag["xenon_quantity_mol"], 0.01)
    assert np.isclose(diag["total_impurity_mol"], 0.13)
