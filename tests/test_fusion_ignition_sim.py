# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests
"""Pytest tests for fusion ignition simulation (DynamicBurnModel)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pytest

import scpn_fusion.core.fusion_ignition_sim as fusion_ignition_sim
from scpn_fusion.core.fusion_ignition_sim import (
    DynamicBurnModel,
    FusionBurnPhysics,
    run_ignition_experiment,
)
from scpn_fusion.core.uncertainty import _dt_reactivity


def _f(result: dict[str, object], key: str) -> float:
    """Coerce one scalar entry of a simulate/scan result to float."""
    return float(np.asarray(result[key], dtype=np.float64).reshape(-1)[-1])


def _a(result: dict[str, object], key: str) -> NDArray[np.float64]:
    """Coerce one array entry of a simulate/scan result to a float array."""
    return np.asarray(result[key], dtype=np.float64)


def test_iter98y2_tau_e_power_degradation() -> None:
    """Confinement time must decrease with increasing heating power (IPB98y2)."""
    model = DynamicBurnModel()
    tau_prev = float("inf")
    for p in [10.0, 30.0, 60.0, 100.0]:
        tau = model.iter98y2_tau_e(p)
        assert tau < tau_prev
        tau_prev = tau


def test_iter98y2_tau_e_matches_iter_reference_point() -> None:
    """Pin the absolute IPB98(y,2) confinement time at the ITER baseline.

    For 15 MA, 5.3 T, n_e20 = 1.0, R = 6.2, a = 2.0, kappa = 1.7, M = 2.5 near
    the 87 MW H-mode loss power, tau_E ~ 3.6 s (ITER design point ~3.7 s). The
    power-degradation test only checks monotonicity, so a wrong 0.0562 prefactor
    would pass it.
    """
    model = DynamicBurnModel(R0=6.2, a=2.0, B_t=5.3, I_p=15.0, kappa=1.7, n_e20=1.0)
    p_loss = 87.0
    n_e19 = model.n_e20 * 10.0
    eps = model.a / model.R0
    expected = (
        0.0562
        * model.I_p**0.93
        * model.B_t**0.15
        * n_e19**0.41
        * p_loss ** (-0.69)
        * model.R0**1.97
        * eps**0.58
        * model.kappa**0.78
        * model.M_eff**0.19
    )
    assert model.iter98y2_tau_e(p_loss) == pytest.approx(expected, rel=1.0e-12)
    # ITER energy confinement time is a few seconds.
    assert 3.0 < model.iter98y2_tau_e(p_loss) < 4.2


def test_bosch_hale_dt_positivity() -> None:
    """Reactivity must be positive for T in [1, 100] keV."""
    for t in np.linspace(1.0, 100.0, 20):
        sv = _dt_reactivity(t)
        val = float(np.asarray(sv).ravel()[0])
        assert val > 0.0, f"Reactivity non-positive at {t} keV"


def test_calculate_thermodynamics_finite() -> None:
    """DynamicBurnModel.simulate() output must contain all-finite values."""
    model = DynamicBurnModel()
    result = model.simulate(
        P_aux_mw=50.0,
        duration_s=1.0,
        dt_s=0.1,
        warn_on_temperature_cap=False,
    )
    for key in (
        "P_fus_MW",
        "P_alpha_MW",
        "P_loss_MW",
        "Q",
        "T_keV",
        "tau_E_s",
        "W_MJ",
    ):
        arr = np.asarray(result[key])
        assert np.all(np.isfinite(arr)), f"{key} contains non-finite values"
    assert _f(result, "Q_final") >= 0.0
    assert _f(result, "T_final_keV") > 0.0


def test_h_mode_threshold() -> None:
    """The H-mode power threshold is positive and finite."""
    model = DynamicBurnModel()
    P_thr = model.h_mode_threshold_mw()
    assert P_thr > 0
    assert np.isfinite(P_thr)


def test_h_mode_threshold_scales_with_field() -> None:
    """The H-mode threshold rises with toroidal field."""
    m_low = DynamicBurnModel(B_t=3.0)
    m_high = DynamicBurnModel(B_t=12.0)
    assert m_high.h_mode_threshold_mw() > m_low.h_mode_threshold_mw()


def test_bosch_hale_peak_near_67_kev() -> None:
    """D-T reactivity peaks near ~67 keV."""
    model = DynamicBurnModel()
    svs = [model.bosch_hale_dt(T) for T in np.linspace(10, 100, 50)]
    peak_T = np.linspace(10, 100, 50)[np.argmax(svs)]
    assert 50 < peak_T < 80


def test_simulate_returns_q_and_power() -> None:
    """A burn simulation reports gain and fusion power."""
    model = DynamicBurnModel()
    result = model.simulate(P_aux_mw=50.0, duration_s=2.0, dt_s=0.1, warn_on_temperature_cap=False)
    assert "Q" in result
    assert "P_fus_MW" in result
    assert _f(result, "Q_final") >= 0.0


def test_simulate_absolute_power_balance_matches_first_principles() -> None:
    """Pin the absolute fusion power, radiated power, and Q at the first step.

    Recompute the headline 0-D outputs from verified primitives (Bosch-Hale
    reactivity, the 5.35e-37 bremsstrahlung coefficient, the 17.6 MeV D-T yield)
    so a wrong coefficient or unit conversion is caught, not only the qualitative
    Q >= 0 behaviour. Order-of-magnitude bounds guard against gross errors of the
    kind that scaling-only tests miss.
    """
    model = DynamicBurnModel()  # default n_e20 = 1.0, Z_eff = 1.65
    t0_keV = 15.0
    result = model.simulate(
        P_aux_mw=50.0,
        T_initial_keV=t0_keV,
        f_he_initial=0.0,  # no helium dilution -> n_d = n_t = n_e / 2
        duration_s=0.1,
        dt_s=0.1,
        warn_on_temperature_cap=False,
    )

    n_e = model.n_e20 * 1e20
    volume = model.V_plasma
    n_d = n_t = 0.5 * n_e
    sigmav = float(model.bosch_hale_dt(t0_keV))

    # Fusion power = n_d n_t <sigma v> * 17.6 MeV * V; this is also the D-T
    # neutron source rate n_d n_t <sigma v> V times the per-reaction energy.
    expected_p_fus_w = n_d * n_t * sigmav * 17.6e6 * 1.602e-19 * volume
    assert _a(result, "P_fus_MW")[0] == pytest.approx(expected_p_fus_w / 1e6, rel=1e-10)

    # Radiated power = bremsstrahlung (5.35e-37) + impurity-line closure.
    p_brems_w = 5.35e-37 * model.Z_eff * n_e**2 * np.sqrt(t0_keV) * volume
    p_line_w = 1e-37 * (model.Z_eff - 1.0) * n_e**2 * volume
    assert _a(result, "P_rad_MW")[0] == pytest.approx((p_brems_w + p_line_w) / 1e6, rel=1e-10)

    # Q = P_fus / P_aux, capped at 15 to suppress 0-D burn artefacts.
    assert _a(result, "Q")[0] == pytest.approx(min(expected_p_fus_w / 50e6, 15.0), rel=1e-10)

    # Order-of-magnitude sanity for an ITER-scale 1e20 m^-3, 15 keV core:
    # fusion power is GW-scale and radiated power is tens of MW.
    assert 1e8 < expected_p_fus_w < 1e10
    assert 1e6 < p_brems_w + p_line_w < 1e8


def test_simulate_stored_energy_uses_total_heat_capacity() -> None:
    """Stored energy uses the total electron+ion heat capacity W = 3 n_e T V.

    An electron-only 1.5 n_e T would halve the heat capacity. ``W_MJ[i]`` is the
    end-of-step energy and ``T_keV[i+1]`` is the temperature recovered from it, so
    the invariant ``W_MJ[i] = 3 n_e T_keV[i+1] V`` pins the factor of 3.
    """
    model = DynamicBurnModel()
    result = model.simulate(
        P_aux_mw=0.0,
        T_initial_keV=8.0,
        duration_s=0.5,
        dt_s=0.05,
        warn_on_temperature_cap=False,
    )

    n_e = model.n_e20 * 1e20
    w_mj = np.asarray(result["W_MJ"])[:-1]
    t_next_keV = np.asarray(result["T_keV"])[1:]
    expected_w_mj = 3.0 * n_e * t_next_keV * 1e3 * 1.602e-19 * model.V_plasma / 1e6
    np.testing.assert_allclose(w_mj, expected_w_mj, rtol=1e-9)


def test_alpha_deposition_remains_nonnegative_for_coarse_burn_timestep() -> None:
    """Alpha slowing-down deposition must preserve non-negative deposited power."""
    model = DynamicBurnModel()
    result = model.simulate(
        P_aux_mw=0.0,
        T_initial_keV=5.0,
        duration_s=5.0,
        dt_s=0.05,
        warn_on_temperature_cap=False,
    )

    p_alpha = np.asarray(result["P_alpha_MW"])
    assert np.all(np.isfinite(p_alpha))
    assert np.all(p_alpha >= 0.0)


def test_simulate_custom_params() -> None:
    """A compact-machine parameter set still yields a non-negative gain."""
    model = DynamicBurnModel(R0=1.85, a=0.6, B_t=12.2, I_p=8.7, kappa=1.97)
    result = model.simulate(P_aux_mw=25.0, duration_s=0.5, dt_s=0.05, warn_on_temperature_cap=False)
    assert _f(result, "Q_final") >= 0.0


def test_plasma_volume() -> None:
    """The toroidal plasma volume matches 2 pi^2 R0 a^2 kappa."""
    model = DynamicBurnModel(R0=6.2, a=2.0, kappa=1.7)
    V = model.V_plasma
    # V = 2 pi^2 R0 a^2 kappa = 2 * 9.87 * 6.2 * 4.0 * 1.7 ≈ 832
    assert 800 < V < 900


def test_rejects_nonpositive_params() -> None:
    """Non-positive machine parameters are rejected."""
    import pytest as _pt

    with _pt.raises(ValueError):
        DynamicBurnModel(R0=0.0)
    with _pt.raises(ValueError):
        DynamicBurnModel(B_t=-1.0)


def _bare_burn_lab() -> FusionBurnPhysics:
    """Build a FusionBurnPhysics with grid geometry but no full kernel init."""
    lab = FusionBurnPhysics.__new__(FusionBurnPhysics)
    lab.R = np.linspace(3.0, 9.0, 33)
    lab.Z = np.linspace(-4.0, 4.0, 33)
    lab.dR = lab.R[1] - lab.R[0]
    lab.dZ = lab.Z[1] - lab.Z[0]
    lab.RR, lab.ZZ = np.meshgrid(lab.R, lab.Z)
    lab.Psi = np.exp(-((lab.RR - 6.2) ** 2 + lab.ZZ**2) / 4.0)
    lab.cfg = {
        "physics": {"plasma_current_target": 15.0e6},
        "dimensions": {
            "R_min": 4.0,
            "R_max": 8.4,
            "Z_min": -4.0,
            "Z_max": 4.0,
            "B0": 5.3,
            "R0": 6.2,
            "kappa": 1.7,
        },
    }
    return lab


def test_calculate_thermodynamics_reports_power_balance() -> None:
    """The equilibrium thermodynamics map returns a finite fusion power balance."""
    out = _bare_burn_lab().calculate_thermodynamics(P_aux_MW=50.0)
    for key in ("P_fusion_MW", "P_alpha_MW", "P_loss_MW", "Net_MW", "Q", "W_MJ"):
        assert key in out
        assert np.isfinite(out[key])
    assert out["P_fusion_MW"] > 0.0
    assert out["Q"] > 0.0


def test_calculate_thermodynamics_zero_aux_gives_zero_q() -> None:
    """With no auxiliary heating the gain is defined as zero."""
    out = _bare_burn_lab().calculate_thermodynamics(P_aux_MW=0.0)
    assert out["Q"] == 0.0


def test_calculate_thermodynamics_rejects_negative_aux() -> None:
    """A negative auxiliary-power input is rejected."""
    with pytest.raises(ValueError, match="P_aux_MW"):
        _bare_burn_lab().calculate_thermodynamics(P_aux_MW=-10.0)


def test_calculate_thermodynamics_limiter_boundary_fallback() -> None:
    """A near-flat flux map falls back to the minimum as the plasma boundary."""
    lab = _bare_burn_lab()
    lab.Psi = np.full((33, 33), 0.5)
    out = lab.calculate_thermodynamics(P_aux_MW=50.0)
    assert np.isfinite(out["P_fusion_MW"])


def test_run_ignition_experiment_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """The standalone ignition power-ramp demo runs and renders its plot."""
    import matplotlib.pyplot as plt

    lab = _bare_burn_lab()
    lab.solve_equilibrium = lambda: None  # type: ignore[assignment, misc]
    monkeypatch.setattr(fusion_ignition_sim, "FusionBurnPhysics", lambda _path: lab)

    saved: list[str] = []
    monkeypatch.setattr(plt, "savefig", lambda path, *a, **k: saved.append(str(path)))
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    run_ignition_experiment()

    assert saved
