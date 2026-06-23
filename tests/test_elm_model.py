# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — ELM Model Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.elm_model import (
    ELMCrashModel,
    ELMCycler,
    PeelingBallooningBoundary,
    RMPSuppression,
    elm_power_balance_frequency,
)


def test_pb_boundary_subcritical():
    pb = PeelingBallooningBoundary(q95=3.0, kappa=1.7, delta=0.3, a=2.0, R0=6.2)
    # Very low alpha and j
    assert not pb.is_unstable(alpha_edge=0.01, j_edge=1e4, s_edge=2.0)
    assert pb.stability_margin(alpha_edge=0.01, j_edge=1e4, s_edge=2.0) > 0.0


def test_pb_boundary_supercritical():
    pb = PeelingBallooningBoundary(q95=3.0, kappa=1.7, delta=0.3, a=2.0, R0=6.2)
    # High alpha
    assert pb.is_unstable(alpha_edge=25.0, j_edge=1e6, s_edge=2.0)
    assert pb.stability_margin(alpha_edge=25.0, j_edge=1e6, s_edge=2.0) < 0.0


def test_peeling_limit_uses_mode_and_shape_terms():
    round_low_mode = PeelingBallooningBoundary(q95=3.0, kappa=1.0, delta=0.0, a=2.0, R0=6.2)
    shaped_high_mode = PeelingBallooningBoundary(q95=3.0, kappa=1.8, delta=0.4, a=2.0, R0=6.2)

    low_mode_limit = round_low_mode.peeling_limit(j_edge=1e5, n_mode=5)
    high_mode_limit = round_low_mode.peeling_limit(j_edge=1e5, n_mode=25)
    shaped_limit = shaped_high_mode.peeling_limit(j_edge=1e5, n_mode=5)

    assert high_mode_limit > low_mode_limit
    assert shaped_limit > low_mode_limit
    with pytest.raises(ValueError, match="n_mode"):
        round_low_mode.peeling_limit(j_edge=1e5, n_mode=0)


def test_pb_coupling_reduces_stability_margin_near_dual_drive():
    pb = PeelingBallooningBoundary(q95=3.0, kappa=1.7, delta=0.3, a=2.0, R0=6.2)

    j_crit = pb.peeling_limit(j_edge=1e5)
    alpha_crit = pb.ballooning_limit(s_edge=2.0)

    assert not pb.is_unstable(alpha_edge=0.4 * alpha_crit, j_edge=0.4 * j_crit, s_edge=2.0)
    assert pb.is_unstable(alpha_edge=0.72 * alpha_crit, j_edge=0.72 * j_crit, s_edge=2.0)


def test_elm_crash_model():
    crash = ELMCrashModel(f_elm_fraction=0.1)

    W_ped = 100.0  # MJ
    T_ped = 5.0
    n_ped = 5.0

    res = crash.crash(T_ped, n_ped, W_ped)

    # 10% of W_ped
    assert np.isclose(res.delta_W_MJ, 10.0)
    assert res.T_ped_post < T_ped
    assert res.n_ped_post < n_ped


def test_crash_profile_flattening():
    crash = ELMCrashModel(f_elm_fraction=0.1)

    rho = np.linspace(0, 1, 100)
    Te = np.ones(100) * 5.0
    ne = np.ones(100) * 5.0

    Te_new, ne_new = crash.apply_to_profiles(rho, Te, ne, rho_ped=0.9)

    # Core unchanged
    assert Te_new[0] == 5.0
    # Edge drops
    assert Te_new[95] < 5.0


def test_elm_frequency_power_balance():
    f = elm_power_balance_frequency(P_SOL_MW=100.0, W_ped_MJ=100.0, f_elm_fraction=0.1)
    # 100 / (0.1 * 100) = 100 / 10 = 10 Hz
    assert np.isclose(f, 10.0)


def test_rmp_suppression():
    rmp = RMPSuppression()
    q = np.linspace(1, 3, 50)
    rho = np.linspace(0, 1, 50)

    # Very weak B_r
    chir = rmp.chirikov_parameter(q, rho, delta_B_r=1e-6, B0=5.3, R0=6.2)
    assert not rmp.suppressed(chir)
    assert rmp.density_pump_out(chir) == 0.0

    # Strong B_r (overlap > 1)
    chir2 = rmp.chirikov_parameter(q, rho, delta_B_r=1e-1, B0=5.3, R0=6.2)
    assert rmp.suppressed(chir2)
    assert rmp.density_pump_out(chir2) > 0.0


def test_rmp_chirikov_counts_outer_resonance_overlap_and_validates_profiles():
    rmp = RMPSuppression(n_toroidal=3)
    rho = np.linspace(0.55, 1.0, 80)
    q_sparse = 2.1 + 0.9 * (rho - rho[0]) / (rho[-1] - rho[0])
    q_dense = 2.1 + 3.9 * (rho - rho[0]) / (rho[-1] - rho[0])

    sparse = rmp.chirikov_parameter(q_sparse, rho, delta_B_r=2e-3, B0=5.3, R0=6.2)
    dense = rmp.chirikov_parameter(q_dense, rho, delta_B_r=2e-3, B0=5.3, R0=6.2)

    assert dense > sparse > 0.0
    with pytest.raises(ValueError, match="monotonic"):
        rmp.chirikov_parameter(q_dense[::-1], rho, delta_B_r=2e-3, B0=5.3, R0=6.2)


def test_elm_cycler():
    pb = PeelingBallooningBoundary(q95=3.0, kappa=1.7, delta=0.3, a=2.0, R0=6.2)
    crash = ELMCrashModel(f_elm_fraction=0.1)

    cycler = ELMCycler(pb, crash)

    # Step 1: Stable
    ev = cycler.step(0.1, alpha_edge=0.1, j_edge=1e4, s_edge=2.0, T_ped=5.0, n_ped=5.0, W_ped=100.0)
    assert ev is None

    # Step 2: Unstable
    ev2 = cycler.step(
        0.1, alpha_edge=10.0, j_edge=1e6, s_edge=2.0, T_ped=5.0, n_ped=5.0, W_ped=100.0
    )
    assert ev2 is not None
    assert ev2.delta_W_MJ == 10.0


def test_elm_require_positive_rejects_bad_values() -> None:
    from scpn_fusion.core.elm_model import _require_positive
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            _require_positive("x", bad)


def test_rmp_chirikov_parameter_validates_profiles() -> None:
    from scpn_fusion.core.elm_model import RMPSuppression
    rmp = RMPSuppression()
    rho = np.linspace(0.0, 1.0, 5)
    q = np.linspace(1.0, 3.0, 5)
    with pytest.raises(ValueError, match="equal length"):
        rmp.chirikov_parameter(q[:4], rho, delta_B_r=1e-3, B0=5.3, R0=6.2)
    with pytest.raises(ValueError, match="at least three"):
        rmp.chirikov_parameter(q[:2], rho[:2], delta_B_r=1e-3, B0=5.3, R0=6.2)
    with pytest.raises(ValueError, match="finite"):
        rmp.chirikov_parameter(np.array([1.0, np.nan, 3.0]), rho[:3], delta_B_r=1e-3, B0=5.3, R0=6.2)
    with pytest.raises(ValueError, match="monotonic"):
        rmp.chirikov_parameter(np.array([3.0, 2.0, 1.0]), rho[:3], delta_B_r=1e-3, B0=5.3, R0=6.2)


def test_rmp_chirikov_returns_zero_for_degenerate_cases() -> None:
    from scpn_fusion.core.elm_model import RMPSuppression
    rho = np.linspace(0.0, 1.0, 5)
    rmp = RMPSuppression()
    q = np.linspace(1.0, 3.0, 5)
    # Non-positive perturbation field -> zero Chirikov parameter.
    assert rmp.chirikov_parameter(q, rho, delta_B_r=0.0, B0=5.3, R0=6.2) == 0.0
    # Extremely narrow q-range: floor(n*q_max) < ceil(n*q_min) -> no resonances.
    q_narrow = np.array([2.40, 2.43, 2.46])
    assert rmp.chirikov_parameter(q_narrow, rho[:3], delta_B_r=1e-3, B0=5.3, R0=6.2) == 0.0


def test_rmp_density_pump_out_and_transport_enhancement() -> None:
    from scpn_fusion.core.elm_model import RMPSuppression
    rmp = RMPSuppression()
    assert rmp.density_pump_out(2.0) >= 0.0
    assert rmp.density_pump_out(0.5) >= 0.0
    assert rmp.pedestal_transport_enhancement(2.0) >= 1.0
    assert rmp.pedestal_transport_enhancement(0.5) == 1.0


def test_elm_power_balance_frequency() -> None:
    from scpn_fusion.core.elm_model import elm_power_balance_frequency
    assert elm_power_balance_frequency(10.0, 0.5, 0.08) > 0.0
    assert elm_power_balance_frequency(10.0, 0.0, 0.08) == 0.0


def test_rmp_chirikov_single_resonance_branch() -> None:
    from scpn_fusion.core.elm_model import RMPSuppression
    rmp = RMPSuppression()  # n_toroidal = 3
    # q in [2.9, 3.1] yields a single rational surface (m = 9) -> the <2-widths branch.
    q = np.array([2.9, 3.0, 3.1])
    rho = np.linspace(0.0, 1.0, 3)
    assert rmp.chirikov_parameter(q, rho, delta_B_r=1e-3, B0=5.3, R0=6.2) >= 0.0
