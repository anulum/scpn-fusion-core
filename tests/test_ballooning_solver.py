# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Ballooning Equation Solver Tests
from __future__ import annotations

import numpy as np

from scpn_fusion.core.ballooning_solver import (
    BallooningEquation,
    BallooningStabilityAnalysis,
    compute_stability_diagram,
    find_marginal_stability,
)
from scpn_fusion.core.stability_mhd import QProfile


def test_ballooning_equation_functions():
    eq = BallooningEquation(s=1.0, alpha=0.5)
    assert np.isclose(eq.f(0.0), 1.0)
    assert np.isclose(eq.g(0.0), 0.5)


def test_s_zero():
    # s = 0: alpha_crit ~ 0
    alpha_crit = find_marginal_stability(0.0)
    assert alpha_crit == 0.0


def test_s_0_5():
    # s = 0.5: alpha_crit is approx 0.3-0.4
    alpha_crit = find_marginal_stability(0.5)
    assert 0.3 < alpha_crit < 0.45


def test_s_1_0():
    # s = 1.0: alpha_crit is approx 0.6
    alpha_crit = find_marginal_stability(1.0)
    assert 0.55 < alpha_crit < 0.65


def test_s_2_0():
    # s = 2.0: alpha_crit is large (deep into second stability)
    # The analytic CHT puts it around 1.2
    alpha_crit = find_marginal_stability(2.0)
    assert 1.0 < alpha_crit < 1.4


def test_eigenfunction_decays_stable():
    # For alpha < alpha_crit, mode is stable (does NOT cross zero)
    eq = BallooningEquation(s=1.0, alpha=0.2)
    res = eq.solve()
    assert res.is_stable
    assert np.all(res.xi > -1e-6)  # Stays positive


def test_eigenfunction_grows_unstable():
    # For alpha >> alpha_crit (~0.6 at s=1), mode is unstable (CROSSES zero)
    eq = BallooningEquation(s=1.0, alpha=1.5)
    res = eq.solve()
    assert not res.is_stable
    assert np.any(res.xi <= 0)


def test_compute_stability_diagram():
    s_range = np.array([0.5, 1.0, 1.5])
    diagram = compute_stability_diagram(s_range)
    assert len(diagram) == 3
    assert diagram[0] < diagram[1] < diagram[2]


def test_ballooning_stability_analysis():
    rho = np.linspace(0, 1, 5)
    q = np.array([1.0, 1.5, 2.0, 3.0, 4.0])
    shear = np.array([-0.1, 0.5, 1.0, 1.5, 2.0])
    alpha_mhd = np.array([0.0, 0.1, 0.8, 0.5, 2.0])  # s=1.0, alpha=0.8 is unstable

    q_prof = QProfile(
        rho=rho,
        q=q,
        shear=shear,
        alpha_mhd=alpha_mhd,
        q_min=1.0,
        q_min_rho=0.0,
        q_edge=4.0,
    )

    analyzer = BallooningStabilityAnalysis()
    margin = analyzer.analyze(q_prof)

    assert len(margin) == 5
    # For s=-0.1, margin should be 0.0 - 0.0 = 0.0
    assert margin[0] == 0.0
    # For s=0.5, alpha=0.1, it should be stable (margin > 0)
    assert margin[1] > 0.0
    # For s=1.0, alpha=0.8, it should be unstable (margin < 0)
    assert margin[2] < 0.0
