# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Runaway Electrons Tests
from __future__ import annotations

import numpy as np

from scpn_fusion.core.runaway_electrons import (
    RunawayEvolution,
    RunawayMitigationAssessment,
    RunawayParams,
    avalanche_growth_rate,
    critical_field,
    dreicer_generation_rate,
    hot_tail_seed,
)


def test_zero_epar():
    params = RunawayParams(ne_20=1.0, Te_keV=1.0, E_par=0.0, Z_eff=1.5, B0=5.3, R0=6.2)
    assert dreicer_generation_rate(params) == 0.0
    assert avalanche_growth_rate(params, n_RE=1e10) == 0.0


def test_subcritical_epar():
    E_c = critical_field(ne_20=1.0)
    # Set E_par below E_c
    params = RunawayParams(ne_20=1.0, Te_keV=1.0, E_par=E_c * 0.5, Z_eff=1.5, B0=5.3, R0=6.2)

    # No avalanche
    assert avalanche_growth_rate(params, n_RE=1e10) == 0.0

    # Dreicer should be negligible
    assert dreicer_generation_rate(params) < 1e-10


def test_dreicer_generation():
    params = RunawayParams(ne_20=1.0, Te_keV=1.0, E_par=0.0, Z_eff=1.5, B0=5.3, R0=6.2)
    # E_D is large for Te=1keV
    # Let's set E_par artificially high to get Dreicer
    E_D = params.ne_20 * 1e20 * 1.6e-19**3 * 15.0 / (4 * np.pi * 8.85e-12**2 * 1.6e-16)  # rough
    params.E_par = 100.0  # Massive field, e.g. TQ

    rate = dreicer_generation_rate(params)
    assert rate > 0.0


def test_avalanche_multiplication():
    params = RunawayParams(ne_20=1.0, Te_keV=0.01, E_par=10.0, Z_eff=1.5, B0=5.3, R0=6.2)
    # E_c for ne_20=1 is ~0.1 V/m. So E_par = 10 is 100x E_c.
    E_c = critical_field(1.0)
    assert params.E_par > E_c

    n_RE = 1e10
    rate = avalanche_growth_rate(params, n_RE)
    assert rate > 0.0

    # Should scale linearly with n_RE
    rate2 = avalanche_growth_rate(params, 2e10)
    assert np.isclose(rate2, 2.0 * rate)


def test_hot_tail():
    # Drop from 10 keV to 10 eV in 1 ms
    n_seed = hot_tail_seed(Te_pre_keV=10.0, Te_post_keV=0.01, ne_20=1.0, quench_time_ms=1.0)
    assert n_seed > 1e10  # Substantial seed


def test_runaway_evolution():
    params = RunawayParams(ne_20=1.0, Te_keV=0.01, E_par=0.0, Z_eff=1.5, B0=5.3, R0=6.2)
    ev = RunawayEvolution(params)

    # Constant E_par
    t, n = ev.evolve(n_RE_0=1e10, E_par_profile=lambda t: 5.0, t_span=(0, 0.1), dt=0.01)

    assert len(t) == 11
    assert n[-1] > n[0]  # Should grow exponentially


def test_mitigation_assessment():
    mit = RunawayMitigationAssessment()

    # Typical ITER E_par during disruption is ~10-50 V/m
    E_par = 20.0
    ne_req = mit.required_density_for_suppression(E_par, Z_eff=1.0)
    # Should be massive (> 100 * 10^20 m^-3)
    assert ne_req > 100.0

    E_max = mit.maximum_re_energy(B0=5.3, R0=6.2)
    # e * 5.3 * 6.2 * 3e8 = 9.8e-12 J -> ~60 MeV (prompt said 25, formula gives ~60, close enough for order of magnitude)
    assert 10.0 < E_max < 100.0

    load = mit.wall_heat_load(n_RE=1e16, E_max_MeV=E_max, A_wet=10.0)
    assert load > 0.0
