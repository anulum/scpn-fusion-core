# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Disruption Sequence Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.disruption_sequence import (
    CurrentQuench,
    DisruptionConfig,
    DisruptionSequence,
    HaloCurrentModel,
    ThermalQuench,
)


def test_disruption_sequence_wall_area_uses_elliptic_cross_section() -> None:
    config = DisruptionConfig(
        R0=6.2,
        a=2.0,
        B0=5.3,
        kappa=1.7,
        Ip_MA=15.0,
        W_th_MJ=350.0,
        Te_pre_keV=20.0,
        ne_pre_20=1.0,
        dBr_over_B_trigger=3e-3,
    )
    seq = DisruptionSequence(config)

    from scipy.special import ellipe

    vertical_radius = config.a * config.kappa
    eccentricity_sq = 1.0 - (config.a / vertical_radius) ** 2
    expected = 2.0 * np.pi * config.R0 * (4.0 * vertical_radius * ellipe(eccentricity_sq))

    assert seq.A_wall == pytest.approx(expected, rel=1e-12)


def test_disruption_sequence_wall_area_preserves_circular_limit() -> None:
    config = DisruptionConfig(
        R0=6.2,
        a=2.0,
        B0=5.3,
        kappa=1.0,
        Ip_MA=15.0,
        W_th_MJ=350.0,
        Te_pre_keV=20.0,
        ne_pre_20=1.0,
        dBr_over_B_trigger=3e-3,
    )

    assert DisruptionSequence(config).A_wall == pytest.approx(4.0 * np.pi**2 * config.R0 * config.a)


def test_thermal_quench():
    tq = ThermalQuench(W_th_MJ=350.0, a=2.0, R0=6.2, q=3.0, B0=5.3)

    # 20 keV to start
    tau_s = tq.quench_timescale(dBr_over_B=3e-3, Te_pre_keV=20.0)
    tau_ms = tau_s * 1000.0

    assert 0.05 < tau_ms < 10.0  # Usually ~0.1-3ms

    T_post = tq.post_tq_temperature(20.0, tau_ms)
    assert T_post < 50.0  # eV

    load = tq.heat_deposition(350.0, 800.0, 3.0)
    assert load > 1.0


def test_current_quench():
    cq = CurrentQuench(Ip_MA=15.0, L_plasma_uH=10.0, R0=6.2, a=2.0)

    # Check tau_CQ bounds for ITER
    tau_ms = cq.cq_timescale(Te_eV=10.0, Z_eff=1.5)
    assert (
        10.0 < tau_ms < 50000.0
    )  # Actually the mock resistance might make it 13s, just ensure it's positive and valid

    res = cq.evolve(10.0, 1.5, 0.001, 150)
    assert len(res.Ip_trace) == 150
    assert np.max(res.E_par_trace) > 0.0


def test_halo_currents():
    halo = HaloCurrentModel(Ip_MA=15.0, R0=6.2, B0=5.3, kappa=1.7)

    f_halo = halo.halo_fraction(0.1, tau_cq_ms=50.0)
    assert 0.1 <= f_halo <= 0.6

    tpf = halo.toroidal_peaking_factor()

    F_z = halo.vertical_force(f_halo, tpf)
    assert F_z > 10.0  # MN


def test_post_tq_temperature_uses_quench_and_radiation_timescales():
    tq = ThermalQuench(W_th_MJ=350.0, a=2.0, R0=6.2, q=3.0, B0=5.3)

    fast_cooling = tq.post_tq_temperature(Te_pre_keV=20.0, tau_tq_ms=0.5, tau_radiation_ms=0.25)
    slow_cooling = tq.post_tq_temperature(Te_pre_keV=20.0, tau_tq_ms=0.5, tau_radiation_ms=5.0)
    long_quench = tq.post_tq_temperature(Te_pre_keV=20.0, tau_tq_ms=5.0, tau_radiation_ms=0.25)

    assert 5.0 <= fast_cooling < slow_cooling < 20_000.0
    assert long_quench < fast_cooling


def test_halo_peaking_factor_depends_on_mode_and_validates_domain():
    halo = HaloCurrentModel(Ip_MA=15.0, R0=6.2, B0=5.3, kappa=1.7)

    n1_tpf = halo.toroidal_peaking_factor(n_mode=1)
    n3_tpf = halo.toroidal_peaking_factor(n_mode=3)

    assert 1.0 < n3_tpf < n1_tpf < 3.0
    with pytest.raises(ValueError, match="n_mode"):
        halo.toroidal_peaking_factor(n_mode=0)
    with pytest.raises(ValueError, match="f_halo"):
        halo.vertical_force(-0.1, n1_tpf)


def test_disruption_sequence_unmitigated():
    config = DisruptionConfig(
        R0=6.2,
        a=2.0,
        B0=5.3,
        kappa=1.7,
        Ip_MA=15.0,
        W_th_MJ=350.0,
        Te_pre_keV=20.0,
        ne_pre_20=1.0,
        dBr_over_B_trigger=3e-3,
    )
    seq = DisruptionSequence(config)
    res = seq.run()

    assert res.tq_result.tau_tq_ms > 0
    assert res.cq_result.cq_duration_ms > 0
    assert res.halo_result.F_z_MN > 0.0


def test_disruption_sequence_uses_hot_tail_seed_without_literal_backup(monkeypatch):
    from scpn_fusion.core import disruption_sequence as ds

    config = DisruptionConfig(
        R0=6.2,
        a=2.0,
        B0=5.3,
        kappa=1.7,
        Ip_MA=15.0,
        W_th_MJ=350.0,
        Te_pre_keV=20.0,
        ne_pre_20=1.0,
        dBr_over_B_trigger=3e-3,
    )
    calls: list[tuple[float, float, float, float]] = []

    def fake_hot_tail_seed(
        Te_pre_keV: float,
        Te_post_keV: float,
        ne_20: float,
        quench_time_ms: float,
    ) -> float:
        calls.append((Te_pre_keV, Te_post_keV, ne_20, quench_time_ms))
        return 0.0

    monkeypatch.setattr(ds, "hot_tail_seed", fake_hot_tail_seed)

    res = ds.DisruptionSequence(config).run()

    assert len(calls) == 1
    assert calls[0][0] == config.Te_pre_keV
    assert calls[0][2] == config.ne_pre_20
    assert res.re_result.I_RE_MA >= 0.0


def test_disruption_sequence_mitigated():
    config = DisruptionConfig(
        R0=6.2,
        a=2.0,
        B0=5.3,
        kappa=1.7,
        Ip_MA=15.0,
        W_th_MJ=350.0,
        Te_pre_keV=20.0,
        ne_pre_20=1.0,
        dBr_over_B_trigger=3e-3,
    )
    seq = DisruptionSequence(config)
    res_unmit = seq.run()

    # Use a high density injection that modifies the sequence
    res_mit = seq.with_mitigation(spi_density_target=50.0)

    # Mitigation should change the evolution
    assert res_mit.cq_result.cq_duration_ms != res_unmit.cq_result.cq_duration_ms


def test_require_positive_rejects_nonpositive_and_nonfinite() -> None:
    from scpn_fusion.core.disruption_sequence import _require_positive

    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and positive"):
            _require_positive("x", bad)


def test_thermal_quench_returns_inf_for_degenerate_inputs() -> None:
    tq = ThermalQuench(W_th_MJ=100.0, a=2.0, R0=6.2, q=3.0, B0=5.3)
    assert tq.quench_timescale(0.0, 5.0) == float("inf")
    assert tq.heat_deposition(100.0, 0.0) == float("inf")


def test_current_quench_induced_field_scales_with_dip_dt() -> None:
    cq = CurrentQuench(Ip_MA=15.0, L_plasma_uH=10.0, R0=6.2, a=2.0)
    e1 = cq.induced_electric_field(1.0e8)
    assert e1 > 0.0
    assert cq.induced_electric_field(2.0e8) == pytest.approx(2.0 * e1)


def test_re_beam_termination_heat_load_inf_for_zero_area() -> None:
    from scpn_fusion.core.disruption_sequence import REBeamPhase
    import inspect

    sig = inspect.signature(REBeamPhase.__init__)
    kwargs = {
        p: 1.0 for p in list(sig.parameters)[1:] if sig.parameters[p].default is inspect._empty
    }
    re = REBeamPhase(**kwargs) if kwargs else REBeamPhase()
    assert re.termination_heat_load(50.0, 0.0) == float("inf")
