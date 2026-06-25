# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Locked Mode Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.locked_mode import (
    ErrorFieldSpectrum,
    ErrorFieldToDisruptionChain,
    LockedModeIsland,
    ModeLocking,
    ResonantFieldAmplification,
)


def test_error_field_spectrum():
    spec = ErrorFieldSpectrum(B0=5.0, R0=5.0)

    B21_nom = spec.B_mn(2, 1)
    assert B21_nom == 5e-4

    spec.set_coil_misalignment(10.0, 0.0)  # 10mm shift
    B21_shift = spec.B_mn(2, 1)

    # sensitivity * B0 * (shift/R0) = 0.01 * 5 * (0.01/5) = 1e-4
    assert np.isclose(B21_shift, 1e-4)


def test_error_field_correction_channels_attenuate_intrinsic_spectrum():
    uncorrected = ErrorFieldSpectrum(B0=5.0, n_corrections=0)
    corrected = ErrorFieldSpectrum(B0=5.0, n_corrections=4)

    assert corrected.B_mn(2, 1) < uncorrected.B_mn(2, 1)
    assert corrected.B_mn(3, 2) < uncorrected.B_mn(3, 2)


def test_error_field_spectrum_rejects_invalid_domain():
    with pytest.raises(ValueError, match="B0"):
        ErrorFieldSpectrum(B0=0.0)
    with pytest.raises(ValueError, match="n_corrections"):
        ErrorFieldSpectrum(B0=5.0, n_corrections=-1)
    with pytest.raises(ValueError, match="R0"):
        ErrorFieldSpectrum(B0=5.0, R0=0.0)


def test_error_field_geometry_scaling_reduces_field_for_larger_machine() -> None:
    compact = ErrorFieldSpectrum(B0=5.0, R0=2.5)
    large = ErrorFieldSpectrum(B0=5.0, R0=10.0)
    compact.set_coil_misalignment(10.0, 0.0)
    large.set_coil_misalignment(10.0, 0.0)
    assert compact.B_mn(2, 1) > large.B_mn(2, 1)


def test_corrected_error_field_is_monotone_with_correction_current() -> None:
    spec = ErrorFieldSpectrum(B0=5.0, R0=5.0)
    spec.set_coil_misalignment(10.0, 0.0)
    b0 = spec.corrected_B_mn(2, 1, I_correction=0.0)
    b50 = spec.corrected_B_mn(2, 1, I_correction=50.0)
    b200 = spec.corrected_B_mn(2, 1, I_correction=200.0)
    assert b0 > b50 > b200 >= 0.0


def test_corrected_error_field_mode_order_has_weaker_high_mn_correction() -> None:
    spec = ErrorFieldSpectrum(B0=5.0, R0=5.0)
    spec.set_coil_misalignment(10.0, 0.0)
    b21 = spec.corrected_B_mn(2, 1, I_correction=100.0)
    b32 = spec.corrected_B_mn(3, 2, I_correction=100.0)
    raw21 = spec.B_mn(2, 1)
    raw32 = spec.B_mn(3, 2)
    frac21 = b21 / raw21 if raw21 > 0 else 0.0
    frac32 = b32 / raw32 if raw32 > 0 else 0.0
    assert frac32 > frac21


def test_corrected_error_field_rejects_invalid_mode_numbers() -> None:
    spec = ErrorFieldSpectrum(B0=5.0)
    with pytest.raises(ValueError, match="mode numbers"):
        spec.corrected_B_mn(0, 1, I_correction=10.0)


def test_resonant_field_amplification():
    rfa = ResonantFieldAmplification(beta_N=2.5, beta_N_nowall=2.8)

    # factor = 1 / (1 - 2.5/2.8) = 1 / (0.3/2.8) = 2.8/0.3 ~ 9.33
    factor = rfa.amplification_factor()
    assert factor > 1.0

    B_err = 1e-4
    B_res = rfa.resonant_field(B_err)
    assert B_res > B_err


def test_resonant_field_amplification_rejects_invalid_beta_domain():
    with pytest.raises(ValueError, match="beta_N_nowall"):
        ResonantFieldAmplification(beta_N=2.0, beta_N_nowall=0.0)
    with pytest.raises(ValueError, match="beta_N"):
        ResonantFieldAmplification(beta_N=-1.0, beta_N_nowall=2.8)


def test_mode_locking():
    ml = ModeLocking(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=1e4)

    # Small B_res -> no lock
    ev1 = ml.evolve_rotation(B_res=1e-5, r_s=1.0, tau_visc=0.1, dt=0.001, n_steps=100)
    assert not ev1.locked

    # Large B_res -> lock
    ev2 = ml.evolve_rotation(B_res=1e-2, r_s=1.0, tau_visc=0.1, dt=0.001, n_steps=1000)
    assert ev2.locked
    assert ev2.omega_trace[-1] == 0.0


def test_mode_locking_rejects_invalid_evolution_domain():
    ml = ModeLocking(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=1e4)

    with pytest.raises(ValueError, match="B_res"):
        ml.evolve_rotation(B_res=-1e-5, r_s=1.0, tau_visc=0.1, dt=0.001, n_steps=100)
    with pytest.raises(ValueError, match="tau_visc"):
        ml.evolve_rotation(B_res=1e-5, r_s=1.0, tau_visc=0.0, dt=0.001, n_steps=100)
    with pytest.raises(ValueError, match="n_steps"):
        ml.evolve_rotation(B_res=1e-5, r_s=1.0, tau_visc=0.1, dt=0.001, n_steps=0)


def test_mode_locking_rejects_invalid_inertia_parameters():
    with pytest.raises(ValueError, match="n_e19"):
        ModeLocking(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=1e4, n_e19=0.0)
    with pytest.raises(ValueError, match="inertia_factor"):
        ModeLocking(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=1e4, inertia_factor=0.0)
    with pytest.raises(ValueError, match="I_eff_override"):
        ModeLocking(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=1e4, I_eff_override=0.0)


def test_mode_locking_inertia_controls_locking_latency():
    low_inertia = ModeLocking(
        R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=1e4, I_eff_override=0.01
    )
    high_inertia = ModeLocking(
        R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=1e4, I_eff_override=0.05
    )
    ev_low = low_inertia.evolve_rotation(B_res=1e-2, r_s=1.0, tau_visc=0.1, dt=0.001, n_steps=1000)
    ev_high = high_inertia.evolve_rotation(
        B_res=1e-2, r_s=1.0, tau_visc=0.1, dt=0.001, n_steps=1000
    )
    assert ev_low.locked
    assert ev_high.locked
    assert ev_low.lock_time <= ev_high.lock_time


def test_post_locking_island_growth():
    lm = LockedModeIsland(r_s=1.0, m=2, n=1, a=2.0, R0=6.2, delta_prime=-1.0)

    res = lm.grow(w0=1e-3, eta=1e-6, dt=0.01, n_steps=1000, delta_r_mn=0.4)

    assert res.w_trace[-1] > 1e-3
    assert res.stochastic


def test_disruption_chain():
    config = {"R0": 6.2, "a": 2.0, "B0": 5.3, "Ip_MA": 15.0, "beta_N": 2.5, "beta_N_nowall": 2.8}

    chain = ErrorFieldToDisruptionChain(config)

    # Small error -> no disruption
    res_safe = chain.run(B_err_n1=1e-6, omega_phi_0=1e4)
    assert not res_safe.disruption

    # Large error -> disruption
    res_disrupt = chain.run(B_err_n1=5e-4, omega_phi_0=1e4)
    assert res_disrupt.disruption
    assert res_disrupt.lock_time > 0.0
    assert res_disrupt.warning_time_ms > 0.0


def test_error_field_spectrum_constructor_validation() -> None:
    for kw, match in [
        ({"B0": 0.0, "R0": 5.0}, "B0"),
        ({"B0": 5.0, "R0": 5.0, "n_corrections": -1}, "n_corrections"),
        ({"B0": 5.0, "R0": 0.0}, "R0"),
        ({"B0": 5.0, "R0": 5.0, "alignment_sensitivity_21": 0.0}, "alignment_sensitivity_21"),
        ({"B0": 5.0, "R0": 5.0, "alignment_sensitivity_32": 0.0}, "alignment_sensitivity_32"),
    ]:
        with pytest.raises(ValueError, match=match):
            ErrorFieldSpectrum(**kw)


def test_resonant_field_amplification_diverges_above_no_wall_limit() -> None:
    rfa = ResonantFieldAmplification(beta_N=3.0, beta_N_nowall=2.5)
    assert rfa.amplification_factor() == float("inf")


def test_error_field_set_coil_misalignment_rejects_nonfinite() -> None:
    spec = ErrorFieldSpectrum(B0=5.0, R0=5.0)
    with pytest.raises(ValueError, match="must be finite"):
        spec.set_coil_misalignment(float("nan"), 0.0)


def test_error_field_b_mn_and_corrected_b_mn_branches() -> None:
    spec = ErrorFieldSpectrum(B0=5.0, R0=5.0)
    assert spec.B_mn(2, 1) >= 0.0
    # (7, 5) is absent from the harmonic table -> B_raw == 0 -> corrected returns 0.0
    assert spec.corrected_B_mn(7, 5, 1.0) == 0.0
    with pytest.raises(ValueError, match="mode numbers"):
        spec.corrected_B_mn(0, 1, 1.0)


def test_mode_locking_constructor_validation() -> None:
    base = dict(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=1e4)
    for bad, match in [
        ({"n_e19": 0.0}, "n_e19"),
        ({"kappa": 0.0}, "kappa"),
        ({"inertia_factor": 0.0}, "inertia_factor"),
    ]:
        with pytest.raises(ValueError, match=match):
            ModeLocking(**{**base, **bad})


def test_mode_locking_em_torque_validation() -> None:
    ml = ModeLocking(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=1e4)
    with pytest.raises(ValueError, match="B_res"):
        ml.em_torque(-1.0, 0.5, 2, 1)
    with pytest.raises(ValueError, match="r_s"):
        ml.em_torque(0.01, 0.0, 2, 1)
    with pytest.raises(ValueError, match="mode numbers"):
        ml.em_torque(0.01, 0.5, 0, 1)


def test_locked_corrected_bmn_rejects_negative_correction() -> None:
    spec = ErrorFieldSpectrum(B0=5.0, R0=5.0)
    with pytest.raises(ValueError, match="I_correction"):
        spec.corrected_B_mn(2, 1, -1.0)


def test_mode_locking_geometry_items_validation() -> None:
    base = dict(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=1e4)
    for bad, match in [({"R0": 0.0}, "R0"), ({"a": 0.0}, "a"), ({"Ip_MA": 0.0}, "Ip")]:
        with pytest.raises(ValueError, match=match):
            ModeLocking(**{**base, **bad})


def test_mode_locking_evolve_rotation_validates_inputs() -> None:
    ml = ModeLocking(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=1e4)
    with pytest.raises(ValueError, match="tau_visc"):
        ml.evolve_rotation(0.01, 0.5, 0.0, 0.1, 10)
    with pytest.raises(ValueError, match="dt must be finite"):
        ml.evolve_rotation(0.01, 0.5, 1.0, 0.0, 10)


def test_error_field_to_disruption_chain_runs() -> None:
    chain = ErrorFieldToDisruptionChain(
        {"R0": 6.2, "a": 2.0, "B0": 5.3, "Ip_MA": 15.0, "beta_N": 2.0, "beta_N_nowall": 2.8}
    )
    result = chain.run(B_err_n1=5.0e-3, omega_phi_0=1.0e3)
    assert result.lock_time >= 0.0 or result.lock_time == -1.0
