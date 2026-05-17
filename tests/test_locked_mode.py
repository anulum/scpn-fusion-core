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
    spec = ErrorFieldSpectrum(B0=5.0)

    B21_nom = spec.B_mn(2, 1)
    assert B21_nom == 5e-4

    spec.set_coil_misalignment(10.0, 0.0)  # 10mm shift
    B21_shift = spec.B_mn(2, 1)

    # 0.01 * 5.0 * 0.01 = 5e-4
    assert np.isclose(B21_shift, 5e-4)


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
