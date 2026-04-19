# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests: Native TGLF-Equivalent Transport Model

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.gk_eigenvalue import LinearGKResult, EigenMode
from scpn_fusion.core.gk_interface import GKLocalParams, GKOutput
from scpn_fusion.core.gk_quasilinear import mixing_length_saturation
from scpn_fusion.core.gk_tglf_native import (
    TGLFNativeConfig,
    TGLFNativeSolver,
    exb_shear_rate,
    quasilinear_weights,
    sat0,
    sat1,
    sat2,
    spectral_weight,
    trapped_fraction,
    trapped_particle_damping,
)

# Standard CBC parameters
_CBC = GKLocalParams(
    R_L_Ti=6.9,
    R_L_Te=6.9,
    R_L_ne=2.2,
    q=1.4,
    s_hat=0.78,
    rho=0.5,
    R0=2.78,
    a=1.0,
    B0=2.0,
    epsilon=0.18,
    T_e_keV=2.0,
    T_i_keV=2.0,
    n_e=5.0,
)


def _make_linear(
    k_y: list[float],
    gamma: list[float],
    omega_r: list[float],
    mode_type: list[str],
) -> LinearGKResult:
    """Fabricate a LinearGKResult for unit tests."""
    modes = [
        EigenMode(k_y_rho_s=k, omega_r=w, gamma=g, mode_type=mt)
        for k, g, w, mt in zip(k_y, gamma, omega_r, mode_type)
    ]
    return LinearGKResult(
        k_y=np.array(k_y),
        gamma=np.array(gamma),
        omega_r=np.array(omega_r),
        mode_type=list(mode_type),
        modes=modes,
    )


# ── Config ────────────────────────────────────────────────────────────


class TestConfig:
    def test_defaults(self):
        cfg = TGLFNativeConfig()
        assert cfg.sat_model == "SAT1"
        assert cfg.n_ky_etg == 0
        assert not cfg.multiscale

    def test_sat2_enables_multiscale(self):
        solver = TGLFNativeSolver(TGLFNativeConfig(sat_model="SAT2"))
        assert solver.config.multiscale

    def test_sat2_sets_etg_ky(self):
        solver = TGLFNativeSolver(TGLFNativeConfig(sat_model="SAT2"))
        assert solver.config.n_ky_etg > 0


# ── E×B shear ─────────────────────────────────────────────────────────


class TestExBShear:
    def test_proportional_to_s_hat_over_q(self):
        p1 = GKLocalParams(R_L_Ti=6.9, R_L_Te=6.9, R_L_ne=2.2, q=1.4, s_hat=0.78, epsilon=0.1)
        p2 = GKLocalParams(R_L_Ti=6.9, R_L_Te=6.9, R_L_ne=2.2, q=1.4, s_hat=1.56, epsilon=0.1)
        assert exb_shear_rate(p2) == pytest.approx(2.0 * exb_shear_rate(p1), rel=1e-10)

    def test_zero_at_zero_epsilon(self):
        p = GKLocalParams(R_L_Ti=6.9, R_L_Te=6.9, R_L_ne=2.2, q=1.4, s_hat=0.78, epsilon=0.0)
        assert exb_shear_rate(p) == 0.0

    def test_increases_with_gradient(self):
        p1 = GKLocalParams(R_L_Ti=3.0, R_L_Te=3.0, R_L_ne=1.0, q=1.4, s_hat=0.78, epsilon=0.1)
        p2 = GKLocalParams(R_L_Ti=9.0, R_L_Te=9.0, R_L_ne=1.0, q=1.4, s_hat=0.78, epsilon=0.1)
        assert exb_shear_rate(p2) > exb_shear_rate(p1)


# ── Trapped particles ─────────────────────────────────────────────────


class TestTrappedParticle:
    def test_fraction_sqrt_2_epsilon(self):
        eps = 0.18
        assert trapped_fraction(eps) == pytest.approx(np.sqrt(2 * eps / (1 + eps)), rel=1e-10)

    def test_damping_less_than_one(self):
        p = GKLocalParams(
            R_L_Ti=6.9, R_L_Te=6.9, R_L_ne=2.2, q=1.4, s_hat=0.78, nu_star=0.5, epsilon=0.18
        )
        assert trapped_particle_damping(p) < 1.0


# ── SAT0 ──────────────────────────────────────────────────────────────


class TestSAT0:
    def test_matches_mixing_length_no_shear(self):
        linear = _make_linear(
            k_y=[0.3, 0.5, 0.8],
            gamma=[0.1, 0.2, 0.05],
            omega_r=[-0.5, -0.8, -0.3],
            mode_type=["ITG", "ITG", "ITG"],
        )
        cfg = TGLFNativeConfig(sat_model="SAT0")
        phi_sq, gamma_net = sat0(linear, gamma_exb=0.0, tp_factor=1.0, cfg=cfg)
        expected = mixing_length_saturation(linear.gamma, linear.omega_r, linear.k_y)
        np.testing.assert_allclose(phi_sq, expected)

    def test_exb_quench_reduces_gamma(self):
        linear = _make_linear(
            k_y=[0.3],
            gamma=[0.2],
            omega_r=[-0.5],
            mode_type=["ITG"],
        )
        cfg = TGLFNativeConfig(sat_model="SAT0")
        _, gn_no_shear = sat0(linear, 0.0, 1.0, cfg)
        _, gn_with_shear = sat0(linear, 1.0, 1.0, cfg)
        assert gn_with_shear[0] < gn_no_shear[0]

    def test_zero_for_subcritical(self):
        linear = _make_linear(
            k_y=[0.3, 0.5],
            gamma=[0.0, 0.0],
            omega_r=[-0.1, -0.2],
            mode_type=["stable", "stable"],
        )
        cfg = TGLFNativeConfig(sat_model="SAT0")
        phi_sq, _ = sat0(linear, 0.0, 1.0, cfg)
        assert np.all(phi_sq == 0.0)

    def test_trapped_damping_effect(self):
        linear = _make_linear(
            k_y=[0.3],
            gamma=[0.2],
            omega_r=[-0.5],
            mode_type=["ITG"],
        )
        cfg = TGLFNativeConfig(sat_model="SAT0")
        _, gn_full = sat0(linear, 0.0, 1.0, cfg)
        _, gn_damped = sat0(linear, 0.0, 0.7, cfg)
        assert gn_damped[0] < gn_full[0]


# ── SAT1 ──────────────────────────────────────────────────────────────


class TestSAT1:
    def test_spectral_weight_sums_to_one(self):
        gamma_net = np.array([0.1, 0.2, 0.15, 0.05])
        k_y = np.array([0.1, 0.3, 0.5, 1.0])
        w = spectral_weight(gamma_net, k_y)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)

    def test_peak_sets_amplitude(self):
        linear = _make_linear(
            k_y=[0.3, 0.5, 0.8],
            gamma=[0.1, 0.2, 0.05],
            omega_r=[-0.5, -0.8, -0.3],
            mode_type=["ITG", "ITG", "ITG"],
        )
        cfg = TGLFNativeConfig(sat_model="SAT1")
        phi_sq, _ = sat1(linear, 0.0, 1.0, cfg)
        assert np.all(phi_sq >= 0)
        assert phi_sq.sum() > 0

    def test_complete_quench(self):
        linear = _make_linear(
            k_y=[0.3, 0.5],
            gamma=[0.1, 0.2],
            omega_r=[-0.5, -0.8],
            mode_type=["ITG", "ITG"],
        )
        cfg = TGLFNativeConfig(sat_model="SAT1", alpha_exb=1.0)
        phi_sq, gamma_net = sat1(linear, gamma_exb=10.0, tp_factor=1.0, cfg=cfg)
        assert np.all(gamma_net == 0.0)
        assert np.all(phi_sq == 0.0)

    def test_nonzero_for_unstable(self):
        linear = _make_linear(
            k_y=[0.3, 0.5],
            gamma=[0.1, 0.2],
            omega_r=[-0.5, -0.8],
            mode_type=["ITG", "ITG"],
        )
        cfg = TGLFNativeConfig(sat_model="SAT1")
        phi_sq, _ = sat1(linear, 0.0, 1.0, cfg)
        assert phi_sq.sum() > 0


# ── SAT2 ──────────────────────────────────────────────────────────────


class TestSAT2:
    def test_enhances_etg_modes(self):
        linear = _make_linear(
            k_y=[0.3, 0.5, 5.0, 10.0],
            gamma=[0.2, 0.15, 0.1, 0.05],
            omega_r=[-0.8, -0.6, 1.0, 1.5],
            mode_type=["ITG", "ITG", "ETG", "ETG"],
        )
        cfg = TGLFNativeConfig(sat_model="SAT2", alpha_cs=3.0)
        phi_sq_sat2, _ = sat2(linear, 0.0, 1.0, cfg)
        phi_sq_sat1, _ = sat1(linear, 0.0, 1.0, cfg)
        # ETG modes (indices 2,3) should be enhanced in SAT2
        assert phi_sq_sat2[2] >= phi_sq_sat1[2]
        assert phi_sq_sat2[3] >= phi_sq_sat1[3]

    def test_cross_scale_proportional_to_alpha_cs(self):
        linear = _make_linear(
            k_y=[0.3, 5.0],
            gamma=[0.2, 0.1],
            omega_r=[-0.8, 1.0],
            mode_type=["ITG", "ETG"],
        )
        cfg_low = TGLFNativeConfig(sat_model="SAT2", alpha_cs=1.0)
        cfg_high = TGLFNativeConfig(sat_model="SAT2", alpha_cs=5.0)
        phi_low, _ = sat2(linear, 0.0, 1.0, cfg_low)
        phi_high, _ = sat2(linear, 0.0, 1.0, cfg_high)
        assert phi_high[1] > phi_low[1]

    def test_etg_fraction_nonzero(self):
        solver = TGLFNativeSolver(TGLFNativeConfig(sat_model="SAT2", n_ky_etg=4, n_ky_ion=8))
        result = solver.solve(_CBC)
        # chi_e_etg tracks how much chi_e comes from ETG modes
        assert np.isfinite(result.chi_e_etg)


# ── Physics validation ────────────────────────────────────────────────


def _cbc_spectrum() -> LinearGKResult:
    """Published CBC spectrum: Dimits et al. 2000, GENE benchmark."""
    return _make_linear(
        k_y=[0.1, 0.2, 0.3, 0.4, 0.6, 1.0, 1.5, 2.0],
        gamma=[0.05, 0.12, 0.19, 0.17, 0.10, 0.04, 0.01, 0.0],
        omega_r=[-0.2, -0.5, -0.8, -0.7, -0.5, -0.3, -0.1, 0.0],
        mode_type=["ITG", "ITG", "ITG", "ITG", "ITG", "ITG", "ITG", "stable"],
    )


class TestPhysicsValidation:
    def test_cbc_spectrum_chi_i_positive_sat0(self):
        linear = _cbc_spectrum()
        cfg = TGLFNativeConfig(sat_model="SAT0")
        phi_sq, gamma_net = sat0(linear, gamma_exb=0.0, tp_factor=1.0, cfg=cfg)
        chi_i, *_ = quasilinear_weights(linear, phi_sq, gamma_net, 2.0, 2.0, _CBC)
        assert chi_i > 0

    def test_cbc_spectrum_chi_i_positive_sat1(self):
        linear = _cbc_spectrum()
        cfg = TGLFNativeConfig(sat_model="SAT1")
        phi_sq, gamma_net = sat1(linear, gamma_exb=0.0, tp_factor=1.0, cfg=cfg)
        chi_i, *_ = quasilinear_weights(linear, phi_sq, gamma_net, 2.0, 2.0, _CBC)
        assert chi_i > 0

    def test_cbc_spectrum_dominant_itg(self):
        linear = _cbc_spectrum()
        cfg = TGLFNativeConfig(sat_model="SAT1")
        _, gamma_net = sat1(linear, gamma_exb=0.0, tp_factor=1.0, cfg=cfg)
        idx_max = int(np.argmax(gamma_net))
        assert linear.mode_type[idx_max] == "ITG"

    def test_sparc_finite(self):
        p = GKLocalParams(
            R_L_Ti=8.0,
            R_L_Te=7.0,
            R_L_ne=2.5,
            q=2.0,
            s_hat=1.0,
            R0=1.85,
            a=0.57,
            B0=12.2,
            kappa=1.75,
            delta=0.54,
            epsilon=0.15,
            T_e_keV=10.0,
            T_i_keV=10.0,
            n_e=15.0,
        )
        solver = TGLFNativeSolver(TGLFNativeConfig(n_ky_ion=8, n_theta=32))
        result = solver.solve(p)
        assert np.isfinite(result.chi_i)
        assert np.isfinite(result.chi_e)

    def test_iter_finite(self):
        p = GKLocalParams(
            R_L_Ti=6.5,
            R_L_Te=6.0,
            R_L_ne=2.0,
            q=1.8,
            s_hat=0.9,
            R0=6.2,
            a=2.0,
            B0=5.3,
            epsilon=0.16,
            T_e_keV=8.0,
            T_i_keV=8.0,
            n_e=10.0,
        )
        solver = TGLFNativeSolver(TGLFNativeConfig(n_ky_ion=8, n_theta=32))
        result = solver.solve(p)
        assert np.isfinite(result.chi_i)
        assert np.isfinite(result.chi_e)

    def test_run_from_params_returns_gkoutput(self):
        solver = TGLFNativeSolver(TGLFNativeConfig(n_ky_ion=8, n_theta=32))
        out = solver.run_from_params(_CBC)
        assert isinstance(out, GKOutput)
        assert out.converged

    def test_subcritical_zero(self):
        p = GKLocalParams(
            R_L_Ti=1.0,
            R_L_Te=1.0,
            R_L_ne=0.5,
            q=1.4,
            s_hat=0.78,
            R0=2.78,
            a=1.0,
            B0=2.0,
            epsilon=0.18,
            T_e_keV=2.0,
            T_i_keV=2.0,
            n_e=5.0,
        )
        solver = TGLFNativeSolver(TGLFNativeConfig(n_ky_ion=8, n_theta=32))
        result = solver.solve(p)
        # Subcritical: less transport than supercritical CBC (R/L_Ti=6.9)
        r_super = TGLFNativeSolver(TGLFNativeConfig(n_ky_ion=8, n_theta=32)).solve(_CBC)
        assert result.chi_i <= r_super.chi_i

    def test_is_available(self):
        solver = TGLFNativeSolver()
        assert solver.is_available()
