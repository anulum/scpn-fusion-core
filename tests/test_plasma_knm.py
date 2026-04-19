# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Plasma Knm coupling matrix tests
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scpn_fusion.phase.plasma_knm import (
    OMEGA_PLASMA_8,
    PLASMA_LAYER_NAMES,
    build_knm_plasma,
    build_knm_plasma_from_config,
    plasma_omega,
)
from scpn_fusion.phase.upde import UPDESystem

# ── Matrix invariants ───────────────────────────────────────────────


class TestPlasmaKnmInvariants:
    @pytest.mark.parametrize("mode", ["baseline", "elm", "ntm", "sawtooth", "hybrid"])
    def test_symmetric(self, mode):
        spec = build_knm_plasma(mode=mode)
        K = np.asarray(spec.K)
        np.testing.assert_allclose(K, K.T, atol=1e-14)

    @pytest.mark.parametrize("mode", ["baseline", "elm", "ntm", "sawtooth", "hybrid"])
    def test_nonnegative(self, mode):
        spec = build_knm_plasma(mode=mode)
        assert np.all(np.asarray(spec.K) >= 0.0)

    def test_shape_default(self):
        spec = build_knm_plasma()
        assert spec.K.shape == (8, 8)
        assert spec.L == 8

    @pytest.mark.parametrize("L", [2, 4, 6, 8])
    def test_shape_parametric(self, L):
        spec = build_knm_plasma(L=L)
        assert spec.K.shape == (L, L)

    def test_layer_names_default(self):
        spec = build_knm_plasma()
        assert spec.layer_names is not None
        assert len(spec.layer_names) == 8
        assert spec.layer_names[0] == "micro_turbulence"
        assert spec.layer_names[7] == "plasma_wall"

    def test_layer_names_truncated(self):
        spec = build_knm_plasma(L=4)
        assert len(spec.layer_names) == 4

    def test_layer_names_custom(self):
        names = ["a", "b", "c"]
        spec = build_knm_plasma(L=3, layer_names=names)
        assert list(spec.layer_names) == names


# ── Physics coupling values ─────────────────────────────────────────


class TestPhysicsCouplings:
    def test_micro_zonal_coupling(self):
        """Diamond 2005: drift-wave / zonal-flow predator-prey."""
        spec = build_knm_plasma()
        K = np.asarray(spec.K)
        assert K[0, 1] == pytest.approx(0.42, abs=1e-10)
        assert K[1, 0] == pytest.approx(0.42, abs=1e-10)

    def test_zonal_transport_coupling(self):
        """Terry 2000: E×B shear suppression."""
        spec = build_knm_plasma()
        K = np.asarray(spec.K)
        assert K[1, 4] == pytest.approx(0.28, abs=1e-10)

    def test_ntm_current_coupling(self):
        """La Haye 2006: NTM ↔ bootstrap current."""
        spec = build_knm_plasma()
        K = np.asarray(spec.K)
        assert K[2, 5] == pytest.approx(0.35, abs=1e-10)

    def test_sawtooth_current_coupling(self):
        """Porcelli 1996: sawtooth ↔ current redistribution."""
        spec = build_knm_plasma()
        K = np.asarray(spec.K)
        assert K[3, 5] == pytest.approx(0.30, abs=1e-10)

    def test_elm_transport_coupling(self):
        """ELM crash depletes pedestal."""
        spec = build_knm_plasma()
        K = np.asarray(spec.K)
        assert K[3, 4] == pytest.approx(0.32, abs=1e-10)

    def test_transport_equilibrium_coupling(self):
        spec = build_knm_plasma()
        K = np.asarray(spec.K)
        assert K[4, 6] == pytest.approx(0.25, abs=1e-10)

    def test_pwi_edge_coupling(self):
        spec = build_knm_plasma()
        K = np.asarray(spec.K)
        assert K[7, 4] == pytest.approx(0.20, abs=1e-10)


# ── Mode biases ─────────────────────────────────────────────────────


class TestModeBias:
    def test_elm_mode_amplifies_sawtooth_transport(self):
        base = build_knm_plasma(mode="baseline")
        elm = build_knm_plasma(mode="elm")
        assert np.asarray(elm.K)[3, 4] > np.asarray(base.K)[3, 4]

    def test_ntm_mode_amplifies_mhd_current(self):
        base = build_knm_plasma(mode="baseline")
        ntm = build_knm_plasma(mode="ntm")
        assert np.asarray(ntm.K)[2, 5] > np.asarray(base.K)[2, 5]

    def test_sawtooth_mode_amplifies_saw_current(self):
        base = build_knm_plasma(mode="baseline")
        saw = build_knm_plasma(mode="sawtooth")
        assert np.asarray(saw.K)[3, 5] > np.asarray(base.K)[3, 5]

    def test_hybrid_mode_elevates_all(self):
        base = build_knm_plasma(mode="baseline")
        hybrid = build_knm_plasma(mode="hybrid")
        K_b = np.asarray(base.K)
        K_h = np.asarray(hybrid.K)
        # Overall sum increases
        assert K_h.sum() > K_b.sum()

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown plasma mode"):
            build_knm_plasma(mode="nonsense")


# ── Custom overrides ────────────────────────────────────────────────


class TestCustomOverrides:
    def test_override_applies(self):
        spec = build_knm_plasma(custom_overrides={(0, 7): 0.99})
        K = np.asarray(spec.K)
        assert K[0, 7] == pytest.approx(0.99, abs=1e-10)
        assert K[7, 0] == pytest.approx(0.99, abs=1e-10)

    def test_override_out_of_range_raises(self):
        with pytest.raises(IndexError, match="out of range"):
            build_knm_plasma(custom_overrides={(0, 99): 0.5})


# ── Zeta / global driver ───────────────────────────────────────────


class TestZeta:
    def test_zeta_none_default(self):
        spec = build_knm_plasma()
        assert spec.zeta is None

    def test_zeta_populated(self):
        spec = build_knm_plasma(zeta_uniform=0.5)
        assert spec.zeta is not None
        np.testing.assert_allclose(spec.zeta, 0.5)


# ── Omega frequencies ──────────────────────────────────────────────


class TestPlasmaOmega:
    def test_omega_8(self):
        w = plasma_omega(8)
        assert w.shape == (8,)
        np.testing.assert_array_equal(w, OMEGA_PLASMA_8)

    def test_omega_fewer_layers(self):
        w = plasma_omega(4)
        assert w.shape == (4,)
        np.testing.assert_array_equal(w, OMEGA_PLASMA_8[:4])

    def test_omega_more_layers_interpolated(self):
        w = plasma_omega(12)
        assert w.shape == (12,)
        assert w[0] == pytest.approx(OMEGA_PLASMA_8[0], rel=1e-6)
        assert w[-1] == pytest.approx(OMEGA_PLASMA_8[-1], rel=1e-6)
        # Monotonically decreasing (log-linear from fast to slow)
        assert np.all(np.diff(w) < 0)

    def test_omega_all_positive(self):
        for L in [2, 4, 8, 16]:
            w = plasma_omega(L)
            assert np.all(w > 0)


# ── Config-based builder ───────────────────────────────────────────


class TestBuildFromConfig:
    def test_iter_baseline(self):
        spec = build_knm_plasma_from_config(
            R0=6.2,
            a=2.0,
            B0=5.3,
            Ip=15.0,
            n_e=10.1,
        )
        assert spec.L == 8
        K = np.asarray(spec.K)
        np.testing.assert_allclose(K, K.T, atol=1e-14)
        assert np.all(K >= 0)

    def test_sparc_baseline(self):
        spec = build_knm_plasma_from_config(
            R0=1.85,
            a=0.57,
            B0=12.2,
            Ip=8.7,
            n_e=30.0,
        )
        assert spec.L == 8

    def test_high_beta_stronger_coupling(self):
        low_beta = build_knm_plasma_from_config(
            R0=6.2,
            a=2.0,
            B0=10.0,
            Ip=15.0,
            n_e=3.0,
        )
        high_beta = build_knm_plasma_from_config(
            R0=6.2,
            a=2.0,
            B0=2.0,
            Ip=15.0,
            n_e=20.0,
        )
        assert np.asarray(high_beta.K).sum() > np.asarray(low_beta.K).sum()

    def test_low_q_triggers_sawtooth(self):
        """q_cyl < 1 should auto-select sawtooth mode."""
        # q_cyl = 5 * a² * B0 / (R0 * Ip)
        # With a=0.5, B0=1.0, R0=6.0, Ip=5.0 → q_cyl ≈ 0.42
        spec = build_knm_plasma_from_config(
            R0=6.0,
            a=0.5,
            B0=1.0,
            Ip=5.0,
            n_e=5.0,
        )
        # Sawtooth-current coupling (P3↔P5) should be amplified
        K = np.asarray(spec.K)
        base = build_knm_plasma(mode="baseline")
        K_base = np.asarray(base.K)
        assert K[3, 5] > K_base[3, 5]


# ── UPDE integration ───────────────────────────────────────────────


class TestUPDEIntegration:
    def test_plasma_knm_drives_upde(self):
        """Plasma Knm plugs into UPDESystem and produces valid output."""
        spec = build_knm_plasma(mode="baseline", zeta_uniform=0.3)
        upde = UPDESystem(spec=spec, dt=1e-3, psi_mode="external")
        rng = np.random.default_rng(42)
        L = spec.L
        N_osc = 16
        theta = [rng.uniform(-np.pi, np.pi, N_osc) for _ in range(L)]
        omega_base = plasma_omega(L)
        omega = [omega_base[m] + rng.normal(0, 0.1, N_osc) for m in range(L)]
        out = upde.step(theta, omega, psi_driver=0.0)
        assert 0.0 <= out["R_global"] <= 1.0 + 1e-12
        assert len(out["theta1"]) == L
        assert out["R_layer"].shape == (L,)

    def test_plasma_knm_run_converges(self):
        """Multi-step UPDE run with plasma Knm should not diverge."""
        spec = build_knm_plasma(mode="baseline", zeta_uniform=0.5)
        upde = UPDESystem(spec=spec, dt=5e-4, psi_mode="external")
        rng = np.random.default_rng(7)
        L = spec.L
        N_osc = 32
        theta = [rng.uniform(-np.pi, np.pi, N_osc) for _ in range(L)]
        omega_base = plasma_omega(L)
        omega = [omega_base[m] + rng.normal(0, 0.05, N_osc) for m in range(L)]
        result = upde.run(200, theta, omega, psi_driver=0.0)
        R_hist = result["R_global_hist"]
        # Should not blow up
        assert np.all(np.isfinite(R_hist))
        assert np.all(R_hist >= 0.0)
        assert np.all(R_hist <= 1.0 + 1e-12)

    def test_plasma_lyapunov_run(self):
        """Lyapunov tracking with plasma Knm."""
        spec = build_knm_plasma(mode="elm", zeta_uniform=0.4)
        upde = UPDESystem(spec=spec, dt=1e-3, psi_mode="external")
        rng = np.random.default_rng(99)
        L = spec.L
        N_osc = 20
        theta = [rng.uniform(-np.pi, np.pi, N_osc) for _ in range(L)]
        omega_base = plasma_omega(L)
        omega = [omega_base[m] + rng.normal(0, 0.1, N_osc) for m in range(L)]
        result = upde.run_lyapunov(100, theta, omega, psi_driver=0.0)
        assert result["V_layer_hist"].shape == (100, L)
        assert np.all(np.isfinite(result["V_global_hist"]))
        assert result["lambda_layer"].shape == (L,)

    @pytest.mark.parametrize("mode", ["baseline", "elm", "ntm", "sawtooth", "hybrid"])
    def test_all_modes_run_stable(self, mode):
        """Every mode should produce stable UPDE trajectories."""
        spec = build_knm_plasma(mode=mode, zeta_uniform=0.3)
        upde = UPDESystem(spec=spec, dt=1e-3, psi_mode="external")
        rng = np.random.default_rng(123)
        L = spec.L
        theta = [rng.uniform(-np.pi, np.pi, 10) for _ in range(L)]
        omega = [plasma_omega(L)[m] * np.ones(10) for m in range(L)]
        result = upde.run(50, theta, omega, psi_driver=0.0)
        assert np.all(np.isfinite(result["R_global_hist"]))


# ── Hypothesis property tests ──────────────────────────────────────


@given(
    L=st.integers(min_value=2, max_value=8),
    K_base=st.floats(0.05, 1.0),
    mode=st.sampled_from(["baseline", "elm", "ntm", "sawtooth", "hybrid"]),
)
@settings(max_examples=40)
def test_plasma_knm_always_symmetric(L, K_base, mode):
    spec = build_knm_plasma(mode=mode, L=L, K_base=K_base)
    K = np.asarray(spec.K)
    np.testing.assert_allclose(K, K.T, atol=1e-14)


@given(
    L=st.integers(min_value=2, max_value=8),
    K_base=st.floats(0.05, 1.0),
)
@settings(max_examples=20)
def test_plasma_knm_always_nonneg(L, K_base):
    spec = build_knm_plasma(L=L, K_base=K_base)
    assert np.all(np.asarray(spec.K) >= 0.0)


# ── Constants ───────────────────────────────────────────────────────


class TestConstants:
    def test_omega_shape(self):
        assert OMEGA_PLASMA_8.shape == (8,)

    def test_omega_positive(self):
        assert np.all(OMEGA_PLASMA_8 > 0)

    def test_omega_monotone_decreasing(self):
        """Frequencies decrease from fast (micro) to slow (PWI)."""
        assert np.all(np.diff(OMEGA_PLASMA_8) < 0)

    def test_layer_names_count(self):
        assert len(PLASMA_LAYER_NAMES) == 8

    def test_layer_names_unique(self):
        assert len(set(PLASMA_LAYER_NAMES)) == 8
