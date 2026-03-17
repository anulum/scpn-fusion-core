# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — UPDE System Tests
"""
Focused tests for the UPDESystem class beyond what test_phase_kuramoto.py
covers.  Emphasis on edge cases, K_override, actuation_gain, alpha/zeta
interactions, and Lyapunov trajectory tracking.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.phase.knm import KnmSpec, build_knm_paper27
from scpn_fusion.phase.upde import UPDESystem


def _layers(L, N, seed=42):
    rng = np.random.default_rng(seed)
    theta = [rng.uniform(-np.pi, np.pi, N) for _ in range(L)]
    omega = [rng.normal(0, 0.3, N) for _ in range(L)]
    return theta, omega


# ── Construction ─────────────────────────────────────────────────────


class TestUPDEConstruction:
    def test_default_fields(self):
        spec = build_knm_paper27(L=3)
        sys = UPDESystem(spec=spec)
        assert sys.dt == 1e-3
        assert sys.psi_mode == "external"
        assert sys.wrap is True

    def test_custom_dt(self):
        spec = build_knm_paper27(L=3)
        sys = UPDESystem(spec=spec, dt=0.01)
        assert sys.dt == 0.01


# ── Step edge cases ──────────────────────────────────────────────────


class TestUPDEStep:
    def test_external_psi_required(self):
        spec = build_knm_paper27(L=2)
        sys = UPDESystem(spec=spec, psi_mode="external")
        theta, omega = _layers(2, 10)
        with pytest.raises(ValueError, match="psi_driver"):
            sys.step(theta, omega)

    def test_unknown_psi_mode(self):
        spec = build_knm_paper27(L=2)
        sys = UPDESystem(spec=spec, psi_mode="invalid")
        theta, omega = _layers(2, 10)
        with pytest.raises(ValueError, match="Unknown psi_mode"):
            sys.step(theta, omega, psi_driver=0.0)

    def test_global_mean_field_mode(self):
        spec = build_knm_paper27(L=3)
        sys = UPDESystem(spec=spec, psi_mode="global_mean_field")
        theta, omega = _layers(3, 15)
        out = sys.step(theta, omega)
        assert "theta1" in out
        assert len(out["theta1"]) == 3

    def test_no_wrap_option(self):
        K = np.eye(2) * 0.1
        spec = KnmSpec(K=K)
        sys = UPDESystem(spec=spec, dt=0.1, psi_mode="global_mean_field", wrap=False)
        theta = [np.array([3.0, 3.1]), np.array([3.0, 3.1])]
        omega = [np.array([1.0, 1.0]), np.array([1.0, 1.0])]
        out = sys.step(theta, omega)
        # Without wrap, phases can exceed pi
        for t in out["theta1"]:
            assert t.max() > 3.0

    def test_actuation_gain_zero_freezes_coupling(self):
        spec = build_knm_paper27(L=2, zeta_uniform=0.0)
        sys = UPDESystem(spec=spec, dt=0.01, psi_mode="global_mean_field")
        theta, omega = _layers(2, 20)
        # With gain=0, only omega drives evolution
        out = sys.step(theta, omega, actuation_gain=0.0)
        for m in range(2):
            expected = np.asarray(theta[m]) + 0.01 * np.asarray(omega[m])
            # Wrapped comparison
            diff = np.abs(out["theta1"][m] - ((expected + np.pi) % (2 * np.pi) - np.pi))
            assert np.all(diff < 1e-10)

    def test_K_override(self):
        L = 3
        spec = build_knm_paper27(L=L)
        sys = UPDESystem(spec=spec, dt=0.005, psi_mode="global_mean_field")
        theta, omega = _layers(L, 20, seed=7)

        K_zeros = np.zeros((L, L))
        out_zero = sys.step(theta, omega, K_override=K_zeros)
        out_default = sys.step(theta, omega)

        # K_override=zeros should differ from default K
        diff = sum(
            float(np.max(np.abs(a - b))) for a, b in zip(out_zero["theta1"], out_default["theta1"])
        )
        assert diff > 0.0

    def test_alpha_matrix(self):
        L = 2
        K = np.ones((L, L)) * 0.5
        alpha = np.ones((L, L)) * 0.3
        spec = KnmSpec(K=K, alpha=alpha)
        sys = UPDESystem(spec=spec, dt=0.01, psi_mode="global_mean_field")
        theta, omega = _layers(L, 20)
        out = sys.step(theta, omega)
        assert len(out["theta1"]) == L

    def test_zeta_per_layer(self):
        L = 3
        K = np.eye(L) * 0.1
        zeta = np.array([0.0, 1.0, 2.0])
        spec = KnmSpec(K=K, zeta=zeta)
        sys = UPDESystem(spec=spec, dt=0.01, psi_mode="external")
        theta = [np.array([1.0, 2.0]) for _ in range(L)]
        omega = [np.zeros(2) for _ in range(L)]
        out = sys.step(theta, omega, psi_driver=0.0)
        # Layer 0 (zeta=0) should differ less from initial than layer 2 (zeta=2)
        d0 = float(np.max(np.abs(out["theta1"][0] - theta[0])))
        d2 = float(np.max(np.abs(out["theta1"][2] - theta[2])))
        assert d2 > d0

    def test_output_keys(self):
        spec = build_knm_paper27(L=2, zeta_uniform=0.5)
        sys = UPDESystem(spec=spec, dt=1e-3, psi_mode="external")
        theta, omega = _layers(2, 10)
        out = sys.step(theta, omega, psi_driver=0.0)
        for key in (
            "theta1",
            "R_layer",
            "Psi_layer",
            "R_global",
            "Psi_global",
            "V_layer",
            "V_global",
        ):
            assert key in out

    def test_r_global_in_unit_interval(self):
        spec = build_knm_paper27(L=4)
        sys = UPDESystem(spec=spec, dt=1e-3, psi_mode="global_mean_field")
        theta, omega = _layers(4, 30)
        out = sys.step(theta, omega)
        assert 0.0 <= out["R_global"] <= 1.0

    def test_v_global_in_range(self):
        spec = build_knm_paper27(L=4, zeta_uniform=1.0)
        sys = UPDESystem(spec=spec, dt=1e-3, psi_mode="external")
        theta, omega = _layers(4, 30)
        out = sys.step(theta, omega, psi_driver=0.0)
        assert 0.0 <= out["V_global"] <= 2.0


# ── run() ────────────────────────────────────────────────────────────


class TestUPDERun:
    def test_shape(self):
        spec = build_knm_paper27(L=3)
        sys = UPDESystem(spec=spec, dt=1e-3, psi_mode="global_mean_field")
        theta, omega = _layers(3, 10)
        result = sys.run(50, theta, omega)
        assert result["R_layer_hist"].shape == (50, 3)
        assert result["R_global_hist"].shape == (50,)
        assert len(result["theta_final"]) == 3

    def test_convergence_strong_coupling(self):
        L = 2
        K = np.eye(L) * 5.0
        spec = KnmSpec(K=K, zeta=np.array([2.0, 2.0]))
        sys = UPDESystem(spec=spec, dt=0.005, psi_mode="external")
        theta, omega = _layers(L, 40, seed=7)
        omega = [np.zeros(40) for _ in range(L)]
        result = sys.run(500, theta, omega, psi_driver=0.0)
        assert result["R_global_hist"][-1] > result["R_global_hist"][0]


# ── run_lyapunov() ───────────────────────────────────────────────────


class TestUPDERunLyapunov:
    def test_returns_lambda(self):
        spec = build_knm_paper27(L=3, zeta_uniform=2.0)
        sys = UPDESystem(spec=spec, dt=0.005, psi_mode="external")
        theta, omega = _layers(3, 20)
        omega = [np.zeros(20) for _ in range(3)]
        result = sys.run_lyapunov(100, theta, omega, psi_driver=0.0)
        assert result["lambda_layer"].shape == (3,)
        assert isinstance(result["lambda_global"], float)
        assert result["V_layer_hist"].shape == (100, 3)
        assert result["V_global_hist"].shape == (100,)

    def test_stable_system_negative_lambda(self):
        spec = build_knm_paper27(L=2, zeta_uniform=3.0)
        sys = UPDESystem(spec=spec, dt=0.005, psi_mode="external")
        theta = [np.array([0.5, 0.6, 0.7]) for _ in range(2)]
        omega = [np.zeros(3) for _ in range(2)]
        result = sys.run_lyapunov(200, theta, omega, psi_driver=0.5)
        assert result["lambda_global"] < 0.0

    def test_pac_gamma_changes_lambda(self):
        spec = build_knm_paper27(L=3, zeta_uniform=1.0)
        sys = UPDESystem(spec=spec, dt=0.005, psi_mode="external")
        theta, omega = _layers(3, 15, seed=11)
        omega = [np.zeros(15) for _ in range(3)]
        r0 = sys.run_lyapunov(80, theta, omega, psi_driver=0.0, pac_gamma=0.0)
        r1 = sys.run_lyapunov(80, theta, omega, psi_driver=0.0, pac_gamma=2.0)
        assert r0["lambda_global"] != pytest.approx(r1["lambda_global"], abs=1e-8)
