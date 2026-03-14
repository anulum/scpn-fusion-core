# ──────────────────────────────────────────────────────────────────────
# Property-based tests for phase/ module
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from scpn_fusion.phase.kuramoto import (
    kuramoto_sakaguchi_step,
    order_parameter,
    lyapunov_v,
    wrap_phase,
)
from scpn_fusion.phase.knm import KnmSpec, build_knm_paper27
from scpn_fusion.phase.upde import UPDESystem


# ── Strategies ───────────────────────────────────────────────────────

phase_arrays = st.integers(min_value=2, max_value=200).flatmap(
    lambda n: arrays(np.float64, n, elements=st.floats(-np.pi, np.pi))
)

omega_arrays = st.integers(min_value=2, max_value=200).flatmap(
    lambda n: arrays(np.float64, n, elements=st.floats(-5.0, 5.0))
)


# ── Order parameter invariants ───────────────────────────────────────


@given(theta=phase_arrays)
@settings(max_examples=100)
def test_order_parameter_range(theta):
    """R must be in [0, 1] for any input phases."""
    assume(np.all(np.isfinite(theta)))
    R, Psi = order_parameter(theta)
    assert 0.0 <= R <= 1.0 + 1e-12, f"R={R} out of bounds"
    assert -np.pi <= Psi <= np.pi + 1e-12, f"Psi={Psi} out of bounds"


@given(n=st.integers(min_value=2, max_value=500))
@settings(max_examples=30)
def test_uniform_phases_low_R(n):
    """Uniformly spaced phases should give R ≈ 0."""
    theta = np.linspace(-np.pi, np.pi, n, endpoint=False)
    R, _ = order_parameter(theta)
    assert R < 0.2, f"Uniform phases gave R={R}, expected near 0"


@given(n=st.integers(min_value=2, max_value=500), phase=st.floats(-np.pi, np.pi))
@settings(max_examples=30)
def test_synchronized_phases_high_R(n, phase):
    """All oscillators at the same phase should give R ≈ 1."""
    assume(np.isfinite(phase))
    theta = np.full(n, phase)
    R, _ = order_parameter(theta)
    assert R > 0.99, f"Synchronized phases gave R={R}, expected ~1"


# ── Lyapunov V invariants ────────────────────────────────────────────


@given(theta=phase_arrays)
@settings(max_examples=50)
def test_lyapunov_v_nonnegative(theta):
    """V(t) >= 0 always (it's a sum of 1-cos terms)."""
    assume(np.all(np.isfinite(theta)))
    V = lyapunov_v(theta, psi=0.0)
    assert V >= -1e-12, f"V={V} is negative"


@given(theta=phase_arrays)
@settings(max_examples=50)
def test_lyapunov_v_bounded(theta):
    """V(t) <= 2 (maximum when all phases are anti-aligned to Ψ)."""
    assume(np.all(np.isfinite(theta)))
    V = lyapunov_v(theta, psi=0.0)
    assert V <= 2.0 + 1e-12, f"V={V} exceeds theoretical max of 2"


# ── Phase wrap invariant ─────────────────────────────────────────────


@given(theta=phase_arrays)
@settings(max_examples=100)
def test_wrap_phase_range(theta):
    """Wrapped phases must be in [-pi, pi]."""
    assume(np.all(np.isfinite(theta)))
    wrapped = wrap_phase(theta)
    assert np.all(wrapped >= -np.pi - 1e-12)
    assert np.all(wrapped <= np.pi + 1e-12)


# ── Kuramoto step invariants ─────────────────────────────────────────


@given(
    n=st.integers(min_value=5, max_value=100),
    K=st.floats(0.1, 10.0),
    zeta=st.floats(0.0, 2.0),
)
@settings(max_examples=30)
def test_kuramoto_step_preserves_array_shape(n, K, zeta):
    """Output theta1 must have same shape as input."""
    rng = np.random.default_rng(42)
    theta = rng.uniform(-np.pi, np.pi, n)
    omega = rng.normal(0, 0.3, n)
    result = kuramoto_sakaguchi_step(
        theta,
        omega,
        dt=1e-3,
        K=K,
        zeta=zeta,
        psi_driver=0.0,
        psi_mode="external",
    )
    assert result["theta1"].shape == theta.shape


# ── KnmSpec symmetry ─────────────────────────────────────────────────


@given(
    L=st.integers(min_value=2, max_value=16),
    zeta=st.floats(0.0, 2.0),
)
@settings(max_examples=20)
def test_knm_symmetric(L, zeta):
    """Knm coupling matrix must be symmetric."""
    spec = build_knm_paper27(L=L, zeta_uniform=zeta)
    K = np.asarray(spec.K)
    assert K.shape == (L, L)
    np.testing.assert_allclose(K, K.T, atol=1e-12)


@given(L=st.integers(min_value=2, max_value=16))
@settings(max_examples=20)
def test_knm_nonnegative(L):
    """All coupling strengths must be >= 0."""
    spec = build_knm_paper27(L=L)
    K = np.asarray(spec.K)
    assert np.all(K >= 0), f"Negative coupling found: min={K.min()}"


# ── UPDE step invariants ─────────────────────────────────────────────


@given(
    L=st.integers(min_value=2, max_value=8),
    N_per=st.integers(min_value=5, max_value=30),
)
@settings(max_examples=15, deadline=10000)
def test_upde_step_r_range(L, N_per):
    """R_global from UPDE step must be in [0, 1]."""
    spec = build_knm_paper27(L=L, zeta_uniform=0.5)
    upde = UPDESystem(spec=spec, dt=1e-3, psi_mode="external")
    rng = np.random.default_rng(42)
    theta = [rng.uniform(-np.pi, np.pi, N_per) for _ in range(L)]
    omega = [rng.normal(0, 0.3, N_per) for _ in range(L)]
    out = upde.step(theta, omega, psi_driver=0.0)
    assert 0.0 <= out["R_global"] <= 1.0 + 1e-12


@given(
    L=st.integers(min_value=2, max_value=8),
    N_per=st.integers(min_value=5, max_value=30),
)
@settings(max_examples=15, deadline=10000)
def test_upde_step_output_shapes(L, N_per):
    """UPDE step outputs must have correct shapes."""
    spec = build_knm_paper27(L=L, zeta_uniform=0.5)
    upde = UPDESystem(spec=spec, dt=1e-3, psi_mode="external")
    rng = np.random.default_rng(42)
    theta = [rng.uniform(-np.pi, np.pi, N_per) for _ in range(L)]
    omega = [rng.normal(0, 0.3, N_per) for _ in range(L)]
    out = upde.step(theta, omega, psi_driver=0.0)
    assert len(out["theta1"]) == L
    assert all(t.shape == (N_per,) for t in out["theta1"])
    assert out["R_layer"].shape == (L,)
    assert out["V_layer"].shape == (L,)


# ── KnmSpec constructor validation ──────────────────────────────────


class TestKnmSpecValidation:
    def test_rejects_non_square_K(self):
        with pytest.raises(ValueError, match="square"):
            KnmSpec(K=np.ones((3, 4)))

    def test_rejects_1d_K(self):
        with pytest.raises(ValueError, match="square"):
            KnmSpec(K=np.ones(4))

    def test_rejects_alpha_shape_mismatch(self):
        with pytest.raises(ValueError, match="alpha shape"):
            KnmSpec(K=np.eye(3), alpha=np.zeros((2, 2)))

    def test_rejects_zeta_shape_mismatch(self):
        with pytest.raises(ValueError, match="zeta shape"):
            KnmSpec(K=np.eye(4), zeta=np.zeros(3))

    def test_rejects_layer_names_length_mismatch(self):
        with pytest.raises(ValueError, match="layer_names"):
            KnmSpec(K=np.eye(3), layer_names=["a", "b"])

    def test_valid_full_spec(self):
        spec = KnmSpec(
            K=np.eye(3),
            alpha=np.zeros((3, 3)),
            zeta=np.ones(3),
            layer_names=["L1", "L2", "L3"],
        )
        assert spec.L == 3

    def test_L_property(self):
        spec = KnmSpec(K=np.eye(5))
        assert spec.L == 5


class TestBuildKnmPaper27:
    def test_small_L_skips_cross_hierarchy_boosts(self):
        spec = build_knm_paper27(L=4)
        assert spec.L == 4
        assert spec.K.shape == (4, 4)

    def test_calibration_anchors_applied(self):
        spec = build_knm_paper27(L=16)
        K = np.asarray(spec.K)
        assert K[0, 1] == pytest.approx(0.302)
        assert K[1, 0] == pytest.approx(0.302)
        assert K[1, 2] == pytest.approx(0.201)
        assert K[2, 3] == pytest.approx(0.252)

    def test_cross_hierarchy_l1_l16(self):
        spec = build_knm_paper27(L=16)
        K = np.asarray(spec.K)
        assert K[0, 15] >= 0.05
        assert K[15, 0] >= 0.05

    def test_cross_hierarchy_l5_l7(self):
        spec = build_knm_paper27(L=16)
        K = np.asarray(spec.K)
        assert K[4, 6] >= 0.15
        assert K[6, 4] >= 0.15

    def test_zeta_none_when_uniform_zero(self):
        spec = build_knm_paper27(L=8, zeta_uniform=0.0)
        assert spec.zeta is None

    def test_zeta_populated_when_nonzero(self):
        spec = build_knm_paper27(L=8, zeta_uniform=0.3)
        assert spec.zeta is not None
        np.testing.assert_allclose(spec.zeta, 0.3)
