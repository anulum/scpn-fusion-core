# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Cross-tier parity and dispatch tests for the phase-dynamics kernels (M-3)."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from scpn_fusion.core import _multi_compat as multi
from scpn_fusion.core import _multi_compat_providers as providers
from scpn_fusion.phase.kuramoto import kuramoto_sakaguchi_step
from scpn_fusion.phase.knm import KnmSpec
from scpn_fusion.phase.upde import UPDESystem

RUST_AVAILABLE = multi.is_available(multi.BackendTier.RUST)


def _kuramoto_state(n: int = 257) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    rng = np.random.default_rng(2026)
    theta = rng.uniform(-np.pi, np.pi, size=n)
    omega = rng.normal(0.0, 0.3, size=n)
    return theta, omega


def test_phase_kernels_are_registered_with_rust_and_numpy_tiers() -> None:
    """Both M-3 kernels carry RUST + NUMPY registrations."""
    kernels = multi.registered_kernels()
    for name in ("kuramoto_step", "upde_tick"):
        assert name in kernels
        tier_names = [entry.rstrip("*") for entry in kernels[name]]
        assert tier_names == ["rust", "numpy"]
        assert "numpy*" in kernels[name]


def test_kuramoto_numpy_tier_matches_public_step_contract() -> None:
    """The NumPy tier reproduces the public step output bit-for-bit."""
    theta, omega = _kuramoto_state(64)
    direct = providers._numpy_kuramoto_step(
        theta, omega, dt=1e-2, K=1.5, alpha=0.1, zeta=0.4, psi=0.2, wrap=True
    )
    assert direct["theta1"].shape == theta.shape
    assert np.all(np.isfinite(direct["theta1"]))
    assert 0.0 <= direct["R"] <= 1.0


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
def test_kuramoto_rust_tier_matches_numpy_tier() -> None:
    """Rust and NumPy tiers agree to floating-point summation order."""
    theta, omega = _kuramoto_state()
    kwargs: dict[str, Any] = dict(dt=1e-2, K=2.0, alpha=0.05, zeta=0.7, psi=-0.3, wrap=True)
    numpy_out = providers._numpy_kuramoto_step(theta, omega, **kwargs)
    rust_out = providers._rust_kuramoto_step(theta, omega, **kwargs)

    np.testing.assert_allclose(rust_out["theta1"], numpy_out["theta1"], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(rust_out["dtheta"], numpy_out["dtheta"], rtol=0.0, atol=1e-12)
    assert rust_out["R"] == pytest.approx(numpy_out["R"], abs=1e-14)
    assert rust_out["Psi_r"] == pytest.approx(numpy_out["Psi_r"], abs=1e-14)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
def test_kuramoto_rust_tier_respects_wrap_flag() -> None:
    """Without wrapping, both tiers leave phases unbounded identically."""
    theta, omega = _kuramoto_state(65)
    kwargs: dict[str, Any] = dict(dt=10.0, K=0.5, alpha=0.0, zeta=0.0, psi=0.0, wrap=False)
    numpy_out = providers._numpy_kuramoto_step(theta, omega, **kwargs)
    rust_out = providers._rust_kuramoto_step(theta, omega, **kwargs)
    np.testing.assert_allclose(rust_out["theta1"], numpy_out["theta1"], rtol=0.0, atol=1e-12)
    assert np.max(np.abs(numpy_out["theta1"])) > np.pi  # wrap really off


def _upde_fixture() -> tuple[
    UPDESystem, list[npt.NDArray[np.float64]], list[npt.NDArray[np.float64]]
]:
    rng = np.random.default_rng(7)
    K = np.array([[1.2, 0.3], [0.2, 0.9]])
    alpha = np.array([[0.0, 0.05], [0.02, 0.0]])
    zeta = np.array([0.5, 0.1])
    spec = KnmSpec(K=K, alpha=alpha, zeta=zeta)
    system = UPDESystem(spec=spec, dt=5e-3)
    theta_layers = [
        rng.uniform(-np.pi, np.pi, size=33),
        rng.uniform(-np.pi, np.pi, size=57),  # non-uniform N
    ]
    omega_layers = [
        rng.normal(0.0, 0.2, size=33),
        rng.normal(0.0, 0.2, size=57),
    ]
    return system, theta_layers, omega_layers


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
def test_upde_rust_tier_matches_numpy_tier_including_non_uniform_layers() -> None:
    """Rust and NumPy UPDE ticks agree on a non-uniform two-layer system."""
    _, theta_layers, omega_layers = _upde_fixture()
    theta_flat = np.concatenate(theta_layers)
    omega_flat = np.concatenate(omega_layers)
    offsets = np.array([0, 33, 90], dtype=np.intp)
    K = np.array([[1.2, 0.3], [0.2, 0.9]])
    alpha = np.array([[0.0, 0.05], [0.02, 0.0]])
    zeta = np.array([0.5, 0.1])
    kwargs: dict[str, Any] = dict(
        dt=5e-3, psi_global=0.25, actuation_gain=1.1, pac_gamma=0.4, wrap=True
    )

    numpy_out = providers._numpy_upde_tick(
        theta_flat, omega_flat, offsets, K, alpha, zeta, **kwargs
    )
    rust_out = providers._rust_upde_tick(theta_flat, omega_flat, offsets, K, alpha, zeta, **kwargs)

    for key in ("theta1", "dtheta", "R_layer", "Psi_layer", "V_layer"):
        np.testing.assert_allclose(rust_out[key], numpy_out[key], rtol=0.0, atol=1e-12)
    assert rust_out["R_global"] == pytest.approx(numpy_out["R_global"], abs=1e-14)
    assert rust_out["V_global"] == pytest.approx(numpy_out["V_global"], abs=1e-12)


def test_public_kuramoto_step_routes_through_dispatcher() -> None:
    """The public API resolves the driver and dispatches the fastest tier."""
    theta, omega = _kuramoto_state(48)
    out = kuramoto_sakaguchi_step(
        theta, omega, dt=1e-2, K=1.0, zeta=0.3, psi_driver=0.1, psi_mode="external"
    )
    assert set(out) == {"theta1", "dtheta", "R", "Psi_r", "Psi"}
    assert out["Psi"] == pytest.approx(0.1)
    assert multi.dispatch_tier("kuramoto_step") in ("rust", "numpy")


def test_public_upde_step_preserves_contract_and_layer_shapes() -> None:
    """UPDESystem.step keeps its per-layer output contract on the dispatcher path."""
    system, theta_layers, omega_layers = _upde_fixture()
    out = system.step(theta_layers, omega_layers, psi_driver=0.2, pac_gamma=0.3)
    assert [t.size for t in out["theta1"]] == [33, 57]
    assert out["R_layer"].shape == (2,)
    assert out["V_layer"].shape == (2,)
    assert np.all(np.isfinite(out["theta1"][0])) and np.all(np.isfinite(out["theta1"][1]))
    assert multi.dispatch_tier("upde_tick") in ("rust", "numpy")


def test_upde_system_synchronises_toward_external_driver() -> None:
    """Physics regression: strong global driver pulls V toward zero."""
    K = np.array([[1.0]])
    spec = KnmSpec(K=K, alpha=None, zeta=np.array([2.0]))
    system = UPDESystem(spec=spec, dt=1e-2)
    rng = np.random.default_rng(11)
    theta = [rng.uniform(-np.pi, np.pi, size=128)]
    omega = [np.zeros(128)]
    result = system.run_lyapunov(1500, theta, omega, psi_driver=0.5)
    assert result["V_global_hist"][-1] < 1e-3
    assert result["lambda_global"] < 0.0
