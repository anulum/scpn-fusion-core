# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the multi-language backend tier providers and bootstrap wiring."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.core import _multi_compat as multi
from scpn_fusion.core import _multi_compat_providers as providers

FloatArray = NDArray[np.float64]


def test_equilibrium_kernel_class_dispatches_to_python_without_rust() -> None:
    """The equilibrium kernel class falls back to Python when Rust is unavailable."""
    cls = multi.dispatch_kernel_class("equilibrium_kernel")
    assert isinstance(cls, type)
    tiers = [multi._TIER_NAMES[t] for t, _ in multi._class_registry["equilibrium_kernel"]]
    assert tiers == ["rust", "numpy"]
    if not multi.is_available(multi.BackendTier.RUST):
        from scpn_fusion.core.fusion_kernel import FusionKernel

        assert cls is FusionKernel


def test_rust_equilibrium_loader_returns_the_rust_kernel_class() -> None:
    """The Rust equilibrium loader returns the Rust-accelerated class."""
    from scpn_fusion.core._rust_compat import RustAcceleratedKernel

    assert providers._load_rust_equilibrium_kernel() is RustAcceleratedKernel


def test_numpy_equilibrium_loader_returns_the_python_kernel_class() -> None:
    """The NumPy equilibrium loader returns the pure-Python kernel class."""
    from scpn_fusion.core.fusion_kernel import FusionKernel

    assert providers._load_numpy_equilibrium_kernel() is FusionKernel


def test_shafranov_bv_bootstrap_registers_both_tiers() -> None:
    """The function bootstrap registers Rust and NumPy tiers for shafranov_bv."""
    tiers = multi.registered_kernels().get("shafranov_bv")
    assert tiers is not None
    names = {tier.rstrip("*") for tier in tiers}
    assert "numpy" in names
    assert "rust" in names


def test_numpy_shafranov_bv_provider_matches_reference() -> None:
    """The NumPy-tier provider returns the canonical free-function value."""
    from scpn_fusion.control.analytic_solver import shafranov_bv as reference

    assert providers._numpy_shafranov_bv(6.2, 2.0, 15.0) == reference(6.2, 2.0, 15.0)
    assert providers._numpy_shafranov_bv(1.7, 0.5, 1.0, beta_p=0.3, li=1.1) == reference(
        1.7, 0.5, 1.0, beta_p=0.3, li=1.1
    )


def test_shafranov_bv_dispatch_resolves_to_numpy_without_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With Rust unavailable, dispatch("shafranov_bv") falls to the NumPy tier."""
    from scpn_fusion.control.analytic_solver import shafranov_bv as reference

    multi._ensure_probed()
    monkeypatch.setitem(multi._availability, multi.BackendTier.RUST, False)
    with multi._registry_lock:
        multi._dispatch_cache.pop("shafranov_bv", None)
    try:
        impl = multi.dispatch("shafranov_bv")
        assert multi.dispatch_tier("shafranov_bv") == "numpy"
        assert impl(6.2, 2.0, 15.0) == reference(6.2, 2.0, 15.0)
        assert impl(6.2, 2.0, 15.0, beta_p=0.9, li=1.2) == reference(
            6.2, 2.0, 15.0, beta_p=0.9, li=1.2
        )
    finally:
        with multi._registry_lock:
            multi._dispatch_cache.pop("shafranov_bv", None)


def test_rust_shafranov_bv_provider_matches_reference() -> None:
    """When Rust is built, its tier provider is bit-exact with the NumPy tier."""
    pytest.importorskip("scpn_fusion_rs")
    from scpn_fusion.control.analytic_solver import shafranov_bv as reference

    assert providers._rust_shafranov_bv(6.2, 2.0, 15.0) == reference(6.2, 2.0, 15.0)
    assert providers._rust_shafranov_bv(3.0, 1.0, 8.0, beta_p=0.9, li=0.6) == reference(
        3.0, 1.0, 8.0, beta_p=0.9, li=0.6
    )


def test_solve_coil_currents_bootstrap_registers_both_tiers() -> None:
    """The function bootstrap registers Rust and NumPy tiers for solve_coil_currents."""
    tiers = multi.registered_kernels().get("solve_coil_currents")
    assert tiers is not None
    names = {tier.rstrip("*") for tier in tiers}
    assert "numpy" in names
    assert "rust" in names


def test_numpy_solve_coil_currents_provider_matches_reference() -> None:
    """The NumPy-tier provider returns the canonical free-function currents."""
    from scpn_fusion.control.analytic_solver import solve_coil_currents as reference

    green = [0.01, 0.02, 0.015]
    np.testing.assert_array_equal(
        providers._numpy_solve_coil_currents(green, -0.05), reference(green, -0.05)
    )
    np.testing.assert_array_equal(
        providers._numpy_solve_coil_currents(green, -0.05, ridge_lambda=1e-3),
        reference(green, -0.05, ridge_lambda=1e-3),
    )


def test_solve_coil_currents_dispatch_resolves_to_numpy_without_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With Rust unavailable, dispatch falls to the NumPy coil-current tier."""
    from scpn_fusion.control.analytic_solver import solve_coil_currents as reference

    multi._ensure_probed()
    monkeypatch.setitem(multi._availability, multi.BackendTier.RUST, False)
    with multi._registry_lock:
        multi._dispatch_cache.pop("solve_coil_currents", None)
    try:
        impl = multi.dispatch("solve_coil_currents")
        assert multi.dispatch_tier("solve_coil_currents") == "numpy"
        np.testing.assert_array_equal(impl([0.01, 0.02], -0.03), reference([0.01, 0.02], -0.03))
    finally:
        with multi._registry_lock:
            multi._dispatch_cache.pop("solve_coil_currents", None)


def test_rust_solve_coil_currents_provider_matches_reference() -> None:
    """When Rust is built, its tier is tolerance-aware equivalent to the NumPy tier.

    The Green's-norm reduction is not bit-reproducible across the backends, so a
    tight relative tolerance is used rather than exact equality.
    """
    pytest.importorskip("scpn_fusion_rs")
    from scpn_fusion.control.analytic_solver import solve_coil_currents as reference

    green = [0.01, 0.02, 0.015, 0.005, 0.01]
    np.testing.assert_allclose(
        providers._rust_solve_coil_currents(green, -0.05),
        reference(green, -0.05),
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        providers._rust_solve_coil_currents(green, -0.05, ridge_lambda=1e-3),
        reference(green, -0.05, ridge_lambda=1e-3),
        rtol=1e-12,
        atol=1e-15,
    )


def test_measure_magnetics_bootstrap_registers_both_tiers() -> None:
    """The function bootstrap registers Rust and NumPy tiers for measure_magnetics."""
    tiers = multi.registered_kernels().get("measure_magnetics")
    assert tiers is not None
    names = {tier.rstrip("*") for tier in tiers}
    assert "numpy" in names
    assert "rust" in names


def test_numpy_measure_magnetics_provider_matches_reference() -> None:
    """The NumPy-tier provider returns the canonical free-function measurements."""
    from scpn_fusion.diagnostics.synthetic_sensors import measure_magnetics as reference

    psi = np.full((33, 33), 1.5, dtype=np.float64)
    np.testing.assert_array_equal(
        providers._numpy_measure_magnetics(psi, 33, 33, 3.0, 9.0, -3.5, 3.5),
        reference(psi, 33, 33, 3.0, 9.0, -3.5, 3.5),
    )


def test_measure_magnetics_dispatch_resolves_to_numpy_without_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With Rust unavailable, dispatch falls to the NumPy magnetics tier."""
    from scpn_fusion.diagnostics.synthetic_sensors import measure_magnetics as reference

    multi._ensure_probed()
    monkeypatch.setitem(multi._availability, multi.BackendTier.RUST, False)
    with multi._registry_lock:
        multi._dispatch_cache.pop("measure_magnetics", None)
    try:
        impl = multi.dispatch("measure_magnetics")
        assert multi.dispatch_tier("measure_magnetics") == "numpy"
        psi = np.full((33, 33), 0.7, dtype=np.float64)
        np.testing.assert_array_equal(
            impl(psi, 33, 33, 3.0, 9.0, -3.5, 3.5),
            reference(psi, 33, 33, 3.0, 9.0, -3.5, 3.5),
        )
    finally:
        with multi._registry_lock:
            multi._dispatch_cache.pop("measure_magnetics", None)


def test_rust_measure_magnetics_provider_matches_reference() -> None:
    """When Rust is built, its magnetics tier is tolerance-aware equivalent."""
    pytest.importorskip("scpn_fusion_rs")
    from scpn_fusion.diagnostics.synthetic_sensors import measure_magnetics as reference

    nr = nz = 65
    r_axis = np.linspace(3.0, 9.0, nr)
    z_axis = np.linspace(-5.0, 5.0, nz)
    rr, zz = np.meshgrid(r_axis, z_axis)
    psi = np.asarray(np.exp(-((rr - 6.0) ** 2 + zz**2) / 8.0), dtype=np.float64)
    np.testing.assert_allclose(
        providers._rust_measure_magnetics(psi, nr, nz, 3.0, 9.0, -5.0, 5.0),
        reference(psi, nr, nz, 3.0, 9.0, -5.0, 5.0),
        rtol=1e-9,
        atol=1e-9,
    )


def _multigrid_problem() -> tuple[FloatArray, FloatArray, float, float, float, float, int, int]:
    """Return a deterministic multigrid fixture for dispatcher provider tests."""
    nr = nz = 33
    r_min, r_max, z_min, z_max = 1.2, 2.2, -0.5, 0.5
    rr, zz = np.meshgrid(np.linspace(r_min, r_max, nr), np.linspace(z_min, z_max, nz))
    source = -rr * np.exp(-((rr - 1.7) ** 2 + zz**2) / 0.05)
    psi_bc = np.zeros((nz, nr))
    return source, psi_bc, r_min, r_max, z_min, z_max, nr, nz


def test_multigrid_solve_bootstrap_registers_both_tiers() -> None:
    """The function bootstrap registers Rust and NumPy tiers for multigrid_solve."""
    tiers = multi.registered_kernels().get("multigrid_solve")
    assert tiers is not None
    names = {tier.rstrip("*") for tier in tiers}
    assert "numpy" in names
    assert "rust" in names


def test_numpy_multigrid_solve_provider_matches_reference() -> None:
    """The NumPy-tier provider returns the canonical free-function solve."""
    from scpn_fusion.core.multigrid_solve import multigrid_solve as reference

    source, psi_bc, r_min, r_max, z_min, z_max, nr, nz = _multigrid_problem()
    psi_p, res_p, nc_p, conv_p = providers._numpy_multigrid_solve(
        source, psi_bc, r_min, r_max, z_min, z_max, nr, nz, tol=1e-6, max_cycles=200
    )
    psi_r, res_r, nc_r, conv_r = reference(
        source, psi_bc, r_min, r_max, z_min, z_max, nr, nz, tol=1e-6, max_cycles=200
    )
    np.testing.assert_array_equal(psi_p, psi_r)
    assert (res_p, nc_p, conv_p) == (res_r, nc_r, conv_r)


def test_multigrid_solve_dispatch_resolves_to_numpy_without_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With Rust unavailable, dispatch falls to the NumPy multigrid tier."""
    from scpn_fusion.core.multigrid_solve import multigrid_solve as reference

    multi._ensure_probed()
    monkeypatch.setitem(multi._availability, multi.BackendTier.RUST, False)
    with multi._registry_lock:
        multi._dispatch_cache.pop("multigrid_solve", None)
    try:
        impl = multi.dispatch("multigrid_solve")
        assert multi.dispatch_tier("multigrid_solve") == "numpy"
        source, psi_bc, r_min, r_max, z_min, z_max, nr, nz = _multigrid_problem()
        psi, _res, _nc, conv = impl(
            source, psi_bc, r_min, r_max, z_min, z_max, nr, nz, tol=1e-6, max_cycles=200
        )
        expected = reference(
            source, psi_bc, r_min, r_max, z_min, z_max, nr, nz, tol=1e-6, max_cycles=200
        )
        assert conv is True
        np.testing.assert_array_equal(psi, expected[0])
    finally:
        with multi._registry_lock:
            multi._dispatch_cache.pop("multigrid_solve", None)


def test_rust_multigrid_solve_provider_matches_reference() -> None:
    """When Rust is built, its multigrid tier converges to the same flux map."""
    pytest.importorskip("scpn_fusion_rs")
    from scpn_fusion.core.multigrid_solve import multigrid_solve as reference

    source, psi_bc, r_min, r_max, z_min, z_max, nr, nz = _multigrid_problem()
    psi_rs, _res_rs, _nc_rs, conv_rs = providers._rust_multigrid_solve(
        source, psi_bc, r_min, r_max, z_min, z_max, nr, nz, tol=1e-6, max_cycles=200
    )
    psi_py, _res_py, _nc_py, conv_py = reference(
        source, psi_bc, r_min, r_max, z_min, z_max, nr, nz, tol=1e-6, max_cycles=200
    )
    assert conv_rs and conv_py
    np.testing.assert_allclose(psi_rs, psi_py, rtol=1e-6, atol=1e-9)


def test_rust_multigrid_solve_rejects_unavailable_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scpn_fusion.core._rust_compat as rust_compat

    source, psi_bc, r_min, r_max, z_min, z_max, nr, nz = _multigrid_problem()
    monkeypatch.setattr(
        rust_compat,
        "rust_multigrid_vcycle",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="unavailable"):
        providers._rust_multigrid_solve(source, psi_bc, r_min, r_max, z_min, z_max, nr, nz)


def test_simulate_tearing_mode_bootstrap_registers_both_tiers() -> None:
    """The function bootstrap registers Rust and NumPy tiers for simulate_tearing_mode."""
    tiers = multi.registered_kernels().get("simulate_tearing_mode")
    assert tiers is not None
    names = {tier.rstrip("*") for tier in tiers}
    assert "numpy" in names
    assert "rust" in names


def test_numpy_simulate_tearing_mode_provider_is_seed_reproducible() -> None:
    """The NumPy-tier provider is deterministic for a given seed."""
    sig1, lbl1, ttd1 = providers._numpy_simulate_tearing_mode(500, seed=2026)
    sig2, lbl2, ttd2 = providers._numpy_simulate_tearing_mode(500, seed=2026)
    np.testing.assert_array_equal(sig1, sig2)
    assert (lbl1, ttd1) == (lbl2, ttd2)
    assert np.all(np.isfinite(sig1))


def test_simulate_tearing_mode_dispatch_resolves_to_numpy_without_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With Rust unavailable, dispatch falls to the NumPy tearing-mode tier."""
    multi._ensure_probed()
    monkeypatch.setitem(multi._availability, multi.BackendTier.RUST, False)
    with multi._registry_lock:
        multi._dispatch_cache.pop("simulate_tearing_mode", None)
    try:
        impl = multi.dispatch("simulate_tearing_mode")
        assert multi.dispatch_tier("simulate_tearing_mode") == "numpy"
        sig, label, _ttd = impl(1000, seed=7)
        assert sig.shape[0] > 0
        assert label in (0, 1)
    finally:
        with multi._registry_lock:
            multi._dispatch_cache.pop("simulate_tearing_mode", None)


def test_rust_simulate_tearing_mode_provider_is_seed_reproducible() -> None:
    """When Rust is built, its tearing-mode tier is reproducible for a seed."""
    pytest.importorskip("scpn_fusion_rs")
    sig1, lbl1, ttd1 = providers._rust_simulate_tearing_mode(500, seed=2026)
    sig2, lbl2, ttd2 = providers._rust_simulate_tearing_mode(500, seed=2026)
    np.testing.assert_array_equal(sig1, sig2)
    assert (lbl1, ttd1) == (lbl2, ttd2)
    assert np.all(np.isfinite(sig1))


def test_gs_rb_sor_smooth_registered_with_gpu_and_numpy_tiers() -> None:
    """The W-2 smoother kernel carries GPU and NumPy tier registrations."""
    kernels = multi.registered_kernels()
    assert "gs_rb_sor_smooth" in kernels
    tier_names = [name.rstrip("*") for name in kernels["gs_rb_sor_smooth"]]
    assert tier_names == ["gpu", "numpy"]
    assert "numpy*" in kernels["gs_rb_sor_smooth"]


def _smooth_problem(n: int) -> tuple[FloatArray, FloatArray]:
    """Build a small seeded GS* smoothing problem for tier tests."""
    rng = np.random.default_rng(2026)
    r_axis = np.linspace(4.0, 8.0, n)
    z_axis = np.linspace(-4.0, 4.0, n)
    r_grid, z_grid = np.meshgrid(r_axis, z_axis)
    source = -np.exp(-((r_grid - 6.0) ** 2 + z_grid**2) / 0.5)
    psi0 = rng.normal(0.0, 1e-3, size=(n, n))
    psi0[0, :] = psi0[-1, :] = psi0[:, 0] = psi0[:, -1] = 0.0
    return psi0, source


def test_numpy_gs_rb_sor_smooth_reduces_residual_and_preserves_boundary() -> None:
    """The NumPy smoother tier contracts the GS* residual on a seeded problem."""
    from scpn_fusion.core.multigrid_solve import mg_residual

    n = 33
    psi0, source = _smooth_problem(n)
    psi0_copy = psi0.copy()
    smoothed = providers._numpy_gs_rb_sor_smooth(
        psi0, source, 4.0, 8.0, -4.0, 4.0, omega=1.3, n_sweeps=100
    )

    np.testing.assert_array_equal(psi0, psi0_copy)  # input not mutated
    np.testing.assert_array_equal(smoothed[0, :], 0.0)
    np.testing.assert_array_equal(smoothed[-1, :], 0.0)
    np.testing.assert_array_equal(smoothed[:, 0], 0.0)
    np.testing.assert_array_equal(smoothed[:, -1], 0.0)

    r_axis = np.linspace(4.0, 8.0, n)
    z_axis = np.linspace(-4.0, 4.0, n)
    r_grid, _ = np.meshgrid(r_axis, z_axis)
    dr = 4.0 / (n - 1)
    dz = 8.0 / (n - 1)
    res_before = float(np.max(np.abs(mg_residual(psi0, source, r_grid, dr, dz))))
    res_after = float(np.max(np.abs(mg_residual(smoothed, source, r_grid, dr, dz))))
    assert res_after < 0.5 * res_before


def test_gpu_gs_rb_sor_smooth_matches_numpy_tier() -> None:
    """The GPU tier agrees with the float64 reference to f32 round-off."""
    if not multi.is_available(multi.BackendTier.GPU):
        pytest.skip("GPU tier unavailable (extension without --features gpu or no adapter)")

    n = 65
    psi0, source = _smooth_problem(n)
    reference = providers._numpy_gs_rb_sor_smooth(
        psi0, source, 4.0, 8.0, -4.0, 4.0, omega=1.3, n_sweeps=50
    )
    gpu_result = providers._gpu_gs_rb_sor_smooth(
        psi0, source, 4.0, 8.0, -4.0, 4.0, omega=1.3, n_sweeps=50
    )

    assert gpu_result.shape == reference.shape
    rel_l2 = float(np.linalg.norm(gpu_result - reference)) / max(
        float(np.linalg.norm(reference)), 1e-30
    )
    assert rel_l2 < 1e-4

    key = (n, n, 4.0, 8.0, -4.0, 4.0)
    assert key in providers._gpu_gs_solver_cache  # geometry-keyed device reuse
    again = providers._gpu_gs_rb_sor_smooth(
        psi0, source, 4.0, 8.0, -4.0, 4.0, omega=1.3, n_sweeps=50
    )
    np.testing.assert_allclose(again, gpu_result, rtol=0.0, atol=0.0)


def test_rust_hall_mhd_loader_rejects_non_class_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Rust Hall-MHD loader rejects a ``PyHallMHD`` symbol that is not a class."""
    scpn_fusion_rs = pytest.importorskip("scpn_fusion_rs")
    monkeypatch.setattr(scpn_fusion_rs, "PyHallMHD", 42, raising=False)
    with pytest.raises(TypeError, match="PyHallMHD is not a class"):
        providers._load_rust_hall_mhd()
