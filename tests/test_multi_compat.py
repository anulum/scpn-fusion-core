# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the multi-language backend dispatcher."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator

import numpy as np
import pytest

from scpn_fusion.core import _multi_compat as multi


def _clear_kernel(name: str) -> None:
    with multi._registry_lock:
        multi._registry.pop(name, None)
        multi._dispatch_cache.pop(name, None)


@pytest.fixture
def kernel_name(request: pytest.FixtureRequest) -> Iterator[str]:
    name = f"test_{request.node.name}"
    _clear_kernel(name)
    yield name
    _clear_kernel(name)


def test_numpy_backend_is_always_available() -> None:
    assert multi.available_backends()["numpy"] is True
    assert multi.is_available(multi.BackendTier.NUMPY) is True


def test_dispatch_selects_registered_numpy_kernel(kernel_name: str) -> None:
    def numpy_impl(value: int) -> int:
        return value + 1

    multi.register_kernel(kernel_name, multi.BackendTier.NUMPY, numpy_impl)

    selected = multi.dispatch(kernel_name)

    assert selected(4) == 5
    assert multi.dispatch_tier(kernel_name) == "numpy"
    assert multi.registered_kernels()[kernel_name] == ["numpy*"]


def test_dispatch_falls_back_to_available_lower_tier(kernel_name: str) -> None:
    def unavailable_impl() -> str:
        return "unavailable"

    def numpy_impl() -> str:
        return "numpy"

    multi.register_kernel(kernel_name, multi.BackendTier.MOJO, unavailable_impl)
    multi.register_kernel(kernel_name, multi.BackendTier.NUMPY, numpy_impl)

    assert multi.dispatch(kernel_name)() == "numpy"
    assert multi.dispatch_tier(kernel_name) == "numpy"
    assert multi.registered_kernels()[kernel_name] == ["mojo", "numpy*"]


def test_dispatch_raises_for_unregistered_kernel() -> None:
    with pytest.raises(KeyError, match="No implementations registered"):
        multi.dispatch("test_missing_kernel")


def test_dispatch_raises_when_all_registered_tiers_are_unavailable(kernel_name: str) -> None:
    multi.register_kernel(kernel_name, multi.BackendTier.MOJO, lambda: None)

    with pytest.raises(RuntimeError, match="All registered backends"):
        multi.dispatch(kernel_name)


def test_dispatch_fallback_survives_telemetry_failure(
    kernel_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_mod = types.ModuleType("scpn_fusion.fallback_telemetry")

    def _raise(*_args, **_kwargs):
        raise RuntimeError("telemetry unavailable")

    fake_mod.record_fallback_event = _raise
    monkeypatch.setitem(sys.modules, "scpn_fusion.fallback_telemetry", fake_mod)

    multi.register_kernel(kernel_name, multi.BackendTier.MOJO, lambda: "mojo")
    multi.register_kernel(kernel_name, multi.BackendTier.NUMPY, lambda: "numpy")

    assert multi.dispatch(kernel_name)() == "numpy"


def _clear_kernel_class(name: str) -> None:
    with multi._class_registry_lock:
        multi._class_registry.pop(name, None)
        multi._class_dispatch_cache.pop(name, None)


@pytest.fixture
def class_kernel_name(request: pytest.FixtureRequest) -> Iterator[str]:
    name = f"testcls_{request.node.name}"
    _clear_kernel_class(name)
    yield name
    _clear_kernel_class(name)


class _NumpyKernelStub:
    pass


class _MojoKernelStub:
    pass


def test_register_and_dispatch_kernel_class(class_kernel_name: str) -> None:
    multi.register_kernel_class(
        class_kernel_name, multi.BackendTier.NUMPY, lambda: _NumpyKernelStub
    )
    assert multi.dispatch_kernel_class(class_kernel_name) is _NumpyKernelStub
    # Second call hits the dispatch cache.
    assert multi.dispatch_kernel_class(class_kernel_name) is _NumpyKernelStub


def test_dispatch_kernel_class_prefers_available_lower_tier(class_kernel_name: str) -> None:
    multi.register_kernel_class(class_kernel_name, multi.BackendTier.MOJO, lambda: _MojoKernelStub)
    multi.register_kernel_class(
        class_kernel_name, multi.BackendTier.NUMPY, lambda: _NumpyKernelStub
    )
    # MOJO is unavailable in CI, so dispatch falls back to the NumPy tier.
    assert multi.dispatch_kernel_class(class_kernel_name) is _NumpyKernelStub


def test_dispatch_kernel_class_raises_for_unregistered() -> None:
    with pytest.raises(KeyError, match="No kernel class registered"):
        multi.dispatch_kernel_class("testcls_missing_kernel_class")


def test_dispatch_kernel_class_raises_when_all_unavailable(class_kernel_name: str) -> None:
    multi.register_kernel_class(class_kernel_name, multi.BackendTier.MOJO, lambda: _MojoKernelStub)
    with pytest.raises(RuntimeError, match="All registered backends"):
        multi.dispatch_kernel_class(class_kernel_name)


def test_equilibrium_kernel_class_dispatches_to_python_without_rust() -> None:
    cls = multi.dispatch_kernel_class("equilibrium_kernel")
    assert isinstance(cls, type)
    tiers = [multi._TIER_NAMES[t] for t, _ in multi._class_registry["equilibrium_kernel"]]
    assert tiers == ["rust", "numpy"]
    if not multi.is_available(multi.BackendTier.RUST):
        from scpn_fusion.core.fusion_kernel import FusionKernel

        assert cls is FusionKernel


def test_rust_equilibrium_loader_returns_the_rust_kernel_class() -> None:
    from scpn_fusion.core._rust_compat import RustAcceleratedKernel

    assert multi._load_rust_equilibrium_kernel() is RustAcceleratedKernel


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

    assert multi._numpy_shafranov_bv(6.2, 2.0, 15.0) == reference(6.2, 2.0, 15.0)
    assert multi._numpy_shafranov_bv(1.7, 0.5, 1.0, beta_p=0.3, li=1.1) == reference(
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

    assert multi._rust_shafranov_bv(6.2, 2.0, 15.0) == reference(6.2, 2.0, 15.0)
    assert multi._rust_shafranov_bv(3.0, 1.0, 8.0, beta_p=0.9, li=0.6) == reference(
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
        multi._numpy_solve_coil_currents(green, -0.05), reference(green, -0.05)
    )
    np.testing.assert_array_equal(
        multi._numpy_solve_coil_currents(green, -0.05, ridge_lambda=1e-3),
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
        multi._rust_solve_coil_currents(green, -0.05),
        reference(green, -0.05),
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        multi._rust_solve_coil_currents(green, -0.05, ridge_lambda=1e-3),
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
        multi._numpy_measure_magnetics(psi, 33, 33, 3.0, 9.0, -3.5, 3.5),
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
        multi._rust_measure_magnetics(psi, nr, nz, 3.0, 9.0, -5.0, 5.0),
        reference(psi, nr, nz, 3.0, 9.0, -5.0, 5.0),
        rtol=1e-9,
        atol=1e-9,
    )


def _multigrid_problem() -> tuple:
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
    psi_p, res_p, nc_p, conv_p = multi._numpy_multigrid_solve(
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
    psi_rs, _res_rs, _nc_rs, conv_rs = multi._rust_multigrid_solve(
        source, psi_bc, r_min, r_max, z_min, z_max, nr, nz, tol=1e-6, max_cycles=200
    )
    psi_py, _res_py, _nc_py, conv_py = reference(
        source, psi_bc, r_min, r_max, z_min, z_max, nr, nz, tol=1e-6, max_cycles=200
    )
    assert conv_rs and conv_py
    np.testing.assert_allclose(psi_rs, psi_py, rtol=1e-6, atol=1e-9)


def test_simulate_tearing_mode_bootstrap_registers_both_tiers() -> None:
    """The function bootstrap registers Rust and NumPy tiers for simulate_tearing_mode."""
    tiers = multi.registered_kernels().get("simulate_tearing_mode")
    assert tiers is not None
    names = {tier.rstrip("*") for tier in tiers}
    assert "numpy" in names
    assert "rust" in names


def test_numpy_simulate_tearing_mode_provider_is_seed_reproducible() -> None:
    """The NumPy-tier provider is deterministic for a given seed."""
    sig1, lbl1, ttd1 = multi._numpy_simulate_tearing_mode(500, seed=2026)
    sig2, lbl2, ttd2 = multi._numpy_simulate_tearing_mode(500, seed=2026)
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
    sig1, lbl1, ttd1 = multi._rust_simulate_tearing_mode(500, seed=2026)
    sig2, lbl2, ttd2 = multi._rust_simulate_tearing_mode(500, seed=2026)
    np.testing.assert_array_equal(sig1, sig2)
    assert (lbl1, ttd1) == (lbl2, ttd2)
    assert np.all(np.isfinite(sig1))
