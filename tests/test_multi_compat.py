# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the multi-language backend dispatcher."""

from __future__ import annotations

import builtins
import ctypes
import sys
import types
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.core import _multi_compat as multi

FloatArray = NDArray[np.float64]


def _clear_kernel(name: str) -> None:
    """Remove a test function-kernel registration and dispatch cache entry."""
    with multi._registry_lock:
        multi._registry.pop(name, None)
        multi._dispatch_cache.pop(name, None)


@pytest.fixture
def kernel_name(request: pytest.FixtureRequest) -> Iterator[str]:
    """Provide an isolated function-kernel name for dispatcher tests."""
    name = f"test_{request.node.name}"
    _clear_kernel(name)
    yield name
    _clear_kernel(name)


@pytest.fixture(autouse=True)
def _clear_rust_symbol_cache_between_tests() -> Iterator[None]:
    """Prevent extension-module id reuse from leaking cached Rust symbols."""
    multi._rust_symbol_cache.clear()
    yield
    multi._rust_symbol_cache.clear()


def test_numpy_backend_is_always_available() -> None:
    """NumPy remains the guaranteed fallback backend tier."""
    assert multi.available_backends()["numpy"] is True
    assert multi.is_available(multi.BackendTier.NUMPY) is True


def test_ensure_probed_returns_when_parallel_probe_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe guard exits when another thread completes probing under lock."""

    class _ProbeLock:
        def __enter__(self) -> None:
            multi._probed = True

        def __exit__(
            self,
            _exc_type: object,
            _exc: object,
            _traceback: object,
        ) -> None:
            return None

    monkeypatch.setattr(multi, "_probed", False)
    monkeypatch.setattr(multi, "_probe_lock", _ProbeLock())

    multi._ensure_probed()

    assert multi._probed is True


def test_jax_probe_treats_broken_optional_import_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken JAX import is treated as an unavailable optional backend."""
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals_: Mapping[str, object] | None = None,
        locals_: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        if name == "jax":
            raise ValueError("numpy dtype ABI mismatch")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert multi._probe_jax() is False


def test_jax_probe_accepts_importable_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """An importable JAX module marks the optional backend as available."""
    real_import = builtins.__import__
    fake_jax = types.ModuleType("jax")

    def fake_import(
        name: str,
        globals_: Mapping[str, object] | None = None,
        locals_: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        if name == "jax":
            return fake_jax
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert multi._probe_jax() is True


def test_dispatch_selects_registered_numpy_kernel(kernel_name: str) -> None:
    """Dispatch selects and caches a registered NumPy-tier kernel."""

    def numpy_impl(value: int) -> int:
        return value + 1

    multi.register_kernel(kernel_name, multi.BackendTier.NUMPY, numpy_impl)

    selected = multi.dispatch(kernel_name)

    assert selected(4) == 5
    assert multi.dispatch(kernel_name) is selected
    assert multi.dispatch_tier(kernel_name) == "numpy"
    assert multi.registered_kernels()[kernel_name] == ["numpy*"]


def test_dispatch_tier_populates_cache_for_registered_kernel(kernel_name: str) -> None:
    """Requesting the selected tier populates the dispatch cache."""

    def numpy_impl(value: int) -> int:
        return value * 2

    multi.register_kernel(kernel_name, multi.BackendTier.NUMPY, numpy_impl)

    assert multi.dispatch_tier(kernel_name) == "numpy"
    assert multi._dispatch_cache[kernel_name][1] is numpy_impl


def test_dispatch_returns_cache_populated_while_waiting_for_lock(
    kernel_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatch rechecks the cache after acquiring the registry lock."""

    def numpy_impl() -> str:
        return "numpy"

    class _RegistryLock:
        def __enter__(self) -> None:
            multi._dispatch_cache[kernel_name] = (multi.BackendTier.NUMPY, numpy_impl)

        def __exit__(
            self,
            _exc_type: object,
            _exc: object,
            _traceback: object,
        ) -> None:
            return None

    monkeypatch.setattr(multi, "_registry_lock", _RegistryLock())

    assert multi.dispatch(kernel_name) is numpy_impl


def test_dispatch_tier_rejects_dispatch_without_cache_population(
    kernel_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatch-tier lookup fails closed if dispatch never records a cache entry."""
    monkeypatch.setattr(multi, "dispatch", lambda _name: None)

    with pytest.raises(RuntimeError, match="dispatch_tier failed"):
        multi.dispatch_tier(kernel_name)


def test_dispatch_falls_back_to_available_lower_tier(kernel_name: str) -> None:
    """Dispatch falls back from an unavailable fast tier to NumPy."""

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
    """Dispatch rejects unknown kernel names."""
    with pytest.raises(KeyError, match="No implementations registered"):
        multi.dispatch("test_missing_kernel")


def test_dispatch_rust_symbol_resolves_from_extension_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rust symbol dispatch resolves callables from the extension module."""
    fake_extension = types.ModuleType("scpn_fusion_rs")

    def rust_symbol() -> str:
        return "rust-symbol"

    fake_extension.__dict__["rust_symbol"] = rust_symbol
    monkeypatch.setitem(sys.modules, "scpn_fusion_rs", fake_extension)

    resolved = multi.dispatch_rust_symbol("rust_symbol")

    assert resolved is rust_symbol
    assert resolved() == "rust-symbol"


def test_dispatch_rust_symbol_uses_cache_for_same_extension_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rust symbol dispatch reuses cache entries for the same module object."""
    fake_extension = types.ModuleType("scpn_fusion_rs")

    def rust_symbol() -> str:
        return "cached-rust-symbol"

    fake_extension.__dict__["rust_symbol"] = rust_symbol
    monkeypatch.setitem(sys.modules, "scpn_fusion_rs", fake_extension)
    multi._rust_symbol_cache.pop("rust_symbol", None)

    resolved = multi.dispatch_rust_symbol("rust_symbol")

    assert multi.dispatch_rust_symbol("rust_symbol") is resolved


def test_dispatch_rust_symbol_refreshes_cache_when_module_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rust symbol dispatch refreshes cached callables when the module changes."""
    first_extension = types.ModuleType("scpn_fusion_rs")
    second_extension = types.ModuleType("scpn_fusion_rs")

    def first_symbol() -> str:
        return "first"

    def second_symbol() -> str:
        return "second"

    first_extension.__dict__["rust_symbol"] = first_symbol
    second_extension.__dict__["rust_symbol"] = second_symbol
    multi._rust_symbol_cache.pop("rust_symbol", None)

    monkeypatch.setitem(sys.modules, "scpn_fusion_rs", first_extension)
    assert multi.dispatch_rust_symbol("rust_symbol") is first_symbol

    monkeypatch.setitem(sys.modules, "scpn_fusion_rs", second_extension)
    assert multi.dispatch_rust_symbol("rust_symbol") is second_symbol


def test_dispatch_rust_symbol_rejects_missing_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rust symbol dispatch preserves missing-symbol errors."""
    fake_extension = types.ModuleType("scpn_fusion_rs")
    monkeypatch.setitem(sys.modules, "scpn_fusion_rs", fake_extension)

    with pytest.raises(AttributeError, match="missing_symbol"):
        multi.dispatch_rust_symbol("missing_symbol")


def test_dispatch_rust_symbol_rejects_non_callable_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rust symbol dispatch rejects exported non-callable objects."""
    fake_extension = types.ModuleType("scpn_fusion_rs")
    fake_extension.__dict__["rust_symbol"] = 42
    monkeypatch.setitem(sys.modules, "scpn_fusion_rs", fake_extension)

    with pytest.raises(TypeError, match="not callable"):
        multi.dispatch_rust_symbol("rust_symbol")


def test_a4_production_surfaces_use_dispatcher_for_rust_symbols() -> None:
    """Production Rust-only surfaces import through the dispatcher boundary."""
    production_files = [
        Path("src/scpn_fusion/phase/kuramoto.py"),
        Path("src/scpn_fusion/phase/upde.py"),
        Path("src/scpn_fusion/scpn/controller_runtime_backend.py"),
        Path("src/scpn_fusion/control/rust_flight_sim_wrapper.py"),
        Path("src/scpn_fusion/core/integrated_transport_solver.py"),
    ]

    for path in production_files:
        source = path.read_text(encoding="utf-8")
        assert "import scpn_fusion_rs" not in source
        assert "from scpn_fusion_rs import" not in source


def test_dispatch_raises_when_all_registered_tiers_are_unavailable(kernel_name: str) -> None:
    """Dispatch rejects kernels whose registered tiers are all unavailable."""
    multi.register_kernel(kernel_name, multi.BackendTier.MOJO, lambda: None)

    with pytest.raises(RuntimeError, match="All registered backends"):
        multi.dispatch(kernel_name)


def test_optional_backend_probes_respect_disable_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional backend probes honor environment disable flags."""
    monkeypatch.setenv("SCPN_DISABLE_MOJO", "true")
    monkeypatch.setenv("SCPN_DISABLE_JULIA", "1")
    monkeypatch.setenv("SCPN_DISABLE_GO", "yes")

    assert multi._probe_mojo() is False
    assert multi._probe_julia() is False
    assert multi._probe_go() is False


def test_julia_probe_accepts_importable_juliacall(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Julia probe succeeds when juliacall is importable."""
    fake_juliacall = types.ModuleType("juliacall")
    fake_juliacall.__dict__["Main"] = object()
    monkeypatch.delenv("SCPN_DISABLE_JULIA", raising=False)
    monkeypatch.setitem(sys.modules, "juliacall", fake_juliacall)

    assert multi._probe_julia() is True


def test_go_probe_accepts_loadable_shared_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Go probe succeeds when the shared library can be loaded."""
    monkeypatch.delenv("SCPN_DISABLE_GO", raising=False)
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: object())

    assert multi._probe_go() is True


def test_dispatch_fallback_survives_telemetry_failure(
    kernel_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fallback telemetry failures do not block dispatch fallback."""
    fake_mod = types.ModuleType("scpn_fusion.fallback_telemetry")

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("telemetry unavailable")

    fake_mod.__dict__["record_fallback_event"] = _raise
    monkeypatch.setitem(sys.modules, "scpn_fusion.fallback_telemetry", fake_mod)

    multi.register_kernel(kernel_name, multi.BackendTier.MOJO, lambda: "mojo")
    multi.register_kernel(kernel_name, multi.BackendTier.NUMPY, lambda: "numpy")

    assert multi.dispatch(kernel_name)() == "numpy"


def _clear_kernel_class(name: str) -> None:
    """Remove a test kernel-class registration and dispatch cache entry."""
    with multi._class_registry_lock:
        multi._class_registry.pop(name, None)
        multi._class_dispatch_cache.pop(name, None)


@pytest.fixture
def class_kernel_name(request: pytest.FixtureRequest) -> Iterator[str]:
    """Provide an isolated kernel-class name for dispatcher tests."""
    name = f"testcls_{request.node.name}"
    _clear_kernel_class(name)
    yield name
    _clear_kernel_class(name)


class _NumpyKernelStub:
    """Dummy NumPy-tier kernel class for class-dispatch tests."""

    pass


class _MojoKernelStub:
    """Dummy unavailable Mojo-tier kernel class for class-dispatch tests."""

    pass


def test_register_and_dispatch_kernel_class(class_kernel_name: str) -> None:
    """Kernel-class registration dispatches and caches the NumPy class."""
    multi.register_kernel_class(
        class_kernel_name, multi.BackendTier.NUMPY, lambda: _NumpyKernelStub
    )
    assert multi.dispatch_kernel_class(class_kernel_name) is _NumpyKernelStub
    # Second call hits the dispatch cache.
    assert multi.dispatch_kernel_class(class_kernel_name) is _NumpyKernelStub


def test_dispatch_kernel_class_prefers_available_lower_tier(class_kernel_name: str) -> None:
    """Kernel-class dispatch falls back to an available lower tier."""
    multi.register_kernel_class(class_kernel_name, multi.BackendTier.MOJO, lambda: _MojoKernelStub)
    multi.register_kernel_class(
        class_kernel_name, multi.BackendTier.NUMPY, lambda: _NumpyKernelStub
    )
    # MOJO is unavailable in CI, so dispatch falls back to the NumPy tier.
    assert multi.dispatch_kernel_class(class_kernel_name) is _NumpyKernelStub


def test_dispatch_kernel_class_raises_for_unregistered() -> None:
    """Kernel-class dispatch rejects unknown class names."""
    with pytest.raises(KeyError, match="No kernel class registered"):
        multi.dispatch_kernel_class("testcls_missing_kernel_class")


def test_dispatch_kernel_class_raises_when_all_unavailable(class_kernel_name: str) -> None:
    """Kernel-class dispatch rejects registrations with no available tier."""
    multi.register_kernel_class(class_kernel_name, multi.BackendTier.MOJO, lambda: _MojoKernelStub)
    with pytest.raises(RuntimeError, match="All registered backends"):
        multi.dispatch_kernel_class(class_kernel_name)


def test_dispatch_kernel_class_returns_cache_populated_while_waiting_for_lock(
    class_kernel_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kernel-class dispatch rechecks the cache after acquiring the lock."""

    class _ClassRegistryLock:
        def __enter__(self) -> None:
            multi._class_dispatch_cache[class_kernel_name] = (
                multi.BackendTier.NUMPY,
                _NumpyKernelStub,
            )

        def __exit__(
            self,
            _exc_type: object,
            _exc: object,
            _traceback: object,
        ) -> None:
            return None

    monkeypatch.setattr(multi, "_class_registry_lock", _ClassRegistryLock())

    assert multi.dispatch_kernel_class(class_kernel_name) is _NumpyKernelStub


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

    assert multi._load_rust_equilibrium_kernel() is RustAcceleratedKernel


def test_numpy_equilibrium_loader_returns_the_python_kernel_class() -> None:
    """The NumPy equilibrium loader returns the pure-Python kernel class."""
    from scpn_fusion.core.fusion_kernel import FusionKernel

    assert multi._load_numpy_equilibrium_kernel() is FusionKernel


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
        multi._rust_multigrid_solve(source, psi_bc, r_min, r_max, z_min, z_max, nr, nz)


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


def test_gpu_probe_disabled_by_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCPN_DISABLE_GPU short-circuits the GPU probe to unavailable."""
    monkeypatch.setenv("SCPN_DISABLE_GPU", "1")
    assert multi._probe_gpu() is False


def test_gpu_probe_without_gpu_feature_symbol_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An extension built without --features gpu reports the tier unavailable."""
    monkeypatch.delenv("SCPN_DISABLE_GPU", raising=False)
    fake_ext = types.ModuleType("scpn_fusion_rs")
    monkeypatch.setitem(sys.modules, "scpn_fusion_rs", fake_ext)
    assert multi._probe_gpu() is False


def test_gpu_probe_delegates_to_extension_adapter_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the gpu feature built, availability follows py_gpu_available()."""
    monkeypatch.delenv("SCPN_DISABLE_GPU", raising=False)
    fake_ext = types.ModuleType("scpn_fusion_rs")
    fake_ext.py_gpu_available = lambda: True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scpn_fusion_rs", fake_ext)
    assert multi._probe_gpu() is True
    fake_ext.py_gpu_available = lambda: False  # type: ignore[attr-defined]
    assert multi._probe_gpu() is False


def test_gpu_probe_treats_broken_extension_import_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising extension import degrades the GPU tier to unavailable."""
    monkeypatch.delenv("SCPN_DISABLE_GPU", raising=False)
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals_: Mapping[str, object] | None = None,
        locals_: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        if name == "scpn_fusion_rs":
            raise ImportError("extension not built")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "scpn_fusion_rs", raising=False)
    assert multi._probe_gpu() is False


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
    smoothed = multi._numpy_gs_rb_sor_smooth(
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
    reference = multi._numpy_gs_rb_sor_smooth(
        psi0, source, 4.0, 8.0, -4.0, 4.0, omega=1.3, n_sweeps=50
    )
    gpu_result = multi._gpu_gs_rb_sor_smooth(
        psi0, source, 4.0, 8.0, -4.0, 4.0, omega=1.3, n_sweeps=50
    )

    assert gpu_result.shape == reference.shape
    rel_l2 = float(np.linalg.norm(gpu_result - reference)) / max(
        float(np.linalg.norm(reference)), 1e-30
    )
    assert rel_l2 < 1e-4

    key = (n, n, 4.0, 8.0, -4.0, 4.0)
    assert key in multi._gpu_gs_solver_cache  # geometry-keyed device reuse
    again = multi._gpu_gs_rb_sor_smooth(psi0, source, 4.0, 8.0, -4.0, 4.0, omega=1.3, n_sweeps=50)
    np.testing.assert_allclose(again, gpu_result, rtol=0.0, atol=0.0)
