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
