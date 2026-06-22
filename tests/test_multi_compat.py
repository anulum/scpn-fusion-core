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
