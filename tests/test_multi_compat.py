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

    setattr(fake_mod, "record_fallback_event", _raise)
    monkeypatch.setitem(sys.modules, "scpn_fusion.fallback_telemetry", fake_mod)

    multi.register_kernel(kernel_name, multi.BackendTier.MOJO, lambda: "mojo")
    multi.register_kernel(kernel_name, multi.BackendTier.NUMPY, lambda: "numpy")

    assert multi.dispatch(kernel_name)() == "numpy"
