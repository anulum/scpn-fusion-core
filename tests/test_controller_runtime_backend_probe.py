# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for optional Rust SCPN runtime probing."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable

import numpy as np
import pytest

from scpn_fusion.core import _multi_compat as multi
from scpn_fusion.scpn.controller_runtime_backend import (
    FloatArray,
    probe_rust_runtime_bindings,
)


def test_probe_rust_runtime_bindings_handles_missing_module() -> None:
    """Report no runtime when the optional Rust extension is unavailable."""
    has_runtime, dense_fn, update_fn, sample_fn = probe_rust_runtime_bindings()

    assert isinstance(has_runtime, bool)
    if not has_runtime:
        assert dense_fn is None
        assert update_fn is None
        assert sample_fn is None


def test_probe_rust_runtime_bindings_uses_available_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return every required callable when the Rust extension exposes them."""
    fake = types.ModuleType("scpn_fusion_rs")

    def dense(*_args: object, **_kwargs: object) -> None:
        return None

    def update(*_args: object, **_kwargs: object) -> None:
        return None

    def sample(*_args: object, **_kwargs: object) -> None:
        return None

    fake.__dict__["scpn_dense_activations"] = dense
    fake.__dict__["scpn_marking_update"] = update
    fake.__dict__["scpn_sample_firing"] = sample
    monkeypatch.setitem(sys.modules, "scpn_fusion_rs", fake)

    has_runtime, dense_fn, update_fn, sample_fn = probe_rust_runtime_bindings()

    assert has_runtime is True
    assert dense_fn is dense
    assert update_fn is update
    assert sample_fn is sample


def test_probe_rust_runtime_bindings_uses_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve all Rust callables through the shared compatibility dispatcher."""
    calls: list[str] = []

    def fake_dispatch(symbol_name: str) -> Callable[..., object]:
        calls.append(symbol_name)
        return lambda *_args, **_kwargs: symbol_name

    monkeypatch.setattr(multi, "dispatch_rust_symbol", fake_dispatch)

    has_runtime, dense_fn, update_fn, sample_fn = probe_rust_runtime_bindings()

    assert has_runtime is True
    assert calls == [
        "scpn_dense_activations",
        "scpn_marking_update",
        "scpn_sample_firing",
    ]
    assert dense_fn is not None
    assert update_fn is not None
    assert sample_fn is not None
    assert (
        dense_fn(
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
        )
        == "scpn_dense_activations"
    )


def test_probe_rust_runtime_bindings_returns_disabled_when_dispatch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed without returning partial callables when dispatch fails."""

    def fake_dispatch(symbol_name: str) -> Callable[..., object]:
        if symbol_name == "scpn_marking_update":
            raise AttributeError(symbol_name)
        return lambda *_args, **_kwargs: symbol_name

    monkeypatch.setattr(multi, "dispatch_rust_symbol", fake_dispatch)

    has_runtime, dense_fn, update_fn, sample_fn = probe_rust_runtime_bindings()

    assert has_runtime is False
    assert dense_fn is None
    assert update_fn is None
    assert sample_fn is None


def test_probe_rust_runtime_bindings_disables_non_callable_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat non-callable Rust exports as an unavailable runtime."""

    def fake_dispatch(symbol_name: str) -> Callable[..., object]:
        if symbol_name == "scpn_sample_firing":
            raise TypeError("not callable")
        return lambda *_args, **_kwargs: symbol_name

    monkeypatch.setattr(multi, "dispatch_rust_symbol", fake_dispatch)

    has_runtime, dense_fn, update_fn, sample_fn = probe_rust_runtime_bindings()

    assert has_runtime is False
    assert dense_fn is None
    assert update_fn is None
    assert sample_fn is None


def test_float_array_alias_accepts_numpy_float_vectors() -> None:
    """Keep the exported runtime array alias importable for typed call sites."""
    vector: FloatArray = np.array([1.0, 2.0], dtype=np.float64)

    assert vector.dtype == np.float64
    assert vector.shape == (2,)
