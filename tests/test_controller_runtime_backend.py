from __future__ import annotations

import sys
import types

from scpn_fusion.scpn.controller_runtime_backend import probe_rust_runtime_bindings


def test_probe_rust_runtime_bindings_handles_missing_module() -> None:
    has_runtime, dense_fn, update_fn, sample_fn = probe_rust_runtime_bindings()
    assert isinstance(has_runtime, bool)
    if not has_runtime:
        assert dense_fn is None
        assert update_fn is None
        assert sample_fn is None


def test_probe_rust_runtime_bindings_uses_available_bindings(
    monkeypatch,  # pytest fixture (left untyped by convention)
) -> None:
    fake = types.ModuleType("scpn_fusion_rs")
    fake.scpn_dense_activations = lambda *_args, **_kwargs: None
    fake.scpn_marking_update = lambda *_args, **_kwargs: None
    fake.scpn_sample_firing = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "scpn_fusion_rs", fake)

    has_runtime, dense_fn, update_fn, sample_fn = probe_rust_runtime_bindings()
    assert has_runtime is True
    assert dense_fn is fake.scpn_dense_activations
    assert update_fn is fake.scpn_marking_update
    assert sample_fn is fake.scpn_sample_firing
