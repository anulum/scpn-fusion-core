# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FNO Turbulence Dispatch Tests
"""Rust <-> NumPy dispatch tests for the FNO turbulence surrogate kernel.

These exercise the pure-NumPy ``MultiLayerFNO`` backbone and the Rust
``PyFnoController`` binding; they do not require the optional legacy JAX path,
so they run in the core-only CI matrix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core.fno_training import (
    FnoKernel,
    MultiLayerFNO,
    _fno_suppression,
    create_fno_controller,
)


def _write_weights(tmp_path: Path, *, seed: int = 123) -> Path:
    """Write a deterministic FNO weight archive and return its path."""
    model = MultiLayerFNO(modes=12, width=32, n_layers=4, seed=seed)
    path = tmp_path / "fno_weights.npz"
    model.save_weights(path)
    return path


def test_fno_dispatch_registers_both_tiers() -> None:
    """The class-kernel registry carries RUST and NUMPY FNO tiers."""
    from scpn_fusion.core import _multi_compat as multi

    kernels = multi.registered_kernel_classes()
    assert "fno_turbulence" in kernels
    tiers = [tier.rstrip("*") for tier in kernels["fno_turbulence"]]
    assert "rust" in tiers
    assert "numpy" in tiers


def test_fno_numpy_floor_without_rust(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The factory resolves to the NumPy FNO kernel when Rust is unavailable."""
    from scpn_fusion.core import _multi_compat as multi

    weights = _write_weights(tmp_path)
    multi._ensure_probed()
    monkeypatch.setitem(multi._availability, multi.BackendTier.RUST, False)
    monkeypatch.delitem(multi._class_dispatch_cache, "fno_turbulence", raising=False)
    try:
        controller = create_fno_controller(weights)
        assert isinstance(controller, FnoKernel)
        field = np.zeros((16, 16), dtype=np.float64)
        prediction = controller.predict(field)
        assert prediction.shape == (16, 16)
        assert bool(np.all(np.isfinite(prediction)))
    finally:
        multi._class_dispatch_cache.pop("fno_turbulence", None)


def test_fno_kernel_matches_model(tmp_path: Path) -> None:
    """The NumPy adapter reproduces the wrapped model's forward pass exactly."""
    weights = _write_weights(tmp_path)
    kernel = FnoKernel(weights)
    model = MultiLayerFNO()
    model.load_weights(weights)
    rng = np.random.default_rng(7)
    field = rng.standard_normal((16, 16)) * 0.5
    np.testing.assert_array_equal(kernel.predict(field), model.forward(field))
    suppression, prediction = kernel.predict_and_suppress(field)
    np.testing.assert_array_equal(prediction, model.forward(field))
    assert suppression == _fno_suppression(model.forward(field))


def test_fno_suppression_matches_reference() -> None:
    """The suppression factor is the clamped tanh of mean-square energy."""
    prediction = np.array([[0.3, -0.2], [0.1, 0.4]], dtype=np.float64)
    energy = float(np.mean(prediction**2))
    assert _fno_suppression(prediction) == pytest.approx(
        float(min(max(np.tanh(energy * 10.0), 0.0), 1.0))
    )


def test_create_fno_controller_returns_predict_surface(tmp_path: Path) -> None:
    """The dispatched controller exposes the predict/suppress protocol surface."""
    weights = _write_weights(tmp_path)
    controller = create_fno_controller(weights)
    field = np.zeros((16, 16), dtype=np.float64)
    assert callable(controller.predict)
    assert callable(controller.predict_and_suppress)
    suppression, prediction = controller.predict_and_suppress(field)
    assert 0.0 <= suppression <= 1.0
    assert prediction.shape == (16, 16)


def test_fno_rust_numpy_predict_parity(tmp_path: Path) -> None:
    """Rust and NumPy tiers agree on the FNO forward over identical weights.

    Both tiers run the identical spectral FNO forward (lift -> Fourier spectral
    convolution + pointwise skip + GELU per layer -> project) over the same
    weight archive, so the prediction and suppression factor are bit-exact up
    to floating-point round-off.
    """
    pytest.importorskip("scpn_fusion_rs")
    from scpn_fusion.core import _multi_compat_providers as providers

    weights = _write_weights(tmp_path)
    numpy_kernel = providers._load_numpy_fno()(weights)
    rust_kernel = providers._load_rust_fno()(weights)
    rng = np.random.default_rng(2026)
    for grid in (16, 24):
        field = rng.standard_normal((grid, grid)) * 0.5
        pred_numpy = numpy_kernel.predict(field)
        pred_rust = rust_kernel.predict(field)
        np.testing.assert_allclose(pred_numpy, pred_rust, rtol=1e-9, atol=1e-12)
        sup_numpy, _ = numpy_kernel.predict_and_suppress(field)
        sup_rust, _ = rust_kernel.predict_and_suppress(field)
        assert sup_numpy == pytest.approx(sup_rust, rel=1e-9, abs=1e-12)
