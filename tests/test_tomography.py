# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Tomography Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for PlasmaTomography geometry/reconstruction paths."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_fusion.diagnostics.tomography as tomography_mod
from scpn_fusion.diagnostics.tomography import PlasmaTomography


class _KernelStub:
    def __init__(self) -> None:
        self.R = np.linspace(4.0, 8.0, 33)
        self.Z = np.linspace(-2.0, 2.0, 33)


class _SensorStub:
    def __init__(self) -> None:
        self.kernel = _KernelStub()
        origin = np.array([6.0, 5.0])
        targets_r = np.linspace(3.5, 8.5, 8)
        self.bolo_chords = [
            (origin, np.array([float(r), -4.0])) for r in targets_r
        ]


def test_geometry_matrix_shape_and_support() -> None:
    tomo = PlasmaTomography(_SensorStub(), grid_res=12, verbose=False)
    assert tomo.A.shape == (8, 144)
    assert int(np.count_nonzero(tomo.A)) > 0


def test_reconstruct_returns_nonnegative_deterministic_grid() -> None:
    tomo = PlasmaTomography(_SensorStub(), grid_res=10, verbose=False)
    signals = np.linspace(0.2, 1.0, 8, dtype=np.float64)
    a = tomo.reconstruct(signals)
    b = tomo.reconstruct(signals)
    assert a.shape == (10, 10)
    assert np.min(a) >= 0.0
    np.testing.assert_allclose(a, b, rtol=0.0, atol=0.0)


def test_reconstruct_rejects_signal_length_mismatch() -> None:
    tomo = PlasmaTomography(_SensorStub(), grid_res=10, verbose=False)
    with pytest.raises(ValueError, match="signals length mismatch"):
        tomo.reconstruct(np.array([1.0, 2.0, 3.0], dtype=np.float64))


def test_reconstruct_falls_back_when_lsq_linear_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tomography_mod, "lsq_linear", None)
    tomo = tomography_mod.PlasmaTomography(_SensorStub(), grid_res=8, verbose=False)
    out = tomo.reconstruct(np.linspace(0.1, 0.8, 8, dtype=np.float64))
    assert out.shape == (8, 8)
    assert np.min(out) >= 0.0
