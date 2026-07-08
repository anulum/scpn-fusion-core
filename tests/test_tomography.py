# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tomography Tests
"""Tests for PlasmaTomography geometry/reconstruction paths."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray

import scpn_fusion.diagnostics.tomography as tomography_mod
from scpn_fusion.diagnostics.tomography import PlasmaTomography

FloatArray = NDArray[np.float64]


class _KernelStub:
    """Minimal kernel grid exposing the tomography coordinate axes."""

    def __init__(self) -> None:
        """Create a deterministic rectangular reconstruction domain."""
        self.R = np.linspace(4.0, 8.0, 33)
        self.Z = np.linspace(-2.0, 2.0, 33)


class _SensorStub:
    """Minimal bolometer sensor geometry for tomography tests."""

    def __init__(self) -> None:
        """Create diagonal bolometer chords crossing the plasma domain."""
        self.kernel = _KernelStub()
        origin = np.array([6.0, 5.0])
        targets_r = np.linspace(3.5, 8.5, 8)
        self.bolo_chords = [(origin, np.array([float(r), -4.0])) for r in targets_r]


def test_geometry_matrix_shape_and_support() -> None:
    """Geometry assembly creates a supported chord-by-pixel matrix."""
    tomo = PlasmaTomography(_SensorStub(), grid_res=12, verbose=False)
    assert tomo.A.shape == (8, 144)
    assert int(np.count_nonzero(tomo.A)) > 0


def test_reconstruct_returns_nonnegative_deterministic_grid() -> None:
    """Automatic reconstruction returns repeatable non-negative grids."""
    tomo = PlasmaTomography(_SensorStub(), grid_res=10, verbose=False)
    signals = np.linspace(0.2, 1.0, 8, dtype=np.float64)
    a = tomo.reconstruct(signals)
    b = tomo.reconstruct(signals)
    assert a.shape == (10, 10)
    assert np.min(a) >= 0.0
    np.testing.assert_allclose(a, b, rtol=0.0, atol=0.0)


def test_reconstruct_rejects_signal_length_mismatch() -> None:
    """Reconstruction rejects a signal vector that cannot match chords."""
    tomo = PlasmaTomography(_SensorStub(), grid_res=10, verbose=False)
    with pytest.raises(ValueError, match="signals length mismatch"):
        tomo.reconstruct(np.array([1.0, 2.0, 3.0], dtype=np.float64))


def _force_rust_unavailable(monkeypatch: pytest.MonkeyPatch, tomo: PlasmaTomography) -> None:
    """Make the Rust tomography backend unresolvable for fallback tests."""

    def _raise(self: PlasmaTomography) -> None:
        raise ImportError("forced-unavailable Rust tomography (test)")

    monkeypatch.setattr(PlasmaTomography, "_load_rust_backend", _raise)
    tomo._rust_backend = None


def test_reconstruct_falls_back_when_lsq_linear_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto reconstruction falls back to SART when Rust and SciPy are absent."""
    monkeypatch.setattr(tomography_mod, "lsq_linear", None)
    tomo = tomography_mod.PlasmaTomography(_SensorStub(), grid_res=8, verbose=False)
    _force_rust_unavailable(monkeypatch, tomo)
    out = tomo.reconstruct(np.linspace(0.1, 0.8, 8, dtype=np.float64))
    assert out.shape == (8, 8)
    assert np.min(out) >= 0.0


def test_auto_falls_back_to_lsq_linear_without_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto method degrades to the SciPy path when Rust is unavailable."""
    tomo = PlasmaTomography(_SensorStub(), grid_res=8, verbose=False)
    _force_rust_unavailable(monkeypatch, tomo)
    out = tomo.reconstruct(np.linspace(0.1, 0.8, 8, dtype=np.float64))
    assert out.shape == (8, 8)
    assert np.min(out) >= 0.0


def test_rust_backend_matches_scipy_reconstruction() -> None:
    """The Rust Tikhonov-NNLS solution matches the SciPy lsq_linear solution.

    Both backends assemble the identical endpoint-inclusive geometry matrix and
    minimise the same convex objective, so the reconstructions of a synthetic
    phantom agree to solver tolerance.
    """
    pytest.importorskip("scpn_fusion_rs")
    tomo = PlasmaTomography(_SensorStub(), grid_res=10, verbose=False)
    rng = np.random.default_rng(7)
    phantom = rng.uniform(0.0, 1.0, size=tomo.n_pixels)
    signals = tomo.A @ phantom

    rust_solution = tomo.reconstruct(signals, method="rust")
    scipy_solution = tomo.reconstruct(signals, method="lsq_linear")

    assert rust_solution.shape == scipy_solution.shape == (10, 10)
    scale = float(np.linalg.norm(scipy_solution))
    rel_l2 = float(np.linalg.norm(rust_solution - scipy_solution)) / max(scale, 1e-30)
    assert rel_l2 < 5.0e-2, f"rust vs scipy rel L2 {rel_l2}"


def test_rust_reconstruction_fits_signals_like_scipy() -> None:
    """Forward-projected Rust reconstruction fits the measured signals.

    Cross-checks the geometry contract indirectly: projecting the Rust
    reconstruction through the Python geometry matrix must reproduce the
    signals about as well as the SciPy reconstruction does.
    """
    pytest.importorskip("scpn_fusion_rs")
    tomo = PlasmaTomography(_SensorStub(), grid_res=10, verbose=False)
    rng = np.random.default_rng(11)
    phantom = rng.uniform(0.0, 1.0, size=tomo.n_pixels)
    signals = tomo.A @ phantom

    rust_solution = tomo.reconstruct(signals, method="rust").reshape(-1)
    scipy_solution = tomo.reconstruct(signals, method="lsq_linear").reshape(-1)
    rust_residual = float(np.linalg.norm(tomo.A @ rust_solution - signals))
    scipy_residual = float(np.linalg.norm(tomo.A @ scipy_solution - signals))
    assert rust_residual <= scipy_residual * 1.5 + 1e-9


def test_auto_prefers_rust_when_available() -> None:
    """Auto method selects the Rust backend when the extension resolves."""
    pytest.importorskip("scpn_fusion_rs")
    tomo = PlasmaTomography(_SensorStub(), grid_res=8, verbose=False)
    signals = np.linspace(0.2, 1.0, 8, dtype=np.float64)
    auto_solution = tomo.reconstruct(signals, method="auto")
    rust_solution = tomo.reconstruct(signals, method="rust")
    np.testing.assert_allclose(auto_solution, rust_solution, rtol=0.0, atol=0.0)


def test_verbose_geometry_logs_progress(capsys: pytest.CaptureFixture[str]) -> None:
    """Verbose construction emits geometry-build progress output."""
    PlasmaTomography(_SensorStub(), grid_res=8, verbose=True)
    assert "[Tomography] Building Geometry Matrix A..." in capsys.readouterr().out


def test_ridge_reconstruction_returns_clipped_grid() -> None:
    """Analytic ridge fallback returns a non-negative reconstruction grid."""
    tomo = PlasmaTomography(_SensorStub(), grid_res=8, verbose=False)
    out = tomo.reconstruct(np.linspace(0.1, 0.8, 8, dtype=np.float64), method="ridge")
    assert out.shape == (8, 8)
    assert np.min(out) >= 0.0


def test_ridge_reconstruction_uses_lstsq_when_solve_is_singular(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Analytic ridge fallback uses least squares when direct solve fails."""
    tomo = PlasmaTomography(_SensorStub(), grid_res=8, verbose=False)

    def raise_singular(_lhs: FloatArray, _rhs: FloatArray) -> FloatArray:
        raise np.linalg.LinAlgError("forced singular ridge system")

    def fallback_lstsq(
        lhs: FloatArray,
        rhs: FloatArray,
        rcond: float | None = None,
    ) -> tuple[FloatArray, FloatArray, np.int32, FloatArray]:
        assert lhs.shape == (tomo.n_pixels, tomo.n_pixels)
        assert rhs.shape == (tomo.n_pixels,)
        assert rcond is None
        solution = np.linspace(-0.5, 0.5, tomo.n_pixels, dtype=np.float64)
        return solution, np.empty(0, dtype=np.float64), np.int32(tomo.n_pixels), np.empty(0, dtype=np.float64)

    monkeypatch.setattr(np.linalg, "solve", raise_singular)
    monkeypatch.setattr(np.linalg, "lstsq", fallback_lstsq)

    out = tomo.reconstruct(np.linspace(0.1, 0.8, 8, dtype=np.float64), method="ridge")
    assert out.shape == (8, 8)
    assert np.min(out) == pytest.approx(0.0)
    assert np.max(out) == pytest.approx(0.5)


def test_plot_reconstruction_returns_two_panel_figure() -> None:
    """Plotting returns a figure with ground-truth and reconstruction axes."""
    tomo = PlasmaTomography(_SensorStub(), grid_res=8, verbose=False)
    ground_truth = np.eye(8, dtype=np.float64)
    reconstruction = np.fliplr(ground_truth)

    fig = tomo.plot_reconstruction(ground_truth, reconstruction)
    try:
        assert isinstance(fig, Figure)
        assert [axis.get_title() for axis in fig.axes] == [
            "Ground Truth (Phantom)",
            "Tomographic Reconstruction",
        ]
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("grid_res", "lambda_reg", "match"),
    [
        (3, 0.1, "grid_res"),
        (8, -1.0e-6, "lambda_reg"),
        (8, float("nan"), "lambda_reg"),
    ],
)
def test_constructor_rejects_invalid_inputs(
    grid_res: int,
    lambda_reg: float,
    match: str,
) -> None:
    """Constructor rejects invalid grid and regularization settings."""
    with pytest.raises(ValueError, match=match):
        PlasmaTomography(
            _SensorStub(),
            grid_res=grid_res,
            lambda_reg=lambda_reg,
            verbose=False,
        )
