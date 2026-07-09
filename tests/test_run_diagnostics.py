# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Run Diagnostics Tests
"""Deterministic tests for diagnostics demo runtime entry point."""

from __future__ import annotations

import importlib
import inspect
import logging
import types
from pathlib import Path
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_fusion.diagnostics.run_diagnostics as run_diagnostics_mod
from scpn_fusion.diagnostics.run_diagnostics import _build_sensor_suite, run_diag_demo

FloatArray = NDArray[np.float64]


class _DummyKernel:
    def __init__(self, _config_path: str) -> None:
        self.R = np.linspace(4.0, 8.0, 33)
        self.Z = np.linspace(-2.0, 2.0, 33)
        self.NR = len(self.R)
        self.NZ = len(self.Z)
        self.dR = float(self.R[1] - self.R[0])
        self.dZ = float(self.Z[1] - self.Z[0])
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.exp(-((self.RR - 6.0) ** 2 + self.ZZ**2))

    def solve_equilibrium(self) -> None:
        return None


class _ZeroAxisKernel(_DummyKernel):
    """Kernel whose magnetic axis flux exercises the zero-axis fallback."""

    def __init__(self, _config_path: str) -> None:
        """Create a valid kernel with an all-zero Psi grid."""
        super().__init__(_config_path)
        self.Psi = np.zeros_like(self.Psi, dtype=np.float64)


class _EmptyPsiKernel(_DummyKernel):
    """Kernel with an invalid empty Psi grid."""

    def __init__(self, _config_path: str) -> None:
        """Create an invalid empty equilibrium grid."""
        super().__init__(_config_path)
        self.Psi = np.array([], dtype=np.float64)


class _DummySensorSuite:
    def __init__(self, kernel: _DummyKernel) -> None:
        self.kernel = kernel
        self.bolo_chords: list[tuple[FloatArray, FloatArray]] = [
            (np.array([6.0, 5.0]), np.array([4.0, -4.0])),
            (np.array([6.0, 5.0]), np.array([5.5, -4.0])),
            (np.array([6.0, 5.0]), np.array([7.0, -4.0])),
            (np.array([6.0, 5.0]), np.array([8.5, -4.0])),
        ]

    def measure_magnetics(self) -> FloatArray:
        return np.linspace(0.1, 2.0, 20, dtype=np.float64)

    def measure_bolometer(self, emission_profile: FloatArray) -> FloatArray:
        base = float(np.mean(np.asarray(emission_profile, dtype=np.float64)))
        return np.array([base + 0.01 * i for i in range(len(self.bolo_chords))], dtype=np.float64)

    def visualize_setup(self) -> Figure:
        fig, ax = plt.subplots()
        ax.plot([4.0, 8.0], [0.0, 0.0], "k-")
        ax.set_title("Dummy Sensor Setup")
        return fig


class _DummyTomography:
    def __init__(self, sensors: _DummySensorSuite) -> None:
        self.sensors = sensors

    def reconstruct(self, signals: FloatArray) -> FloatArray:
        level = float(np.mean(np.asarray(signals, dtype=np.float64)))
        return np.full(
            (self.sensors.kernel.NZ, self.sensors.kernel.NR),
            level,
            dtype=np.float64,
        )

    def plot_reconstruction(self, ground_truth: FloatArray, reconstruction: FloatArray) -> Figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        ax1.imshow(ground_truth, origin="lower", cmap="hot")
        ax2.imshow(reconstruction, origin="lower", cmap="hot")
        return fig


class _PlotFailingTomography(_DummyTomography):
    """Tomography double that fails during plot generation."""

    def plot_reconstruction(self, ground_truth: FloatArray, reconstruction: FloatArray) -> Figure:
        """Raise the deterministic plotting failure used by tests."""
        _ = ground_truth, reconstruction
        raise RuntimeError("forced tomography plot failure")


def test_run_diag_demo_returns_finite_summary_without_figures() -> None:
    summary = run_diag_demo(
        config_path="dummy.json",
        output_dir="unused",
        seed=123,
        save_figures=False,
        verbose=False,
        kernel_factory=_DummyKernel,
        sensor_factory=_DummySensorSuite,
        tomography_factory=_DummyTomography,
    )
    for key in (
        "seed",
        "config_path",
        "mag_channels",
        "bolo_channels",
        "phantom_sum",
        "reconstruction_sum",
        "reconstruction_rmse",
        "plot_saved",
    ):
        assert key in summary
    assert summary["seed"] == 123
    assert summary["config_path"] == "dummy.json"
    assert summary["mag_channels"] == 20
    assert summary["bolo_channels"] == 4
    assert summary["plot_saved"] is False
    assert summary["plot_error"] is None
    assert summary["tomography_path"] is None
    assert summary["sensor_geometry_path"] is None
    assert np.isfinite(summary["phantom_sum"])
    assert np.isfinite(summary["reconstruction_rmse"])


def test_run_diag_demo_is_deterministic_for_fixed_seed() -> None:
    a = run_diag_demo(
        config_path="dummy.json",
        output_dir="unused",
        seed=77,
        save_figures=False,
        verbose=False,
        kernel_factory=_DummyKernel,
        sensor_factory=_DummySensorSuite,
        tomography_factory=_DummyTomography,
    )
    b = run_diag_demo(
        config_path="dummy.json",
        output_dir="unused",
        seed=77,
        save_figures=False,
        verbose=False,
        kernel_factory=_DummyKernel,
        sensor_factory=_DummySensorSuite,
        tomography_factory=_DummyTomography,
    )
    for key in (
        "phantom_sum",
        "reconstruction_sum",
        "reconstruction_rmse",
    ):
        assert a[key] == pytest.approx(b[key], rel=0.0, abs=0.0)


def test_run_diag_demo_saves_expected_figures(tmp_path: Path) -> None:
    summary = run_diag_demo(
        config_path="dummy.json",
        output_dir=tmp_path,
        seed=9,
        save_figures=True,
        verbose=False,
        kernel_factory=_DummyKernel,
        sensor_factory=_DummySensorSuite,
        tomography_factory=_DummyTomography,
    )
    assert summary["plot_saved"] is True
    assert summary["plot_error"] is None
    tomo_path = Path(str(summary["tomography_path"]))
    geom_path = Path(str(summary["sensor_geometry_path"]))
    assert tomo_path.exists()
    assert geom_path.exists()


def test_run_diag_demo_does_not_mutate_global_numpy_rng_state() -> None:
    np.random.seed(97531)
    state = np.random.get_state()

    run_diag_demo(
        config_path="dummy.json",
        output_dir="unused",
        seed=101,
        save_figures=False,
        verbose=False,
        kernel_factory=_DummyKernel,
        sensor_factory=_DummySensorSuite,
        tomography_factory=_DummyTomography,
    )

    observed = float(np.random.random())
    np.random.set_state(state)
    expected = float(np.random.random())
    assert observed == expected


def test_sensor_suite_builder_passes_rng_when_supported() -> None:
    """Sensor factories declaring rng receive the scoped generator directly."""
    rng = np.random.default_rng(8)
    kernel = object()

    def factory(
        kernel_arg: object, *, rng: np.random.Generator
    ) -> tuple[object, np.random.Generator]:
        return kernel_arg, rng

    built = _build_sensor_suite(factory, kernel, seed=123, rng=rng)
    assert built == (kernel, rng)


def test_sensor_suite_builder_passes_seed_when_supported() -> None:
    """Sensor factories declaring seed receive the deterministic seed."""
    kernel = object()

    def factory(kernel_arg: object, *, seed: int) -> tuple[object, int]:
        return kernel_arg, seed

    built = _build_sensor_suite(factory, kernel, seed=321, rng=np.random.default_rng(9))
    assert built == (kernel, 321)


def test_sensor_suite_builder_falls_back_when_signature_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uninspectable factories still get the runtime rng fallback."""
    rng = np.random.default_rng(10)
    kernel = object()

    def raise_signature(_factory: object) -> None:
        raise ValueError("forced signature failure")

    def factory(
        kernel_arg: object, *, rng: np.random.Generator
    ) -> tuple[object, np.random.Generator]:
        return kernel_arg, rng

    monkeypatch.setattr(inspect, "signature", raise_signature)
    built = _build_sensor_suite(factory, kernel, seed=77, rng=rng)
    assert built == (kernel, rng)


def test_run_diag_demo_rejects_empty_psi_grid() -> None:
    """Diagnostics reject a kernel with no equilibrium grid values."""
    with pytest.raises(ValueError, match="Kernel Psi grid is empty"):
        run_diag_demo(
            config_path="dummy.json",
            output_dir="unused",
            seed=1,
            save_figures=False,
            verbose=False,
            kernel_factory=_EmptyPsiKernel,
            sensor_factory=_DummySensorSuite,
            tomography_factory=_DummyTomography,
        )


def test_run_diag_demo_handles_zero_axis_flux() -> None:
    """A zero magnetic-axis flux uses the bounded phantom fallback."""
    summary = run_diag_demo(
        config_path="dummy.json",
        output_dir="unused",
        seed=2,
        save_figures=False,
        verbose=False,
        kernel_factory=_ZeroAxisKernel,
        sensor_factory=_DummySensorSuite,
        tomography_factory=_DummyTomography,
    )
    assert np.isfinite(summary["phantom_sum"])


def test_run_diag_demo_verbose_plot_save_logs_paths(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Verbose figure saves report generated artifacts through logging."""
    caplog.set_level(logging.INFO, logger="scpn_fusion.diagnostics.run_diagnostics")
    summary = run_diag_demo(
        config_path="dummy.json",
        output_dir=tmp_path,
        seed=3,
        save_figures=True,
        verbose=True,
        kernel_factory=_DummyKernel,
        sensor_factory=_DummySensorSuite,
        tomography_factory=_DummyTomography,
    )
    assert summary["plot_saved"] is True
    assert "Tomography_Result.png" in caplog.text
    assert "Sensor_Geometry.png" in caplog.text


def test_run_diag_demo_records_plot_error() -> None:
    """Plot failures are captured in the summary instead of escaping."""
    summary = run_diag_demo(
        config_path="dummy.json",
        output_dir="unused",
        seed=4,
        save_figures=True,
        verbose=False,
        kernel_factory=_DummyKernel,
        sensor_factory=_DummySensorSuite,
        tomography_factory=_PlotFailingTomography,
    )
    assert summary["plot_saved"] is False
    assert summary["plot_error"] == "RuntimeError: forced tomography plot failure"


def test_fusion_kernel_import_falls_back_when_rust_symbol_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The module-level FusionKernel import falls back to the Python kernel."""

    class _FallbackKernel:
        """Fallback kernel sentinel used during module reload."""

    original_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name == "scpn_fusion.core._rust_compat":
            raise ImportError("forced rust import failure")
        if name == "scpn_fusion.core.fusion_kernel":
            return types.SimpleNamespace(FusionKernel=_FallbackKernel)
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    reloaded = importlib.reload(run_diagnostics_mod)
    try:
        assert reloaded.FusionKernel is _FallbackKernel
    finally:
        monkeypatch.undo()
        importlib.reload(run_diagnostics_mod)


def test_run_diag_demo_verbose_path_logs_progress(caplog: pytest.LogCaptureFixture) -> None:
    """Verbose diagnostics progress is routed through structured logging."""
    caplog.set_level(logging.INFO, logger="scpn_fusion.diagnostics.run_diagnostics")

    summary = run_diag_demo(
        config_path="dummy.json",
        output_dir="unused",
        seed=55,
        save_figures=False,
        verbose=True,
        kernel_factory=_DummyKernel,
        sensor_factory=_DummySensorSuite,
        tomography_factory=_DummyTomography,
    )

    assert summary["mag_channels"] == 20
    assert "Measuring Signals" in caplog.text
    assert "Magnetic Probes: 20 channels" in caplog.text
