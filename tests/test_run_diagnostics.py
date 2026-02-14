# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Run Diagnostics Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Deterministic tests for diagnostics demo runtime entry point."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scpn_fusion.diagnostics.run_diagnostics import run_diag_demo


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


class _DummySensorSuite:
    def __init__(self, kernel: _DummyKernel) -> None:
        self.kernel = kernel
        self.bolo_chords = [
            (np.array([6.0, 5.0]), np.array([4.0, -4.0])),
            (np.array([6.0, 5.0]), np.array([5.5, -4.0])),
            (np.array([6.0, 5.0]), np.array([7.0, -4.0])),
            (np.array([6.0, 5.0]), np.array([8.5, -4.0])),
        ]

    def measure_magnetics(self) -> np.ndarray:
        return np.linspace(0.1, 2.0, 20, dtype=np.float64)

    def measure_bolometer(self, emission_profile: np.ndarray) -> np.ndarray:
        base = float(np.mean(np.asarray(emission_profile, dtype=np.float64)))
        return np.array([base + 0.01 * i for i in range(len(self.bolo_chords))], dtype=np.float64)

    def visualize_setup(self):  # type: ignore[no-untyped-def]
        fig, ax = plt.subplots()
        ax.plot([4.0, 8.0], [0.0, 0.0], "k-")
        ax.set_title("Dummy Sensor Setup")
        return fig


class _DummyTomography:
    def __init__(self, sensors: _DummySensorSuite) -> None:
        self.sensors = sensors

    def reconstruct(self, signals: np.ndarray) -> np.ndarray:
        level = float(np.mean(np.asarray(signals, dtype=np.float64)))
        return np.full(
            (self.sensors.kernel.NZ, self.sensors.kernel.NR),
            level,
            dtype=np.float64,
        )

    def plot_reconstruction(self, ground_truth: np.ndarray, reconstruction: np.ndarray):  # type: ignore[no-untyped-def]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        ax1.imshow(ground_truth, origin="lower", cmap="hot")
        ax2.imshow(reconstruction, origin="lower", cmap="hot")
        return fig


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
    kwargs = dict(
        config_path="dummy.json",
        output_dir="unused",
        seed=77,
        save_figures=False,
        verbose=False,
        kernel_factory=_DummyKernel,
        sensor_factory=_DummySensorSuite,
        tomography_factory=_DummyTomography,
    )
    a = run_diag_demo(**kwargs)
    b = run_diag_demo(**kwargs)
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
