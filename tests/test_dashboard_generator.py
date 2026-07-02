# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the dashboard generator plotting orchestration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import matplotlib
import numpy as np
import pytest
from numpy.typing import NDArray

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import scpn_fusion.ui.dashboard_generator as dashboard_module
from scpn_fusion.ui.dashboard_generator import DashboardGenerator
from scpn_fusion.ui.dashboard_generator import run_dashboard


FloatArray = NDArray[np.float64]


class _DummyKernel:
    """Synthetic solved equilibrium used by dashboard tests."""

    cfg: dict[str, Any]
    R: FloatArray
    Z: FloatArray
    NR: int
    NZ: int
    dR: float
    dZ: float
    RR: FloatArray
    ZZ: FloatArray
    Psi: FloatArray
    config_path: str | None
    solved: bool

    def __init__(self, config_path: str | None = None) -> None:
        self.cfg = {"dimensions": {"R_min": 1.0, "R_max": 5.0, "Z_min": -2.0, "Z_max": 2.0}}
        self.R = np.linspace(1.0, 5.0, 60, dtype=np.float64)
        self.Z = np.linspace(-2.0, 2.0, 60, dtype=np.float64)
        self.NR = self.R.size
        self.NZ = self.Z.size
        self.dR = float(self.R[1] - self.R[0])
        self.dZ = float(self.Z[1] - self.Z[0])
        rr, zz = np.meshgrid(self.R, self.Z)
        self.RR = cast(FloatArray, rr)
        self.ZZ = cast(FloatArray, zz)
        self.Psi = cast(FloatArray, (self.RR - 3.0) ** 2 + (self.ZZ / 1.5) ** 2)
        self.config_path = config_path
        self.solved = False

    def solve_equilibrium(self) -> None:
        """Record that the dashboard runner invoked the solver."""
        self.solved = True


class _IncompleteKernel:
    """Kernel fixture intentionally missing dashboard-required fields."""

    cfg: dict[str, Any]

    def __init__(self) -> None:
        self.cfg = {"dimensions": {"R_min": 1.0, "R_max": 5.0, "Z_min": -2.0, "Z_max": 2.0}}


def test_generate_poincare_plot_traces_fieldline_points() -> None:
    """The plot contains one rendered trajectory for each requested seed line."""
    kernel = _DummyKernel()
    fig = DashboardGenerator(kernel).generate_poincare_plot(n_lines=4, n_transits=25)
    try:
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert len(ax.lines) == 4
        for line in ax.lines:
            x_data = np.asarray(line.get_xdata())
            y_data = np.asarray(line.get_ydata())
            assert x_data.size >= 2
            assert y_data.size >= 2
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("n_lines", "n_transits", "message"),
    [
        (0, 1, "n_lines"),
        (1, 0, "n_transits"),
    ],
)
def test_generate_poincare_plot_rejects_non_positive_counts(
    n_lines: int,
    n_transits: int,
    message: str,
) -> None:
    """Seed-line and transit counts must be positive."""
    generator = DashboardGenerator(_DummyKernel())

    with pytest.raises(ValueError, match=message):
        generator.generate_poincare_plot(n_lines=n_lines, n_transits=n_transits)


def test_constructor_rejects_incomplete_kernel() -> None:
    """A malformed kernel fails before plotting starts."""
    with pytest.raises(ValueError, match="required fields"):
        DashboardGenerator(cast(Any, _IncompleteKernel()))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_dimension", "R_min"),
        ("nonnumeric_dimension", "numeric"),
        ("nonfinite_dimension", "finite"),
        ("missing_dimensions_mapping", "dimensions mapping"),
        ("inverted_dimensions", "R_min < R_max"),
        ("wrong_array_rank", "non-empty 2D array"),
        ("nonfinite_array", "finite values"),
        ("axis_too_short", "at least two points"),
        ("decreasing_axis", "strictly increasing"),
        ("declared_count", "declared count"),
        ("invalid_spacing", "finite and positive"),
        ("mismatched_spacing", "uniform grid spacing"),
        ("field_shape", "grid shape"),
    ],
)
def test_constructor_rejects_invalid_kernel_contracts(case: str, message: str) -> None:
    """Dashboard kernels must expose finite, uniform, shape-consistent fields."""
    kernel = _DummyKernel()

    if case == "missing_dimension":
        del kernel.cfg["dimensions"]["R_min"]
    elif case == "nonnumeric_dimension":
        kernel.cfg = cast(
            Any,
            {"dimensions": {"R_min": "bad", "R_max": 5.0, "Z_min": -2.0, "Z_max": 2.0}},
        )
    elif case == "nonfinite_dimension":
        kernel.cfg["dimensions"]["R_min"] = np.inf
    elif case == "missing_dimensions_mapping":
        kernel.cfg = cast(Any, {})
    elif case == "inverted_dimensions":
        kernel.cfg["dimensions"]["R_max"] = 0.5
    elif case == "wrong_array_rank":
        kernel.RR = np.array([1.0, 2.0], dtype=np.float64)
    elif case == "nonfinite_array":
        kernel.Psi[0, 0] = np.nan
    elif case == "axis_too_short":
        kernel.R = np.array([1.0], dtype=np.float64)
    elif case == "decreasing_axis":
        kernel.R = np.flip(kernel.R).copy()
    elif case == "declared_count":
        kernel.NR += 1
    elif case == "invalid_spacing":
        kernel.dR = 0.0
    elif case == "mismatched_spacing":
        kernel.dR *= 2.0
    elif case == "field_shape":
        kernel.Psi = kernel.Psi[:-1, :]
    else:  # pragma: no cover - parametrization guard.
        raise AssertionError(f"unhandled validation case: {case}")

    with pytest.raises(ValueError, match=message):
        DashboardGenerator(kernel)


def test_get_psi_clamps_coordinates_to_grid_bounds() -> None:
    """Nearest-neighbour Psi lookup clamps out-of-domain coordinates."""
    kernel = _DummyKernel()
    generator = DashboardGenerator(kernel)

    assert generator._get_psi(r=-100.0, z=100.0) == pytest.approx(float(kernel.Psi[-1, 0]))


def test_load_fusion_kernel_returns_none_on_missing_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional FusionKernel shim fails closed when the core import is absent."""

    def _raise_import_error(_module_name: str) -> object:
        raise ImportError("missing test backend")

    monkeypatch.setattr(
        "scpn_fusion.ui.dashboard_generator.importlib.import_module",
        _raise_import_error,
    )

    assert dashboard_module._load_fusion_kernel() is None


def test_load_fusion_kernel_returns_none_without_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional FusionKernel shim requires the imported module attribute."""
    monkeypatch.setattr(
        "scpn_fusion.ui.dashboard_generator.importlib.import_module",
        lambda _module_name: SimpleNamespace(),
    )

    assert dashboard_module._load_fusion_kernel() is None


def test_default_config_path_uses_packaged_iter_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI helper resolves through the data-path adapter."""
    expected = Path("/tmp/scpn-test/iter_config.toml")
    monkeypatch.setattr(dashboard_module, "default_iter_config_path", lambda: expected)

    assert dashboard_module._default_config_path() == str(expected)


def test_run_dashboard_requires_fusion_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dashboard runner fails explicitly when the optional kernel is unavailable."""
    monkeypatch.setattr(dashboard_module, "FusionKernel", None)

    with pytest.raises(RuntimeError, match="FusionKernel"):
        run_dashboard("config.toml")


def test_run_dashboard_solves_saves_and_closes_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner solves, writes the requested PNG, closes the figure, and returns the path."""
    created: list[_DummyKernel] = []

    class _KernelFactory:
        """Factory matching the optional FusionKernel constructor contract."""

        def __call__(self, config_path: str) -> _DummyKernel:
            kernel = _DummyKernel(config_path)
            created.append(kernel)
            return kernel

    closed: list[Figure] = []
    original_close = plt.close

    def _record_close(fig: Figure | None = None) -> None:
        if isinstance(fig, Figure):
            closed.append(fig)
        original_close(fig)

    output_path = tmp_path / "nested" / "Poincare_Topology.png"
    monkeypatch.setattr(dashboard_module, "FusionKernel", _KernelFactory())
    monkeypatch.setattr(plt, "close", _record_close)

    result = run_dashboard(
        "synthetic-config.toml",
        output_path=output_path,
        n_lines=3,
        n_transits=4,
    )

    assert result == output_path
    assert output_path.is_file()
    assert created[0].config_path == "synthetic-config.toml"
    assert created[0].solved is True
    assert closed
