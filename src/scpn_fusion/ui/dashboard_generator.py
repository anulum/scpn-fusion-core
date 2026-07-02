# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Visualization helpers for SCPN Fusion dashboards.

The generator produces high-level geometry diagnostics derived from a solved
`FusionKernel` equilibrium instance.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from scpn_fusion._data_paths import default_iter_config_path

logger = logging.getLogger(__name__)

FloatArray: TypeAlias = NDArray[np.float64]


class _DashboardKernel(Protocol):
    """Structural contract consumed by the dashboard renderer."""

    RR: FloatArray
    ZZ: FloatArray
    Psi: FloatArray
    cfg: dict[str, Any]
    R: FloatArray
    Z: FloatArray
    NR: int
    NZ: int
    dR: float
    dZ: float


class _SolvableDashboardKernel(_DashboardKernel, Protocol):
    """Dashboard kernel contract required by the CLI runner."""

    def solve_equilibrium(self) -> None:
        """Solve or refresh the equilibrium before rendering."""


class _KernelFactory(Protocol):
    """Callable constructor contract for the optional FusionKernel import."""

    def __call__(self, config_path: str) -> _SolvableDashboardKernel:
        """Create a dashboard-compatible kernel for ``config_path``."""


@dataclass(frozen=True, slots=True)
class _DimensionBounds:
    """Finite plotting bounds from the kernel configuration."""

    r_min: float
    r_max: float
    z_min: float
    z_max: float


@dataclass(frozen=True, slots=True)
class _ValidatedKernelFields:
    """Typed arrays and metadata validated for dashboard rendering."""

    bounds: _DimensionBounds
    r_grid: FloatArray
    z_grid: FloatArray
    rr: FloatArray
    zz: FloatArray
    psi: FloatArray
    d_r: float
    d_z: float


def _load_fusion_kernel() -> _KernelFactory | None:
    """Load the optional ``FusionKernel`` constructor if the core backend exists."""
    try:
        module = importlib.import_module("scpn_fusion.core.fusion_kernel")
        return cast(_KernelFactory, module.FusionKernel)
    except (AttributeError, ImportError):
        return None


FusionKernel: _KernelFactory | None = _load_fusion_kernel()


def _finite_dimension(dimensions: Mapping[str, Any], name: str) -> float:
    """Return one finite numeric dimension from a kernel config mapping."""
    if name not in dimensions:
        raise ValueError(f"kernel cfg dimensions must include {name}")
    try:
        value = float(dimensions[name])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"kernel cfg dimension {name} must be numeric") from exc
    if not np.isfinite(value):
        raise ValueError(f"kernel cfg dimension {name} must be finite")
    return value


def _require_dimension_bounds(cfg: Mapping[str, Any]) -> _DimensionBounds:
    """Validate and return the plotting bounds declared by a kernel config."""
    dimensions = cfg.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise ValueError("kernel cfg must include a dimensions mapping")
    dimension_map = cast(Mapping[str, Any], dimensions)
    bounds = _DimensionBounds(
        r_min=_finite_dimension(dimension_map, "R_min"),
        r_max=_finite_dimension(dimension_map, "R_max"),
        z_min=_finite_dimension(dimension_map, "Z_min"),
        z_max=_finite_dimension(dimension_map, "Z_max"),
    )
    if not bounds.r_min < bounds.r_max or not bounds.z_min < bounds.z_max:
        raise ValueError("kernel cfg dimensions must satisfy R_min < R_max and Z_min < Z_max")
    return bounds


def _coerce_float_array(name: str, values: object, *, ndim: int) -> FloatArray:
    """Return a finite float64 array with the requested dimensionality."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim or array.size == 0:
        raise ValueError(f"{name} must be a non-empty {ndim}D array")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_axis(name: str, axis: FloatArray, *, declared_count: int, spacing: float) -> None:
    """Validate one uniform coordinate axis used by nearest-neighbour lookup."""
    if axis.size < 2:
        raise ValueError(f"{name} must contain at least two points")
    diffs = np.diff(axis)
    if not bool(np.all(diffs > 0.0)):
        raise ValueError(f"{name} must be strictly increasing")
    if declared_count != axis.size:
        raise ValueError(f"{name} declared count must match the grid size")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"{name} spacing must be finite and positive")
    if not bool(np.allclose(diffs, spacing, rtol=1e-6, atol=1e-12)):
        raise ValueError(f"{name} spacing must match the uniform grid spacing")


def _validate_field_shapes(
    *,
    rr: FloatArray,
    zz: FloatArray,
    psi: FloatArray,
    expected_shape: tuple[int, int],
) -> None:
    """Validate two-dimensional dashboard fields against the kernel grid shape."""
    for name, field in (("RR", rr), ("ZZ", zz), ("Psi", psi)):
        if field.shape != expected_shape:
            raise ValueError(f"{name} shape must match (NZ, NR) grid shape {expected_shape}")


def _validate_kernel_fields(kernel: _DashboardKernel) -> _ValidatedKernelFields:
    """Validate and normalize the kernel surface consumed by the dashboard."""
    bounds = _require_dimension_bounds(kernel.cfg)
    r_grid = _coerce_float_array("R", kernel.R, ndim=1)
    z_grid = _coerce_float_array("Z", kernel.Z, ndim=1)
    rr = _coerce_float_array("RR", kernel.RR, ndim=2)
    zz = _coerce_float_array("ZZ", kernel.ZZ, ndim=2)
    psi = _coerce_float_array("Psi", kernel.Psi, ndim=2)
    _validate_axis("R", r_grid, declared_count=kernel.NR, spacing=kernel.dR)
    _validate_axis("Z", z_grid, declared_count=kernel.NZ, spacing=kernel.dZ)
    _validate_field_shapes(rr=rr, zz=zz, psi=psi, expected_shape=(z_grid.size, r_grid.size))
    return _ValidatedKernelFields(
        bounds=bounds,
        r_grid=r_grid,
        z_grid=z_grid,
        rr=rr,
        zz=zz,
        psi=psi,
        d_r=float(kernel.dR),
        d_z=float(kernel.dZ),
    )


class DashboardGenerator:
    """Build and render Poincaré-style diagnostic plots.

    The class acts as a compatibility-safe visualization layer over a solved
    equilibrium object. It deliberately avoids solving physics itself and only
    renders diagnostics based on already computed equilibrium fields.

    Attributes
    ----------
    kernel : object
        Solved equilibrium kernel exposing ``RR``, ``ZZ``, ``Psi``, ``cfg``,
        ``R``, ``Z``, ``NR``, ``NZ``, ``dR``, and ``dZ``.
    """

    def __init__(self, kernel: _DashboardKernel) -> None:
        """Create a generator bound to an already-solved equilibrium kernel.

        Parameters
        ----------
        kernel : object
            Object exposing the equilibrium fields used by the plotting path.

        Raises
        ------
        ValueError
            If the object does not expose the required field layout.
        """
        required = {"RR", "ZZ", "Psi", "cfg", "R", "Z", "NR", "NZ", "dR", "dZ"}
        missing = sorted(name for name in required if not hasattr(kernel, name))
        if missing:
            raise ValueError(
                "Kernel does not expose required fields for dashboard generation: "
                f"{', '.join(missing)}"
            )
        self.kernel = kernel
        self._fields = _validate_kernel_fields(kernel)

    def generate_poincare_plot(self, n_lines: int = 20, n_transits: int = 500) -> Figure:
        """Generate a lightweight Poincaré-like magnetic topology figure.

        The implementation is a surrogate map suitable for quick quality checks. It
        plots both sampled field-line-like trajectories and flux-surface contours.

        Parameters
        ----------
        n_lines : int, default=20
            Number of seeded trajectories.
        n_transits : int, default=500
            Iteration count per trajectory.

        Returns
        -------
        Figure
            Matplotlib figure containing the generated topology view.
        """
        if n_lines <= 0:
            raise ValueError("n_lines must be greater than zero")
        if n_transits <= 0:
            raise ValueError("n_transits must be greater than zero")

        logger.info("Generating Poincaré plot...")

        # Seed points along the midplane (Z=0, R=R_min..R_max)
        bounds = self._fields.bounds
        R_start = np.linspace(
            bounds.r_min + 0.1,
            bounds.r_max - 0.1,
            n_lines,
        )
        r_min = bounds.r_min
        r_max = bounds.r_max
        z_min = bounds.z_min
        z_max = bounds.z_max
        r0 = 0.5 * (r_min + r_max)
        z0 = 0.5 * (z_min + z_max)
        max_psi = max(float(np.max(np.abs(self._fields.psi))), 1.0e-9)
        max_radius = max(r_max - r0, r0 - r_min, z_max - z0, z0 - z_min)

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.contour(
            self._fields.rr,
            self._fields.zz,
            self._fields.psi,
            levels=20,
            colors="gray",
            alpha=0.2,
        )

        for i in range(n_lines):
            r, z = float(R_start[i]), 0.0
            path_r = [r]
            path_z = [z]

            for _ in range(n_transits):
                psi_val = self._get_psi(r, z)
                q = 1.0 + 3.0 * (float(psi_val) / max_psi) ** 2

                k = 2e-4
                theta = np.arctan2(z - z0, r - r0)
                theta_new = theta + 2 * np.pi / q + k * np.sin(2 * theta)
                radial_kick = (k / 10.0) * np.sin(2 * theta)

                radius = float(np.hypot(r - r0, z - z0))
                radius = float(np.clip(radius + radial_kick, 0.02, max_radius))
                r = float(np.clip(r0 + radius * np.cos(theta_new), r_min, r_max))
                z = float(np.clip(z0 + radius * np.sin(theta_new), z_min, z_max))

                path_r.append(r)
                path_z.append(z)

            ax.plot(path_r, path_z, ".", ms=0.8, alpha=0.35)

        # High-fidelity contour proxy for axisymmetric topology.
        ax.contour(self._fields.rr, self._fields.zz, self._fields.psi, levels=50, cmap="nipy_spectral")
        ax.set_title("Magnetic Topology (Flux Surfaces)")
        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")
        ax.set_aspect("equal")
        return fig

    def _get_psi(self, r: float, z: float) -> float:
        """Return nearest-neighbour Psi value for a coordinate.

        Parameters
        ----------
        r : float
            Radial coordinate in meters.
        z : float
            Vertical coordinate in meters.

        Returns
        -------
        float
            Interpolated nearest-neighbour flux value.
        """
        ir = int((r - self._fields.r_grid[0]) / self._fields.d_r)
        iz = int((z - self._fields.z_grid[0]) / self._fields.d_z)
        ir = int(np.clip(ir, 0, self._fields.r_grid.size - 1))
        iz = int(np.clip(iz, 0, self._fields.z_grid.size - 1))
        return float(self._fields.psi[iz, ir])


def run_dashboard(
    config_path: str | Path,
    output_path: str | Path = "Poincare_Topology.png",
    *,
    n_lines: int = 20,
    n_transits: int = 500,
) -> Path:
    """Run the Poincaré dashboard end-to-end from a config file.

    This helper is intentionally defensive for CLI use and exits with explicit
    diagnostics when the compatibility shim is unavailable or the config is not
    loadable.

    Parameters
    ----------
    config_path : str or Path
        Path to solver configuration.
    output_path : str or Path, default="Poincare_Topology.png"
        PNG path to write. Parent directories are created when needed.
    n_lines : int, default=20
        Number of seeded trajectories in the generated plot.
    n_transits : int, default=500
        Iteration count per trajectory.

    Returns
    -------
    Path
        Resolved output path used for the PNG render.

    Raises
    ------
    RuntimeError
        If ``FusionKernel`` is unavailable in the runtime environment.
    """
    if FusionKernel is None:
        raise RuntimeError("FusionKernel is not available in this environment")

    output = Path(output_path)
    kernel = FusionKernel(str(config_path))
    kernel.solve_equilibrium()

    gen = DashboardGenerator(kernel)
    fig = gen.generate_poincare_plot(n_lines=n_lines, n_transits=n_transits)

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
    finally:
        plt.close(fig)
    logger.info("Dashboard saved: %s", output)
    return output


def _default_config_path() -> str:
    """Resolve the bundled default ITER configuration path for CLI execution."""
    return str(default_iter_config_path())


if __name__ == "__main__":
    cfg = _default_config_path()
    run_dashboard(cfg)
