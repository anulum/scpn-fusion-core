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
from pathlib import Path
from typing import Any

import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

try:
    FusionKernel: Any | None = importlib.import_module(
        "scpn_fusion.core.fusion_kernel"
    ).FusionKernel
except (AttributeError, ImportError):
    FusionKernel = None

logger = logging.getLogger(__name__)


class DashboardGenerator:
    """Build and render Poincaré-style diagnostic plots.

    The class acts as a compatibility-safe visualization layer over a solved
    equilibrium object. It deliberately avoids solving physics itself and only
    renders diagnostics based on already computed equilibrium fields.

    Attributes
    ----------
        kernel: A solved equilibrium kernel exposing `RR`, `ZZ`, `Psi`, `cfg`, `R`,
            `Z`, `NR`, `NZ`, `dR`, and `dZ`.
    """

    def __init__(self, kernel: Any) -> None:
        """Create a generator bound to an already-solved equilibrium kernel.

        Args:
            kernel: Object exposing the equilibrium fields used by the plotting path.

        Raises
        ------
            ValueError: If the object does not expose the required field layout.
        """
        required = {"RR", "ZZ", "Psi", "cfg", "R", "Z", "NR", "NZ", "dR", "dZ"}
        missing = sorted(name for name in required if not hasattr(kernel, name))
        if missing:
            raise ValueError(
                "Kernel does not expose required fields for dashboard generation: "
                f"{', '.join(missing)}"
            )
        self.kernel = kernel

    def generate_poincare_plot(self, n_lines: int = 20, n_transits: int = 500) -> Figure:
        """Generate a lightweight Poincaré-like magnetic topology figure.

        The implementation is a surrogate map suitable for quick quality checks. It
        plots both sampled field-line-like trajectories and flux-surface contours.

        Args:
            n_lines: Number of seeded trajectories.
            n_transits: Iteration count per trajectory.

        Returns
        -------
            Matplotlib `Figure` object containing the generated topology view.
        """
        if n_lines <= 0:
            raise ValueError("n_lines must be greater than zero")
        if n_transits <= 0:
            raise ValueError("n_transits must be greater than zero")

        logger.info("Generating Poincaré plot...")

        # Seed points along the midplane (Z=0, R=R_min..R_max)
        R_start = np.linspace(
            float(self.kernel.cfg["dimensions"]["R_min"]) + 0.1,
            float(self.kernel.cfg["dimensions"]["R_max"]) - 0.1,
            n_lines,
        )
        r_min = float(self.kernel.cfg["dimensions"]["R_min"])
        r_max = float(self.kernel.cfg["dimensions"]["R_max"])
        z_min = float(self.kernel.cfg["dimensions"]["Z_min"])
        z_max = float(self.kernel.cfg["dimensions"]["Z_max"])
        r0 = 0.5 * (r_min + r_max)
        z0 = 0.5 * (z_min + z_max)
        max_psi = max(float(np.max(np.abs(self.kernel.Psi))), 1.0e-9)
        max_radius = max(r_max - r0, r0 - r_min, z_max - z0, z0 - z_min)

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.contour(
            self.kernel.RR, self.kernel.ZZ, self.kernel.Psi, levels=20, colors="gray", alpha=0.2
        )

        for i in range(n_lines):
            r, z = float(R_start[i]), 0.0
            path_r = [r]
            path_z = [z]

            for _ in range(n_transits):
                psi_val = self._get_psi(r, z)
                q = 1.0 + 3.0 * (float(psi_val) / max_psi) ** 2
                if q <= 0:
                    q = 1.0

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
        ax.contour(self.kernel.RR, self.kernel.ZZ, self.kernel.Psi, levels=50, cmap="nipy_spectral")
        ax.set_title("Magnetic Topology (Flux Surfaces)")
        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")
        ax.set_aspect("equal")
        return fig

    def _get_psi(self, r: float, z: float) -> float:
        """Return nearest-neighbour Psi value for a coordinate.

        Args:
            r: Radial coordinate in meters.
            z: Vertical coordinate in meters.

        Returns
        -------
            Interpolated (nearest neighbour) flux value.
        """
        ir = int((r - self.kernel.R[0]) / self.kernel.dR)
        iz = int((z - self.kernel.Z[0]) / self.kernel.dZ)
        ir = int(np.clip(ir, 0, self.kernel.NR - 1))
        iz = int(np.clip(iz, 0, self.kernel.NZ - 1))
        return float(self.kernel.Psi[iz, ir])


def run_dashboard(config_path: str | Path) -> None:
    """Run the Poincaré dashboard end-to-end from a config file.

    This helper is intentionally defensive for CLI use and exits with explicit
    diagnostics when the compatibility shim is unavailable or the config is not
    loadable.

    Args:
        config_path: Path to solver configuration.
    """
    if FusionKernel is None:
        raise RuntimeError("FusionKernel is not available in this environment")

    kernel = FusionKernel(str(config_path))
    kernel.solve_equilibrium()

    gen = DashboardGenerator(kernel)
    fig = gen.generate_poincare_plot()

    fig.savefig("Poincare_Topology.png")
    logger.info("Dashboard saved: Poincare_Topology.png")


def _default_config_path() -> str:
    """Resolve repository-root default config path."""
    return str(Path(__file__).resolve().parents[3] / "iter_config.json")


if __name__ == "__main__":
    cfg = _default_config_path()
    run_dashboard(cfg)
