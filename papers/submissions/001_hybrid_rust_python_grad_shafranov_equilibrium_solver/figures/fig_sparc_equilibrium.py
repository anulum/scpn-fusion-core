#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — manufactured SPARC-geometry flux illustration.
"""Plot a dimensionless manufactured flux field in SPARC-like geometry."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from style import COLORS, SINGLE_COL, apply_style, figsize


def _manufactured_shaped_flux(
    radius: NDArray[np.float64],
    height: NDArray[np.float64],
    *,
    major_radius_m: float,
    minor_radius_m: float,
    elongation: float,
    triangularity: float,
) -> NDArray[np.float64]:
    """Return a dimensionless D-shaped field without claiming a GS solution."""
    radial = (radius - major_radius_m) / minor_radius_m
    vertical = height / (minor_radius_m * elongation)
    shaped_radial = radial + 0.28 * triangularity * vertical**2
    return (
        0.5 * (shaped_radial**2 + vertical**2)
        - 0.08 * shaped_radial**3
        - 0.06 * vertical**4
        + 0.04 * shaped_radial**2 * vertical**2
    )


def main() -> None:
    """Generate PDF and PNG versions of the manufactured geometry figure."""
    apply_style()
    output_dir = Path(__file__).resolve().parent
    major_radius_m = 1.85
    minor_radius_m = 0.57
    elongation = 1.97
    triangularity = 0.54

    radius = np.linspace(
        major_radius_m - 1.2 * minor_radius_m,
        major_radius_m + 1.2 * minor_radius_m,
        257,
    )
    height = np.linspace(
        -1.3 * elongation * minor_radius_m,
        1.3 * elongation * minor_radius_m,
        257,
    )
    radius_grid, height_grid = np.meshgrid(radius, height, indexing="xy")
    flux = _manufactured_shaped_flux(
        radius_grid,
        height_grid,
        major_radius_m=major_radius_m,
        minor_radius_m=minor_radius_m,
        elongation=elongation,
        triangularity=triangularity,
    )
    inside = ((radius_grid - major_radius_m) / minor_radius_m) ** 2 + (
        height_grid / (elongation * minor_radius_m)
    ) ** 2 < 0.8
    masked_flux = np.where(inside, flux, np.inf)
    axis_index = np.unravel_index(np.argmin(masked_flux), flux.shape)
    axis_radius = float(radius_grid[axis_index])
    axis_height = float(height_grid[axis_index])
    axis_flux = float(flux[axis_index])
    reference_level = float(np.quantile(flux[inside], 0.9))
    core_levels = np.linspace(axis_flux, reference_level, 13)

    figure, axis = plt.subplots(figsize=figsize(SINGLE_COL, 1.3))
    colors = axis.contourf(radius_grid, height_grid, flux, levels=40, cmap="RdYlBu_r", alpha=0.6)
    axis.contour(radius_grid, height_grid, flux, levels=core_levels, colors="k", linewidths=0.6)
    axis.contour(
        radius_grid,
        height_grid,
        flux,
        levels=[reference_level],
        colors=COLORS["red"],
        linewidths=2.0,
    )
    axis.plot(
        axis_radius,
        axis_height,
        "o",
        color=COLORS["blue"],
        markersize=7,
        markeredgecolor="k",
        markeredgewidth=0.5,
        zorder=5,
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    wall_radius = major_radius_m + 1.15 * minor_radius_m * np.cos(
        theta + triangularity * np.sin(theta)
    )
    wall_height = 1.15 * elongation * minor_radius_m * np.sin(theta)
    axis.plot(wall_radius, wall_height, "k-", linewidth=2.0, label="Manufactured boundary")

    colorbar = figure.colorbar(colors, ax=axis, shrink=0.85, pad=0.02)
    colorbar.set_label(r"Dimensionless manufactured $\psi$", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    axis.set_xlabel(r"$R$ (m)")
    axis.set_ylabel(r"$Z$ (m)")
    axis.set_title("Manufactured flux in SPARC-like geometry")
    axis.set_aspect("equal")
    annotation = (
        f"geometry only\n$R_0={major_radius_m}$ m\n$a={minor_radius_m}$ m\n"
        f"$\\kappa={elongation}$\n$\\delta={triangularity}$"
    )
    axis.text(
        0.97,
        0.97,
        annotation,
        transform=axis.transAxes,
        fontsize=7,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    for extension in ("pdf", "png"):
        figure.savefig(output_dir / f"fig_sparc_equilibrium.{extension}")
    plt.close(figure)
    print("  [OK] fig_sparc_equilibrium (manufactured geometry)")


if __name__ == "__main__":
    main()
