#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — manufactured inverse-layout illustration.
"""Plot a deterministic manufactured inverse-reconstruction data layout."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from style import COLORS, DOUBLE_COL, apply_style, figsize


def _manufactured_flux_field(
    radius: NDArray[np.float64],
    height: NDArray[np.float64],
    *,
    major_radius_m: float,
    minor_radius_m: float,
    elongation: float,
    triangularity: float,
    quadratic_weight: float,
    quartic_weight: float,
) -> NDArray[np.float64]:
    """Return a dimensionless shaped field for visual layout testing."""
    radial = (radius - major_radius_m) / minor_radius_m
    vertical = height / (minor_radius_m * elongation)
    return (
        quadratic_weight * (radial**2 + vertical**2)
        + quartic_weight * radial**4
        + 0.05 * radial**2 * vertical**2
        - 0.1 * vertical**4
        + triangularity * 0.2 * radial * vertical**2
        - 0.06 * radial**3
    )


def main() -> None:
    """Generate PDF and PNG versions of the manufactured three-panel figure."""
    apply_style()
    output_dir = Path(__file__).resolve().parent
    major_radius_m = 1.85
    minor_radius_m = 0.57
    elongation = 1.97
    triangularity = 0.54

    radius = np.linspace(
        major_radius_m - 1.1 * minor_radius_m,
        major_radius_m + 1.1 * minor_radius_m,
        65,
    )
    height = np.linspace(
        -1.2 * elongation * minor_radius_m,
        1.2 * elongation * minor_radius_m,
        65,
    )
    radius_grid, height_grid = np.meshgrid(radius, height, indexing="xy")
    target = _manufactured_flux_field(
        radius_grid,
        height_grid,
        major_radius_m=major_radius_m,
        minor_radius_m=minor_radius_m,
        elongation=elongation,
        triangularity=triangularity,
        quadratic_weight=0.5,
        quartic_weight=-0.12,
    )
    comparison = _manufactured_flux_field(
        radius_grid,
        height_grid,
        major_radius_m=major_radius_m,
        minor_radius_m=minor_radius_m,
        elongation=elongation,
        triangularity=triangularity,
        quadratic_weight=0.498,
        quartic_weight=-0.119,
    )
    rng = np.random.default_rng(123)
    comparison += gaussian_filter(rng.normal(0.0, 0.002, comparison.shape), sigma=3.0)
    residual = np.abs(target - comparison)

    field_min = min(float(target.min()), float(comparison.min()))
    field_max = max(float(target.max()), float(comparison.max()))
    levels = np.linspace(field_min, field_max, 20)
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    wall_radius = major_radius_m + 1.1 * minor_radius_m * np.cos(
        theta + triangularity * np.sin(theta)
    )
    wall_height = 1.1 * elongation * minor_radius_m * np.sin(theta)
    probe_theta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    probe_radius = major_radius_m + 1.05 * minor_radius_m * np.cos(
        probe_theta + triangularity * np.sin(probe_theta)
    )
    probe_height = 1.05 * elongation * minor_radius_m * np.sin(probe_theta)

    figure, axes = plt.subplots(1, 3, figsize=figsize(DOUBLE_COL, 0.45), sharey=True)
    titles = (
        "(a) Manufactured target",
        "(b) Manufactured comparison",
        r"(c) Residual $|\Delta\psi|$",
    )
    for index, (axis, title, field) in enumerate(zip(axes, titles, (target, comparison, residual))):
        if index < 2:
            axis.contourf(
                radius_grid, height_grid, field, levels=levels, cmap="RdYlBu_r", alpha=0.7
            )
            axis.contour(radius_grid, height_grid, field, levels=levels, colors="k", linewidths=0.4)
            axis.plot(wall_radius, wall_height, "k-", linewidth=1.5)
            if index == 0:
                axis.plot(
                    probe_radius,
                    probe_height,
                    "v",
                    color=COLORS["green"],
                    markersize=4,
                    markeredgecolor="k",
                    markeredgewidth=0.3,
                    label="Synthetic probes",
                    zorder=5,
                )
                axis.legend(loc="upper right", fontsize=7, markerscale=1.2)
        else:
            clipped = np.clip(residual, 1e-5, None)
            colors = axis.pcolormesh(
                radius_grid,
                height_grid,
                clipped,
                norm=LogNorm(vmin=1e-4, vmax=float(residual.max())),
                cmap="inferno",
                shading="auto",
            )
            axis.plot(wall_radius, wall_height, "w-", linewidth=1.5)
            colorbar = figure.colorbar(colors, ax=axis, shrink=0.8, pad=0.03)
            colorbar.set_label(r"Dimensionless $|\Delta\psi|$", fontsize=8)
            colorbar.ax.tick_params(labelsize=7)

        axis.set_title(title, fontsize=9)
        axis.set_xlabel(r"$R$ (m)")
        axis.set_aspect("equal")

    axes[0].set_ylabel(r"$Z$ (m)")
    rms = float(np.sqrt(np.mean(residual**2)))
    axes[2].text(
        0.95,
        0.05,
        f"manufactured RMS = {rms:.2e}",
        transform=axes[2].transAxes,
        fontsize=8,
        horizontalalignment="right",
        verticalalignment="bottom",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "0.5", "alpha": 0.9},
    )
    figure.subplots_adjust(wspace=0.08)
    for extension in ("pdf", "png"):
        figure.savefig(output_dir / f"fig_inverse_reconstruction.{extension}")
    plt.close(figure)
    print("  [OK] fig_inverse_reconstruction (manufactured illustration)")


if __name__ == "__main__":
    main()
