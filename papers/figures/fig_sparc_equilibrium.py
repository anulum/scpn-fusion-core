#!/usr/bin/env python3
"""
Figure: 2D tokamak equilibrium contour plot (SPARC-like parameters).

Shows poloidal flux contours with separatrix, X-point, and magnetic axis
marked.  Uses an analytical Solov'ev-type solution to generate a realistic
D-shaped equilibrium with lower single-null geometry.

Output: fig_sparc_equilibrium.pdf / .png
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import apply_style, figsize, SINGLE_COL, COLORS

apply_style()
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def _solovev_equilibrium(R, Z, R0=1.85, a=0.57, kappa=1.97, delta=0.54,
                         A=-0.15, C=0.08):
    """
    Generate a Solov'ev-type analytical equilibrium with D-shaping.

    Parameters are loosely based on SPARC (R0=1.85m, a=0.57m, kappa=1.97,
    delta=0.54).  The solution is approximate but produces visually
    correct flux contours with a lower X-point.
    """
    # Normalised coordinates
    r = (R - R0) / a
    z = Z / (a * kappa)

    # D-shaped flux function (Miller-like parameterisation)
    # Combines Solov'ev core with shaping terms
    psi = (
        0.5 * (r**2 + z**2)
        - 0.12 * r**4
        + 0.06 * r**2 * z**2
        - 0.15 * z**4
        + delta * 0.3 * r * z**2
        - 0.08 * r**3
        + A * R**2 / (R0**2)
        + C * z**2 * (1 + 0.5 * r)
    )

    # Add X-point by superposing a lower-null field
    z_x = -kappa * a * 1.05  # X-point vertical position
    r_x = 0.0
    dx = r - r_x
    dz = z - (z_x / (a * kappa))
    psi += 0.02 * np.log(dx**2 + dz**2 + 0.01)

    return psi


def main():
    outdir = os.path.dirname(__file__)

    # SPARC-like parameters
    R0 = 1.85   # m
    a = 0.57    # m
    kappa = 1.97
    delta = 0.54

    # Computational grid
    NR, NZ = 257, 257
    R = np.linspace(R0 - 1.2 * a, R0 + 1.2 * a, NR)
    Z = np.linspace(-1.3 * kappa * a, 1.3 * kappa * a, NZ)
    RR, ZZ = np.meshgrid(R, Z, indexing='xy')

    psi = _solovev_equilibrium(RR, ZZ, R0=R0, a=a, kappa=kappa, delta=delta)

    # Find magnetic axis (minimum of psi in the plasma region)
    mask = ((RR - R0)**2 / a**2 + ZZ**2 / (kappa * a)**2) < 0.8
    psi_masked = np.where(mask, psi, np.inf)
    idx_min = np.unravel_index(np.argmin(psi_masked), psi.shape)
    R_ax = RR[idx_min]
    Z_ax = ZZ[idx_min]
    psi_ax = psi[idx_min]

    # Separatrix level: use a contour slightly above the X-point value
    # Find approximate X-point (saddle point near bottom)
    z_region = Z < -0.5 * kappa * a
    if np.any(z_region):
        psi_bottom = psi[z_region[np.newaxis, :].flatten() if False else
                         (ZZ < -0.5 * kappa * a)]
        # Use gradient-based search in the lower region
        lower_mask = (ZZ < -0.3 * kappa * a) & (np.abs(RR - R0) < 0.5 * a)
        psi_lower = np.where(lower_mask, psi, -np.inf)
        idx_xpt = np.unravel_index(np.argmax(psi_lower), psi.shape)
        R_xpt = RR[idx_xpt]
        Z_xpt = ZZ[idx_xpt]
        psi_xpt = psi[idx_xpt]
    else:
        psi_xpt = psi_ax + 0.95 * (np.max(psi[mask]) - psi_ax)
        R_xpt = R0
        Z_xpt = -kappa * a

    # Normalise psi for contour levels
    psi_sep = psi_ax + 0.85 * (psi_xpt - psi_ax) if psi_xpt > psi_ax else psi_xpt

    # Contour levels
    n_core = 12
    n_sol = 5
    levels_core = np.linspace(psi_ax, psi_sep, n_core + 1)
    levels_sol = np.linspace(psi_sep, psi_sep + 0.3 * abs(psi_sep - psi_ax), n_sol)
    levels_all = np.concatenate([levels_core, levels_sol[1:]])

    # ----- Plot -----
    fig, ax = plt.subplots(figsize=figsize(SINGLE_COL, 1.3))

    # Filled contours (soft blue-to-red)
    cf = ax.contourf(RR, ZZ, psi, levels=40, cmap='RdYlBu_r', alpha=0.6)

    # Line contours (closed flux surfaces)
    cs_core = ax.contour(RR, ZZ, psi, levels=levels_core,
                         colors='k', linewidths=0.6, linestyles='-')

    # Separatrix (thick red)
    ax.contour(RR, ZZ, psi, levels=[psi_sep],
               colors=COLORS["red"], linewidths=2.0, linestyles='-')

    # SOL contours (dashed)
    ax.contour(RR, ZZ, psi, levels=levels_sol[1:],
               colors='grey', linewidths=0.5, linestyles='--')

    # Magnetic axis marker
    ax.plot(R_ax, Z_ax, 'o', color=COLORS["blue"], markersize=7,
            markeredgecolor='k', markeredgewidth=0.5, zorder=5)
    ax.annotate('Magnetic\naxis', xy=(R_ax, Z_ax),
                xytext=(R_ax + 0.15 * a, Z_ax + 0.4 * kappa * a),
                fontsize=8, ha='left',
                arrowprops=dict(arrowstyle='->', color='k', lw=0.8))

    # X-point marker
    ax.plot(R_xpt, Z_xpt, 'x', color=COLORS["red"], markersize=9,
            markeredgewidth=2, zorder=5)
    ax.annotate('X-point', xy=(R_xpt, Z_xpt),
                xytext=(R_xpt + 0.2 * a, Z_xpt - 0.15 * kappa * a),
                fontsize=8, ha='left',
                arrowprops=dict(arrowstyle='->', color=COLORS["red"], lw=0.8))

    # Vessel wall (simplified D-shape)
    theta = np.linspace(0, 2 * np.pi, 200)
    R_wall = R0 + 1.15 * a * np.cos(theta + delta * np.sin(theta))
    Z_wall = 1.15 * kappa * a * np.sin(theta)
    ax.plot(R_wall, Z_wall, 'k-', lw=2.0, label='Vessel wall')

    # Colorbar
    cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r'$\Psi$ (Wb/rad)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Labels
    ax.set_xlabel(r'$R$ (m)')
    ax.set_ylabel(r'$Z$ (m)')
    ax.set_title(r'SPARC-like equilibrium ($I_p = 8.7$ MA, $B_T = 12.2$ T)')
    ax.set_aspect('equal')

    # Annotation for key parameters
    text_box = (f"$R_0 = {R0}$ m\n"
                f"$a = {a}$ m\n"
                f"$\\kappa = {kappa}$\n"
                f"$\\delta = {delta}$")
    ax.text(0.97, 0.97, text_box, transform=ax.transAxes,
            fontsize=7, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='0.7', alpha=0.9))

    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(outdir, f'fig_sparc_equilibrium.{ext}'))
    plt.close(fig)
    print('  [OK] fig_sparc_equilibrium')


if __name__ == '__main__':
    main()
