# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Repository Header Generator
# Generates an authentic Grad-Shafranov equilibrium visualization
# using the same physics as FusionKernel (D-shaped tokamak flux surfaces).
# ──────────────────────────────────────────────────────────────────────
"""Generate repo_header.png (1280x640) for GitHub README."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Tokamak geometry (ITER-like D-shape parameters)
# ---------------------------------------------------------------------------
R0 = 5.0        # Major radius [m]
a = 2.5          # Minor radius [m]
kappa = 1.7      # Elongation
delta = 0.4      # Triangularity

# Grid matching FusionKernel resolution
NR, NZ = 256, 128
r = np.linspace(R0 - 1.5 * a, R0 + 1.5 * a, NR)
z = np.linspace(-kappa * a * 1.2, kappa * a * 1.2, NZ)
R, Z = np.meshgrid(r, z)

# ---------------------------------------------------------------------------
# Simplified Grad-Shafranov flux function with Shafranov shift + triangularity
# psi(R,Z) ~ (R - R0 - delta_shift)^2 + (Z/kappa)^2   (normalised)
# ---------------------------------------------------------------------------
delta_shift = delta * a * (Z / (a * kappa)) ** 2
psi = (R - R0 - delta_shift) ** 2 + (Z / kappa) ** 2
psi = psi / psi.max()

# Toroidal current density (proportional to dpsi: peaks at magnetic axis)
j_phi = np.exp(-psi / 0.08) * R / R0


def generate_header() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    W, H, DPI = 12.8, 6.4, 100

    # Custom plasma colormap: dark blue -> cyan -> magenta -> white core
    plasma_cmap = LinearSegmentedColormap.from_list(
        "scpn_plasma",
        ["#00050a", "#001428", "#003366", "#0066aa", "#00ccff",
         "#ff00ff", "#ff66ff", "#ffffff"],
    )

    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor="#00050a")

    # --- Left panel: flux surfaces + current density (70% width) ----------
    ax1 = fig.add_axes([0.0, 0.0, 0.68, 1.0], facecolor="#00050a")
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # Current density heatmap (the plasma glow)
    ax1.contourf(R, Z, j_phi, levels=40, cmap=plasma_cmap, alpha=0.85)

    # Flux surface contours
    levels_fine = np.linspace(0.01, 0.95, 18)
    ax1.contour(R, Z, psi, levels=levels_fine, colors="cyan", linewidths=0.5, alpha=0.35)

    # Key surfaces: pedestal, mid, core
    cs_key = ax1.contour(
        R, Z, psi,
        levels=[0.05, 0.15, 0.35, 0.60, 0.85],
        colors="cyan", linewidths=1.2, alpha=0.7,
    )

    # Separatrix (LCFS)
    psi_lcfs = 0.95
    ax1.contour(R, Z, psi, levels=[psi_lcfs], colors="#ff3366", linewidths=2.0, alpha=0.9)

    # Magnetic axis marker
    ax1.plot(R0, 0, "o", color="#ff00ff", markersize=5, alpha=0.9, zorder=5)

    # X-point markers (approximate)
    z_xpt = kappa * a * 0.98
    ax1.plot(R0 - delta * a * 0.3, -z_xpt, "x", color="#ff3366", markersize=7,
             markeredgewidth=1.5, alpha=0.8, zorder=5)
    ax1.plot(R0 - delta * a * 0.3, z_xpt, "x", color="#ff3366", markersize=7,
             markeredgewidth=1.5, alpha=0.8, zorder=5)

    # Vessel wall (elliptical outline)
    theta_wall = np.linspace(0, 2 * np.pi, 200)
    r_wall = R0 + (a * 1.15) * np.cos(theta_wall + delta * np.sin(theta_wall))
    z_wall = kappa * a * 1.15 * np.sin(theta_wall)
    ax1.plot(r_wall, z_wall, color="#334466", linewidth=1.5, alpha=0.6)

    # Simulated magnetic probe positions (dots around the vessel)
    n_probes = 24
    theta_probes = np.linspace(0, 2 * np.pi, n_probes, endpoint=False)
    r_probes = R0 + (a * 1.20) * np.cos(theta_probes + delta * np.sin(theta_probes))
    z_probes = kappa * a * 1.20 * np.sin(theta_probes)
    ax1.scatter(r_probes, z_probes, s=8, c="#66aaff", alpha=0.6, zorder=4)

    # Neural spike traces (SNN controller activity)
    rng = np.random.default_rng(42)
    for _ in range(50):
        sx = rng.uniform(r.min(), r.min() + 0.6)
        sy = rng.uniform(z.min() + 1, z.max() - 1)
        dy = rng.normal(0, 0.15)
        ax1.plot([sx, sx + 0.08], [sy, sy + dy], color="white", alpha=0.2, lw=0.4)

    ax1.set_xlim(r.min(), r.max())
    ax1.set_ylim(z.min(), z.max())

    # --- Right panel: text overlay (30% width) ---------------------------
    ax2 = fig.add_axes([0.62, 0.0, 0.38, 1.0], facecolor="#00050a")
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # Title
    ax2.text(0.08, 0.82, "SCPN", color="white", fontsize=36,
             fontweight="bold", fontfamily="monospace", alpha=0.95)
    ax2.text(0.08, 0.72, "FUSION CORE", color="white", fontsize=36,
             fontweight="bold", fontfamily="monospace", alpha=0.95)

    # Subtitle
    ax2.text(0.08, 0.63, "Neuro-Symbolic Tokamak Control", color="#00ccff",
             fontsize=13, fontfamily="monospace", alpha=0.8)

    # Key metrics
    metrics = [
        ("Grad-Shafranov Solver", "Multigrid + SOR + GMRES"),
        ("SNN Controller", "Petri Net \u2192 LIF Neurons"),
        ("AI Surrogates", "FNO + Neural Transport"),
        ("Validation", "8 SPARC shots, ITPA DB"),
        ("Languages", "Python + Rust (PyO3)"),
        ("Tests", "205+ Rust | 60+ Python"),
    ]

    y_start = 0.50
    for i, (label, value) in enumerate(metrics):
        y = y_start - i * 0.065
        ax2.text(0.08, y, f"\u25b8 {label}", color="#6688aa", fontsize=9,
                 fontfamily="monospace", alpha=0.9)
        ax2.text(0.10, y - 0.028, value, color="#99bbdd", fontsize=8,
                 fontfamily="monospace", alpha=0.7)

    # Decorative separator line
    ax2.plot([0.08, 0.85], [0.57, 0.57], color="#334466", linewidth=0.8, alpha=0.5)

    # Attribution
    ax2.text(0.08, 0.06, "\u00a9 1998\u20132026 Miroslav \u0160otek", color="#445566",
             fontsize=7, fontfamily="monospace", alpha=0.6)
    ax2.text(0.08, 0.03, "anulum.li | AGPL-3.0", color="#445566",
             fontsize=7, fontfamily="monospace", alpha=0.5)

    # Save
    out = "docs/assets/repo_header.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0, facecolor="#00050a")
    plt.close(fig)
    print(f"Generated {out} ({W * DPI:.0f}x{H * DPI:.0f}px)")


if __name__ == "__main__":
    generate_header()
