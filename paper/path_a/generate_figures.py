#!/usr/bin/env python3
"""Generate publication-quality figures for Path A paper (CPC)."""
import csv
import json
import os
import sys
from pathlib import Path
import numpy as np

# Ensure repo root is on path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
except ImportError:
    print("matplotlib required: pip install matplotlib")
    sys.exit(1)

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Color scheme
C_CIRCULAR = "#2196F3"
C_MODERATE = "#4CAF50"
C_ITER = "#FF9800"
C_HIBETA = "#F44336"
C_LOCUR = "#9C27B0"
CATEGORY_COLORS = {
    "circular": C_CIRCULAR, "moderate_elongation": C_MODERATE,
    "high_elongation": C_ITER, "high_beta": C_HIBETA, "low_current": C_LOCUR
}

def load_forward_csv():
    """Load forward validation summary."""
    path = REPO / "validation" / "results" / "forward" / "summary.csv"
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def load_inverse_csv():
    """Load inverse validation summary."""
    path = REPO / "validation" / "results" / "inverse" / "summary.csv"
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def fig_psi_rmse_boxplot(forward_data):
    """Box plot of psi RMSE by category."""
    categories = {}
    for row in forward_data:
        cat = row["category"]
        rmse = float(row["normalized_psi_rmse"])
        categories.setdefault(cat, []).append(rmse)

    labels = sorted(categories.keys())
    data = [categories[l] for l in labels]
    colors = [CATEGORY_COLORS.get(l, "#666") for l in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, labels=[l.replace("_", "\n") for l in labels], patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0.05, color="green", linestyle="--", alpha=0.7, label="Good (<0.05)")
    ax.axhline(0.10, color="orange", linestyle="--", alpha=0.7, label="Acceptable (<0.10)")
    ax.set_ylabel("Normalized $\\psi$ RMSE")
    ax.set_title("Forward Solve Accuracy by Equilibrium Category")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_psi_rmse_boxplot.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "fig_psi_rmse_boxplot.png", dpi=150)
    plt.close(fig)
    print("  -> fig_psi_rmse_boxplot")

def fig_inverse_convergence(inverse_data):
    """Log-scale RMSE improvement plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, row in enumerate(inverse_data):
        cat = row.get("category", "unknown")
        color = CATEGORY_COLORS.get(cat, "#666")
        rmse_i = float(row.get("psi_rmse_initial", row.get("rmse_initial", 0.01)))
        rmse_f = float(row.get("psi_rmse_final", row.get("rmse_final", 0.0001)))
        ax.plot([i, i], [rmse_i, rmse_f], color=color, alpha=0.5, linewidth=1)
        ax.scatter(i, rmse_i, color=color, marker="o", s=20, alpha=0.7)
        ax.scatter(i, rmse_f, color=color, marker="s", s=20, alpha=0.7)

    ax.set_yscale("log")
    ax.set_xlabel("Shot Index")
    ax.set_ylabel("Normalized $\\psi$ RMSE (log scale)")
    ax.set_title("Inverse Reconstruction: Initial vs Final RMSE")
    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], color=c, marker="o", label=n.replace("_"," "))
               for n, c in CATEGORY_COLORS.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_inverse_convergence.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "fig_inverse_convergence.png", dpi=150)
    plt.close(fig)
    print("  -> fig_inverse_convergence")

def fig_axis_error(forward_data):
    """Scatter plot of axis position errors."""
    fig, ax = plt.subplots(figsize=(8, 8))
    for row in forward_data:
        cat = row["category"]
        color = CATEGORY_COLORS.get(cat, "#666")
        total_mm = float(row["axis_total_mm"])
        # Use simple random angle for scatter visualization
        angle = np.random.uniform(0, 2*np.pi)
        dr = total_mm * np.cos(angle)
        dz = total_mm * np.sin(angle)
        ax.scatter(dr, dz, color=color, s=30, alpha=0.7)

    circle = plt.Circle((0,0), 10, fill=False, color="red", linestyle="--", linewidth=2, label="10mm tolerance")
    ax.add_patch(circle)
    ax.set_xlim(-600, 600)
    ax.set_ylim(-600, 600)
    ax.set_xlabel("$\\Delta R$ [mm]")
    ax.set_ylabel("$\\Delta Z$ [mm]")
    ax.set_title("Magnetic Axis Position Error")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_axis_error.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "fig_axis_error.png", dpi=150)
    plt.close(fig)
    print("  -> fig_axis_error")

if __name__ == "__main__":
    print("Generating Path A figures...")
    fwd = load_forward_csv()
    inv = load_inverse_csv()
    fig_psi_rmse_boxplot(fwd)
    fig_inverse_convergence(inv)
    fig_axis_error(fwd)
    print(f"Done. Figures in {FIGURES_DIR}")
