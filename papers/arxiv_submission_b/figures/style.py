"""
Shared matplotlib style configuration for SCPN-Fusion-Core paper figures.

Publication-quality settings targeting Nuclear Fusion / IEEE TPS journals.
All figure scripts import this module for consistent appearance.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Global rcParams — serif fonts, proper sizing, high DPI
# ---------------------------------------------------------------------------
STYLE = {
    # Font
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "font.size": 10,
    "mathtext.fontset": "cm",

    # Axes
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.linewidth": 0.8,
    "axes.grid": False,

    # Ticks
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.top": True,
    "ytick.right": True,

    # Legend
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.7",
    "legend.fancybox": False,

    # Lines
    "lines.linewidth": 1.5,
    "lines.markersize": 5,

    # Figure
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,

    # PDF backend for text-as-paths (safer for journal submission)
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style():
    """Apply publication style to the current matplotlib session."""
    mpl.rcParams.update(STYLE)


# ---------------------------------------------------------------------------
# Journal column widths (inches)
# ---------------------------------------------------------------------------
SINGLE_COL = 3.5    # single-column figure
DOUBLE_COL = 7.0    # double-column figure
GOLDEN = (1 + np.sqrt(5)) / 2  # golden ratio ~ 1.618


def figsize(width=SINGLE_COL, ratio=1.0 / GOLDEN):
    """Return (width, height) tuple with given width and aspect ratio."""
    return (width, width * ratio)


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------
# Colourblind-safe palette (Wong 2011, Nature Methods 8, 441)
COLORS = {
    "blue":    "#0072B2",
    "orange":  "#E69F00",
    "green":   "#009E73",
    "red":     "#D55E00",
    "purple":  "#CC79A7",
    "cyan":    "#56B4E9",
    "yellow":  "#F0E442",
    "black":   "#000000",
}

# Ordered list for sequential use
COLOR_CYCLE = [
    COLORS["blue"],
    COLORS["red"],
    COLORS["green"],
    COLORS["orange"],
    COLORS["purple"],
    COLORS["cyan"],
]


def get_colors(n):
    """Return the first n colours from the cycle."""
    return COLOR_CYCLE[:n]
