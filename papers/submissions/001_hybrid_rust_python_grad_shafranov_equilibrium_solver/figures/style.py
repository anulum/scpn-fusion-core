# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — shared equilibrium-paper figure style.
"""Shared print-ready Matplotlib configuration for Paper 001."""

from __future__ import annotations

import logging

import matplotlib as mpl
import numpy as np


class _UnixFontTimestampFilter(logging.Filter):
    """Silence fontTools' harmless Unix-epoch normalization diagnostic."""

    _MESSAGE_SUFFIX = "timestamp seems very low; regarding as unix timestamp"

    def filter(self, record: logging.LogRecord) -> bool:
        """Keep every fontTools record except the known epoch diagnostic."""
        return not record.getMessage().endswith(self._MESSAGE_SUFFIX)


# Some distribution fonts encode head-table dates as Unix timestamps. fontTools
# normalizes those values before embedding; the verifier independently checks
# embedded fonts and byte-identical regeneration.
logging.getLogger("fontTools.ttLib.tables._h_e_a_d").addFilter(_UnixFontTimestampFilter())

STYLE: dict[str, object] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "font.size": 10,
    "mathtext.fontset": "cm",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.linewidth": 0.8,
    "axes.grid": False,
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
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.7",
    "legend.fancybox": False,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

SINGLE_COL = 3.5
DOUBLE_COL = 7.0
GOLDEN = (1.0 + np.sqrt(5.0)) / 2.0

COLORS: dict[str, str] = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "cyan": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}


def apply_style() -> None:
    """Apply the paper's deterministic Matplotlib style."""
    mpl.rcParams.update(STYLE)


def figsize(width: float = SINGLE_COL, ratio: float = 1.0 / GOLDEN) -> tuple[float, float]:
    """Return a figure size with the requested width and aspect ratio.

    Parameters
    ----------
    width:
        Figure width in inches.
    ratio:
        Height divided by width.

    Returns
    -------
    tuple[float, float]
        Width and height in inches.
    """
    return width, width * ratio
