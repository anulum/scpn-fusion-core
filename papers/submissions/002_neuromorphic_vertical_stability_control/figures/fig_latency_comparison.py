#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — historical controller-latency figure.
"""Plot the exact latency aggregates emitted by the legacy campaign harness.

The old harness timed a lane-specific complete shot and divided by its step
count. It did not use one common plant and did not serialize the exact invocation
episode count. This figure preserves historical evidence; it is not a current
controller ranking or a portable performance claim.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import COLORS, DOUBLE_COL, apply_style, figsize

apply_style()
import matplotlib.pyplot as plt

_ARTIFACT = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "evidence",
        "historical_controller_latency.json",
    )
)

_LANES = [
    ("PID", "PID\n(Python)", COLORS["blue"]),
    ("Rust-PID", "PID\n(Rust; separate plant)", COLORS["cyan"]),
    ("H-infinity", "H-infinity\n(invalidated)", COLORS["purple"]),
    ("NMPC-JAX", "NMPC\n(JAX; uncalibrated)", COLORS["red"]),
    ("Nengo-SNN", "SNN\n(Nengo; radial only)", COLORS["green"]),
]


def _load_lanes() -> tuple[list[tuple[str, float, float, str]], str]:
    """Load exact historical lane aggregates and their host label."""
    with open(_ARTIFACT, encoding="utf-8") as handle:
        data = json.load(handle)
    controllers = data["results"]
    lanes = []
    for key, label, colour in _LANES:
        record = controllers.get(key)
        if not isinstance(record, dict):
            raise ValueError(f"Missing historical latency record for {key}")
        p50 = float(record["p50_latency_us"])
        p95 = float(record["p95_latency_us"])
        if not math.isfinite(p50) or not math.isfinite(p95) or p50 <= 0.0 or p95 < p50:
            raise ValueError(f"Invalid historical latency record for {key}: {record}")
        lanes.append((label, p50, p95, colour))
    methodology = data["methodology"]
    host_label = f"{methodology['host']}; non-isolated; invocation count not serialized"
    return lanes, host_label


def _fmt(microseconds: float) -> str:
    """Format a microsecond latency for a compact bar label."""
    if microseconds >= 1000.0:
        return f"{microseconds / 1000.0:.1f} ms"
    if microseconds >= 1.0:
        return f"{microseconds:.1f} us"
    return f"{microseconds * 1000.0:.0f} ns"


def main() -> None:
    """Write PDF and PNG copies of the historical non-comparable figure."""
    outdir = os.path.dirname(__file__)
    lanes, host_label = _load_lanes()
    labels = [lane[0] for lane in lanes]
    p50 = np.array([lane[1] for lane in lanes])
    p95 = np.array([lane[2] for lane in lanes])
    colours = [lane[3] for lane in lanes]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=figsize(DOUBLE_COL, 0.55))
    yerr = np.vstack([np.zeros_like(p50), np.maximum(p95 - p50, 0.0)])
    ax.bar(
        x,
        p50,
        0.6,
        color=colours,
        edgecolor="k",
        linewidth=0.4,
        zorder=3,
        yerr=yerr,
        error_kw={"ecolor": "0.3", "elinewidth": 0.8, "capsize": 3, "zorder": 4},
    )
    for x_position, value in zip(x, p50, strict=True):
        ax.text(
            x_position,
            value * 1.35,
            _fmt(float(value)),
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )

    ax.set_yscale("log")
    ax.set_ylabel("Legacy latency (microseconds)\n(p50, p95 whisker)")
    ax.set_xlabel("Legacy lane (not accuracy-equivalent)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0.1, max(p95) * 6.0)
    ax.set_title("Historical legacy-harness latency output")
    ax.yaxis.grid(True, which="major", alpha=0.3, zorder=0)
    ax.text(
        0.015,
        0.965,
        host_label,
        transform=ax.transAxes,
        fontsize=5.6,
        color="0.4",
        style="italic",
        ha="left",
        va="top",
    )

    for extension in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_latency_comparison.{extension}"))
    plt.close(fig)
    print("  [OK] fig_latency_comparison (historical non-comparable lanes)")


if __name__ == "__main__":
    main()
