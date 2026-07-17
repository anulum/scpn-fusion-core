#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
"""
Figure: Controller per-update latency — measured lanes only.

Bars show the measured per-control-step wall-clock latency (p50, with the
p95 whisker) of the controller lanes that were actually benchmarked on this
project's development workstation.  Source of the numbers is the committed
provenance artifact ``validation/reports/stress_test_campaign.json``; the
figure loads it rather than hard-coding values.

HONESTY NOTE (K17-B4 correction).  An earlier version of this figure and of
Paper B Table 5 attributed the CPU timings to an "AMD Ryzen 9 7950X" and
included FPGA (Xilinx Alveo U250) and neuromorphic (Intel Loihi) bars.  The
Ryzen attribution was fabricated: the real measurement host is the Intel Core
i5-11600K development workstation (confirmed via /proc/cpuinfo; the numbers
themselves were real, only the CPU label was wrong).  The FPGA and Loihi bars
were projections with no hardware measurement behind them and are retracted —
they are not shown here.  Only measured lanes are plotted.

Output: fig_latency_comparison.pdf / .png
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import apply_style, figsize, DOUBLE_COL, COLORS

apply_style()
import matplotlib.pyplot as plt

# Real measurement host (Intel Core i5-11600K, 6C/12T).  This is the actual
# workstation that produced stress_test_campaign.json; it replaces the
# fabricated "AMD Ryzen 9 7950X" attribution from the retracted revision.
HOST_LABEL = "Intel Core i5-11600K (6C/12T), local non-isolated wall-clock"

# Repo-root-relative path to the committed measured artifact.
_ARTIFACT = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "validation", "reports", "stress_test_campaign.json",
    )
)

# Measured lanes to show, in display order.  Each entry maps an artifact key
# to a (display label, colour).  H-infinity is intentionally excluded: its
# committed record carries diagnostic_status="invalidated_..." (stale scalar
# plant calibration), so it is not a defensible measured lane.
_LANES = [
    ("PID",       "PID\n(Python)",      COLORS["blue"]),
    ("Rust-PID",  "PID\n(Rust)",        COLORS["cyan"]),
    ("NMPC-JAX",  "NMPC\n(JAX, CPU)",   COLORS["red"]),
    ("Nengo-SNN", "SNN\n(Nengo, CPU)",  COLORS["green"]),
]


def _load_lanes():
    """Load measured (label, p50_us, p95_us, colour) tuples from the artifact."""
    with open(_ARTIFACT, encoding="utf-8") as handle:
        data = json.load(handle)
    lanes = []
    for key, label, colour in _LANES:
        record = data.get(key)
        if not isinstance(record, dict) or "p50_latency_us" not in record:
            continue
        if str(record.get("diagnostic_status", "")).startswith("invalidated"):
            continue
        lanes.append(
            (
                label,
                float(record["p50_latency_us"]),
                float(record.get("p95_latency_us", record["p50_latency_us"])),
                colour,
            )
        )
    if not lanes:
        raise RuntimeError(f"No measured lanes found in {_ARTIFACT}")
    return lanes


def _fmt(us):
    """Format a microsecond latency for a compact bar label."""
    if us >= 1000.0:
        return f"{us / 1000.0:.1f} ms"
    if us >= 1.0:
        return f"{us:.1f} us"
    return f"{us * 1000.0:.0f} ns"


def main():
    outdir = os.path.dirname(__file__)
    lanes = _load_lanes()

    labels = [lane[0] for lane in lanes]
    p50 = np.array([lane[1] for lane in lanes])
    p95 = np.array([lane[2] for lane in lanes])
    colours = [lane[3] for lane in lanes]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=figsize(DOUBLE_COL, 0.5))

    # p50 bars with an asymmetric whisker up to p95 (both measured).
    yerr = np.vstack([np.zeros_like(p50), np.maximum(p95 - p50, 0.0)])
    ax.bar(
        x, p50, 0.6, color=colours, edgecolor="k", linewidth=0.4, zorder=3,
        yerr=yerr, error_kw=dict(ecolor="0.3", elinewidth=0.8, capsize=3, zorder=4),
    )
    for xi, v in zip(x, p50):
        ax.text(xi, v * 1.35, _fmt(v), ha="center", va="bottom",
                fontsize=7, fontweight="bold")

    ax.set_yscale("log")
    ax.set_ylabel(r"Per-control-step latency ($\mu$s)" + "\n(p50, p95 whisker)")
    ax.set_xlabel("Controller lane (measured)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.1, max(p95) * 6.0)
    ax.set_title("Measured controller latency")

    # 1 ms VDE deadline reference.
    ax.axhline(1000.0, color="grey", ls=":", lw=0.6, zorder=0)
    ax.text(0.02, 1000.0 * 1.25, "1 ms (VDE deadline)",
            fontsize=6.5, color="grey", ha="left")

    ax.yaxis.grid(True, which="major", alpha=0.3, zorder=0)

    # Real host provenance, stamped on the figure itself (top-left, inside axes).
    ax.text(0.015, 0.965, HOST_LABEL, transform=ax.transAxes,
            fontsize=6, color="0.4", style="italic", ha="left", va="top")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_latency_comparison.{ext}"))
    plt.close(fig)
    print("  [OK] fig_latency_comparison (measured lanes, host: i5-11600K)")


if __name__ == "__main__":
    main()
