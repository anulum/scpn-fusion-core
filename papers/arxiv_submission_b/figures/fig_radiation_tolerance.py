#!/usr/bin/env python3
"""
Figure: Radiation tolerance — bit-flip rate vs control error.

Two-panel figure:
  (a) Settling time increase (%) vs bit-flips per tick for SNN (stochastic
      computing) vs hypothetical binary controller degradation.
  (b) Control value error (%) vs bit-flips per tick.

Demonstrates the graceful degradation property of stochastic computing:
a single bit flip in 1024-bit stream changes the value by only 0.1%,
whereas a single bit flip in a 32-bit binary register can corrupt the
value by up to 50% (MSB flip).

Output: fig_radiation_tolerance.pdf / .png
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import apply_style, figsize, DOUBLE_COL, COLORS

apply_style()
import matplotlib.pyplot as plt


def main():
    outdir = os.path.dirname(__file__)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize(DOUBLE_COL, 0.5))

    # Data from Table 7 of Paper B
    flips = np.array([0, 1, 10, 50, 100])
    value_error_sc = np.array([0, 0.1, 1.0, 4.9, 9.8])  # % for 1024-bit
    settling_increase_sc = np.array([0, 0.5, 5, 15, 30])  # %

    # Binary controller: catastrophic at even 1 flip
    # Worst case: MSB flip in 32-bit → 50% error
    # Average case: random bit → ~6.25% per flip (expected over all bits)
    flips_binary = np.array([0, 1, 2, 5, 10])
    value_error_binary = np.array([0, 50, 75, 100, 100])  # worst case
    settling_binary = np.array([0, 100, 100, 100, 100])  # system failure

    # Extended SC curve for smooth interpolation
    flips_fine = np.linspace(0, 120, 200)
    # SC: error scales as n_flips / L
    sc_error_fine = flips_fine / 1024 * 100
    sc_settling_fine = 0.3 * flips_fine  # approximately linear

    # ---- Panel (a): Settling time degradation ----
    ax1.plot(flips_fine, sc_settling_fine, '-', color=COLORS["green"],
             lw=1.5, label='SNN (stochastic, $L=1024$)')
    ax1.plot(flips, settling_increase_sc, 'o', color=COLORS["green"],
             markersize=6, markeredgecolor='k', markeredgewidth=0.4,
             zorder=5)

    # Binary: show catastrophic failure
    ax1.plot(flips_binary, settling_binary, 's--', color=COLORS["red"],
             lw=1.5, markersize=6, markeredgecolor='k',
             markeredgewidth=0.4, label='Binary (32-bit)', zorder=5)

    # Failure region
    ax1.axhspan(80, 110, color=COLORS["red"], alpha=0.08, zorder=0)
    ax1.text(60, 95, 'System failure', fontsize=8, color=COLORS["red"],
             ha='center', style='italic')

    ax1.set_xlabel('Bit flips per control tick')
    ax1.set_ylabel('Settling time increase (%)')
    ax1.set_title('(a) Control performance degradation')
    ax1.set_xlim(-2, 120)
    ax1.set_ylim(-5, 110)
    ax1.legend(loc='center right', fontsize=8)

    # ---- Panel (b): Control value error ----
    ax2.plot(flips_fine, sc_error_fine, '-', color=COLORS["green"],
             lw=1.5, label='SNN (stochastic, $L=1024$)')
    ax2.plot(flips, value_error_sc, 'o', color=COLORS["green"],
             markersize=6, markeredgecolor='k', markeredgewidth=0.4,
             zorder=5)

    ax2.plot(flips_binary, value_error_binary, 's--', color=COLORS["red"],
             lw=1.5, markersize=6, markeredgecolor='k',
             markeredgewidth=0.4, label='Binary (32-bit, worst case)',
             zorder=5)

    # Annotations
    ax2.annotate('Single bit flip:\nSC = 0.1%, binary = 50%',
                 xy=(1, 50), xytext=(30, 70),
                 fontsize=7, ha='left',
                 arrowprops=dict(arrowstyle='->', color='0.4', lw=0.8),
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                           edgecolor='0.5', alpha=0.9))

    ax2.set_xlabel('Bit flips per control tick')
    ax2.set_ylabel('Control value error (%)')
    ax2.set_title('(b) Value corruption magnitude')
    ax2.set_xlim(-2, 120)
    ax2.set_ylim(-5, 110)
    ax2.legend(loc='center right', fontsize=8)

    for ax in (ax1, ax2):
        ax.yaxis.grid(True, which='major', alpha=0.3, zorder=0)

    fig.subplots_adjust(wspace=0.3)

    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(outdir, f'fig_radiation_tolerance.{ext}'))
    plt.close(fig)
    print('  [OK] fig_radiation_tolerance')


if __name__ == '__main__':
    main()
