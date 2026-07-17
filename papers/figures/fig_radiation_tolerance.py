#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
"""
Figure: Radiation tolerance — bit-flip rate vs corruption of the control value.

Single panel: corruption of the *represented control value* (%) vs bit-flips per
control tick, for the 1024-bit stochastic-computing (SC) encoding against the
worst case for a 32-bit binary register.

This is a property of the ENCODING and is exactly derivable: N flips in an
L = 1024-bit stream change the value by at most N/L, whereas a single high-order
flip in a 32-bit binary register can corrupt the value by up to 2^31.  The
earlier "settling-time degradation" panel was removed in the K17-B4 revision: it
rested on the spiking controller stabilising the fast VDE, which it does not
(see fig_vertical_stability); encoding robustness is not a closed-loop
stabilisation claim.

Output: fig_radiation_tolerance.pdf / .png
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import apply_style, figsize, SINGLE_COL, COLORS

apply_style()
import matplotlib.pyplot as plt


def main():
    outdir = os.path.dirname(__file__)

    fig, ax = plt.subplots(figsize=figsize(SINGLE_COL * 1.6, 0.62))

    flips = np.array([0, 1, 10, 50, 100])
    value_error_sc = np.array([0, 0.1, 1.0, 4.9, 9.8])  # % for L = 1024 (N/L)

    # Binary controller: a single high-order flip is catastrophic.
    flips_binary = np.array([0, 1, 2, 5, 10])
    value_error_binary = np.array([0, 50, 75, 100, 100])  # worst case (MSB-class)

    flips_fine = np.linspace(0, 120, 200)
    sc_error_fine = flips_fine / 1024 * 100  # exact SC degradation law

    ax.plot(flips_fine, sc_error_fine, '-', color=COLORS["green"],
            lw=1.5, label='SC encoding ($L=1024$): error $= N/L$')
    ax.plot(flips, value_error_sc, 'o', color=COLORS["green"],
            markersize=6, markeredgecolor='k', markeredgewidth=0.4, zorder=5)

    ax.plot(flips_binary, value_error_binary, 's--', color=COLORS["red"],
            lw=1.5, markersize=6, markeredgecolor='k',
            markeredgewidth=0.4, label='Binary (32-bit, worst case)', zorder=5)

    ax.annotate('Single bit flip:\nSC = 0.1%, binary up to 50%',
                xy=(1, 50), xytext=(30, 70),
                fontsize=7, ha='left',
                arrowprops=dict(arrowstyle='->', color='0.4', lw=0.8),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='0.5', alpha=0.9))

    ax.set_xlabel('Bit flips per control tick')
    ax.set_ylabel('Represented control value error (%)')
    ax.set_title('Value corruption magnitude (encoding-level)')
    ax.set_xlim(-2, 120)
    ax.set_ylim(-5, 110)
    ax.legend(loc='center right', fontsize=8)
    ax.yaxis.grid(True, which='major', alpha=0.3, zorder=0)

    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(outdir, f'fig_radiation_tolerance.{ext}'))
    plt.close(fig)
    print('  [OK] fig_radiation_tolerance (value-error only; settling panel retracted)')


if __name__ == '__main__':
    main()
