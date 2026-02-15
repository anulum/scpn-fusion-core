#!/usr/bin/env python3
"""
Figure: Validation RMSE — SPARC/ITER/ITPA comparison.

Two-panel figure:
  (a) SPARC GEQDSK magnetic axis position error (bar chart per shot)
  (b) ITPA confinement scaling error by machine (grouped bar chart)

Values correspond to Tables 3 and 4 in Paper A.

Output: fig_validation_rmse.pdf / .png
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

    # ---- Panel (a): SPARC axis position error ----
    shots = [
        'sparc\n1300', 'sparc\n1305', 'sparc\n1310',
        'sparc\n1315', 'sparc\n1349',
        'lmode\nhv', 'lmode\nvh', 'lmode\nvv',
    ]
    # Axis error in mm (from Table 3)
    delta_R_mm = [146, 7, 2, 4, 3, 178, 204, 209]  # mm
    Ip_MA = [0.2, 5.0, 8.7, 8.7, 8.0, 8.5, 8.5, 8.5]

    x = np.arange(len(shots))
    colors_shot = [COLORS["orange"] if dr > 50 else COLORS["blue"]
                   for dr in delta_R_mm]

    bars = ax1.bar(x, delta_R_mm, color=colors_shot, edgecolor='k',
                   linewidth=0.4, width=0.7, zorder=3)

    ax1.set_yscale('log')
    ax1.set_ylabel(r'$\Delta R_{\mathrm{axis}}$ (mm)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(shots, fontsize=7)
    ax1.set_ylim(0.8, 500)
    ax1.set_title('(a) SPARC magnetic axis error')

    # 10 mm reference line
    ax1.axhline(10, color='grey', ls='--', lw=0.8, zorder=0)
    ax1.text(len(shots) - 0.5, 12, '10 mm', fontsize=7, color='grey',
             ha='right')

    # Label bars
    for bar, val in zip(bars, delta_R_mm):
        ax1.text(bar.get_x() + bar.get_width() / 2, val * 1.2,
                 f'{val}', ha='center', va='bottom', fontsize=6.5)

    # Colour legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["blue"], edgecolor='k', linewidth=0.4,
              label=r'Full current ($I_p > 5$ MA)'),
        Patch(facecolor=COLORS["orange"], edgecolor='k', linewidth=0.4,
              label='Low current / L-mode'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=7)

    ax1.yaxis.grid(True, which='major', alpha=0.3, zorder=0)

    # ---- Panel (b): ITPA confinement scaling error ----
    machines = ['JET', 'DIII-D', 'ASDEX-U', 'C-Mod', 'ITER\n(pred.)']
    error_min = [5, 6, 4, 3, 8]
    error_max = [8, 10, 9, 7, 15]
    error_mid = [(lo + hi) / 2 for lo, hi in zip(error_min, error_max)]
    error_err = [(hi - lo) / 2 for lo, hi in zip(error_min, error_max)]

    x2 = np.arange(len(machines))
    bar_colors = [COLORS["blue"], COLORS["red"], COLORS["green"],
                  COLORS["purple"], COLORS["orange"]]

    bars2 = ax2.bar(x2, error_mid, yerr=error_err, color=bar_colors,
                    edgecolor='k', linewidth=0.4, width=0.6,
                    capsize=3, error_kw={'lw': 0.8}, zorder=3)

    # 10% reference line
    ax2.axhline(10, color='grey', ls='--', lw=0.8, zorder=0)
    ax2.text(len(machines) - 0.5, 10.5, '10%', fontsize=7, color='grey',
             ha='right')

    ax2.set_ylabel(r'IPB98(y,2) error (\%)')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(machines, fontsize=9)
    ax2.set_ylim(0, 20)
    ax2.set_title(r'(b) Confinement scaling $\tau_E$ error')

    # Value labels
    for bar, mid in zip(bars2, error_mid):
        ax2.text(bar.get_x() + bar.get_width() / 2, mid + error_err[bars2.index(bar)] + 0.8,
                 f'{mid:.0f}%', ha='center', va='bottom', fontsize=7)

    ax2.yaxis.grid(True, which='major', alpha=0.3, zorder=0)

    fig.subplots_adjust(wspace=0.35)

    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(outdir, f'fig_validation_rmse.{ext}'))
    plt.close(fig)
    print('  [OK] fig_validation_rmse')


if __name__ == '__main__':
    main()
