#!/usr/bin/env python3
"""
Figure: Performance scaling — Python vs Rust vs Rust+multigrid.

Grouped bar chart showing forward equilibrium solve times at three grid
sizes (33x33, 65x65, 128x128).  Log-scale y-axis.  Values correspond to
Table 2 in Paper A.

Output: fig_performance_scaling.pdf / .png
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

    grids = ['$33{\\times}33$', '$65{\\times}65$', '$128{\\times}128$']
    x = np.arange(len(grids))
    bar_width = 0.22

    # Solve times (milliseconds) from Table 2 of the paper
    python_ms  = [800,   5000,    30000]     # NumPy path
    rust_ms    = [2,     100,     1000]      # Rust SOR
    rust_mg_ms = [0.8,   15,      120]       # Rust multigrid (projected)

    fig, ax = plt.subplots(figsize=figsize(DOUBLE_COL, 0.5))

    bars1 = ax.bar(x - bar_width, python_ms, bar_width,
                   color=COLORS["blue"], edgecolor='k', linewidth=0.4,
                   label='Python (NumPy)', zorder=3)
    bars2 = ax.bar(x, rust_ms, bar_width,
                   color=COLORS["red"], edgecolor='k', linewidth=0.4,
                   label='Rust (SOR)', zorder=3)
    bars3 = ax.bar(x + bar_width, rust_mg_ms, bar_width,
                   color=COLORS["green"], edgecolor='k', linewidth=0.4,
                   label='Rust (multigrid)', zorder=3)

    ax.set_yscale('log')
    ax.set_ylabel('Solve time (ms)')
    ax.set_xlabel('Grid resolution')
    ax.set_xticks(x)
    ax.set_xticklabels(grids)
    ax.set_ylim(0.3, 100000)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('Forward equilibrium solve time')

    # Add value labels on bars
    def _label_bar(bars, fmt='{:.0f}'):
        for bar in bars:
            h = bar.get_height()
            if h < 1:
                label = f'{h:.1f}'
            elif h < 100:
                label = f'{h:.0f}'
            else:
                label = f'{h / 1000:.1f}k' if h >= 1000 else f'{h:.0f}'
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.3,
                    label, ha='center', va='bottom', fontsize=7,
                    rotation=0)

    _label_bar(bars1)
    _label_bar(bars2)
    _label_bar(bars3)

    # Speedup annotations
    for i in range(len(grids)):
        speedup_rust = python_ms[i] / rust_ms[i]
        speedup_mg = python_ms[i] / rust_mg_ms[i]
        ax.annotate(f'{speedup_rust:.0f}x',
                    xy=(x[i], rust_ms[i]),
                    xytext=(x[i] - 0.08, rust_ms[i] * 0.25),
                    fontsize=7, color=COLORS["red"], ha='center',
                    fontweight='bold')

    ax.yaxis.grid(True, which='major', alpha=0.3, zorder=0)

    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(outdir, f'fig_performance_scaling.{ext}'))
    plt.close(fig)
    print('  [OK] fig_performance_scaling')


if __name__ == '__main__':
    main()
