#!/usr/bin/env python3
"""
Figure: Latency comparison — PID / MPC / SNN across hardware platforms.

Grouped bar chart with log-scale y-axis showing per-update latency for
each controller type on CPU (Python), CPU (Rust), GPU (JAX), FPGA, and
neuromorphic hardware.  Values from Table 5 in Paper B.

Output: fig_latency_comparison.pdf / .png
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

    # Data from Table 5 of Paper B (latency in microseconds)
    platforms = [
        'CPU\n(Python)',
        'CPU\n(Rust)',
        'GPU\n(JAX)',
        'FPGA\n(proj.)',
        'Neuromorphic\n(Loihi, proj.)',
    ]

    # Latencies in microseconds
    pid_us  = [100,    10,     None,    None,   None]
    mpc_us  = [1000,   None,   50,      None,   None]
    snn_us  = [50,     5,      None,    0.1,    0.05]

    # Replace None with 0 for plotting (will not be drawn)
    x = np.arange(len(platforms))
    bar_width = 0.22

    fig, ax = plt.subplots(figsize=figsize(DOUBLE_COL, 0.5))

    def _plot_bars(data, offset, color, label):
        valid_x = []
        valid_v = []
        for i, v in enumerate(data):
            if v is not None:
                valid_x.append(x[i] + offset)
                valid_v.append(v)
        bars = ax.bar(valid_x, valid_v, bar_width,
                      color=color, edgecolor='k', linewidth=0.4,
                      label=label, zorder=3)
        # Value labels
        for bx, bv in zip(valid_x, valid_v):
            if bv >= 1000:
                label_txt = f'{bv / 1000:.0f} ms'
            elif bv >= 1:
                label_txt = f'{bv:.0f} us'
            elif bv >= 0.1:
                label_txt = f'{bv * 1000:.0f} ns'
            else:
                label_txt = f'{bv * 1000:.0f} ns'
            ax.text(bx, bv * 1.4, label_txt, ha='center', va='bottom',
                    fontsize=6.5, fontweight='bold')
        return bars

    _plot_bars(pid_us, -bar_width, COLORS["blue"], 'PID')
    _plot_bars(mpc_us, 0,          COLORS["red"],  'MPC')
    _plot_bars(snn_us, bar_width,  COLORS["green"], 'SNN')

    ax.set_yscale('log')
    ax.set_ylabel(r'Per-update latency ($\mu$s)')
    ax.set_xlabel('Hardware platform')
    ax.set_xticks(x)
    ax.set_xticklabels(platforms, fontsize=8)
    ax.set_ylim(0.02, 5000)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('Controller latency across hardware platforms')

    # Reference lines
    ax.axhline(1000, color='grey', ls=':', lw=0.6, zorder=0)
    ax.text(4.5, 1200, '1 ms (VDE deadline)', fontsize=6.5, color='grey',
            ha='right')
    ax.axhline(1, color='grey', ls=':', lw=0.6, zorder=0)
    ax.text(4.5, 1.2, r'1 $\mu$s', fontsize=6.5, color='grey', ha='right')

    ax.yaxis.grid(True, which='major', alpha=0.3, zorder=0)

    # Power consumption annotation (secondary info)
    powers = ['~100 W', '~100 W', '~300 W', '~5 W', '<1 W']
    for i, pw in enumerate(powers):
        ax.text(x[i], 0.03, pw, ha='center', va='top', fontsize=6,
                color='0.5', style='italic')

    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(outdir, f'fig_latency_comparison.{ext}'))
    plt.close(fig)
    print('  [OK] fig_latency_comparison')


if __name__ == '__main__':
    main()
