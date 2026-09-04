#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
"""
Figure: Compilation pipeline block diagram.

Block diagram: Petri Net (Packet A) -> Compiler (Packet B) -> SNN (runtime)
  -> Hardware targets (CPU/FPGA/Neuromorphic).

Shows the three-stage compilation flow with intermediate representations:
  StochasticPetriNet -> FusionCompiler -> CompiledNet -> Controller -> Actuators

Output: fig_compilation_pipeline.pdf / .png
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from style import apply_style, figsize, DOUBLE_COL, COLORS

apply_style()
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _draw_block(ax, xy, w, h, label, sublabel=None, color='#E8F0FE',
                edgecolor='k', fontsize=8, text_color='k'):
    """Draw a rounded rectangle with label."""
    rect = FancyBboxPatch(
        (xy[0] - w / 2, xy[1] - h / 2), w, h,
        boxstyle='round,pad=0.08',
        facecolor=color, edgecolor=edgecolor,
        linewidth=1.2, zorder=3
    )
    ax.add_patch(rect)
    if sublabel:
        ax.text(xy[0], xy[1] + 0.09, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color,
                zorder=4)
        ax.text(xy[0], xy[1] - 0.19, sublabel, ha='center', va='center',
                fontsize=fontsize - 2, color='0.4', zorder=4,
                style='italic')
    else:
        ax.text(xy[0], xy[1], label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color,
                zorder=4)


def _arrow(ax, start, end, label=None, color='k'):
    """Draw a thick arrow between blocks."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0',
        color=color, linewidth=1.8, zorder=2,
        mutation_scale=15,
    )
    ax.add_patch(arrow)
    if label:
        mid_x = 0.5 * (start[0] + end[0])
        mid_y = 0.5 * (start[1] + end[1]) + 0.15
        ax.text(mid_x, mid_y, label, ha='center', va='bottom',
                fontsize=5.5, color='0.3', style='italic')


def main():
    outdir = os.path.dirname(__file__)

    fig, ax = plt.subplots(figsize=figsize(DOUBLE_COL, 0.42))

    # Main pipeline blocks (horizontal flow)
    y_main = 1.0
    blocks = [
        (0.8, y_main, 1.2, 0.65, 'Petri Net\nBuilder',
         'Packet A', '#D6EAF8', COLORS["blue"]),
        (2.6, y_main, 1.2, 0.65, 'Fusion\nCompiler',
         'Packet B', '#D5F5E3', COLORS["green"]),
        (4.4, y_main, 1.2, 0.65, 'Compiled\nNet',
         'SNN artifact', '#FEF9E7', COLORS["orange"]),
        (6.8, y_main, 1.2, 0.65, 'Runtime\nController',
         'Packet C', '#FADBD8', COLORS["red"]),
    ]

    for x, y, w, h, label, sublabel, color, ec in blocks:
        _draw_block(ax, (x, y), w, h, label, sublabel=sublabel,
                    color=color, edgecolor=ec)

    # Arrows between main blocks
    _arrow(ax, (1.4, y_main), (2.0, y_main))
    _arrow(ax, (3.2, y_main), (3.8, y_main))
    _arrow(ax, (5.0, y_main), (6.2, y_main), 'tick(obs)')

    # Input/Output blocks
    _draw_block(ax, (6.8, -0.35), 1.0, 0.35, 'Actuators',
                color='#F5F5F5', edgecolor='0.5', fontsize=7)
    _arrow(ax, (6.8, y_main - 0.33), (6.8, -0.12), 'u(t)', color='0.4')

    # Sensor input
    _draw_block(ax, (6.8, 1.9), 1.0, 0.35, 'Sensors',
                color='#F5F5F5', edgecolor='0.5', fontsize=7)
    _arrow(ax, (6.8, 1.67), (6.8, y_main + 0.33), 'obs(t)', color='0.4')

    # Hardware targets (below compiled net)
    y_hw = -0.35
    hw_targets = [
        (3.3, y_hw, 'CPU\n(Rust)', '#EDE7F6'),
        (4.4, y_hw, 'FPGA\n(Verilog)', '#E0F2F1'),
        (5.5, y_hw, 'Loihi\n(neurom.)', '#FFF3E0'),
    ]
    for x, y, label, color in hw_targets:
        _draw_block(ax, (x, y), 0.9, 0.38, label, color=color,
                    edgecolor='0.5', fontsize=7)

    # Arrows from compiled net to hardware targets
    for x, y, _, _ in hw_targets:
        ax.annotate('', xy=(x, y + 0.19), xytext=(4.4, y_main - 0.33),
                    arrowprops=dict(arrowstyle='-|>', color='0.5',
                                   lw=0.8, connectionstyle='arc3,rad=0'))

    # Title label for hardware targets
    ax.text(4.4, -0.78, 'Prospective deployment targets',
            ha='center', va='top', fontsize=8, color='0.4',
            style='italic')

    # Inner details of compiler
    compiler_details = [
        'Step 1: Matrix extraction',
        'Step 2: LIF instantiation',
        'Step 3: Bitstream encoding',
        'Step 4: Assembly',
    ]
    for i, detail in enumerate(compiler_details):
        ax.text(2.6, 2.04 - i * 0.14, detail,
                ha='center', va='bottom', fontsize=5.5, color=COLORS["green"],
                family='monospace')

    ax.set_xlim(0.0, 7.6)
    ax.set_ylim(-1.0, 2.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Neuro-symbolic compilation pipeline: Petri net to SNN',
                 fontsize=11, pad=12)

    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(outdir, f'fig_compilation_pipeline.{ext}'))
    plt.close(fig)
    print('  [OK] fig_compilation_pipeline')


if __name__ == '__main__':
    main()
