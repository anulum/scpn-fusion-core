#!/usr/bin/env python3
"""
Figure: 8-place, 4-transition Petri net for vertical stability control.

Schematic diagram drawn using matplotlib patches and annotations.
Shows sensor places (left), transitions (center), action places (right),
with weighted arcs.  Uses the bipartite graph structure from Paper B
Section 5.1.

Output: fig_petri_net.pdf / .png
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import apply_style, figsize, DOUBLE_COL, COLORS

apply_style()
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches


def _draw_place(ax, xy, label, token_density=None, color='white',
                edgecolor='k'):
    """Draw a Petri net place (circle)."""
    circle = Circle(xy, 0.18, facecolor=color, edgecolor=edgecolor,
                    linewidth=1.2, zorder=5)
    ax.add_patch(circle)
    ax.text(xy[0], xy[1], label, ha='center', va='center',
            fontsize=6.5, fontweight='bold', zorder=6)
    if token_density is not None:
        ax.text(xy[0], xy[1] - 0.26, f'{token_density}',
                ha='center', va='top', fontsize=6, color='0.4')


def _draw_transition(ax, xy, label, color='0.2'):
    """Draw a Petri net transition (filled rectangle)."""
    rect = FancyBboxPatch((xy[0] - 0.12, xy[1] - 0.07), 0.24, 0.14,
                          boxstyle='round,pad=0.02',
                          facecolor=color, edgecolor='k',
                          linewidth=1.0, zorder=5)
    ax.add_patch(rect)
    ax.text(xy[0], xy[1], label, ha='center', va='center',
            fontsize=6, color='white', fontweight='bold', zorder=6)


def _draw_arc(ax, start, end, weight=None, color='k', style='-',
              connectionstyle='arc3,rad=0.08'):
    """Draw a weighted arc between two nodes."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        connectionstyle=connectionstyle,
        color=color, linewidth=1.0, zorder=3,
        mutation_scale=12,
        shrinkA=18, shrinkB=18,
    )
    ax.add_patch(arrow)
    if weight is not None:
        mid_x = 0.5 * (start[0] + end[0])
        mid_y = 0.5 * (start[1] + end[1])
        ax.text(mid_x, mid_y + 0.08, str(weight),
                ha='center', va='bottom', fontsize=6, color=color,
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                          edgecolor='none', alpha=0.8))


def main():
    outdir = os.path.dirname(__file__)

    fig, ax = plt.subplots(figsize=figsize(DOUBLE_COL, 0.55))

    # Layout coordinates
    # Sensor places (left column)
    sensor_x = 0.8
    sensor_places = {
        'z_pos':  (sensor_x, 3.5),
        'z_neg':  (sensor_x, 2.5),
        'dz_pos': (sensor_x, 1.5),
        'dz_neg': (sensor_x, 0.5),
    }

    # Transitions (center column)
    trans_x = 2.5
    transitions = {
        'T_corr_up':   (trans_x, 3.2),
        'T_corr_dn':   (trans_x, 2.2),
        'T_heat_up':   (trans_x, 1.2),
        'T_heat_dn':   (trans_x, 0.5),
    }

    # Action places (right column)
    action_x = 4.2
    action_places = {
        'coil_up':   (action_x, 3.5),
        'coil_dn':   (action_x, 2.5),
        'heat_up':   (action_x, 1.5),
        'heat_dn':   (action_x, 0.5),
    }

    # Draw sensor places (blue tint)
    sensor_color = '#D6EAF8'
    for name, pos in sensor_places.items():
        nice = name.replace('_', '\\_')
        _draw_place(ax, pos, nice, color=sensor_color,
                    edgecolor=COLORS["blue"])

    # Draw action places (red tint)
    action_color = '#FADBD8'
    for name, pos in action_places.items():
        nice = name.replace('_', '\\_')
        _draw_place(ax, pos, nice, color=action_color,
                    edgecolor=COLORS["red"])

    # Draw transitions (dark bars)
    trans_labels = {
        'T_corr_up': 'T_up',
        'T_corr_dn': 'T_dn',
        'T_heat_up': 'T_h+',
        'T_heat_dn': 'T_h-',
    }
    for name, pos in transitions.items():
        _draw_transition(ax, pos, trans_labels[name], color='0.25')

    # Draw arcs: sensor places -> transitions (input arcs)
    arcs_in = [
        ('z_neg',  'T_corr_up',  '1.0'),
        ('z_pos',  'T_corr_dn',  '1.0'),
        ('dz_neg', 'T_corr_up',  '0.5'),
        ('dz_pos', 'T_corr_dn',  '0.5'),
        ('z_pos',  'T_heat_up',  '0.3'),
        ('z_neg',  'T_heat_up',  '0.3'),
        ('dz_pos', 'T_heat_dn',  '0.8'),
        ('dz_neg', 'T_heat_dn',  '0.2'),
    ]
    for src, tgt, w in arcs_in:
        rad = 0.08
        if abs(sensor_places[src][1] - transitions[tgt][1]) > 1.5:
            rad = 0.15
        _draw_arc(ax, sensor_places[src], transitions[tgt],
                  weight=w, color=COLORS["blue"],
                  connectionstyle=f'arc3,rad={rad}')

    # Draw arcs: transitions -> action places (output arcs)
    arcs_out = [
        ('T_corr_up',  'coil_up',   '1.0'),
        ('T_corr_dn',  'coil_dn',   '1.0'),
        ('T_heat_up',  'heat_up',   '1.0'),
        ('T_heat_dn',  'heat_dn',   '1.0'),
    ]
    for src, tgt, w in arcs_out:
        _draw_arc(ax, transitions[src], action_places[tgt],
                  weight=w, color=COLORS["red"],
                  connectionstyle='arc3,rad=0.05')

    # Column labels
    ax.text(sensor_x, 4.2, 'Sensor Places\n(observations)',
            ha='center', va='bottom', fontsize=9, fontweight='bold',
            color=COLORS["blue"])
    ax.text(trans_x, 4.2, 'Transitions\n(firing gates)',
            ha='center', va='bottom', fontsize=9, fontweight='bold',
            color='0.25')
    ax.text(action_x, 4.2, 'Action Places\n(actuator cmds)',
            ha='center', va='bottom', fontsize=9, fontweight='bold',
            color=COLORS["red"])

    # Threshold annotations
    for name, pos in transitions.items():
        th = '0.3' if 'heat' in name else '0.4'
        ax.text(pos[0], pos[1] - 0.18, f'$\\theta={th}$',
                ha='center', va='top', fontsize=6, color='0.4')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=sensor_color, edgecolor=COLORS["blue"],
                       linewidth=1.2, label='Sensor place'),
        mpatches.Patch(facecolor=action_color, edgecolor=COLORS["red"],
                       linewidth=1.2, label='Action place'),
        mpatches.Patch(facecolor='0.25', edgecolor='k',
                       linewidth=1.0, label='Transition'),
    ]
    ax.legend(handles=legend_elements, loc='lower center',
              ncol=3, fontsize=7, frameon=True)

    ax.set_xlim(0.2, 5.0)
    ax.set_ylim(-0.3, 4.8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('8-place vertical stability Petri net controller',
                 fontsize=11, pad=10)

    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(outdir, f'fig_petri_net.{ext}'))
    plt.close(fig)
    print('  [OK] fig_petri_net')


if __name__ == '__main__':
    main()
