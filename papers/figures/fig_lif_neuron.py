#!/usr/bin/env python3
"""
Figure: LIF neuron membrane potential trace with spikes and threshold.

Shows the leaky integrate-and-fire (LIF) neuron dynamics used in the
SNN compiler.  Demonstrates membrane potential evolution with input
current, threshold crossing, spike generation, and reset.

Two-panel figure:
  (a) LIF neuron trace: membrane potential V(t) with spikes
  (b) Input current I(t) and output spike train

Output: fig_lif_neuron.pdf / .png
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from style import apply_style, figsize, DOUBLE_COL, COLORS

apply_style()
import matplotlib.pyplot as plt


def _simulate_lif(t, I_input, V_rest=0.0, V_th=0.4, V_reset=0.0,
                  tau_mem=10.0, R=1.0):
    """Simulate a LIF neuron and return membrane potential + spike times."""
    dt = t[1] - t[0]
    V = np.zeros_like(t)
    V[0] = V_rest
    spikes = []

    for i in range(1, len(t)):
        # Leak + input integration
        dV = (-V[i - 1] + V_rest + R * I_input[i - 1]) / tau_mem
        V[i] = V[i - 1] + dV * dt

        # Threshold crossing
        if V[i] >= V_th:
            spikes.append(t[i])
            V[i] = V_reset  # reset

    return V, np.array(spikes)


def main():
    outdir = os.path.dirname(__file__)

    # Time
    dt = 0.1  # ms
    t = np.arange(0, 100, dt)  # 100 ms

    # Input current: structured pattern with varying amplitude
    I_input = np.zeros_like(t)
    # Phase 1: sub-threshold constant input
    mask1 = (t >= 5) & (t < 20)
    I_input[mask1] = 0.03

    # Phase 2: supra-threshold steps
    mask2 = (t >= 25) & (t < 45)
    I_input[mask2] = 0.06

    # Phase 3: ramp input
    mask3 = (t >= 50) & (t < 70)
    I_input[mask3] = 0.02 + 0.003 * (t[mask3] - 50)

    # Phase 4: strong burst
    mask4 = (t >= 75) & (t < 90)
    I_input[mask4] = 0.08

    # Add small noise
    rng = np.random.RandomState(17)
    I_input += rng.normal(0, 0.003, len(t))
    I_input = np.maximum(I_input, 0)

    # LIF parameters
    V_rest = 0.0
    V_th = 0.4
    V_reset = 0.0
    tau_mem = 8.0
    R = 10.0

    V, spike_times = _simulate_lif(t, I_input, V_rest=V_rest, V_th=V_th,
                                    V_reset=V_reset, tau_mem=tau_mem, R=R)

    # For visual spike representation, add brief spike peaks
    V_display = V.copy()
    for sp in spike_times:
        idx = np.argmin(np.abs(t - sp))
        if idx > 0:
            V_display[idx] = V_th + 0.15  # overshoot for visibility

    # ---- Figure ----
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize(DOUBLE_COL, 0.6),
        sharex=True, gridspec_kw={'height_ratios': [3, 1]}
    )

    # Panel (a): Membrane potential
    ax1.plot(t, V_display, '-', color=COLORS["blue"], lw=1.2,
             label=r'$V_{\mathrm{mem}}(t)$')

    # Threshold line
    ax1.axhline(V_th, color=COLORS["red"], ls='--', lw=1.0,
                label=r'$V_{\mathrm{th}}$' + f' = {V_th}')

    # Reset level
    ax1.axhline(V_reset, color='grey', ls=':', lw=0.6)
    ax1.text(98, V_reset + 0.02, r'$V_{\mathrm{reset}}$', fontsize=7,
             color='grey', ha='right')

    # Spike markers
    for sp in spike_times:
        idx = np.argmin(np.abs(t - sp))
        ax1.plot([sp, sp], [V_th, V_th + 0.15], '-', color=COLORS["red"],
                 lw=1.5)
        ax1.plot(sp, V_th + 0.15, 'v', color=COLORS["red"], markersize=4)

    # Phase annotations
    phases = [
        (12.5, 'Sub-threshold'),
        (35, 'Regular spiking'),
        (60, 'Ramp response'),
        (82, 'Burst'),
    ]
    for x_pos, label in phases:
        ax1.text(x_pos, V_th + 0.22, label, ha='center', va='bottom',
                 fontsize=6.5, color='0.4', style='italic')

    ax1.set_ylabel(r'$V_{\mathrm{mem}}$')
    ax1.set_ylim(-0.1, V_th + 0.35)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_title('Leaky integrate-and-fire (LIF) neuron dynamics')

    # Panel (b): Input current and output spikes
    ax2.fill_between(t, 0, I_input, color=COLORS["cyan"], alpha=0.5,
                     label=r'$I_{\mathrm{input}}(t)$')
    ax2.plot(t, I_input, '-', color=COLORS["cyan"], lw=0.8)

    # Output spike raster
    for sp in spike_times:
        ax2.axvline(sp, ymin=0.7, ymax=1.0, color=COLORS["red"], lw=1.0)

    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel(r'$I_{\mathrm{in}}$')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.01, 0.12)
    ax2.legend(loc='upper left', fontsize=7)

    # Mark spikes label
    if len(spike_times) > 0:
        ax2.text(spike_times[0], 0.11, 'output\nspikes', fontsize=6,
                 color=COLORS["red"], ha='center', va='top')

    # Equation annotation
    eq_text = (r'$\tau_{\mathrm{mem}} \frac{dV}{dt} = '
               r'-(V - V_{\mathrm{rest}}) + R \cdot I(t)$'
               f'\n$\\tau_{{\\mathrm{{mem}}}} = {tau_mem}$ ms, '
               f'$R = {R}$')
    ax1.text(0.97, 0.3, eq_text, transform=ax1.transAxes,
             fontsize=7, ha='right', va='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='0.7', alpha=0.9))

    fig.subplots_adjust(hspace=0.08)

    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(outdir, f'fig_lif_neuron.{ext}'))
    plt.close(fig)
    print('  [OK] fig_lif_neuron')


if __name__ == '__main__':
    main()
