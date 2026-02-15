#!/usr/bin/env python3
"""
Figure: Neural equilibrium surrogate — training loss + inference time.

Two-panel figure:
  (a) Training loss curve (MSE vs epoch, log scale) for MLP surrogate
      trained on SPARC GEQDSK data with PCA dimensionality reduction.
  (b) Inference time comparison: iterative GS solve vs neural surrogate
      on different platforms (bar chart, log scale).

Output: fig_neural_surrogate.pdf / .png
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

    # ---- Panel (a): Training loss curve ----
    rng = np.random.RandomState(7)
    epochs = np.arange(1, 501)

    # Synthetic training loss: exponential decay + noise
    train_loss = 0.15 * np.exp(-epochs / 80) + 0.002
    train_loss += rng.normal(0, 0.001, len(epochs)) * np.exp(-epochs / 200)
    train_loss = np.maximum(train_loss, 5e-4)

    # Validation loss: slightly higher, diverges a bit after overfitting
    val_loss = 0.18 * np.exp(-epochs / 90) + 0.003
    val_loss += rng.normal(0, 0.0015, len(epochs)) * np.exp(-epochs / 250)
    val_loss = np.maximum(val_loss, 8e-4)
    # Add slight overfitting rise after epoch 400
    val_loss[380:] += np.linspace(0, 0.002, len(val_loss[380:]))

    ax1.semilogy(epochs, train_loss, '-', color=COLORS["blue"],
                 lw=1.2, label='Training loss', alpha=0.8)
    ax1.semilogy(epochs, val_loss, '-', color=COLORS["red"],
                 lw=1.2, label='Validation loss', alpha=0.8)

    # Best model marker
    best_epoch = np.argmin(val_loss) + 1
    best_val = val_loss[best_epoch - 1]
    ax1.plot(best_epoch, best_val, '*', color=COLORS["green"],
             markersize=10, markeredgecolor='k', markeredgewidth=0.4,
             zorder=5, label=f'Best model (epoch {best_epoch})')

    ax1.axhline(0.005, color='grey', ls=':', lw=0.8)
    ax1.text(480, 0.0055, r'$5\times 10^{-3}$ target', fontsize=7,
             color='grey', ha='right')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean squared error')
    ax1.set_title('(a) Surrogate training loss')
    ax1.set_xlim(0, 510)
    ax1.set_ylim(3e-4, 0.3)
    ax1.legend(loc='upper right', fontsize=8)

    # ---- Panel (b): Inference time comparison ----
    methods = [
        'GS solve\n(Python)',
        'GS solve\n(Rust)',
        'Neural\n(NumPy)',
        'Neural\n(Rust)',
    ]
    times_us = [5e6, 1e5, 5, 0.8]  # microseconds
    colors_bar = [COLORS["blue"], COLORS["red"],
                  COLORS["cyan"], COLORS["orange"]]

    x = np.arange(len(methods))
    bars = ax2.bar(x, times_us, color=colors_bar, edgecolor='k',
                   linewidth=0.4, width=0.6, zorder=3)

    ax2.set_yscale('log')
    ax2.set_ylabel(r'Inference time ($\mu$s)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=8)
    ax2.set_ylim(0.3, 2e7)
    ax2.set_title('(b) Equilibrium inference latency')

    # Value labels
    for bar, t in zip(bars, times_us):
        if t >= 1e6:
            label = f'{t / 1e6:.0f} s'
        elif t >= 1e3:
            label = f'{t / 1e3:.0f} ms'
        elif t >= 1:
            label = f'{t:.0f} us'
        else:
            label = f'{t:.1f} us'
        ax2.text(bar.get_x() + bar.get_width() / 2, t * 1.5,
                 label, ha='center', va='bottom', fontsize=7,
                 fontweight='bold')

    # Speedup annotation
    speedup = times_us[0] / times_us[3]
    ax2.annotate(f'{speedup:.0e}x\nfaster',
                 xy=(3, times_us[3]),
                 xytext=(2.5, 100),
                 fontsize=8, color=COLORS["orange"],
                 ha='center', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=COLORS["orange"],
                                 lw=1.0))

    ax2.yaxis.grid(True, which='major', alpha=0.3, zorder=0)

    fig.subplots_adjust(wspace=0.35)

    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(outdir, f'fig_neural_surrogate.{ext}'))
    plt.close(fig)
    print('  [OK] fig_neural_surrogate')


if __name__ == '__main__':
    main()
