#!/usr/bin/env python3
"""
Master script: Generate all publication figures for SCPN-Fusion-Core papers.

Produces PDF and PNG figures in papers/figures/ for:
  - Paper A (Equilibrium Solver): 6 figures
  - Paper B (SNN Controller):    6 figures

Requirements: numpy, matplotlib, scipy (standard scientific Python stack).
No project-specific imports are used — all data is synthetic/analytical.

Usage:
    python papers/figures/generate_all_figures.py

Each figure is also independently runnable:
    python papers/figures/fig_gs_convergence.py
"""

import os
import sys
import time

# Ensure the figures directory is on the import path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Use non-interactive backend for headless rendering
import matplotlib
matplotlib.use('Agg')


def main():
    print('=' * 70)
    print('SCPN-Fusion-Core: Generating publication figures')
    print('=' * 70)
    t0 = time.time()

    # ---- Paper A: Equilibrium Solver ----
    print('\n--- Paper A: Equilibrium Solver ---')

    from fig_gs_convergence import main as gen_gs_convergence
    gen_gs_convergence()

    from fig_sparc_equilibrium import main as gen_sparc_equilibrium
    gen_sparc_equilibrium()

    from fig_inverse_reconstruction import main as gen_inverse_reconstruction
    gen_inverse_reconstruction()

    from fig_performance_scaling import main as gen_performance_scaling
    gen_performance_scaling()

    from fig_neural_surrogate import main as gen_neural_surrogate
    gen_neural_surrogate()

    from fig_validation_rmse import main as gen_validation_rmse
    gen_validation_rmse()

    # ---- Paper B: SNN Controller ----
    print('\n--- Paper B: SNN Controller ---')

    from fig_petri_net import main as gen_petri_net
    gen_petri_net()

    from fig_compilation_pipeline import main as gen_compilation_pipeline
    gen_compilation_pipeline()

    from fig_vertical_stability import main as gen_vertical_stability
    gen_vertical_stability()

    from fig_latency_comparison import main as gen_latency_comparison
    gen_latency_comparison()

    from fig_radiation_tolerance import main as gen_radiation_tolerance
    gen_radiation_tolerance()

    from fig_lif_neuron import main as gen_lif_neuron
    gen_lif_neuron()

    elapsed = time.time() - t0
    print(f'\n{"=" * 70}')
    print(f'All 12 figures generated in {elapsed:.1f}s')
    print(f'Output directory: {SCRIPT_DIR}')
    print(f'{"=" * 70}')

    # List generated files
    pdf_files = sorted(f for f in os.listdir(SCRIPT_DIR) if f.endswith('.pdf'))
    png_files = sorted(f for f in os.listdir(SCRIPT_DIR) if f.endswith('.png'))
    print(f'\nPDF files ({len(pdf_files)}):')
    for f in pdf_files:
        size_kb = os.path.getsize(os.path.join(SCRIPT_DIR, f)) / 1024
        print(f'  {f:40s} {size_kb:6.1f} KB')
    print(f'\nPNG files ({len(png_files)}):')
    for f in png_files:
        size_kb = os.path.getsize(os.path.join(SCRIPT_DIR, f)) / 1024
        print(f'  {f:40s} {size_kb:6.1f} KB')


if __name__ == '__main__':
    main()
