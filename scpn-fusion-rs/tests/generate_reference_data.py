#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Generate Reference Data
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Generate reference baseline data for Rust migration validation.

Run from SCPN-Fusion-Core/ directory:
    python scpn-fusion-rs/tests/generate_reference_data.py

Outputs go to scpn-fusion-rs/tests/reference/
"""

import sys
import os
import json

# Add the source to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'reference')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_elliptic_reference():
    """Generate reference values for elliptic integrals from scipy."""
    try:
        from scipy.special import ellipk, ellipe
    except ImportError:
        print("WARNING: scipy not available, skipping elliptic reference")
        return

    ms = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
    ref = {
        "K": {str(m): float(ellipk(m)) for m in ms},
        "E": {str(m): float(ellipe(m)) for m in ms},
    }
    path = os.path.join(OUTPUT_DIR, 'reference_elliptic.json')
    with open(path, 'w') as f:
        json.dump(ref, f, indent=2)
    print(f"  Elliptic integrals: {path}")


def generate_bosch_hale_reference():
    """Generate reference Bosch-Hale D-T rates."""
    try:
        from scpn_fusion.core.fusion_ignition_sim import FusionBurnPhysics
    except ImportError:
        print("WARNING: scpn_fusion not importable, generating from formula directly")
        import numpy as np

        def bosch_hale_dt(T_keV):
            T = max(T_keV, 0.1)
            return 3.68e-18 / (T ** (2.0 / 3.0)) * np.exp(-19.94 / (T ** (1.0 / 3.0)))

        temps = [0.1, 1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0]
        rates = {str(t): float(bosch_hale_dt(t)) for t in temps}
        path = os.path.join(OUTPUT_DIR, 'reference_bosch_hale.json')
        with open(path, 'w') as f:
            json.dump(rates, f, indent=2)
        print(f"  Bosch-Hale rates: {path}")
        return

    sim = FusionBurnPhysics("iter_config.json")
    temps = [0.1, 1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0]
    rates = {str(t): float(sim.bosch_hale_dt(t)) for t in temps}
    path = os.path.join(OUTPUT_DIR, 'reference_bosch_hale.json')
    with open(path, 'w') as f:
        json.dump(rates, f, indent=2)
    print(f"  Bosch-Hale rates: {path}")


def generate_equilibrium_reference():
    """Generate reference ITER equilibrium from Python solver."""
    try:
        import numpy as np
        from scpn_fusion.core.fusion_kernel import FusionKernel
    except ImportError:
        print("WARNING: scpn_fusion not importable, skipping equilibrium reference")
        return

    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'iter_config.json')
    fk = FusionKernel(config_path)
    fk.solve_equilibrium()

    path = os.path.join(OUTPUT_DIR, 'reference_iter_equilibrium.npz')
    np.savez(
        path,
        Psi=fk.Psi,
        J_phi=fk.J_phi,
        B_R=getattr(fk, 'B_R', None),
        B_Z=getattr(fk, 'B_Z', None),
        R=fk.R,
        Z=fk.Z,
    )
    print(f"  ITER equilibrium: {path}")

    # Vacuum field reference
    psi_vac = fk.calculate_vacuum_field()
    vac_path = os.path.join(OUTPUT_DIR, 'reference_vacuum_field.npz')
    np.savez(vac_path, Psi_vac=psi_vac)
    print(f"  Vacuum field: {vac_path}")


def generate_config_summary():
    """Generate summary of all 6 configs for quick validation."""
    configs_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    configs = [
        ('iter_config.json', 'iter_config.json'),
        ('iter_validated', 'validation/iter_validated_config.json'),
        ('iter_genetic', 'validation/iter_genetic_config.json'),
        ('iter_analytic', 'validation/iter_analytic_config.json'),
        ('iter_force_balanced', 'validation/iter_force_balanced.json'),
        ('default', 'src/scpn_fusion/core/default_config.json'),
    ]

    summary = {}
    for name, rel_path in configs:
        full_path = os.path.join(configs_dir, rel_path)
        with open(full_path) as f:
            cfg = json.load(f)
        summary[name] = {
            'reactor_name': cfg['reactor_name'],
            'grid_resolution': cfg['grid_resolution'],
            'num_coils': len(cfg['coils']),
            'max_iterations': cfg['solver']['max_iterations'],
            'R_range': [cfg['dimensions']['R_min'], cfg['dimensions']['R_max']],
            'Z_range': [cfg['dimensions']['Z_min'], cfg['dimensions']['Z_max']],
        }

    path = os.path.join(OUTPUT_DIR, 'config_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Config summary: {path}")


if __name__ == '__main__':
    print("Generating reference data for SCPN-Fusion-Core Rust migration...")
    print()

    print("[1/4] Config summary")
    generate_config_summary()

    print("[2/4] Elliptic integrals")
    generate_elliptic_reference()

    print("[3/4] Bosch-Hale D-T rates")
    generate_bosch_hale_reference()

    print("[4/4] Equilibrium reference")
    generate_equilibrium_reference()

    print()
    print("Done! Reference data saved to:", OUTPUT_DIR)
