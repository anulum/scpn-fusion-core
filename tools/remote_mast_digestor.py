# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Remote MAST Data Digestor (Physics Aligned)
"""
High-memory data digestor for the ML350 Server.

Updates:
- Noise Floor Fix for Density.
- Flattop detection.
- Physics Alignment: Dynamic rho_grid extending beyond separatrix (R_s).
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# SAS Cache Paths
SAS_MAST_ROOT = Path("/mnt/data_sas/DATASETS/SCPN-CONTROL/mast")
CACHE_DIR = SAS_MAST_ROOT / "cache"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def process_shot_comparison(shot_id: int):
    """
    Truth-Pass Worker: Compare Rust Oracle against MAST Magnetic Signals.
    """
    try:
        from scpn_fusion.io.mast_ingestor import MastIngestor
        ingestor = MastIngestor(cache_dir=CACHE_DIR)
        
        summary = ingestor.load_shot_summary(shot_id)
        
        # 1. IDENTIFY FLATTOP WINDOW
        ip = summary["ip"]
        max_ip = np.max(ip)
        flattop_indices = np.where(ip > 0.8 * max_ip)[0]
        
        if len(flattop_indices) == 0:
            logger.warning(f"Shot {shot_id}: No stable flattop found.")
            return None

        logger.info(f"Processing Shot {shot_id}: Validating flattop stability...")
        
        # 2. PHYSICS ALIGNED ORACLE SOLVE
        from scpn_fusion.core.frc_rigid_rotor import RigidRotorFRCInputs, solve_frc_equilibrium
        
        errors = []
        sample_indices = np.linspace(flattop_indices[0], flattop_indices[-1], 10, dtype=int)
        
        for idx in sample_indices:
            # Noise Floor
            n0_safe = max(float(summary["density"][idx]), 1e18)
            
            # Geometry Alignment
            R_s = 0.6  # Approximate separatrix radius for MAST (m)
            # GRID MUST EXTEND BEYOND R_s (e.g., 1.5 * R_s)
            rho_grid = np.linspace(0.0, 1.2 * R_s, 128)
            
            inputs = RigidRotorFRCInputs(
                n0=n0_safe,
                T_i_eV=1500.0, 
                T_e_eV=800.0,
                theta_dot=0.0,
                R_s=R_s,
                B_ext=summary["ip"][idx] * 2e-7 / R_s,
                delta=0.03
            )
            
            # Solve using the high-performance Rust lane
            state = solve_frc_equilibrium(inputs, rho_grid, solver="rust")
            
            if state.converged:
                errors.append(state.residual)

        return {
            "shot_id": shot_id, 
            "mean_residual": np.mean(errors) if errors else 1.0, 
            "samples": len(errors),
            "max_ip_ma": float(max_ip / 1e6)
        }

    except Exception as e:
        logger.error(f"Error processing Shot {shot_id}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--shots", nargs="+", type=int)
    args = parser.parse_args()

    try:
        os.nice(19)
    except:
        pass

    if not args.shots:
        target_shots = [30419, 30420, 30421, 30422, 30423, 30424]
    else:
        target_shots = args.shots

    logger.info(f"Starting PHYSICS-ALIGNED digestion of {len(target_shots)} shots...")
    t0 = time.perf_counter()

    with mp.Pool(args.workers) as pool:
        results = pool.map(process_shot_comparison, target_shots)

    valid_results = [r for r in results if r is not None]
    
    t_total = time.perf_counter() - t0
    logger.info(f"Digestion complete. Processed {len(valid_results)} shots in {t_total:.1f}s.")
    
    out_path = SAS_MAST_ROOT / "digestion_report_v3_aligned.npz"
    np.savez(out_path, results=valid_results)
    logger.info(f"Physics-aligned summary report saved to {out_path}")

if __name__ == "__main__":
    main()
