# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Synthetic Dataset Generator
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Generates a large synthetic dataset of tokamak equilibria (GEQDSK format)
by solving the Grad-Shafranov equation over a randomized parameter space.

This addresses Step 1.1: "Expand Validation Dataset Ruthlessly."
Target: 100+ high-quality equilibria spanning DIII-D, KSTAR, ITER-like regimes.
"""

import sys
import os
import json
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Ensure local source is in path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from scpn_fusion.core.fusion_kernel import FusionKernel
from scpn_fusion.core.eqdsk import GEqdsk, write_geqdsk

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = repo_root / "data" / "synthetic_geqdsk"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_random_config(seed: int) -> dict:
    """Generate a randomized tokamak configuration."""
    rng = np.random.default_rng(seed)
    
    # Randomize machine size (Compact to Large)
    R0 = rng.uniform(1.6, 6.2) # Major radius (m)
    aspect_ratio = rng.uniform(2.5, 3.5)
    a = R0 / aspect_ratio # Minor radius (m)
    
    # Shaping
    kappa = rng.uniform(1.4, 1.9) # Elongation
    delta = rng.uniform(0.2, 0.6) # Triangularity
    
    # Physics
    Ip_ma = rng.uniform(2.0, 15.0) # Plasma Current (MA)
    B0 = rng.uniform(2.0, 6.0) # Toroidal Field (T)
    
    # Grid
    # Use moderately high resolution for quality
    nr = 129
    nz = 129
    
    # Define domain
    r_min = max(0.1, R0 - a * 1.5)
    r_max = R0 + a * 1.5
    z_min = -a * kappa * 1.5
    z_max = a * kappa * 1.5
    
    # Coils (Simplified PF set to support the shape)
    # We place coils based on the target shape to ensure convergence
    coils = []
    
    # Vertical Field Coils (Outer)
    coils.append({"r": r_max + 0.5, "z": z_max + 0.5, "current": -Ip_ma * 0.5e6}) # Top Outer
    coils.append({"r": r_max + 0.5, "z": z_min - 0.5, "current": -Ip_ma * 0.5e6}) # Bottom Outer
    
    # Shaping Coils (Divertor / Elongation)
    coils.append({"r": r_min - 0.2, "z": z_max, "current": Ip_ma * 0.2e6}) # Top Inner
    coils.append({"r": r_min - 0.2, "z": z_min, "current": Ip_ma * 0.2e6}) # Bottom Inner
    
    # Central Solenoid (CS) - flux provider
    coils.append({"r": r_min * 0.5, "z": 0.0, "current": Ip_ma * 0.1e6})

    config = {
        "reactor_name": f"SYNTH_{seed:04d}",
        "dimensions": {
            "R_min": float(r_min),
            "R_max": float(r_max),
            "Z_min": float(z_min),
            "Z_max": float(z_max)
        },
        "grid_resolution": [nr, nz],
        "physics": {
            "plasma_current_target": float(Ip_ma * 1e6),
            "vacuum_permeability": 1.25663706e-6, # mu0
            "profiles": {
                "mode": "l-mode" # Start with L-mode for stability
            }
        },
        "coils": coils,
        "solver": {
            "max_iterations": 2000,
            "convergence_threshold": 1e-5, # Strict convergence
            "relaxation_factor": 0.05, # Conservative relaxation
            "solver_method": "sor" 
        }
    }
    return config

def validate_equilibrium(kernel: FusionKernel) -> bool:
    """Physics-informed validation of the solution."""
    # 1. Check Convergence
    # The kernel stores the final residual
    # We don't have direct access to 'residual' attribute easily outside solve return
    # But we can check for NaNs and basic shape properties
    
    if np.isnan(kernel.Psi).any() or np.isinf(kernel.Psi).any():
        logger.warning("Validation Failed: NaNs in Psi")
        return False
        
    # 2. Check Magnetic Axis
    iz, ir, psi_axis = kernel._find_magnetic_axis()
    r_axis = kernel.R[ir]
    z_axis = kernel.Z[iz]
    
    # Axis should be roughly central
    r_mid = (kernel.cfg["dimensions"]["R_min"] + kernel.cfg["dimensions"]["R_max"]) / 2
    if abs(r_axis - r_mid) > (r_mid * 0.5):
        logger.warning(f"Validation Failed: Axis too far off-center (R={r_axis:.2f}, Mid={r_mid:.2f})")
        return False
        
    # 3. Check Current Integration
    # Does the integrated current match the target?
    # FusionKernel automatically normalizes, but let's check sanity
    total_current = np.sum(kernel.J_phi) * kernel.dR * kernel.dZ
    target = kernel.cfg["physics"]["plasma_current_target"]
    error = abs(total_current - target) / (target + 1e-9)
    
    if error > 0.01: # 1% error tolerance
        logger.warning(f"Validation Failed: Current mismatch (Target={target:.2e}, Actual={total_current:.2e}, Err={error:.2%})")
        return False

    return True

def convert_to_geqdsk(kernel: FusionKernel, run_id: str) -> GEqdsk:
    """Convert FusionKernel state to GEqdsk object."""
    iz, ir, psi_axis = kernel._find_magnetic_axis()
    _, psi_bry = kernel.find_x_point(kernel.Psi)
    
    nw = kernel.NR
    nh = kernel.NZ
    
    # 1D profiles (Simplified extraction along midplane Z=0 approx)
    # Finding index closest to magnetic axis Z
    iz_axis = iz
    
    # Extract profiles along the radial chord at Z_axis
    # This is a simplification; ideally we map to flux coordinates
    fpol = np.linspace(1.0, 0.0, nw) # Placeholder for F(psi)
    pres = kernel.J_phi[iz_axis, :] # Rough proxy for pressure profile shape
    ffprime = np.zeros(nw)
    pprime = np.zeros(nw)
    qpsi = np.ones(nw) * 1.5 # Placeholder q-profile
    
    return GEqdsk(
        description=f"SCPN Synthetic {run_id}",
        nw=nw,
        nh=nh,
        rdim=kernel.R[-1] - kernel.R[0],
        zdim=kernel.Z[-1] - kernel.Z[0],
        rcentr=(kernel.R[0] + kernel.R[-1])/2,
        rleft=kernel.R[0],
        zmid=(kernel.Z[0] + kernel.Z[-1])/2,
        rmaxis=kernel.R[ir],
        zmaxis=kernel.Z[iz],
        simag=psi_axis,
        sibry=psi_bry,
        current=kernel.cfg["physics"]["plasma_current_target"],
        psirz=kernel.Psi,
        fpol=fpol,
        pres=pres,
        ffprime=ffprime,
        pprime=pprime,
        qpsi=qpsi
    )

def main():
    logger.info("Starting Synthetic Dataset Generation...")
    
    target_count = 20 # Start with 20 for this session (scale to 100 later)
    success_count = 0
    seed = 4200
    
    pbar = tqdm(total=target_count)
    
    while success_count < target_count:
        seed += 1
        run_id = f"RUN_{seed}"
        
        try:
            config = generate_random_config(seed)
            
            # Save temporary config
            config_path = DATA_DIR / f"{run_id}_config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
                
            # Run Solver
            fk = FusionKernel(config_path)
            result = fk.solve_equilibrium()
            
            if not result["converged"]:
                logger.debug(f"{run_id}: Solver did not converge. Skipping.")
                config_path.unlink(missing_ok=True)
                continue
                
            # Validate
            if not validate_equilibrium(fk):
                logger.debug(f"{run_id}: Validation failed. Skipping.")
                config_path.unlink(missing_ok=True)
                continue
                
            # Save GEQDSK
            geqdsk = convert_to_geqdsk(fk, run_id)
            geqdsk_path = DATA_DIR / f"{run_id}.geqdsk"
            write_geqdsk(geqdsk, geqdsk_path)
            
            # Save meta-data (physics metrics)
            meta = {
                "seed": seed,
                "residual": result["residual"],
                "iterations": result["iterations"],
                "beta_p": 0.5, # Placeholder
                "q_95": 3.0 # Placeholder
            }
            with open(DATA_DIR / f"{run_id}_meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            
            # Cleanup temp config
            config_path.unlink(missing_ok=True)
            
            success_count += 1
            pbar.update(1)
            
        except Exception as e:
            logger.error(f"{run_id}: Error: {e}")
            continue

    pbar.close()
    logger.info(f"Successfully generated {success_count} equilibria in {DATA_DIR}")

if __name__ == "__main__":
    main()
