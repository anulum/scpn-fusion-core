# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — TGLF Dataset Generator
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
"""
Generates 10,000+ TGLF-like turbulence fields by perturbing reference cases.
Used to train the FNO on physics-informed data (Roadmap G4).
"""

import json
import logging
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

# Use reference cases from tglf_interface
from scpn_fusion.core.tglf_interface import REFERENCE_CASES

logger = logging.getLogger(__name__)

def generate_tglf_like_dataset(output_path="validation/reference_data/tglf_training_data.npz", n_samples=10000):
    print(f"--- Generating {n_samples} TGLF-like samples ---")
    
    rng = np.random.default_rng(42)
    
    # Grid size for FNO (matches fno_jax_training)
    grid_size = 64
    
    all_fields = []
    all_intensities = [] # Target: Gamma_max
    
    # Base regimes
    regimes = list(REFERENCE_CASES.keys())
    
    for i in range(n_samples):
        regime_name = rng.choice(regimes)
        ref = REFERENCE_CASES[regime_name]
        
        # Pick a random rho point from reference
        idx = rng.integers(0, len(ref["rho_points"]))
        base_gamma = ref["gamma_max"][idx]
        
        # Perturb growth rate
        gamma = base_gamma * rng.uniform(0.5, 2.0)
        
        # Generate a structured turbulence field (spectral noise)
        # Power spectrum ~ k^-alpha
        alpha = 3.0 if "ITG" in regime_name else 2.5
        
        # Frequency domain
        k = np.fft.fftfreq(grid_size)
        kx, ky = np.meshgrid(k, k)
        k_mag = np.sqrt(kx**2 + ky**2)
        k_mag[0, 0] = 1.0 # avoid div by zero
        
        spectrum = (k_mag ** -alpha)
        noise = rng.standard_normal((grid_size, grid_size)) + 1j * rng.standard_normal((grid_size, grid_size))
        field_ft = noise * spectrum
        
        # Back to spatial
        field = np.abs(np.fft.ifft2(field_ft))
        
        # Scale field intensity by gamma
        field = field * gamma * 10.0 # Arbitrary scaling for visibility
        
        all_fields.append(field.astype(np.float32))
        all_intensities.append(gamma)
        
        if i % 1000 == 0:
            print(f"  Progress: {i}/{n_samples}")

    X = np.array(all_fields).reshape(-1, grid_size, grid_size, 1)
    Y = np.array(all_intensities)
    
    print(f"Saving dataset to {output_path}...")
    np.savez(output_path, X=X, Y=Y)
    print("Done.")

if __name__ == "__main__":
    generate_tglf_like_dataset()
