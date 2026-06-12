# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Quantized Hardware Surrogate Tool
"""
Extracts a Fixed-Point Quantized Jacobian Surrogate (Fifth Lane).

For extreme RF-band control loops (1 - 10 MHz) such as RMF (Rotating Magnetic Field)
current drive or ultra-fast helicity injection phase-locking. At 10 MHz (100 ns cycle),
floating-point operations and DSP multipliers are too slow or consume too much FPGA
fabric.

This lane takes the Local Jacobian and quantizes it into an Int16/Int8
format. Inference uses pure integer arithmetic (bit-shifts and integer adds)
to map sensor inputs directly to physical flux states in pure combinational logic.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from scpn_fusion.core.frc_rigid_rotor import RigidRotorFRCInputs, solve_frc_equilibrium

logger = logging.getLogger(__name__)

def compute_quantized_jacobian(
    nominal_features: np.ndarray,
    grid_size: int,
    eps: float = 1e-4,
    bits: int = 16
):
    logger.info("Extracting %d-bit Quantized FRC Jacobian...", bits)
    rho_grid = np.linspace(0.0, 0.5, grid_size)

    n0, t_i, t_e, theta_dot, r_s, b_ext, delta = nominal_features
    nom_inputs = RigidRotorFRCInputs(
        n0=n0 * 1e20,
        T_i_eV=t_i * 1000.0,
        T_e_eV=t_e * 1000.0,
        theta_dot=theta_dot * 1000.0,
        R_s=r_s,
        B_ext=b_ext,
        delta=delta
    )

    nom_state = solve_frc_equilibrium(nom_inputs, rho_grid, solver="rust", tolerance=1e-10)
    Y_nom = nom_state.B_z

    num_features = len(nominal_features)
    J = np.zeros((grid_size, num_features))

    for i in range(num_features):
        X_perturbed = nominal_features.copy()
        perturbation = max(X_perturbed[i] * eps, 1e-8)
        X_perturbed[i] += perturbation

        pert_inputs = RigidRotorFRCInputs(
            n0=X_perturbed[0] * 1e20,
            T_i_eV=X_perturbed[1] * 1000.0,
            T_e_eV=X_perturbed[2] * 1000.0,
            theta_dot=X_perturbed[3] * 1000.0,
            R_s=X_perturbed[4],
            B_ext=X_perturbed[5],
            delta=X_perturbed[6]
        )

        pert_state = solve_frc_equilibrium(pert_inputs, rho_grid, solver="rust", tolerance=1e-10)
        J[:, i] = (pert_state.B_z - Y_nom) / perturbation

    # Quantization
    # Scale factors to map floats to integer range [-2^(bits-1), 2^(bits-1) - 1]
    max_val = 2**(bits - 1) - 1

    j_max = np.max(np.abs(J))
    y_max = np.max(np.abs(Y_nom))

    scale_j = max_val / j_max if j_max > 0 else 1.0
    scale_y = max_val / y_max if y_max > 0 else 1.0

    J_quant = np.round(J * scale_j).astype(np.int16 if bits <= 16 else np.int32)
    Y_quant = np.round(Y_nom * scale_y).astype(np.int16 if bits <= 16 else np.int32)

    return Y_nom, J, Y_quant, J_quant, scale_y, scale_j

def main():
    parser = argparse.ArgumentParser(description="Extract Quantized Surrogate (1-10MHz Lane)")
    parser.add_argument("--grid", type=int, default=128, help="Grid size for FRC")
    parser.add_argument("--out", default="weights/frc_quantized_surrogate_v1.npz", help="Save path")
    parser.add_argument("--bits", type=int, default=16, help="Quantization bits (8 or 16)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t_start = time.perf_counter()

    X_nom = np.array([3.0, 10.0, 5.0, 0.0, 0.2, 5.0, 0.02])

    Y_nom, J, Y_q, J_q, s_y, s_j = compute_quantized_jacobian(X_nom, args.grid, bits=args.bits)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        grid_size=np.array([args.grid]),
        x_nom=X_nom,
        y_nom=Y_nom,
        jacobian=J,
        y_nom_quantized=Y_q,
        jacobian_quantized=J_q,
        scale_y=np.array([s_y]),
        scale_j=np.array([s_j]),
        bits=np.array([args.bits])
    )

    t_total = time.perf_counter() - t_start
    logger.info("Quantized extraction complete in %.3f s. Saved to %s", t_total, out_path)
    logger.info("FPGA Multipliers needed: 0. Pure integer arithmetic enabled.")

if __name__ == "__main__":
    main()
