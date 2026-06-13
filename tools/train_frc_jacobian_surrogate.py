# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Local Jacobian Surrogate Training Tool
"""
Extracts a local Jacobian surrogate for the no-rotation FRC profile.

This computes finite-difference gradients around a nominal operating point
using the Steinhauer analytical limit. The theta_dot column is fixed at zero
because rotating BVP support is intentionally fail-closed.

It is a direct matrix-vector mapping from local state to physical space:
Y_pred = Y_nom + J @ (X - X_nom)
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from scpn_fusion.core.frc_rigid_rotor import RigidRotorFRCInputs, solve_frc_equilibrium

logger = logging.getLogger(__name__)

def compute_local_jacobian(
    nominal_features: np.ndarray,
    grid_size: int,
    eps: float = 1e-4
):
    logger.info("Computing FRC Jacobian around nominal point...")
    rho_grid = np.linspace(0.0, 0.5, grid_size)

    # 1. Unpack nominal features
    n0, t_i, t_e, theta_dot, r_s, b_ext, delta = nominal_features

    # De-normalize inputs for the physics oracle
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

    logger.info("Nominal Y shape: %s", Y_nom.shape)
    logger.info("Extracting %d partial derivatives...", num_features)

    feature_names = ["n0", "T_i", "T_e", "theta_dot", "R_s", "B_ext", "delta"]

    for i in range(num_features):
        logger.info("  Computing d/d(%s)...", feature_names[i])
        if feature_names[i] == "theta_dot":
            J[:, i] = 0.0
            continue

        # Perturb
        X_perturbed = nominal_features.copy()
        # Ensure non-zero perturbation
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
        Y_pert = pert_state.B_z

        # Central difference would be better, but forward difference is enough for the local surrogate matrix
        J[:, i] = (Y_pert - Y_nom) / perturbation

    return Y_nom, J

def main():
    parser = argparse.ArgumentParser(description="Extract FRC local Jacobian surrogate")
    parser.add_argument("--grid", type=int, default=256, help="Grid size for FRC")
    parser.add_argument("--out", default="weights/frc_jacobian_surrogate_v1.npz", help="Save path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t_start = time.perf_counter()

    # Define standard high-density, high-field FRC operating point
    X_nom = np.array([
        3.0,   # n0 = 3e20 m^-3
        10.0,  # T_i = 10 keV
        5.0,   # T_e = 5 keV
        0.0,   # theta_dot = 0 krad/s
        0.2,   # R_s = 0.2 m
        5.0,   # B_ext = 5.0 T
        0.02   # delta = 0.02 m
    ])

    Y_nom, Jacobian = compute_local_jacobian(X_nom, args.grid)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        grid_size=np.array([args.grid]),
        x_nom=X_nom,
        y_nom=Y_nom,
        jacobian=Jacobian
    )

    t_total = time.perf_counter() - t_start
    logger.info("Jacobian extraction complete in %.3f s. Saved to %s", t_total, out_path)

if __name__ == "__main__":
    main()
