# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — FNO Spatial Training Data Generator
# © 1998–2026 Miroslav Sotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
"""
Generate (equilibrium, transport_field) pairs for FNO training.

Uses the trained QLKNN MLP as an oracle: for each equilibrium on a
2D (R,Z) grid, compute local plasma parameters at each grid point,
query the MLP for transport coefficients, and produce paired
(psi_2d, chi_i_2d) spatial data.

This replaces synthetic spectral noise with physics-grounded
training data for the FNO turbulence/transport surrogate.

Usage
-----
    python tools/generate_fno_qlknn_spatial.py
    python tools/generate_fno_qlknn_spatial.py --weights weights/neural_transport_qlknn.npz
    python tools/generate_fno_qlknn_spatial.py --n-equilibria 500 --grid-size 64
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "neural_transport_qlknn.npz"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "fno_qlknn_spatial"


def _make_tokamak_equilibrium(
    nr: int, nz: int,
    R0: float, a: float, kappa: float, delta: float,
    B0: float, Ip: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic tokamak equilibrium psi(R,Z) and profiles.

    Returns (psi_2d, rho_2d, R_grid, Z_grid) all shape (nr, nz).
    """
    R_min, R_max = R0 - 1.5 * a, R0 + 1.5 * a
    Z_min, Z_max = -1.5 * a * kappa, 1.5 * a * kappa

    R = np.linspace(R_min, R_max, nr)
    Z = np.linspace(Z_min, Z_max, nz)
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")

    # Simple analytic equilibrium (Soloviev-like)
    r_local = np.sqrt((RR - R0) ** 2 + (ZZ / kappa) ** 2)
    rho = r_local / a  # normalised radius
    rho = np.clip(rho, 0, 1.5)

    # Poloidal flux (parabolic profile)
    psi = 1.0 - (1.0 - rho ** 2) ** 2
    psi = np.clip(psi, 0, 1.2)

    return psi, rho, RR, ZZ


def _profiles_from_rho(
    rho: np.ndarray,
    Te0: float, Ti0: float, ne0: float,
    q0: float, q_edge: float,
) -> dict[str, np.ndarray]:
    """Generate radial profiles from normalised radius.

    Returns dict with Te, Ti, ne, q, s_hat, beta_e, grad_Te, grad_Ti, grad_ne.
    """
    # Pedestal-like profiles
    alpha_T = 2.0  # temperature profile peakedness
    alpha_n = 1.5  # density profile peakedness

    Te = Te0 * (1 - rho ** alpha_T) ** 1.5
    Ti = Ti0 * (1 - rho ** alpha_T) ** 1.5
    ne = ne0 * (1 - rho ** alpha_n) ** 1.0

    Te = np.maximum(Te, 0.1)  # floor at 100 eV
    Ti = np.maximum(Ti, 0.1)
    ne = np.maximum(ne, 0.1)

    # Safety factor profile
    q = q0 + (q_edge - q0) * rho ** 2

    # Magnetic shear
    s_hat = 2 * (q_edge - q0) * rho ** 2 / np.maximum(q, 0.5)

    # Beta_e
    beta_e = 4.03e-3 * ne * Te

    # Normalised gradients: R/L_X = -R * (1/X) * dX/drho * drho/dr
    # For analytic profiles, compute analytically
    R_major = 6.2
    eps = rho / (R_major / 2.0)

    # dTe/drho
    dTe = -Te0 * 1.5 * alpha_T * rho ** (alpha_T - 1) * (1 - rho ** alpha_T) ** 0.5
    grad_Te = -R_major * dTe / np.maximum(Te, 0.01)
    grad_Te = np.clip(grad_Te, 0, 50)

    dTi = -Ti0 * 1.5 * alpha_T * rho ** (alpha_T - 1) * (1 - rho ** alpha_T) ** 0.5
    grad_Ti = -R_major * dTi / np.maximum(Ti, 0.01)
    grad_Ti = np.clip(grad_Ti, 0, 50)

    dne = -ne0 * 1.0 * alpha_n * rho ** (alpha_n - 1) * (1 - rho ** alpha_n) ** 0.0
    grad_ne = -R_major * dne / np.maximum(ne, 0.01)
    grad_ne = np.clip(grad_ne, 0, 30)

    return {
        "Te": Te, "Ti": Ti, "ne": ne,
        "q": q, "s_hat": s_hat, "beta_e": beta_e,
        "grad_Te": grad_Te, "grad_Ti": grad_Ti, "grad_ne": grad_ne,
    }


def generate(
    weights_path: Path,
    output_dir: Path,
    n_equilibria: int = 200,
    grid_size: int = 64,
    seed: int = 42,
) -> None:
    """Generate FNO training data using QLKNN MLP as oracle."""
    rng = np.random.default_rng(seed)

    # Load the trained QLKNN MLP
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from scpn_fusion.core.neural_transport import NeuralTransportModel

    model = NeuralTransportModel(weights_path)
    if not model.is_neural:
        print(f"WARNING: Could not load neural weights from {weights_path}.")
        print("  Will use critical-gradient fallback (less accurate).")

    print(f"Generating {n_equilibria} equilibria on {grid_size}x{grid_size} grids...")

    all_psi = []
    all_chi = []

    # Parameter ranges for equilibrium variation
    params_ranges = {
        "R0": (5.0, 7.0),       # Major radius [m]
        "a": (1.5, 2.5),        # Minor radius [m]
        "kappa": (1.5, 2.0),    # Elongation
        "delta": (0.2, 0.5),    # Triangularity
        "B0": (4.0, 6.0),       # Toroidal field [T]
        "Ip": (10.0, 20.0),     # Plasma current [MA]
        "Te0": (5.0, 25.0),     # Central Te [keV]
        "Ti0": (5.0, 25.0),     # Central Ti [keV]
        "ne0": (3.0, 15.0),     # Central ne [10^19 m^-3]
        "q0": (0.8, 1.2),       # On-axis q
        "q_edge": (3.0, 6.0),   # Edge q
    }

    t0 = time.monotonic()
    for i in range(n_equilibria):
        # Random equilibrium parameters
        p = {k: rng.uniform(*v) for k, v in params_ranges.items()}

        psi, rho, RR, ZZ = _make_tokamak_equilibrium(
            grid_size, grid_size,
            p["R0"], p["a"], p["kappa"], p["delta"],
            p["B0"], p["Ip"], rng,
        )

        profiles = _profiles_from_rho(
            rho, p["Te0"], p["Ti0"], p["ne0"], p["q0"], p["q_edge"],
        )

        # Build input batch for MLP: (grid_size*grid_size, 10)
        flat_rho = rho.ravel()
        x_batch = np.column_stack([
            flat_rho,
            profiles["Te"].ravel(),
            profiles["Ti"].ravel(),
            profiles["ne"].ravel(),
            profiles["grad_Te"].ravel(),
            profiles["grad_Ti"].ravel(),
            profiles["grad_ne"].ravel(),
            profiles["q"].ravel(),
            profiles["s_hat"].ravel(),
            profiles["beta_e"].ravel(),
        ])

        # Query MLP for chi_i at each grid point
        if model.is_neural and model._weights is not None:
            from scpn_fusion.core.neural_transport import _mlp_forward
            out = _mlp_forward(x_batch, model._weights)  # (N, 3)
            chi_i_2d = out[:, 1].reshape(grid_size, grid_size)
        else:
            # Fallback: use critical gradient model
            from scpn_fusion.core.neural_transport import critical_gradient_model, TransportInputs
            chi_i_flat = np.zeros(len(x_batch))
            for j in range(len(x_batch)):
                inp = TransportInputs(
                    rho=x_batch[j, 0], te_kev=x_batch[j, 1],
                    ti_kev=x_batch[j, 2], ne_19=x_batch[j, 3],
                    grad_te=x_batch[j, 4], grad_ti=x_batch[j, 5],
                    grad_ne=x_batch[j, 6], q=x_batch[j, 7],
                    s_hat=x_batch[j, 8], beta_e=x_batch[j, 9],
                )
                chi_i_flat[j] = critical_gradient_model(inp).chi_i
            chi_i_2d = chi_i_flat.reshape(grid_size, grid_size)

        # Normalise to [0, 1] range for FNO
        psi_norm = psi / max(psi.max(), 1e-8)
        chi_norm = chi_i_2d / max(chi_i_2d.max(), 1e-8)

        all_psi.append(psi_norm.astype(np.float32))
        all_chi.append(chi_norm.astype(np.float32))

        if (i + 1) % 50 == 0:
            elapsed = time.monotonic() - t0
            print(f"  {i+1}/{n_equilibria} ({elapsed:.0f}s)")

    X = np.array(all_psi)  # (N, 64, 64)
    Y = np.array(all_chi)  # (N, 64, 64)

    # Split 85/15
    n_train = int(len(X) * 0.85)
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:], Y[n_train:]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "train.npz", X=X_train, Y=Y_train)
    np.savez(output_dir / "val.npz", X=X_val, Y=Y_val)

    metadata = {
        "source": "QLKNN-MLP oracle on synthetic equilibria",
        "weights_used": str(weights_path),
        "neural_oracle": model.is_neural,
        "n_equilibria": n_equilibria,
        "grid_size": grid_size,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "param_ranges": {k: list(v) for k, v in params_ranges.items()},
        "seed": seed,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"\nSaved to {output_dir}/")
    print(f"  train.npz: X{X_train.shape}, Y{Y_train.shape}")
    print(f"  val.npz:   X{X_val.shape}, Y{Y_val.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-equilibria", type=int, default=200)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(args.weights, args.output_dir, args.n_equilibria, args.grid_size, args.seed)


if __name__ == "__main__":
    main()
