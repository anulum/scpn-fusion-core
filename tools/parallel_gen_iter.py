# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Parallel ITER Data Generation Tool
"""
Parallelized data generation for 2D ITER surrogates.
Uses multiprocessing to saturate all available CPU threads.
"""

import argparse
import logging
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np

from scpn_fusion.core.fusion_kernel import FusionKernel

logger = logging.getLogger(__name__)

def generate_chunk(n_samples: int, config_path: str, seed: int):
    """Worker function for parallel generation."""
    fk = FusionKernel(config_path)
    # Ensure ITER nominals
    fk.cfg["physics"]["B_T"] = 5.3
    fk.cfg["target"] = fk.cfg.get("target", {})
    fk.cfg["target"]["kappa"] = 1.7
    fk.cfg["target"]["R_axis"] = 6.2
    fk.cfg["target"]["Z_axis"] = 0.0

    X, Y = [], []
    base_currents = [float(c["current"]) for c in fk.cfg["coils"]]
    base_ip = float(fk.cfg["physics"]["plasma_current_target"])
    rng = np.random.default_rng(seed)

    for i in range(n_samples):
        # Perturb
        for idx, coil in enumerate(fk.cfg["coils"]):
            coil["current"] = base_currents[idx] * rng.uniform(0.85, 1.15)
        ip = base_ip * rng.uniform(0.8, 1.2)
        fk.cfg["physics"]["plasma_current_target"] = ip

        try:
            fk.solve_equilibrium()
            iz, ir, psi_ax = fk._find_magnetic_axis()
            (rx, zx), psi_x = fk.find_x_point(fk.Psi)
            features = [
                ip / 1e6, 5.3, fk.R[ir], fk.Z[iz],
                1.0, 1.0, psi_ax, psi_x,
                1.7, 0.33, 0.33, 3.0
            ]
            X.append(features)
            Y.append(fk.Psi.ravel())
        except Exception:
            continue
    return np.array(X), np.array(Y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--out", default="data/iter_2d_high_fidelity.npz")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    samples_per_worker = args.samples // args.workers
    remainder = args.samples % args.workers

    tasks = []
    for i in range(args.workers):
        n = samples_per_worker + (1 if i < remainder else 0)
        tasks.append((n, args.config, 42 + i))

    logger.info(f"Starting parallel generation of {args.samples} samples on {args.workers} workers...")
    t0 = time.perf_counter()

    with mp.Pool(args.workers) as pool:
        results = pool.starmap(generate_chunk, tasks)

    X = np.concatenate([r[0] for r in results if len(r[0]) > 0])
    Y = np.concatenate([r[1] for r in results if len(r[1]) > 0])

    t_total = time.perf_counter() - t0
    logger.info(f"Generated {len(X)} valid samples in {t_total:.1f}s ({t_total/len(X):.2f}s/sample avg across all workers)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, X=X, Y=Y)
    logger.info(f"Saved dataset to {args.out}")

if __name__ == "__main__":
    main()
