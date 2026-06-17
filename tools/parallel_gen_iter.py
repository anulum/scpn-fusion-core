# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Parallel ITER Data Generation Tool
"""
Parallelized data generation for 2D ITER surrogates.

Large runs may use the full host, but that must be explicit and justified.
Boundary X-points are rejected by default because they indicate a clipped or
failed equilibrium rather than a clean training sample.
"""

import argparse
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np

from scpn_fusion.core.fusion_kernel import FusionKernel

logger = logging.getLogger(__name__)

DEFAULT_SHARED_WORKERS = 12


def default_worker_count(cpu_count: int | None = None) -> int:
    """Return the shared-host default worker count."""
    count = mp.cpu_count() if cpu_count is None else cpu_count
    return max(1, min(count, DEFAULT_SHARED_WORKERS))


def validate_worker_policy(workers: int, *, allow_full_host: bool, run_justification: str) -> None:
    """Require explicit justification before exceeding the shared-host default."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers > DEFAULT_SHARED_WORKERS and (not allow_full_host or not run_justification.strip()):
        raise ValueError(
            "workers above 12 require --allow-full-host and a non-empty --run-justification"
        )


def is_boundary_xpoint(
    r_x: float,
    z_x: float,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
    *,
    margin_fraction: float = 0.01,
) -> bool:
    """Return True when an X-point sits on the computational boundary."""
    r_margin = max((r_max - r_min) * margin_fraction, 1.0e-12)
    z_margin = max((z_max - z_min) * margin_fraction, 1.0e-12)
    return (
        r_x <= r_min + r_margin
        or r_x >= r_max - r_margin
        or z_x <= z_min + z_margin
        or z_x >= z_max - z_margin
    )


def generate_chunk(n_samples: int, config_path: str, seed: int, allow_boundary_xpoints: bool):
    """Worker function for parallel generation."""
    fk = FusionKernel(config_path)
    # Ensure ITER nominals
    fk.cfg["physics"]["B_T"] = 5.3
    fk.cfg["target"] = fk.cfg.get("target", {})
    fk.cfg["target"]["kappa"] = 1.7
    fk.cfg["target"]["R_axis"] = 6.2
    fk.cfg["target"]["Z_axis"] = 0.0

    X, Y = [], []
    rejected_boundary_xpoints = 0
    failed_solves = 0
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
            if not allow_boundary_xpoints and is_boundary_xpoint(
                float(rx),
                float(zx),
                float(np.min(fk.R)),
                float(np.max(fk.R)),
                float(np.min(fk.Z)),
                float(np.max(fk.Z)),
            ):
                rejected_boundary_xpoints += 1
                continue
            features = [
                ip / 1e6,
                5.3,
                fk.R[ir],
                fk.Z[iz],
                1.0,
                1.0,
                psi_ax,
                psi_x,
                1.7,
                0.33,
                0.33,
                3.0,
            ]
            X.append(features)
            Y.append(fk.Psi.ravel())
        except Exception:
            failed_solves += 1
            continue
    return np.array(X), np.array(Y), rejected_boundary_xpoints, failed_solves


def main():
    """Generate ITER surrogate data chunks with bounded worker policy."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=default_worker_count())
    parser.add_argument("--out", default="data/iter_2d_high_fidelity.npz")
    parser.add_argument("--report", help="JSON generation report path; defaults beside --out")
    parser.add_argument("--allow-boundary-xpoints", action="store_true")
    parser.add_argument("--allow-full-host", action="store_true")
    parser.add_argument("--run-justification", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        validate_worker_policy(
            args.workers,
            allow_full_host=args.allow_full_host,
            run_justification=args.run_justification,
        )
    except ValueError as exc:
        parser.error(str(exc))

    samples_per_worker = args.samples // args.workers
    remainder = args.samples % args.workers

    tasks = []
    for i in range(args.workers):
        n = samples_per_worker + (1 if i < remainder else 0)
        tasks.append((n, args.config, 42 + i, args.allow_boundary_xpoints))

    logger.info(
        f"Starting parallel generation of {args.samples} samples on {args.workers} workers..."
    )
    t0 = time.perf_counter()

    with mp.Pool(args.workers) as pool:
        results = pool.starmap(generate_chunk, tasks)

    valid_chunks = [r for r in results if len(r[0]) > 0]
    if valid_chunks:
        X = np.concatenate([r[0] for r in valid_chunks])
        Y = np.concatenate([r[1] for r in valid_chunks])
    else:
        X = np.empty((0, 12), dtype=np.float64)
        Y = np.empty((0, 0), dtype=np.float64)

    t_total = time.perf_counter() - t0
    rejected_boundary = sum(int(r[2]) for r in results)
    failed_solves = sum(int(r[3]) for r in results)
    avg_s = t_total / len(X) if len(X) else float("inf")
    logger.info(
        f"Generated {len(X)} valid samples in {t_total:.1f}s ({avg_s:.2f}s/sample avg across all workers)"
    )

    report_path = Path(args.report) if args.report else Path(args.out).with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "requested_samples": args.samples,
        "workers": args.workers,
        "allow_full_host": args.allow_full_host,
        "run_justification": args.run_justification,
        "allow_boundary_xpoints": args.allow_boundary_xpoints,
        "valid_samples": int(len(X)),
        "rejected_boundary_xpoints": rejected_boundary,
        "failed_solves": failed_solves,
        "elapsed_s": t_total,
        "status": "passed" if len(X) > 0 else "failed_no_valid_samples",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if len(X) == 0:
        raise RuntimeError(f"no valid samples generated; report written to {report_path}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, X=X, Y=Y)
    logger.info(f"Saved dataset to {args.out}")


if __name__ == "__main__":
    main()
