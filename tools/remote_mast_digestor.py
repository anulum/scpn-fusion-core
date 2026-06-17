# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Remote MAST Data Digestor
"""
MAST shot digestor for local or remote high-memory execution.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from scpn_fusion.io.mast_ingestor import MastIngestor, default_mast_cache_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MastDigestResult = dict[str, float | int]

MAST_DIGEST_CLAIM_BOUNDARY = (
    "MAST digest reports are local ingestion diagnostics only; same-case "
    "magnetic-geometry validation is required before physics-validation claims."
)


def process_shot_comparison(shot_id: int, cache_dir: Path) -> MastDigestResult | None:
    """
    Compare the rigid-rotor equilibrium solver against MAST summary signals.
    """
    try:
        ingestor = MastIngestor(cache_dir=cache_dir)

        summary = ingestor.load_shot_summary(shot_id)

        ip = summary["ip"]
        max_ip = np.max(ip)
        flattop_indices = np.where(ip > 0.8 * max_ip)[0]

        if len(flattop_indices) == 0:
            logger.warning(f"Shot {shot_id}: No stable flattop found.")
            return None

        logger.info(f"Processing Shot {shot_id}: Validating flattop stability...")
        from scpn_fusion.core.frc_rigid_rotor import RigidRotorFRCInputs, solve_frc_equilibrium

        errors = []
        sample_indices = np.linspace(flattop_indices[0], flattop_indices[-1], 10, dtype=int)

        for idx in sample_indices:
            n0_safe = max(float(summary["density"][idx]), 1e18)
            R_s = 0.6
            rho_grid = np.linspace(0.0, 1.2 * R_s, 128)

            inputs = RigidRotorFRCInputs(
                n0=n0_safe,
                T_i_eV=1500.0,
                T_e_eV=800.0,
                theta_dot=0.0,
                R_s=R_s,
                B_ext=summary["ip"][idx] * 2e-7 / R_s,
                delta=0.03,
            )

            state = solve_frc_equilibrium(inputs, rho_grid, solver="rust")

            if state.converged:
                errors.append(state.residual)

        return {
            "shot_id": shot_id,
            "mean_residual": float(np.mean(errors)) if errors else 1.0,
            "samples": len(errors),
            "max_ip_ma": float(max_ip / 1e6),
        }

    except Exception as e:
        logger.error(f"Error processing Shot {shot_id}: {e}")
        return None


def build_digest_report(
    *,
    target_shots: list[int],
    results: list[MastDigestResult | None],
    elapsed_s: float,
) -> dict[str, object]:
    """Summarise remote MAST digest completeness without promoting validation claims."""
    valid_results = [result for result in results if result is not None]
    processed_shots = [int(result["shot_id"]) for result in valid_results]
    processed_set = set(processed_shots)
    missing_shots = [int(shot_id) for shot_id in target_shots if int(shot_id) not in processed_set]

    if missing_shots:
        status = "blocked_incomplete_mast_digest"
    elif any(int(result["samples"]) <= 0 for result in valid_results):
        status = "blocked_no_solver_samples"
    else:
        status = "local_mast_digest_complete_not_physics_validation"

    return {
        "status": status,
        "accepted_full_fidelity_ready": False,
        "claim_boundary": MAST_DIGEST_CLAIM_BOUNDARY,
        "target_shots": [int(shot_id) for shot_id in target_shots],
        "processed_shots": processed_shots,
        "missing_shots": missing_shots,
        "elapsed_s": float(elapsed_s),
        "shot_count": len(target_shots),
        "valid_result_count": len(valid_results),
        "results": valid_results,
    }


def main() -> None:
    """Run bounded remote MAST digestion and write the digest report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--shots", nargs="+", type=int)
    parser.add_argument("--cache-dir", type=Path, default=default_mast_cache_dir())
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    try:
        os.nice(19)
    except OSError:
        pass

    if not args.shots:
        target_shots = [30419, 30420, 30421, 30422, 30423, 30424]
    else:
        target_shots = args.shots

    cache_dir = args.cache_dir.expanduser().resolve()
    out_path = args.out or (cache_dir.parent / "digestion_report_v3_aligned.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting MAST digestion of {len(target_shots)} shots...")
    t0 = time.perf_counter()

    with mp.Pool(args.workers) as pool:
        results = pool.starmap(
            process_shot_comparison,
            [(shot_id, cache_dir) for shot_id in target_shots],
        )

    t_total = time.perf_counter() - t0
    report = build_digest_report(target_shots=target_shots, results=results, elapsed_s=t_total)
    valid_results = report["results"]

    logger.info(
        "Digestion complete. Status=%s. Processed %d/%d shots in %.1fs.",
        report["status"],
        report["valid_result_count"],
        report["shot_count"],
        t_total,
    )

    np.savez(
        out_path,
        results=np.array(valid_results, dtype=object),
        report_json=np.array(json.dumps(report, sort_keys=True)),
    )
    logger.info(f"MAST summary report saved to {out_path}")


if __name__ == "__main__":
    main()
