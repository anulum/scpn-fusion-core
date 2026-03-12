#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Augmented Neural Equilibrium Training
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Train neural equilibrium on ALL reference GEQDSK files (SPARC + DIII-D + JET).

Previous training used only 8 SPARC files × 25 perturbations = 200 samples.
This script uses all 18 GEQDSK files × 25 perturbations = 450 samples,
covering three tokamak families with diverse shapes, currents, and fields.

Usage:
    python tools/train_neural_equilibrium_augmented.py [--n-perturbations 25] [--seed 42]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np


def collect_geqdsk_files(ref_dir: Path) -> list[Path]:
    """Collect GEQDSK/EQDSK files from all machine subdirectories."""
    files: list[Path] = []
    for subdir in sorted(ref_dir.iterdir()):
        if not subdir.is_dir():
            continue
        found = sorted(subdir.glob("*.geqdsk")) + sorted(subdir.glob("*.eqdsk"))
        files.extend(found)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-perturbations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

    ref_dir = REPO_ROOT / "validation" / "reference_data"
    files = collect_geqdsk_files(ref_dir)
    if not files:
        print(f"No GEQDSK/EQDSK files found in {ref_dir}")
        return 1

    print("=" * 60)
    print("Augmented Neural Equilibrium Training")
    print(f"  Files: {len(files)} GEQDSK across {len(set(f.parent.name for f in files))} machines")
    print(f"  Perturbations: {args.n_perturbations} per file")
    print(f"  Expected samples: {len(files) * (1 + args.n_perturbations)}")
    print("=" * 60)

    for f in files:
        print(f"  [{f.parent.name}] {f.name}")

    from scpn_fusion.core.neural_equilibrium import (
        DEFAULT_WEIGHTS_PATH,
        NeuralEquilibriumAccelerator,
    )

    save_path = Path(args.save_path) if args.save_path else DEFAULT_WEIGHTS_PATH

    accel = NeuralEquilibriumAccelerator()
    t0 = time.perf_counter()
    result = accel.train_from_geqdsk(
        files,
        n_perturbations=args.n_perturbations,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - t0

    accel.save_weights(save_path)
    result.weights_path = str(save_path)

    print(f"\nSamples: {result.n_samples}")
    print(f"PCA components: {result.n_components}")
    print(f"Explained variance: {result.explained_variance * 100:.2f}%")
    print(f"Final train loss: {result.final_loss:.6f}")
    print(f"Val loss: {result.val_loss:.6f}")
    print(f"Test MSE: {result.test_mse:.6f}")
    print(f"Test max error: {result.test_max_error:.6f}")
    print(f"Train time: {elapsed:.1f}s")
    print(f"Weights: {result.weights_path}")

    # Quick validation on each machine family
    from scpn_fusion.core.eqdsk import read_geqdsk

    accel_val = NeuralEquilibriumAccelerator()
    accel_val.load_weights(save_path)

    for machine_dir in sorted(ref_dir.iterdir()):
        if not machine_dir.is_dir():
            continue
        gfiles = sorted(machine_dir.glob("*.geqdsk")) + sorted(machine_dir.glob("*.eqdsk"))
        if not gfiles:
            continue
        test_eq = read_geqdsk(gfiles[0])
        kappa = 1.7
        q95 = 3.0
        if hasattr(test_eq, "rbbbs") and test_eq.rbbbs is not None and len(test_eq.rbbbs) > 3:
            r_span = test_eq.rbbbs.max() - test_eq.rbbbs.min()
            kappa = (test_eq.zbbbs.max() - test_eq.zbbbs.min()) / max(r_span, 0.01)
        if hasattr(test_eq, "qpsi") and test_eq.qpsi is not None and len(test_eq.qpsi) > 0:
            idx_95 = int(0.95 * len(test_eq.qpsi))
            q95 = test_eq.qpsi[min(idx_95, len(test_eq.qpsi) - 1)]
        features = np.array(
            [
                test_eq.current / 1e6,
                test_eq.bcentr,
                test_eq.rmaxis,
                test_eq.zmaxis,
                1.0,
                1.0,
                test_eq.simag,
                test_eq.sibry,
                kappa,
                0.3,
                0.3,
                q95,
            ]
        )
        psi_pred = accel_val.predict(features)
        ref_psi = test_eq.psirz[: psi_pred.shape[0], : psi_pred.shape[1]]
        rel_l2 = float(np.linalg.norm(psi_pred - ref_psi) / max(np.linalg.norm(ref_psi), 1e-12))
        print(f"  [{machine_dir.name}] {gfiles[0].name}: rel_L2 = {rel_l2:.4f}")

    bench = accel_val.benchmark(features)
    print(f"\nInference: {bench['mean_ms']:.3f} ms (median {bench['median_ms']:.3f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
