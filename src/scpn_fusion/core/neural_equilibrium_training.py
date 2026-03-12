# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Neural Equilibrium Training Runtime
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Training entrypoints extracted from neural_equilibrium.py."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np


def train_on_sparc(
    sparc_dir: str | Path | None = None,
    save_path: str | Path | None = None,
    n_perturbations: int = 25,
    seed: int = 42,
):
    """
    Train neural equilibrium on SPARC GEQDSK files and save weights.

    Returns the TrainingResult object produced by NeuralEquilibriumAccelerator.
    """
    from .neural_equilibrium import (
        DEFAULT_WEIGHTS_PATH,
        REPO_ROOT,
        NeuralEquilibriumAccelerator,
    )

    if save_path is None:
        save_path = DEFAULT_WEIGHTS_PATH

    if sparc_dir is None:
        sparc_dir = REPO_ROOT / "validation" / "reference_data" / "sparc"
    sparc_dir = Path(sparc_dir)

    files = sorted(sparc_dir.glob("*.geqdsk")) + sorted(sparc_dir.glob("*.eqdsk"))
    if not files:
        raise FileNotFoundError(f"No GEQDSK/EQDSK files in {sparc_dir}")

    accel = NeuralEquilibriumAccelerator()
    result = accel.train_from_geqdsk(
        files,
        n_perturbations=n_perturbations,
        seed=seed,
    )
    accel.save_weights(save_path)
    result.weights_path = str(save_path)
    return result


def run_training_cli() -> int:
    """CLI entrypoint retained for `python neural_equilibrium.py` compatibility."""
    from .neural_equilibrium import NeuralEquilibriumAccelerator, REPO_ROOT

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

    sparc_dir = REPO_ROOT / "validation" / "reference_data" / "sparc"
    if not sparc_dir.exists():
        print(f"SPARC data not found at {sparc_dir}")
        return 1

    print("=" * 60)
    print("Training Neural Equilibrium on SPARC GEQDSKs")
    print("=" * 60)

    result = train_on_sparc(sparc_dir)
    print(f"\nSamples: {result.n_samples}")
    print(f"PCA components: {result.n_components}")
    print(f"Explained variance: {result.explained_variance * 100:.2f}%")
    print(f"Final train loss: {result.final_loss:.6f}")
    print(f"Val loss: {result.val_loss:.6f}")
    print(f"Test MSE: {result.test_mse:.6f}")
    print(f"Test max error: {result.test_max_error:.6f}")
    print(f"Train time: {result.train_time_s:.1f}s")
    print(f"Weights: {result.weights_path}")

    # Quick validation
    accel = NeuralEquilibriumAccelerator()
    accel.load_weights(result.weights_path)

    from scpn_fusion.core.eqdsk import read_geqdsk

    test_eq = read_geqdsk(next(sparc_dir.glob("*.geqdsk")))
    kappa_cli = 1.7
    q95_cli = 3.0
    if hasattr(test_eq, "rbbbs") and test_eq.rbbbs is not None and len(test_eq.rbbbs) > 3:
        r_span = test_eq.rbbbs.max() - test_eq.rbbbs.min()
        kappa_cli = (test_eq.zbbbs.max() - test_eq.zbbbs.min()) / max(r_span, 0.01)
    if hasattr(test_eq, "qpsi") and test_eq.qpsi is not None and len(test_eq.qpsi) > 0:
        idx_95 = int(0.95 * len(test_eq.qpsi))
        q95_cli = test_eq.qpsi[min(idx_95, len(test_eq.qpsi) - 1)]
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
            kappa_cli,
            0.3,
            0.3,
            q95_cli,
        ]
    )

    psi_pred = accel.predict(features)
    ref_psi = test_eq.psirz[: psi_pred.shape[0], : psi_pred.shape[1]]
    diff = psi_pred - ref_psi
    rel_l2 = float(np.linalg.norm(diff) / np.linalg.norm(ref_psi))
    print(f"\nValidation relative L2 on first file: {rel_l2:.6f}")

    bench = accel.benchmark(features)
    print(f"Inference: {bench['mean_ms']:.3f} ms (median: {bench['median_ms']:.3f} ms)")
    return 0
