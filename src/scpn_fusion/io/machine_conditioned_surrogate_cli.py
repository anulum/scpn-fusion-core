# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Machine-Conditioned Surrogate CLI
"""Command-line adapter for the machine-conditioned successor trainer."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable

TrainingCallable = Callable[..., dict[str, Any]]


def run_training_cli(
    train: TrainingCallable,
    *,
    default_epochs: int,
    default_pca_components: int,
    default_pca_oversampling: int,
    default_pca_power_iterations: int,
) -> None:
    """Parse the public CLI contract and invoke the supplied trainer."""
    parser = argparse.ArgumentParser(
        description="Train a recovery-safe machine-conditioned equilibrium surrogate."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--pca-components", type=int, default=default_pca_components)
    parser.add_argument("--pca-oversampling", type=int, default=default_pca_oversampling)
    parser.add_argument("--pca-power-iterations", type=int, default=default_pca_power_iterations)
    parser.add_argument("--pca-chunk-rows", type=int, default=256)
    parser.add_argument("--evaluation-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = train(
        dataset_dir=args.dataset_dir,
        output_path=args.out,
        report_path=args.report,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        n_components=args.pca_components,
        pca_oversampling=args.pca_oversampling,
        pca_power_iterations=args.pca_power_iterations,
        pca_chunk_rows=args.pca_chunk_rows,
        evaluation_every=args.evaluation_every,
        checkpoint_every=args.checkpoint_every,
        early_stopping_patience=args.early_stopping_patience,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


__all__ = ["run_training_cli"]
