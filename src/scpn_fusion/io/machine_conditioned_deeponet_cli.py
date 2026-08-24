# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Machine-Conditioned DeepONet CLI
"""Argument adapter for the manifest-bound equilibrium DeepONet trainer."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Callable

TrainingCallable = Callable[..., dict[str, Any]]
LOGGER = logging.getLogger(__name__)


def run_deeponet_cli(
    train: TrainingCallable,
    *,
    default_basis_width: int,
) -> None:
    """Parse the command-line contract and invoke the DeepONet trainer.

    Parameters
    ----------
    train : TrainingCallable
        Production training entry point accepting the parsed keyword contract.
    default_basis_width : int
        Positive default branch/trunk output width.

    Raises
    ------
    SystemExit
        If command-line arguments are missing or invalid.
    OSError
        If dataset, recovery, artifact, or report storage fails.
    ValueError
        If data, configuration, or recovery authentication fails.
    """
    parser = argparse.ArgumentParser(
        description="Train a recovery-safe fixed-machine equilibrium DeepONet."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--calibration-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--basis-width", type=int, default=default_basis_width)
    parser.add_argument("--shot-batch-size", type=int, default=256)
    parser.add_argument("--coordinate-batch-size", type=int, default=512)
    parser.add_argument("--validation-probe-shots", type=int, default=1024)
    parser.add_argument("--validation-probe-coordinates", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--statistics-chunk-rows", type=int, default=256)
    parser.add_argument("--evaluation-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=40)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = train(
        dataset_dir=args.dataset_dir,
        output_path=args.out,
        report_path=args.report,
        checkpoint_dir=args.checkpoint_dir,
        steps=args.steps,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        calibration_fraction=args.calibration_fraction,
        test_fraction=args.test_fraction,
        basis_width=args.basis_width,
        shot_batch_size=args.shot_batch_size,
        coordinate_batch_size=args.coordinate_batch_size,
        validation_probe_shots=args.validation_probe_shots,
        validation_probe_coordinates=args.validation_probe_coordinates,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        statistics_chunk_rows=args.statistics_chunk_rows,
        evaluation_every=args.evaluation_every,
        checkpoint_every=args.checkpoint_every,
        early_stopping_patience=args.early_stopping_patience,
        resume=args.resume,
    )
    LOGGER.info("%s", report["status"])


__all__ = ["run_deeponet_cli"]
