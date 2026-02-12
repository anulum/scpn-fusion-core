# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Neural Transport Validation CLI
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Validate neural transport weights against analytic fallback outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scpn_fusion.core.neural_transport import (
    NeuralTransportModel,
    critical_gradient_model,
    make_training_dataset,
)

DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "neural_transport_weights.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate neural transport surrogate.")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-rel-l2", type=float, default=0.20)
    return parser.parse_args()


def rel_l2(pred: np.ndarray, target: np.ndarray) -> float:
    num = np.linalg.norm(pred - target)
    den = np.linalg.norm(target) + 1e-12
    return float(num / den)


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights file not found: {weights}")

    model = NeuralTransportModel.from_npz(weights)
    x, y_target = make_training_dataset(n_samples=args.samples, seed=args.seed)
    y_pred = model.predict_profile(x)
    y_ref = critical_gradient_model(x)

    mae = float(np.mean(np.abs(y_pred - y_ref)))
    rmse = float(np.sqrt(np.mean((y_pred - y_ref) ** 2)))
    r2 = 1.0 - float(np.sum((y_pred - y_ref) ** 2) / (np.sum((y_ref - y_ref.mean(axis=0)) ** 2) + 1e-12))
    rel = rel_l2(y_pred, y_ref)

    print(f"Validated weights: {weights}")
    print(f"Samples: {args.samples}")
    print(f"MAE: {mae:.6e}")
    print(f"RMSE: {rmse:.6e}")
    print(f"Relative L2: {rel:.6e}")
    print(f"R^2 (global): {r2:.6f}")

    if rel > args.max_rel_l2:
        raise SystemExit(
            f"Validation failed: relative L2 {rel:.6f} exceeds threshold {args.max_rel_l2:.6f}"
        )
    print("Validation passed.")


if __name__ == "__main__":
    main()
