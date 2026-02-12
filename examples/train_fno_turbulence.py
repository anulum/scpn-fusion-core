# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — FNO Training CLI
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Train and export multi-layer FNO turbulence weights."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scpn_fusion.core.fno_training import DEFAULT_WEIGHTS_PATH, train_fno


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FNO turbulence suppressor (NumPy only).")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--samples", type=int, default=10_000, help="Number of synthetic training samples.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--modes", type=int, default=12, help="Number of Fourier modes.")
    parser.add_argument("--width", type=int, default=32, help="Channel width.")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_WEIGHTS_PATH),
        help="Output .npz weight path.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke mode (caps samples/epochs for fast validation).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 3)
        args.samples = min(args.samples, 24)
        args.modes = min(args.modes, 8)
        args.width = min(args.width, 8)
        args.batch_size = min(args.batch_size, 4)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Training multi-layer FNO turbulence model...")
    print(
        f"epochs={args.epochs}, samples={args.samples}, lr={args.lr}, "
        f"modes={args.modes}, width={args.width}, batch_size={args.batch_size}"
    )

    history = train_fno(
        n_samples=args.samples,
        epochs=args.epochs,
        lr=args.lr,
        modes=args.modes,
        width=args.width,
        save_path=output,
        batch_size=args.batch_size,
        seed=args.seed,
        patience=50 if not args.quick else 5,
    )

    print("Training complete.")
    print(f"Saved weights: {history['saved_path']}")
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Best val loss: {history['best_val_loss']:.6f}")


if __name__ == "__main__":
    main()
