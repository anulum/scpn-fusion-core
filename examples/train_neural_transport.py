# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Neural Transport Training CLI
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Train neural transport weights from analytic critical-gradient data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scpn_fusion.core.neural_transport import (
    HIDDEN1,
    HIDDEN2,
    INPUT_DIM,
    OUTPUT_DIM,
    make_training_dataset,
    relu,
    sigmoid,
    softplus,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "weights" / "neural_transport_weights.npz"


class Adam:
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], lr: float) -> None:
        self.t += 1
        for key, p in params.items():
            g = grads[key]
            if key not in self.m:
                self.m[key] = np.zeros_like(p)
                self.v[key] = np.zeros_like(p)
            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[key] / (1.0 - self.beta1**self.t)
            v_hat = self.v[key] / (1.0 - self.beta2**self.t)
            p -= lr * m_hat / (np.sqrt(v_hat) + self.eps)


def init_params(seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.normal(0.0, 0.08, size=(INPUT_DIM, HIDDEN1)),
        "b1": np.zeros((HIDDEN1,), dtype=np.float64),
        "w2": rng.normal(0.0, 0.08, size=(HIDDEN1, HIDDEN2)),
        "b2": np.zeros((HIDDEN2,), dtype=np.float64),
        "w3": rng.normal(0.0, 0.08, size=(HIDDEN2, OUTPUT_DIM)),
        "b3": np.zeros((OUTPUT_DIM,), dtype=np.float64),
    }


def forward(params: Dict[str, np.ndarray], x_norm: np.ndarray, output_scale: np.ndarray) -> Dict[str, np.ndarray]:
    z1 = x_norm @ params["w1"] + params["b1"][None, :]
    h1 = relu(z1)
    z2 = h1 @ params["w2"] + params["b2"][None, :]
    h2 = relu(z2)
    z3 = h2 @ params["w3"] + params["b3"][None, :]
    y = softplus(z3) * output_scale[None, :]
    return {"z1": z1, "h1": h1, "z2": z2, "h2": h2, "z3": z3, "y": y}


def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    err = pred - target
    return float(np.mean(err * err))


def train(
    n_samples: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> Dict[str, object]:
    x, y = make_training_dataset(n_samples=n_samples, seed=seed)
    split = max(1, int(0.9 * n_samples))
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    input_mean = x_train.mean(axis=0)
    input_std = np.maximum(x_train.std(axis=0), 1e-8)
    output_scale = np.maximum(y_train.mean(axis=0), 1e-4)

    x_train_n = (x_train - input_mean[None, :]) / input_std[None, :]
    x_val_n = (x_val - input_mean[None, :]) / input_std[None, :]

    params = init_params(seed=seed)
    opt = Adam()
    rng = np.random.default_rng(seed + 17)
    best = {k: v.copy() for k, v in params.items()}
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for _epoch in range(epochs):
        order = rng.permutation(len(x_train_n))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = x_train_n[idx]
            yb = y_train[idx]

            cache = forward(params, xb, output_scale)
            pred = cache["y"]
            bsz = max(1, xb.shape[0])

            d_y = (2.0 / (bsz * OUTPUT_DIM)) * (pred - yb)
            d_z3 = d_y * sigmoid(cache["z3"]) * output_scale[None, :]

            grad_w3 = cache["h2"].T @ d_z3
            grad_b3 = d_z3.sum(axis=0)

            d_h2 = d_z3 @ params["w3"].T
            d_z2 = d_h2 * (cache["z2"] > 0.0)
            grad_w2 = cache["h1"].T @ d_z2
            grad_b2 = d_z2.sum(axis=0)

            d_h1 = d_z2 @ params["w2"].T
            d_z1 = d_h1 * (cache["z1"] > 0.0)
            grad_w1 = xb.T @ d_z1
            grad_b1 = d_z1.sum(axis=0)

            grads = {
                "w1": grad_w1,
                "b1": grad_b1,
                "w2": grad_w2,
                "b2": grad_b2,
                "w3": grad_w3,
                "b3": grad_b3,
            }
            opt.step(params, grads, lr=lr)

        train_pred = forward(params, x_train_n, output_scale)["y"]
        val_pred = forward(params, x_val_n, output_scale)["y"] if len(x_val_n) else train_pred[:1]
        train_loss = mse_loss(train_pred, y_train)
        val_loss = mse_loss(val_pred, y_val if len(y_val) else y_train[:1])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best = {k: v.copy() for k, v in params.items()}

    return {
        "params": best,
        "input_mean": input_mean,
        "input_std": input_std,
        "output_scale": output_scale,
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "best_val_loss": best_val,
    }


def save_npz(path: Path, model: Dict[str, object]) -> None:
    p = model["params"]
    assert isinstance(p, dict)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        w1=p["w1"],
        b1=p["b1"],
        w2=p["w2"],
        b2=p["b2"],
        w3=p["w3"],
        b3=p["b3"],
        input_mean=model["input_mean"],
        input_std=model["input_std"],
        output_scale=model["output_scale"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural transport surrogate (NumPy).")
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.samples = min(args.samples, 512)
        args.epochs = min(args.epochs, 10)
        args.batch_size = min(args.batch_size, 32)

    print(
        f"Training neural transport: samples={args.samples}, "
        f"epochs={args.epochs}, lr={args.lr}, batch={args.batch_size}"
    )
    model = train(
        n_samples=args.samples,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    out = Path(args.output)
    save_npz(out, model)
    print(f"Saved weights: {out}")
    print(f"Best validation MSE: {model['best_val_loss']:.6e}")


if __name__ == "__main__":
    main()
