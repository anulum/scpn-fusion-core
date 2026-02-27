# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Pretrained Surrogates Bundle
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Train and bundle lightweight pretrained MLP/FNO surrogates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.eqdsk import read_geqdsk
from scpn_fusion.core.fno_training import AdamOptimizer, MultiLayerFNO


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ITPA_CSV = REPO_ROOT / "validation" / "reference_data" / "itpa" / "hmode_confinement.csv"
DEFAULT_JET_DIR = REPO_ROOT / "validation" / "reference_data" / "jet"
DEFAULT_WEIGHTS_DIR = REPO_ROOT / "weights"
DEFAULT_MLP_PATH = DEFAULT_WEIGHTS_DIR / "pretrained_mlp_itpa.npz"
DEFAULT_FNO_PATH = DEFAULT_WEIGHTS_DIR / "pretrained_fno_eurofusion_jet.npz"
DEFAULT_MANIFEST_PATH = DEFAULT_WEIGHTS_DIR / "pretrained_surrogates_manifest.json"


FloatArray = NDArray[np.float64]


def _default_surrogate_coverage() -> dict[str, Any]:
    shipped = [
        "scpn_fusion.core.pretrained_surrogates:mlp_itpa",
        "scpn_fusion.core.pretrained_surrogates:fno_eurofusion_jet",
        "scpn_fusion.core.neural_equilibrium:sparc",
        "scpn_fusion.core.neural_transport:qlknn",
    ]
    requires_user_training = [
        "scpn_fusion.core.heat_ml_shadow_surrogate",
        "scpn_fusion.core.gyro_swin_surrogate",
        "scpn_fusion.core.turbulence_oracle",
    ]
    total = len(shipped) + len(requires_user_training)
    return {
        "pretrained_shipped": shipped,
        "requires_user_training": requires_user_training,
        "coverage_fraction": float(len(shipped) / max(total, 1)),
        "coverage_percent": float(100.0 * len(shipped) / max(total, 1)),
        "notes": (
            "Pretrained artifacts are bundled for MLP (ITPA), FNO (JET), "
            "neural equilibrium (SPARC GEQDSK), and QLKNN transport. "
            "Remaining surrogate lanes still require facility-specific "
            "user training."
        ),
    }


def get_pretrained_surrogate_coverage(manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    default_cov = _default_surrogate_coverage()
    if not manifest:
        return default_cov
    cov = manifest.get("coverage")
    if not isinstance(cov, dict):
        return default_cov
    merged = dict(default_cov)
    merged.update(cov)
    return merged


def _as_repo_relative(path: Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT.resolve()).as_posix())
    except Exception:
        return str(p.as_posix())


@dataclass(frozen=True)
class PretrainedMLPSurrogate:
    feature_mean: FloatArray
    feature_std: FloatArray
    w1: FloatArray
    b1: FloatArray
    w2: FloatArray
    b2: FloatArray
    target_mean: float
    target_std: float

    def predict(self, features: FloatArray) -> FloatArray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.feature_mean.size:
            raise ValueError(
                f"Expected feature width {self.feature_mean.size}, got {x.shape[1]}."
            )
        x_norm = (x - self.feature_mean[None, :]) / self.feature_std[None, :]
        h = np.tanh(x_norm @ self.w1 + self.b1[None, :])
        y_norm = h @ self.w2 + self.b2
        y = y_norm * self.target_std + self.target_mean
        return np.maximum(y.reshape(-1), 1e-6)


def _load_itpa_training_data(csv_path: Path = DEFAULT_ITPA_CSV) -> tuple[FloatArray, FloatArray]:
    features: list[list[float]] = []
    targets: list[float] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            features.append(
                [
                    float(row["Ip_MA"]),
                    float(row["BT_T"]),
                    float(row["ne19_1e19m3"]),
                    float(row["Ploss_MW"]),
                    float(row["R_m"]),
                    float(row["a_m"]),
                    float(row["kappa"]),
                    float(row["delta"]),
                    float(row["M_AMU"]),
                ]
            )
            targets.append(float(row["tau_E_s"]))
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 8:
        raise ValueError("ITPA dataset must contain at least 8 rows.")
    return x, y


def _train_itpa_mlp(
    *,
    seed: int = 42,
    hidden: int = 32,
    epochs: int = 1200,
    lr: float = 1.2e-2,
    l2: float = 5.0e-4,
    csv_path: Path = DEFAULT_ITPA_CSV,
) -> tuple[PretrainedMLPSurrogate, dict[str, float]]:
    x, y = _load_itpa_training_data(csv_path)
    rng = np.random.default_rng(int(seed))

    x_mu = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    x_n = (x - x_mu[None, :]) / x_std[None, :]

    y_mu = float(np.mean(y))
    y_std = float(np.std(y)) if float(np.std(y)) > 1e-8 else 1.0
    y_n = (y - y_mu) / y_std

    n, d = x_n.shape
    h = int(hidden)
    w1 = rng.normal(0.0, 0.22, size=(d, h))
    b1 = np.zeros((h,), dtype=np.float64)
    w2 = rng.normal(0.0, 0.18, size=(h,))
    b2 = np.array(0.0, dtype=np.float64)

    for _ in range(int(epochs)):
        h_act = np.tanh(x_n @ w1 + b1[None, :])
        pred = h_act @ w2 + b2
        err = pred - y_n

        grad_pred = (2.0 / n) * err
        grad_w2 = h_act.T @ grad_pred + l2 * w2
        grad_b2 = float(np.sum(grad_pred))
        grad_h = grad_pred[:, None] * w2[None, :]
        grad_z1 = grad_h * (1.0 - h_act * h_act)
        grad_w1 = x_n.T @ grad_z1 + l2 * w1
        grad_b1 = np.sum(grad_z1, axis=0)

        w2 -= lr * grad_w2
        b2 -= lr * grad_b2
        w1 -= lr * grad_w1
        b1 -= lr * grad_b1

    model = PretrainedMLPSurrogate(
        feature_mean=x_mu.astype(np.float64),
        feature_std=x_std.astype(np.float64),
        w1=w1.astype(np.float64),
        b1=b1.astype(np.float64),
        w2=w2.astype(np.float64),
        b2=np.asarray(b2, dtype=np.float64),
        target_mean=y_mu,
        target_std=y_std,
    )
    pred_tau = model.predict(x)
    rmse = float(np.sqrt(np.mean((pred_tau - y) ** 2)))
    rmse_pct = float(100.0 * rmse / (np.mean(np.abs(y)) + 1e-12))
    return model, {"train_rmse_s": rmse, "train_rmse_pct": rmse_pct}


def _resize_2d(arr: FloatArray, out_h: int = 64, out_w: int = 64) -> FloatArray:
    src = np.asarray(arr, dtype=np.float64)
    if src.ndim != 2:
        raise ValueError("Expected 2D array.")

    in_h, in_w = src.shape
    x_in = np.linspace(0.0, 1.0, in_w, dtype=np.float64)
    x_out = np.linspace(0.0, 1.0, out_w, dtype=np.float64)
    row_interp = np.empty((in_h, out_w), dtype=np.float64)
    for i in range(in_h):
        row_interp[i] = np.interp(x_out, x_in, src[i])

    y_in = np.linspace(0.0, 1.0, in_h, dtype=np.float64)
    y_out = np.linspace(0.0, 1.0, out_h, dtype=np.float64)
    out = np.empty((out_h, out_w), dtype=np.float64)
    for j in range(out_w):
        out[:, j] = np.interp(y_out, y_in, row_interp[:, j])
    return out


def _build_jet_fno_dataset(
    *,
    jet_dir: Path = DEFAULT_JET_DIR,
    seed: int = 42,
    augment_per_file: int = 12,
) -> tuple[FloatArray, FloatArray]:
    rng = np.random.default_rng(int(seed))
    x_rows: list[FloatArray] = []
    y_rows: list[FloatArray] = []

    files = sorted(Path(jet_dir).glob("*.geqdsk"))
    if not files:
        raise ValueError(f"No JET GEQDSK files found in {jet_dir}.")

    for path in files:
        eq = read_geqdsk(path)
        psi = np.asarray(eq.psirz, dtype=np.float64)
        psi_norm = (psi - float(np.min(psi))) / (float(np.max(psi) - np.min(psi)) + 1e-12)
        base = _resize_2d(psi_norm, out_h=64, out_w=64)
        for _ in range(int(augment_per_file)):
            perturb = rng.normal(0.0, 0.008, size=base.shape)
            rolled = np.roll(base, int(rng.integers(-2, 3)), axis=0)
            rolled = np.roll(rolled, int(rng.integers(-2, 3)), axis=1)
            x = np.clip(rolled + perturb, 0.0, 1.2)

            lap = (
                np.roll(x, 1, axis=0)
                + np.roll(x, -1, axis=0)
                + np.roll(x, 1, axis=1)
                + np.roll(x, -1, axis=1)
                - 4.0 * x
            )
            y = np.clip(x + 0.06 * lap - 0.010 * (x * x), 0.0, 1.2)
            x_rows.append(x.astype(np.float64))
            y_rows.append(y.astype(np.float64))

    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.float64)


def _relative_l2(pred: FloatArray, target: FloatArray) -> float:
    denom = float(np.linalg.norm(target) + 1e-8)
    return float(np.linalg.norm(pred - target) / denom)


def _train_fno_on_jet(
    *,
    save_path: Path,
    seed: int = 42,
    modes: int = 8,
    width: int = 16,
    epochs: int = 24,
    batch_size: int = 8,
    augment_per_file: int = 12,
    jet_dir: Path = DEFAULT_JET_DIR,
) -> dict[str, float]:
    x, y = _build_jet_fno_dataset(
        jet_dir=jet_dir, seed=seed, augment_per_file=augment_per_file
    )
    n = x.shape[0]
    split = max(4, int(0.85 * n))
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    model = MultiLayerFNO(modes=modes, width=width, n_layers=4, seed=seed)
    optim = AdamOptimizer()
    rng = np.random.default_rng(seed + 991)
    best_val = float("inf")
    best_w = model.project_w.copy()
    best_b = model.project_b

    for _ in range(int(epochs)):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), int(batch_size)):
            batch = order[start : start + int(batch_size)]
            if batch.size == 0:
                continue
            grad_w = np.zeros_like(model.project_w)
            grad_b = 0.0
            for i in batch:
                pred, hidden = model.forward_with_hidden(x_train[int(i)])
                target = y_train[int(i)]
                scale = float(np.mean(target * target) + 1e-8)
                err = pred - target
                grad_y = (2.0 / err.size) * err / scale
                grad_w += np.tensordot(hidden, grad_y, axes=([0, 1], [0, 1]))
                grad_b += float(np.sum(grad_y))
            grad_w /= float(batch.size)
            grad_b /= float(batch.size)

            params = {
                "project_w": model.project_w,
                "project_b": np.array([model.project_b], dtype=np.float64),
            }
            grads = {
                "project_w": grad_w,
                "project_b": np.array([grad_b], dtype=np.float64),
            }
            optim.step(params, grads, lr=8e-4)
            model.project_b = float(params["project_b"][0])

        val_losses = [_relative_l2(model.forward(x_val[i]), y_val[i]) for i in range(len(x_val))]
        val = float(np.mean(np.asarray(val_losses, dtype=np.float64)))
        if val < best_val:
            best_val = val
            best_w = model.project_w.copy()
            best_b = model.project_b

    model.project_w = best_w
    model.project_b = best_b
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(save_path)

    train_losses = [_relative_l2(model.forward(x_train[i]), y_train[i]) for i in range(min(24, len(x_train)))]
    val_losses = [_relative_l2(model.forward(x_val[i]), y_val[i]) for i in range(len(x_val))]
    return {
        "train_relative_l2": float(np.mean(np.asarray(train_losses, dtype=np.float64))),
        "val_relative_l2": float(np.mean(np.asarray(val_losses, dtype=np.float64))),
        "dataset_samples": float(n),
    }


def evaluate_pretrained_fno(
    *,
    fno_path: Path = DEFAULT_FNO_PATH,
    jet_dir: Path = DEFAULT_JET_DIR,
    seed: int = 47,
    augment_per_file: int = 8,
    max_samples: int = 16,
) -> dict[str, float]:
    if int(max_samples) <= 0:
        raise ValueError("max_samples must be > 0.")
    x, y = _build_jet_fno_dataset(
        jet_dir=jet_dir, seed=seed, augment_per_file=augment_per_file
    )
    if x.size == 0:
        raise ValueError("FNO evaluation dataset is empty.")
    m = MultiLayerFNO()
    m.load_weights(fno_path)
    n = min(int(max_samples), int(x.shape[0]))
    losses = [_relative_l2(m.forward(x[i]), y[i]) for i in range(n)]
    return {
        "eval_relative_l2_mean": float(np.mean(np.asarray(losses, dtype=np.float64))),
        "eval_relative_l2_p95": float(np.percentile(np.asarray(losses, dtype=np.float64), 95)),
        "eval_samples": float(n),
    }


def save_pretrained_mlp(model: PretrainedMLPSurrogate, path: Path = DEFAULT_MLP_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        feature_mean=model.feature_mean,
        feature_std=model.feature_std,
        w1=model.w1,
        b1=model.b1,
        w2=model.w2,
        b2=model.b2,
        target_mean=np.asarray([model.target_mean], dtype=np.float64),
        target_std=np.asarray([model.target_std], dtype=np.float64),
    )


def load_pretrained_mlp(path: Path = DEFAULT_MLP_PATH) -> PretrainedMLPSurrogate:
    with np.load(path, allow_pickle=False) as data:
        return PretrainedMLPSurrogate(
            feature_mean=np.asarray(data["feature_mean"], dtype=np.float64),
            feature_std=np.asarray(data["feature_std"], dtype=np.float64),
            w1=np.asarray(data["w1"], dtype=np.float64),
            b1=np.asarray(data["b1"], dtype=np.float64),
            w2=np.asarray(data["w2"], dtype=np.float64),
            b2=np.asarray(data["b2"], dtype=np.float64),
            target_mean=float(np.asarray(data["target_mean"], dtype=np.float64).reshape(-1)[0]),
            target_std=float(np.asarray(data["target_std"], dtype=np.float64).reshape(-1)[0]),
        )


def evaluate_pretrained_mlp(
    *,
    model_path: Path = DEFAULT_MLP_PATH,
    csv_path: Path = DEFAULT_ITPA_CSV,
    max_samples: int = 0,
) -> dict[str, float]:
    model = load_pretrained_mlp(path=model_path)
    x, y = _load_itpa_training_data(csv_path)
    if max_samples > 0:
        x = x[:max_samples]
        y = y[:max_samples]
    pred = model.predict(x)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    rmse_pct = float(100.0 * rmse / (np.mean(np.abs(y)) + 1e-12))
    return {"rmse_s": rmse, "rmse_pct": rmse_pct, "samples": float(x.shape[0])}


def bundle_pretrained_surrogates(
    *,
    force_retrain: bool = False,
    seed: int = 42,
    itpa_csv_path: Path = DEFAULT_ITPA_CSV,
    jet_dir: Path = DEFAULT_JET_DIR,
    mlp_hidden: int = 32,
    mlp_epochs: int = 1200,
    mlp_lr: float = 1.2e-2,
    mlp_l2: float = 5.0e-4,
    fno_modes: int = 8,
    fno_width: int = 16,
    fno_epochs: int = 24,
    fno_batch_size: int = 8,
    fno_augment_per_file: int = 12,
    weights_dir: Path = DEFAULT_WEIGHTS_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    mlp_path: Path = DEFAULT_MLP_PATH,
    fno_path: Path = DEFAULT_FNO_PATH,
) -> dict[str, Any]:
    if int(mlp_hidden) <= 0:
        raise ValueError("mlp_hidden must be > 0.")
    if int(mlp_epochs) <= 0:
        raise ValueError("mlp_epochs must be > 0.")
    if int(fno_modes) <= 0 or int(fno_width) <= 0:
        raise ValueError("fno_modes/fno_width must be > 0.")
    if int(fno_epochs) <= 0 or int(fno_batch_size) <= 0:
        raise ValueError("fno_epochs/fno_batch_size must be > 0.")
    if int(fno_augment_per_file) <= 0:
        raise ValueError("fno_augment_per_file must be > 0.")

    if (
        not force_retrain
        and manifest_path.exists()
        and mlp_path.exists()
        and fno_path.exists()
    ):
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    mlp_model, mlp_metrics = _train_itpa_mlp(
        seed=seed,
        hidden=mlp_hidden,
        epochs=mlp_epochs,
        lr=mlp_lr,
        l2=mlp_l2,
        csv_path=itpa_csv_path,
    )
    save_pretrained_mlp(mlp_model, path=mlp_path)

    fno_metrics = _train_fno_on_jet(
        save_path=fno_path,
        seed=seed + 73,
        modes=fno_modes,
        width=fno_width,
        epochs=fno_epochs,
        batch_size=fno_batch_size,
        augment_per_file=fno_augment_per_file,
        jet_dir=jet_dir,
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "task2-pretrained-v1",
        "artifacts": {
            "mlp_itpa": _as_repo_relative(mlp_path),
            "fno_eurofusion_jet": _as_repo_relative(fno_path),
        },
        "datasets": {
            "itpa": _as_repo_relative(itpa_csv_path),
            "eurofusion_proxy_jet": _as_repo_relative(jet_dir),
        },
        "config": {
            "seed": int(seed),
            "mlp_hidden": int(mlp_hidden),
            "mlp_epochs": int(mlp_epochs),
            "fno_modes": int(fno_modes),
            "fno_width": int(fno_width),
            "fno_epochs": int(fno_epochs),
            "fno_batch_size": int(fno_batch_size),
            "fno_augment_per_file": int(fno_augment_per_file),
        },
        "metrics": {
            "mlp": mlp_metrics,
            "fno": fno_metrics,
        },
        "coverage": _default_surrogate_coverage(),
    }
    weights_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
