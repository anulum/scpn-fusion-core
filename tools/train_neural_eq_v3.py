#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — V3 Multi-Machine Neural Equilibrium Training
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""V3: Sample-weighted training with wider perturbation range.

Fixes found in v1/v2:
  - negdelta (delta=-0.39) is an isolated outlier in feature space (dist=4.87
    to nearest neighbour). Standard unweighted MSE makes the majority cluster
    dominate gradient updates.
  - v1: SPARC=0.029 JET=0.024 DIIID=0.253 (negdelta=0.48, hmode_2MA=0.30)
  - v2: SPARC=0.029 JET=0.024 DIIID=0.247 (more epochs but same pattern)

V3 changes:
  1. Per-file equal weight: each GEQDSK file gets 1/18 total gradient influence
  2. Base samples weighted 3x vs perturbations (emphasize exact reconstructions)
  3. Wider perturbation range [0.5, 1.5] vs [0.7, 1.3] (more shape diversity)
  4. Wider network: 512-256-128-64 (~200K params)
  5. Lower initial LR: 5e-4 (less overshooting on outlier gradients)
  6. Patience: 300 epochs
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from numpy.typing import NDArray

TARGET_GRID = 129
HIDDEN = (512, 256, 128, 64)
N_PCA = 50
LR0 = 5e-4
LR_END = 1e-5
WD = 1e-4
EPOCHS = 3000
BS = 32
PATIENCE = 300
MOM = 0.9


class MinimalPCA:
    def __init__(self, k: int = 20) -> None:
        self.k = k
        self.mean_: NDArray | None = None
        self.V: NDArray | None = None
        self.evr: NDArray | None = None

    def fit(self, X: NDArray) -> "MinimalPCA":
        self.mean_ = X.mean(0)
        Xc = X - self.mean_
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(self.k, len(S))
        self.V = Vt[:k]
        self.evr = S[:k] ** 2 / max((S**2).sum(), 1e-15)
        self.k = k
        return self

    def transform(self, X: NDArray) -> NDArray:
        return (X - self.mean_) @ self.V.T

    def inverse_transform(self, Z: NDArray) -> NDArray:
        return Z @ self.V + self.mean_

    def fit_transform(self, X: NDArray) -> NDArray:
        self.fit(X)
        return self.transform(X)


class MLP:
    def __init__(self, sizes: list[int], seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.W: list[NDArray] = []
        self.b: list[NDArray] = []
        for i in range(len(sizes) - 1):
            self.W.append(rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))

    def forward(self, x: NDArray) -> NDArray:
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(0, h)
        return h


def load_data(
    ref_dir: Path,
    n_pert: int,
    rng: np.random.Generator,
) -> tuple[NDArray, NDArray, list[str], NDArray]:
    from scpn_fusion.core.eqdsk import read_geqdsk
    from scipy.interpolate import RectBivariateSpline

    t_rho = np.linspace(0, 1, TARGET_GRID)
    t_zeta = np.linspace(0, 1, TARGET_GRID)

    X_list: list[NDArray] = []
    Y_list: list[NDArray] = []
    labels: list[str] = []
    weights: list[float] = []

    machines: dict[str, list[Path]] = {}
    for d in sorted(ref_dir.iterdir()):
        if not d.is_dir():
            continue
        fs = sorted(d.glob("*.geqdsk")) + sorted(d.glob("*.eqdsk"))
        if fs:
            machines[d.name] = fs

    total_files = sum(len(v) for v in machines.values())

    for machine, paths in machines.items():
        for path in paths:
            eq = read_geqdsk(path)
            r_n = (eq.r - eq.r[0]) / max(eq.r[-1] - eq.r[0], 1e-12)
            z_n = (eq.z - eq.z[0]) / max(eq.z[-1] - eq.z[0], 1e-12)
            spl = RectBivariateSpline(z_n, r_n, eq.psirz, kx=3, ky=3)
            psi = spl(t_zeta, t_rho, grid=True)
            denom = eq.sibry - eq.simag
            if abs(denom) < 1e-12:
                denom = 1.0
            psi_n = (psi - eq.simag) / denom

            kap, du, dl, q95 = 1.7, 0.3, 0.3, 3.0
            if hasattr(eq, "rbdry") and eq.rbdry is not None and len(eq.rbdry) > 3:
                rs = eq.rbdry.max() - eq.rbdry.min()
                if rs > 0.01:
                    kap = (eq.zbdry.max() - eq.zbdry.min()) / rs
                    du = (eq.rmaxis - eq.rbdry[np.argmax(eq.zbdry)]) / (rs / 2)
                    dl = (eq.rmaxis - eq.rbdry[np.argmin(eq.zbdry)]) / (rs / 2)
            if hasattr(eq, "qpsi") and eq.qpsi is not None and len(eq.qpsi) > 0:
                q95 = eq.qpsi[min(int(0.95 * len(eq.qpsi)), len(eq.qpsi) - 1)]

            base = np.array(
                [
                    eq.current / 1e6,
                    eq.bcentr,
                    eq.rmaxis,
                    eq.zmaxis,
                    1.0,
                    1.0,
                    eq.simag,
                    eq.sibry,
                    kap,
                    du,
                    dl,
                    q95,
                ]
            )

            # Per-file weight: each file gets equal total gradient influence
            w_per_sample = 1.0 / (total_files * (1 + n_pert))

            # Base sample: 3x weight (prioritise exact reconstruction)
            X_list.append(base)
            Y_list.append(psi_n.ravel())
            labels.append(machine)
            weights.append(w_per_sample * 3.0)

            for _ in range(n_pert):
                pp = rng.uniform(0.5, 1.5)
                ff = rng.uniform(0.5, 1.5)
                feat = base.copy()
                feat[4] = pp
                feat[5] = ff
                mix = 0.5 * (pp + ff) - 1.0
                pn = psi_n.copy()
                mask = (pn >= 0) & (pn < 1.0)
                pn[mask] += mix * 0.15 * (1.0 - pn[mask])
                X_list.append(feat)
                Y_list.append(pn.ravel())
                labels.append(machine)
                weights.append(w_per_sample)

    return np.array(X_list), np.array(Y_list), labels, np.array(weights)


def stratified_split(
    labels: list[str],
    rng: np.random.Generator,
) -> tuple[NDArray, NDArray, NDArray]:
    machines = sorted(set(labels))
    tri, vai, tei = [], [], []
    for m in machines:
        idx = np.array([i for i, l in enumerate(labels) if l == m])
        rng.shuffle(idx)
        n = len(idx)
        nt = max(1, int(0.70 * n))
        nv = max(1, int(0.15 * n))
        tri.extend(idx[:nt])
        vai.extend(idx[nt : nt + nv])
        tei.extend(idx[nt + nv :])
    return np.array(tri), np.array(vai), np.array(tei)


def validate_per_machine(
    ref_dir: Path,
    mlp: MLP,
    pca: MinimalPCA,
    imean: NDArray,
    istd: NDArray,
) -> dict[str, dict]:
    from scpn_fusion.core.eqdsk import read_geqdsk
    from scipy.interpolate import RectBivariateSpline

    t_rho = np.linspace(0, 1, TARGET_GRID)
    t_zeta = np.linspace(0, 1, TARGET_GRID)
    results: dict[str, dict] = {}

    for d in sorted(ref_dir.iterdir()):
        if not d.is_dir():
            continue
        fs = sorted(d.glob("*.geqdsk")) + sorted(d.glob("*.eqdsk"))
        if not fs:
            continue
        file_res = []
        for f in fs:
            eq = read_geqdsk(f)
            r_n = (eq.r - eq.r[0]) / max(eq.r[-1] - eq.r[0], 1e-12)
            z_n = (eq.z - eq.z[0]) / max(eq.z[-1] - eq.z[0], 1e-12)
            spl = RectBivariateSpline(z_n, r_n, eq.psirz, kx=3, ky=3)
            psi = spl(t_zeta, t_rho, grid=True)
            den = eq.sibry - eq.simag
            if abs(den) < 1e-12:
                den = 1.0
            psi_ref = (psi - eq.simag) / den

            kap, du2, dl2, q95 = 1.7, 0.3, 0.3, 3.0
            if hasattr(eq, "rbdry") and eq.rbdry is not None and len(eq.rbdry) > 3:
                rs = eq.rbdry.max() - eq.rbdry.min()
                if rs > 0.01:
                    kap = (eq.zbdry.max() - eq.zbdry.min()) / rs
                    du2 = (eq.rmaxis - eq.rbdry[np.argmax(eq.zbdry)]) / (rs / 2)
                    dl2 = (eq.rmaxis - eq.rbdry[np.argmin(eq.zbdry)]) / (rs / 2)
            if hasattr(eq, "qpsi") and eq.qpsi is not None and len(eq.qpsi) > 0:
                q95 = eq.qpsi[min(int(0.95 * len(eq.qpsi)), len(eq.qpsi) - 1)]

            feat = np.array(
                [
                    eq.current / 1e6,
                    eq.bcentr,
                    eq.rmaxis,
                    eq.zmaxis,
                    1.0,
                    1.0,
                    eq.simag,
                    eq.sibry,
                    kap,
                    du2,
                    dl2,
                    q95,
                ]
            )
            xn = (feat - imean) / istd
            coeff = mlp.forward(xn[np.newaxis, :])
            pred = pca.inverse_transform(coeff)[0].reshape(TARGET_GRID, TARGET_GRID)
            rl2 = float(np.linalg.norm(pred - psi_ref) / max(np.linalg.norm(psi_ref), 1e-12))
            file_res.append((f.name, rl2))

        ml2 = float(np.mean([x[1] for x in file_res]))
        mx2 = float(np.max([x[1] for x in file_res]))
        results[d.name] = {
            "mean_rel_l2": ml2,
            "max_rel_l2": mx2,
            "per_file": {n: v for n, v in file_res},
        }
        print(f"  [{d.name}] mean={ml2:.4f}  max={mx2:.4f}")
        for n, v in file_res:
            print(f"    {n}: {v:.4f}")

    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    rng = np.random.default_rng(42)
    ref_dir = REPO_ROOT / "validation" / "reference_data"

    print("Loading data...")
    X, Y, labels, sample_weights = load_data(ref_dir, 100, rng)
    print(f"X: {X.shape}, Y: {Y.shape}")
    print(f"Psi range: [{Y.min():.3f}, {Y.max():.3f}]")

    imean = X.mean(0)
    istd = X.std(0)
    istd[istd < 1e-10] = 1.0
    Xn = (X - imean) / istd

    pca = MinimalPCA(N_PCA)
    Yc = pca.fit_transform(Y)
    print(f"PCA: {pca.k} components, {sum(pca.evr) * 100:.1f}% variance")

    tri, vai, tei = stratified_split(labels, rng)
    print(f"Split: {len(tri)}/{len(vai)}/{len(tei)}")

    sw_train = sample_weights[tri]
    sw_train = sw_train / sw_train.mean()

    mlp = MLP([12, *HIDDEN, pca.k], seed=42)
    Xtr, Ytr = Xn[tri], Yc[tri]
    Xva, Yva = Xn[vai], Yc[vai]

    vel = [np.zeros_like(w) for w in mlp.W]
    vel_b = [np.zeros_like(b) for b in mlp.b]
    best_vl = float("inf")
    best_W: list[NDArray] | None = None
    best_b: list[NDArray] | None = None
    patience_ctr = 0
    rng2 = np.random.default_rng(43)
    t0 = time.perf_counter()

    print(f"\nTraining: {EPOCHS} epochs, network {HIDDEN}...")
    for ep in range(EPOCHS):
        lr = LR_END + 0.5 * (LR0 - LR_END) * (1 + np.cos(np.pi * ep / EPOCHS))
        order = rng2.permutation(len(Xtr))
        eloss = 0.0

        for s in range(0, len(Xtr), BS):
            idx = order[s : s + BS]
            xb, yb = Xtr[idx], Ytr[idx]
            wb = sw_train[idx]

            acts = [xb]
            h = xb
            for i, (W, b) in enumerate(zip(mlp.W, mlp.b)):
                z = h @ W + b
                h = np.maximum(0, z) if i < len(mlp.W) - 1 else z
                acts.append(h)

            err = acts[-1] - yb
            w_sqrt = np.sqrt(wb[:, np.newaxis])
            weighted_err = err * w_sqrt
            loss = float(np.mean(weighted_err**2))
            eloss += loss * len(idx)

            delta = 2.0 * weighted_err * w_sqrt / len(idx)
            for i in range(len(mlp.W) - 1, -1, -1):
                gw = acts[i].T @ delta + WD * mlp.W[i]
                gb = delta.sum(0)
                gn = np.linalg.norm(gw)
                if gn > 5.0:
                    gw *= 5.0 / gn
                    gb *= 5.0 / gn
                vel[i] = MOM * vel[i] - lr * gw
                vel_b[i] = MOM * vel_b[i] - lr * gb
                mlp.W[i] += vel[i]
                mlp.b[i] += vel_b[i]
                if i > 0:
                    delta = delta @ mlp.W[i].T
                    delta *= (acts[i] > 0).astype(float)

        eloss /= max(len(Xtr), 1)

        vp = mlp.forward(Xva)
        vl = float(np.mean((vp - Yva) ** 2))
        if vl < best_vl:
            best_vl = vl
            patience_ctr = 0
            best_W = [w.copy() for w in mlp.W]
            best_b = [b.copy() for b in mlp.b]
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stop at epoch {ep} (val={vl:.6f})")
                break

        if ep % 200 == 0:
            print(f"  Epoch {ep:4d}: train={eloss:.6f}  val={vl:.6f}  lr={lr:.2e}")

    train_time = time.perf_counter() - t0
    epochs_run = ep + 1
    if best_W is not None:
        mlp.W = best_W
        mlp.b = best_b

    print(f"\nTraining done in {train_time:.1f}s ({epochs_run} epochs)")
    print(f"  Best val loss: {best_vl:.6f}")

    print("\nPer-machine validation:")
    per_machine = validate_per_machine(ref_dir, mlp, pca, imean, istd)

    all_pass = True
    for machine, metrics in per_machine.items():
        if (
            machine == "sparc"
            and metrics["mean_rel_l2"] > 0.10
            or machine != "sparc"
            and metrics["mean_rel_l2"] > 0.20
        ):
            all_pass = False

    save_path = REPO_ROOT / "weights" / "neural_equilibrium_augmented_v3.npz"
    payload: dict[str, NDArray] = {
        "n_components": np.array([pca.k]),
        "grid_nh": np.array([TARGET_GRID]),
        "grid_nw": np.array([TARGET_GRID]),
        "n_input_features": np.array([12]),
        "pca_mean": pca.mean_,
        "pca_components": pca.V,
        "pca_evr": pca.evr,
        "input_mean": imean,
        "input_std": istd,
        "n_layers": np.array([len(mlp.W)]),
        "psi_normalized": np.array([1]),
    }
    for i, (w, b) in enumerate(zip(mlp.W, mlp.b)):
        payload[f"w{i}"] = w
        payload[f"b{i}"] = b
    np.savez(save_path, **payload)

    metrics_out = {
        "model": "neural_equilibrium_augmented_v3",
        "hidden": list(HIDDEN),
        "n_pca": pca.k,
        "sample_weighted": True,
        "n_samples": len(X),
        "best_val_loss": best_vl,
        "train_time_s": train_time,
        "epochs_run": epochs_run,
        "per_machine": per_machine,
        "acceptance": "PASS" if all_pass else "FAIL",
    }
    with open(save_path.with_suffix(".metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\nWeights: {save_path}")
    if all_pass:
        print("ALL ACCEPTANCE CRITERIA MET")
    else:
        print("ACCEPTANCE CRITERIA NOT MET")
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
