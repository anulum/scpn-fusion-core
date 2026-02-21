# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Tomography
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.optimize import lsq_linear
except Exception:  # pragma: no cover - optional dependency path
    lsq_linear = None


class PlasmaTomography:
    """
    Reconstruct 2D emissivity profile from line-integrated chord signals.
    """

    def __init__(
        self,
        sensors: Any,
        grid_res: int = 20,
        *,
        lambda_reg: float = 0.1,
        verbose: bool = True,
    ) -> None:
        self.sensors = sensors
        grid_res = int(grid_res)
        if grid_res < 4:
            raise ValueError("grid_res must be >= 4.")
        lambda_reg = float(lambda_reg)
        if not np.isfinite(lambda_reg) or lambda_reg < 0.0:
            raise ValueError("lambda_reg must be finite and >= 0.")

        self.res = grid_res
        self.lambda_reg = lambda_reg
        self.verbose = bool(verbose)

        self.R_rec = np.linspace(sensors.kernel.R[0], sensors.kernel.R[-1], self.res)
        self.Z_rec = np.linspace(sensors.kernel.Z[0], sensors.kernel.Z[-1], self.res)
        self.n_pixels = self.res * self.res
        self.A = self._build_geometry_matrix()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _build_geometry_matrix(self) -> np.ndarray:
        self._log("[Tomography] Building Geometry Matrix A...")
        n_chords = len(self.sensors.bolo_chords)
        A = np.zeros((n_chords, self.n_pixels), dtype=np.float64)

        dr = float(self.R_rec[1] - self.R_rec[0])
        dz = float(self.Z_rec[1] - self.Z_rec[0])
        for i, (start, end) in enumerate(self.sensors.bolo_chords):
            num_samples = 100
            r_samples = np.linspace(start[0], end[0], num_samples)
            z_samples = np.linspace(start[1], end[1], num_samples)
            dl = float(np.linalg.norm(np.asarray(end) - np.asarray(start)) / num_samples)

            for k in range(num_samples):
                r = float(r_samples[k])
                z = float(z_samples[k])
                ir = int((r - float(self.R_rec[0])) / dr)
                iz = int((z - float(self.Z_rec[0])) / dz)
                if 0 <= ir < self.res and 0 <= iz < self.res:
                    pixel_idx = iz * self.res + ir
                    A[i, pixel_idx] += dl

        return A

    def reconstruct(self, signals: np.ndarray, method: str = "auto") -> np.ndarray:
        """
        Solve inversion problem Ax=b with regularization.
        Supports 'lsq_linear' (SciPy), 'ridge' (Phillips-Twomey), and 'sart' (Iterative).
        """
        b = np.asarray(signals, dtype=np.float64).reshape(-1)
        if b.size != self.A.shape[0]:
            raise ValueError(
                f"signals length mismatch: expected {self.A.shape[0]}, got {b.size}."
            )
        # Condition signals
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        b = np.maximum(b, 0.0)

        if method == "auto":
            method = "lsq_linear" if lsq_linear is not None else "sart"

        if method == "lsq_linear" and lsq_linear is not None:
            # SciPy Path
            lam = self.lambda_reg
            A_aug = np.vstack([self.A, np.sqrt(lam) * np.eye(self.n_pixels, dtype=np.float64)])
            b_aug = np.concatenate([b, np.zeros(self.n_pixels, dtype=np.float64)])
            res = lsq_linear(A_aug, b_aug, bounds=(0.0, np.inf), tol=1e-4)
            x = np.asarray(res.x, dtype=np.float64)
            
        elif method == "sart":
            # Iterative Reconstruction (Simultaneous Algebraic Reconstruction Technique)
            # x_new = x_old + lambda * A.T * (b - A*x_old) / (A.T * A * 1)
            x = np.zeros(self.n_pixels, dtype=np.float64)
            n_iters = 50
            relax = 0.1
            
            # Precompute weights for SART
            v = np.sum(self.A, axis=0) # Column sums
            h = np.sum(self.A, axis=1) # Row sums
            v = np.where(v > 0, 1.0/v, 0.0)
            h = np.where(h > 0, 1.0/h, 0.0)
            
            for _ in range(n_iters):
                # Back-projection of error
                error = b - self.A @ x
                update = self.A.T @ (h * error)
                x += relax * v * update
                x = np.maximum(x, 0.0) # Non-negativity constraint
                
        else:
            # Analytic Ridge fallback
            lam = self.lambda_reg
            L = np.eye(self.n_pixels, dtype=np.float64) * 4.0
            idx = np.arange(self.n_pixels)
            mask_l = (idx % self.res) > 0
            L[idx[mask_l], idx[mask_l]-1] = -1.0
            mask_r = (idx % self.res) < (self.res - 1)
            L[idx[mask_r], idx[mask_r]+1] = -1.0
            mask_d = idx >= self.res
            L[idx[mask_d], idx[mask_d]-self.res] = -1.0
            mask_u = idx < (self.n_pixels - self.res)
            L[idx[mask_u], idx[mask_u]+self.res] = -1.0
            
            lhs = self.A.T @ self.A + lam * (L.T @ L)
            rhs = self.A.T @ b
            try:
                x = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                x = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            x = np.clip(np.asarray(x, dtype=np.float64), 0.0, np.inf)

        return x.reshape((self.res, self.res))

    def plot_reconstruction(
        self,
        ground_truth: np.ndarray,
        reconstruction: np.ndarray,
    ):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.set_title("Ground Truth (Phantom)")
        ax1.imshow(np.asarray(ground_truth, dtype=np.float64), origin="lower", cmap="hot")

        ax2.set_title("Tomographic Reconstruction")
        ax2.imshow(np.asarray(reconstruction, dtype=np.float64), origin="lower", cmap="hot")

        return fig
