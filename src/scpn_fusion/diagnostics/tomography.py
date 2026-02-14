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

    def reconstruct(self, signals: np.ndarray) -> np.ndarray:
        """
        Solve non-negative regularized inversion from chord signals.
        """
        b = np.asarray(signals, dtype=np.float64).reshape(-1)
        if b.size != self.A.shape[0]:
            raise ValueError(
                f"signals length mismatch: expected {self.A.shape[0]}, got {b.size}."
            )
        if not np.all(np.isfinite(b)):
            raise ValueError("signals must be finite.")

        lam = self.lambda_reg
        A_aug = np.vstack([self.A, np.sqrt(lam) * np.eye(self.n_pixels, dtype=np.float64)])
        b_aug = np.concatenate([b, np.zeros(self.n_pixels, dtype=np.float64)])

        if lsq_linear is not None:
            res = lsq_linear(A_aug, b_aug, bounds=(0.0, np.inf), tol=1e-4)
            x = np.asarray(res.x, dtype=np.float64)
        else:
            x, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
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
