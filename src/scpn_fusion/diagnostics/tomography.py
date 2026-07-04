# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tomography
"""Tomographic reconstruction tools for synthetic diagnostic post-processing."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

try:
    from scipy.optimize import lsq_linear
except Exception:  # pragma: no cover - optional dependency path
    lsq_linear = None

FloatArray = NDArray[np.float64]


class PlasmaTomography:
    """Reconstruct 2D emissivity profile from line-integrated chord signals."""

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
        self._rust_backend: Any = None

    def _load_rust_backend(self) -> Any:
        """Build (once) the Rust tomography twin from the identical geometry.

        Both backends assemble the same endpoint-inclusive chord-sampling
        geometry matrix and solve the same Tikhonov-regularised non-negative
        least-squares problem, so their reconstructions agree to solver
        tolerance.

        Returns
        -------
        object
            The Rust ``PyTomography`` instance.

        Raises
        ------
        ImportError
            If the optional Rust extension is unavailable.
        """
        if self._rust_backend is None:
            from scpn_fusion.core._multi_compat import dispatch_rust_symbol

            tomography_cls = dispatch_rust_symbol("PyTomography")
            chords = [
                ((float(start[0]), float(start[1])), (float(end[0]), float(end[1])))
                for start, end in self.sensors.bolo_chords
            ]
            self._rust_backend = tomography_cls(
                chords,
                (float(self.R_rec[0]), float(self.R_rec[-1])),
                (float(self.Z_rec[0]), float(self.Z_rec[-1])),
                self.res,
                self.lambda_reg,
            )
        return self._rust_backend

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _build_geometry_matrix(self) -> FloatArray:
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
                # floor (not int truncation) so points left of the grid origin
                # land in bin -1 and are excluded instead of aliasing into
                # column 0; matches the Rust geometry assembly exactly.
                ir = int(np.floor((r - float(self.R_rec[0])) / dr))
                iz = int(np.floor((z - float(self.Z_rec[0])) / dz))
                if 0 <= ir < self.res and 0 <= iz < self.res:
                    pixel_idx = iz * self.res + ir
                    A[i, pixel_idx] += dl

        return A

    def reconstruct(self, signals: FloatArray, method: str = "auto") -> FloatArray:
        """Solve inversion problem Ax=b with regularization.

        Supports 'rust' (fastest, Tikhonov-NNLS via accelerated projected
        gradient), 'lsq_linear' (SciPy), 'ridge' (Phillips-Twomey), and 'sart'
        (Iterative). 'auto' prefers the Rust backend and falls back to
        'lsq_linear' then 'sart' when the extension is unavailable.
        """
        b = np.asarray(signals, dtype=np.float64).reshape(-1)
        if b.size != self.A.shape[0]:
            raise ValueError(f"signals length mismatch: expected {self.A.shape[0]}, got {b.size}.")
        # Condition signals
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        b = np.maximum(b, 0.0)

        if method == "auto":
            try:
                self._load_rust_backend()
                method = "rust"
            except (ImportError, AttributeError, TypeError):
                method = "lsq_linear" if lsq_linear is not None else "sart"

        if method == "rust":
            backend = self._load_rust_backend()
            return np.asarray(backend.reconstruct(list(b)), dtype=np.float64)

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
            v = np.sum(self.A, axis=0)  # Column sums
            h = np.sum(self.A, axis=1)  # Row sums
            v = np.divide(1.0, v, out=np.zeros_like(v), where=v > 0.0)
            h = np.divide(1.0, h, out=np.zeros_like(h), where=h > 0.0)

            for _ in range(n_iters):
                # Back-projection of error
                error = b - self.A @ x
                update = self.A.T @ (h * error)
                x += relax * v * update
                x = np.maximum(x, 0.0)  # Non-negativity constraint

        else:
            # Analytic Ridge fallback
            lam = self.lambda_reg
            L = np.eye(self.n_pixels, dtype=np.float64) * 4.0
            idx = np.arange(self.n_pixels)
            mask_l = (idx % self.res) > 0
            L[idx[mask_l], idx[mask_l] - 1] = -1.0
            mask_r = (idx % self.res) < (self.res - 1)
            L[idx[mask_r], idx[mask_r] + 1] = -1.0
            mask_d = idx >= self.res
            L[idx[mask_d], idx[mask_d] - self.res] = -1.0
            mask_u = idx < (self.n_pixels - self.res)
            L[idx[mask_u], idx[mask_u] + self.res] = -1.0

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
        ground_truth: FloatArray,
        reconstruction: FloatArray,
    ) -> Figure:
        """Plot and return a two-panel ghost ground-truth vs reconstruction figure."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.set_title("Ground Truth (Phantom)")
        ax1.imshow(np.asarray(ground_truth, dtype=np.float64), origin="lower", cmap="hot")

        ax2.set_title("Tomographic Reconstruction")
        ax2.imshow(np.asarray(reconstruction, dtype=np.float64), origin="lower", cmap="hot")

        return fig
