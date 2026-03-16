# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — GK Correction Layer
"""
Error quantification and correction for surrogate transport models
validated against GK spot-checks.

Compares surrogate vs GK fluxes at spot-check surfaces and applies
multiplicative, additive, or full-replacement corrections interpolated
to the full radial grid.  Temporal smoothing prevents discontinuities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class CorrectionRecord:
    """Single spot-check comparison result."""

    rho_idx: int
    rho: float
    chi_i_surrogate: float
    chi_i_gk: float
    chi_e_surrogate: float
    chi_e_gk: float
    D_e_surrogate: float
    D_e_gk: float

    @property
    def rel_error_chi_i(self) -> float:
        if abs(self.chi_i_gk) < 1e-10:
            return 0.0
        return (self.chi_i_surrogate - self.chi_i_gk) / self.chi_i_gk

    @property
    def rel_error_chi_e(self) -> float:
        if abs(self.chi_e_gk) < 1e-10:
            return 0.0
        return (self.chi_e_surrogate - self.chi_e_gk) / self.chi_e_gk


@dataclass
class CorrectorConfig:
    """Correction layer parameters."""

    mode: str = "multiplicative"  # "multiplicative" / "additive" / "replace"
    smoothing_alpha: float = 0.3  # EMA weight for new correction factor
    replace_threshold: float = 1.0  # switch to full replacement if |error| > this


class GKCorrector:
    """Apply corrections to surrogate transport based on GK spot-checks."""

    def __init__(self, nr: int, config: CorrectorConfig | None = None) -> None:
        self.nr = nr
        self.config = config or CorrectorConfig()
        self._alpha_chi_i = np.ones(nr)  # multiplicative correction factors
        self._alpha_chi_e = np.ones(nr)
        self._alpha_D_e = np.ones(nr)
        self.history: list[list[CorrectionRecord]] = []

    def update(
        self,
        records: list[CorrectionRecord],
        rho: NDArray[np.float64],
    ) -> None:
        """Incorporate new spot-check results into correction factors.

        Interpolates correction factors from spot-check surfaces to the
        full radial grid with temporal EMA smoothing.
        """
        if not records:
            return

        self.history.append(records)
        alpha = self.config.smoothing_alpha

        spot_rho = np.array([r.rho for r in records])
        spot_alpha_i = np.array([r.chi_i_gk / max(abs(r.chi_i_surrogate), 1e-10) for r in records])
        spot_alpha_e = np.array([r.chi_e_gk / max(abs(r.chi_e_surrogate), 1e-10) for r in records])
        spot_alpha_d = np.array([r.D_e_gk / max(abs(r.D_e_surrogate), 1e-10) for r in records])

        # Interpolate to full grid
        new_alpha_i = np.interp(
            rho, spot_rho, spot_alpha_i, left=spot_alpha_i[0], right=spot_alpha_i[-1]
        )
        new_alpha_e = np.interp(
            rho, spot_rho, spot_alpha_e, left=spot_alpha_e[0], right=spot_alpha_e[-1]
        )
        new_alpha_d = np.interp(
            rho, spot_rho, spot_alpha_d, left=spot_alpha_d[0], right=spot_alpha_d[-1]
        )

        # Temporal EMA smoothing
        self._alpha_chi_i = alpha * new_alpha_i + (1.0 - alpha) * self._alpha_chi_i
        self._alpha_chi_e = alpha * new_alpha_e + (1.0 - alpha) * self._alpha_chi_e
        self._alpha_D_e = alpha * new_alpha_d + (1.0 - alpha) * self._alpha_D_e

    def correct(
        self,
        chi_i: NDArray[np.float64],
        chi_e: NDArray[np.float64],
        D_e: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Apply stored correction factors to surrogate profiles."""
        mode = self.config.mode

        if mode == "multiplicative":
            return (
                chi_i * self._alpha_chi_i,
                chi_e * self._alpha_chi_e,
                D_e * self._alpha_D_e,
            )
        elif mode == "additive":
            # Additive: chi_corrected = chi_surr + (alpha - 1) * chi_surr
            # Equivalent to multiplicative but expressed differently for clarity
            return (
                chi_i + chi_i * (self._alpha_chi_i - 1.0),
                chi_e + chi_e * (self._alpha_chi_e - 1.0),
                D_e + D_e * (self._alpha_D_e - 1.0),
            )
        elif mode == "replace":
            # Only replace where error exceeds threshold
            mask_i = np.abs(self._alpha_chi_i - 1.0) > self.config.replace_threshold
            mask_e = np.abs(self._alpha_chi_e - 1.0) > self.config.replace_threshold

            chi_i_out = chi_i.copy()
            chi_e_out = chi_e.copy()
            D_e_out = D_e.copy()

            chi_i_out[mask_i] = chi_i[mask_i] * self._alpha_chi_i[mask_i]
            chi_e_out[mask_e] = chi_e[mask_e] * self._alpha_chi_e[mask_e]

            return chi_i_out, chi_e_out, D_e_out

        return chi_i, chi_e, D_e

    @property
    def max_correction_factor(self) -> float:
        return float(
            np.max(
                np.abs(
                    np.concatenate(
                        [
                            self._alpha_chi_i - 1.0,
                            self._alpha_chi_e - 1.0,
                        ]
                    )
                )
            )
        )

    @property
    def mean_correction_factor(self) -> float:
        return float(
            np.mean(
                np.abs(
                    np.concatenate(
                        [
                            self._alpha_chi_i - 1.0,
                            self._alpha_chi_e - 1.0,
                        ]
                    )
                )
            )
        )
