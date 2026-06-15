# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Neural Equilibrium Kernel Runtime
"""
Drop-in kernel wrapper around the neural equilibrium accelerator.

This module isolates runtime/orchestration concerns from training code and
keeps the runtime feature contract aligned with the trained 12-feature model.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from .fusion_kernel_numerics import FloatArray

from scpn_fusion.io.safe_loaders import checked_json_load

logger = logging.getLogger(__name__)


def _first_finite_number(
    *sections: Mapping[str, Any],
    keys: tuple[str, ...],
    default: float | None = None,
) -> float:
    for section in sections:
        for key in keys:
            value = section.get(key)
            if value is None:
                continue
            number = float(value)
            if np.isfinite(number):
                return number
            raise ValueError(f"Neural-equilibrium feature {key!r} must be finite.")
    if default is not None:
        return default
    raise KeyError(f"Missing required neural-equilibrium feature key among {keys!r}.")


class NeuralEquilibriumKernel:
    """
    Drop-in replacement for FusionKernel using the neural surrogate.

    Enables ~1000x faster control loops by bypassing the Picard iteration.
    Satisfies the attribute and method interface expected by
    IsoFluxController and TokamakFlightSim.
    """

    config_path: Path
    cfg: dict[str, Any]
    accel: Any
    NR: int
    NZ: int
    dR: float
    dZ: float
    R: NDArray[np.float64]
    Z: NDArray[np.float64]
    RR: NDArray[np.float64]
    ZZ: NDArray[np.float64]
    Psi: NDArray[np.float64]

    FEATURE_NAMES: tuple[str, ...] = (
        "I_p",
        "B_t",
        "R_axis",
        "Z_axis",
        "pprime_scale",
        "ffprime_scale",
        "simag",
        "sibry",
        "kappa",
        "delta_upper",
        "delta_lower",
        "q95",
    )

    def __init__(
        self,
        config_path: str | Path,
        weights_path: str | Path | None = None,
    ) -> None:
        # Delayed import avoids circular imports when re-exporting from
        # scpn_fusion.core.neural_equilibrium.
        from .neural_equilibrium import DEFAULT_WEIGHTS_PATH, NeuralEquilibriumAccelerator

        self.config_path = Path(config_path)
        self.cfg = checked_json_load(self.config_path)

        resolved_weights_path = (
            Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS_PATH
        )

        self.accel = NeuralEquilibriumAccelerator()
        if resolved_weights_path.exists():
            self.accel.load_weights(resolved_weights_path)
        else:
            logger.warning("Weights not found at %s. Prediction will fail.", resolved_weights_path)

        # Mirror dimensions
        dims = self.cfg["dimensions"]
        self.NR = self.accel.cfg.grid_shape[1]
        self.NZ = self.accel.cfg.grid_shape[0]
        self.dR = (dims["R_max"] - dims["R_min"]) / (self.NR - 1)
        self.dZ = (dims["Z_max"] - dims["Z_min"]) / (self.NZ - 1)

        self.R = np.linspace(dims["R_min"], dims["R_max"], self.NR)
        self.Z = np.linspace(dims["Z_min"], dims["Z_max"], self.NZ)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros(self.accel.cfg.grid_shape)

    def _feature_prior(self, name: str) -> float | None:
        input_mean = getattr(self.accel, "_input_mean", None)
        if input_mean is None:
            return None
        index = self.FEATURE_NAMES.index(name)
        if index >= len(input_mean):
            return None
        value = float(input_mean[index])
        return value if np.isfinite(value) else None

    def _feature(
        self,
        name: str,
        *sections: Mapping[str, Any],
        keys: tuple[str, ...],
        default: float | None = None,
    ) -> float:
        prior = self._feature_prior(name)
        return _first_finite_number(
            *sections,
            keys=keys,
            default=default if default is not None else prior,
        )

    def _build_feature_vector(self) -> NDArray[np.float64]:
        physics = self.cfg.get("physics", {})
        target = self.cfg.get("target", {})
        profiles = self.cfg.get("profiles", {})
        equilibrium = self.cfg.get("equilibrium", {})
        reference = self.cfg.get("_reference", {})

        total_ip_ma = _first_finite_number(
            physics,
            target,
            reference,
            keys=("plasma_current_target", "I_p", "Ip", "plasma_current_A"),
            default=sum(float(c.get("current", 0.0)) for c in self.cfg.get("coils", [])),
        )
        if abs(total_ip_ma) > 100_000.0:
            total_ip_ma /= 1e6

        delta_upper = self._feature(
            "delta_upper",
            physics,
            target,
            reference,
            keys=("delta_upper", "delta_up", "triangularity_upper", "delta"),
        )
        delta_lower = self._feature(
            "delta_lower",
            physics,
            target,
            reference,
            keys=("delta_lower", "delta_low", "triangularity_lower", "delta"),
        )

        return np.array(
            [
                total_ip_ma,
                self._feature(
                    "B_t",
                    physics,
                    target,
                    reference,
                    keys=("B_T", "B_t", "B0", "bcentr"),
                ),
                self._feature(
                    "R_axis",
                    target,
                    physics,
                    equilibrium,
                    reference,
                    keys=("R_axis", "rmaxis", "R_major_m", "major_radius"),
                ),
                self._feature(
                    "Z_axis",
                    target,
                    physics,
                    equilibrium,
                    keys=("Z_axis", "zmaxis"),
                    default=0.0,
                ),
                self._feature(
                    "pprime_scale",
                    physics,
                    target,
                    profiles,
                    keys=("pprime_scale", "pprime_s", "pressure_gradient_scale"),
                    default=1.0,
                ),
                self._feature(
                    "ffprime_scale",
                    physics,
                    target,
                    profiles,
                    keys=("ffprime_scale", "ffprime_s", "diamagnetic_scale"),
                    default=1.0,
                ),
                self._feature("simag", physics, target, equilibrium, keys=("simag", "psi_axis")),
                self._feature(
                    "sibry", physics, target, equilibrium, keys=("sibry", "psi_boundary")
                ),
                self._feature(
                    "kappa",
                    physics,
                    target,
                    reference,
                    keys=("kappa", "elongation"),
                ),
                delta_upper,
                delta_lower,
                self._feature(
                    "q95",
                    physics,
                    target,
                    reference,
                    keys=("q95", "q_95", "q95_typical"),
                ),
            ],
            dtype=np.float64,
        )

    def find_magnetic_axis(self) -> tuple[float, float, float]:
        """Find axis position (R, Z) and value Psi_axis with sub-grid precision."""
        # If the surrogate is too biased, we can mix in the target tracking.
        # But for validation, we want to see it respond.
        idx_max = np.argmax(self.Psi)
        iz, ir = np.unravel_index(idx_max, self.Psi.shape)

        # Simple quadratic interpolation if not on boundary.
        if 0 < iz < self.Psi.shape[0] - 1 and 0 < ir < self.Psi.shape[1] - 1:
            p1, p2, p3 = self.Psi[iz, ir - 1], self.Psi[iz, ir], self.Psi[iz, ir + 1]
            denom_r = p1 - 2 * p2 + p3
            dr_shift = 0.5 * (p1 - p3) / (denom_r if abs(denom_r) > 1e-12 else 1e-12)

            q1, q2, q3 = self.Psi[iz - 1, ir], self.Psi[iz, ir], self.Psi[iz + 1, ir]
            denom_z = q1 - 2 * q2 + q3
            dz_shift = 0.5 * (q1 - q3) / (denom_z if abs(denom_z) > 1e-12 else 1e-12)

            r_ax = self.R[ir] + dr_shift * (self.R[1] - self.R[0])
            z_ax = self.Z[iz] + dz_shift * (self.Z[1] - self.Z[0])

            psi_ax = p2 - 0.125 * (p3 - p1) ** 2 / (denom_r if abs(denom_r) > 1e-12 else 1e-12)
            return r_ax, z_ax, psi_ax

        return self.R[ir], self.Z[iz], self.Psi[iz, ir]

    def solve_equilibrium(self, **kwargs: Any) -> dict[str, Any]:
        """Predict Psi from the canonical 12-feature equilibrium descriptor."""
        t0 = time.perf_counter()
        features = self._build_feature_vector()

        self.Psi = self.accel.predict(features)

        return {
            "converged": True,
            "wall_time_s": time.perf_counter() - t0,
            "solver_method": "neural_surrogate",
        }

    def find_x_point(self, Psi: FloatArray) -> tuple[tuple[float, float], float]:
        """Locate X-point on predicted Psi."""
        dPsi_dz, dPsi_dr = np.gradient(Psi, self.Z, self.R)
        grad_mag = np.sqrt(dPsi_dz**2 + dPsi_dr**2)

        # Only look in the lower half for divertor.
        mask = (self.cfg["dimensions"]["Z_min"] * 0.2) > self.ZZ
        if not np.any(mask):
            mask = np.ones_like(Psi, dtype=bool)

        masked_grad = np.where(mask, grad_mag, 1e9)
        idx_min = np.argmin(masked_grad)
        iz, ir = np.unravel_index(idx_min, Psi.shape)

        return (self.R[ir], self.Z[iz]), Psi[iz, ir]
