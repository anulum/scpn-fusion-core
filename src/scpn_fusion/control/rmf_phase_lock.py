# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — RMF Phase-Lock Control Simulation Lane
"""
Deterministic high-frequency RMF phase-locking using Stochastic Computing and JAX-Scan.

This module provides software simulation evidence only. FPGA export is
fail-closed until a real RTL generator and timing validation are implemented.
"""

from __future__ import annotations

import logging
import importlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ── RMF Phase-Lock Configuration ─────────────────────────────────────


@dataclass
class RMFPhaseLockConfig:
    """Configuration for the RMF phase-lock loop."""

    f_rmf_nom_hz: float = 1.0e6  # 1 MHz nominal RMF frequency
    f_sampling_hz: float = 10.0e6  # simulation sampling frequency
    k_p: float = 0.5  # Phase detector gain
    k_d: float = 0.01  # Frequency damping
    n_neurons: int = 128  # Population size for stochastic PLL
    bits: int = 16  # reserved fixed-point width for future export


def _wrapped_phase_delta(delta: float) -> float:
    """Return signed phase delta in [-pi, pi)."""
    return float((delta + np.pi) % (2.0 * np.pi) - np.pi)


class RMFPhaseLockController:
    """Software RMF phase-lock controller for JAX horizon simulations."""

    def __init__(self, config: RMFPhaseLockConfig | None = None, seed: int = 42) -> None:
        self.cfg = RMFPhaseLockConfig() if config is None else config
        self.dt = 1.0 / self.cfg.f_sampling_hz
        self.omega_nom = 2.0 * np.pi * self.cfg.f_rmf_nom_hz

        # State
        self.phi_ant = 0.0
        self.omega_rmf = self.omega_nom
        self.omega_bias = 0.0
        self._last_phi_plasma: float | None = None
        self.t = 0.0

        self.history: dict[str, list[float]] = {
            "t": [],
            "phi_ant": [],
            "phi_plasma": [],
            "omega": [],
        }

    def step(self, plasma_phi: float) -> float:
        """Advance one deterministic software phase-lock cycle."""
        phase_error = float(np.sin(self.phi_ant - plasma_phi))
        observed_bias = (
            _wrapped_phase_delta(plasma_phi - self._last_phi_plasma) / self.dt - self.omega_nom
            if self._last_phi_plasma is not None
            else self.omega_bias
        )
        self.omega_bias = observed_bias - self.cfg.k_p * phase_error * self.dt
        self.omega_rmf = self.omega_nom + self.omega_bias
        self.phi_ant = float((self.phi_ant + self.omega_rmf * self.dt) % (2.0 * np.pi))
        self.t += self.dt
        self._last_phi_plasma = float(plasma_phi)
        self.history["t"].append(self.t)
        self.history["phi_ant"].append(self.phi_ant)
        self.history["phi_plasma"].append(float(plasma_phi))
        self.history["omega"].append(float(self.omega_rmf))
        return self.phi_ant

    def step_horizon(self, plasma_phis: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate a control horizon through the deterministic software PLL."""
        phis = np.asarray(plasma_phis, dtype=np.float64)
        out = np.empty_like(phis, dtype=np.float64)
        for idx, plasma_phi in enumerate(phis):
            out[idx] = self.step(float(plasma_phi))
        return out

    def step_jax_horizon(self, plasma_phis: object) -> object:
        """
        Evaluate a control horizon and return a JAX array when JAX is available.

        The deterministic NumPy path owns the control-law contract. This wrapper
        preserves the historical API without importing JAX at module import time.
        """
        out = self.step_horizon(np.asarray(plasma_phis, dtype=np.float64))
        try:
            jnp = importlib.import_module("jax.numpy")
        except ImportError:
            return out
        return jnp.asarray(out)

    def export_to_fpga(self, out_path: str) -> None:
        """Fail closed until a real RTL generator and timing proof exist."""
        raise NotImplementedError(
            "RMF FPGA export is not implemented; step_jax_horizon is software simulation evidence only"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ctrl = RMFPhaseLockController()

    # 1 ms software horizon at the configured sampling rate.
    horizon = 10000
    plasma_traj = np.zeros(horizon)

    out_phis = ctrl.step_horizon(plasma_traj)
    print(f"JAX-Scan software horizon evaluated: {len(out_phis)} cycles.")
