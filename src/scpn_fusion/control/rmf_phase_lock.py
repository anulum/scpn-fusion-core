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
from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp
import numpy as np
from jax import jit, lax

logger = logging.getLogger(__name__)

# ── RMF Phase-Lock Configuration ─────────────────────────────────────

@dataclass
class RMFPhaseLockConfig:
    """Configuration for the RMF phase-lock loop."""
    f_rmf_nom_hz: float = 1.0e6  # 1 MHz nominal RMF frequency
    f_sampling_hz: float = 10.0e6 # simulation sampling frequency
    k_p: float = 0.5             # Phase detector gain
    k_d: float = 0.01            # Frequency damping
    n_neurons: int = 128         # Population size for stochastic PLL
    bits: int = 16               # reserved fixed-point width for future export


# ── JAX-Scan Accelerated Lane (Fifth Lane) ───────────────────────────

@jit
def _rmf_pll_scan_step(state: Tuple[float, ...], plasma_phi: float) -> Tuple[Tuple[float, ...], float]:
    """A single JIT-compiled cycle of the phase-lock loop (State-Space form)."""
    # Unpack state: (phi_ant, omega, t, dt, k_p, k_d, omega_nom)
    phi_ant, omega, t, dt, k_p, k_d, omega_nom = state

    # 1. Phase Detector (sin correlation)
    error = jnp.sin(phi_ant - plasma_phi)

    # 2. Control Law (Proportional-Derivative)
    d_omega = -k_p * error - k_d * (omega - omega_nom)
    new_omega = omega + d_omega * dt

    # 3. Phase Integration
    new_phi = (phi_ant + new_omega * dt) % (2.0 * jnp.pi)

    new_state = (new_phi, new_omega, t + dt, dt, k_p, k_d, omega_nom)
    return new_state, new_phi


class RMFPhaseLockController:
    """
    Software RMF phase-lock controller for JAX horizon simulations.
    """

    def __init__(
        self,
        config: RMFPhaseLockConfig | None = None,
        seed: int = 42
    ) -> None:
        self.cfg = RMFPhaseLockConfig() if config is None else config
        self.dt = 1.0 / self.cfg.f_sampling_hz
        self.omega_nom = 2.0 * np.pi * self.cfg.f_rmf_nom_hz

        # State
        self.phi_ant = 0.0
        self.omega_rmf = self.omega_nom
        self.t = 0.0

        self.history: dict[str, list[float]] = {
            "t": [],
            "phi_ant": [],
            "phi_plasma": [],
            "omega": []
        }

    def step_jax_horizon(self, plasma_phis: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate a complete control horizon using JAX-Scan.
        """
        init_state = (
            float(self.phi_ant),
            float(self.omega_rmf),
            float(self.t),
            float(self.dt),
            float(self.cfg.k_p),
            float(self.cfg.k_d),
            float(self.omega_nom)
        )

        final_state, phi_history = lax.scan(_rmf_pll_scan_step, init_state, plasma_phis)

        # Update state from horizon end
        self.phi_ant, self.omega_rmf, self.t = final_state[0:3]

        return phi_history

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
    plasma_traj = jnp.zeros(horizon) # static for test

    out_phis = ctrl.step_jax_horizon(plasma_traj)
    print(f"JAX-Scan software horizon evaluated: {len(out_phis)} cycles.")
