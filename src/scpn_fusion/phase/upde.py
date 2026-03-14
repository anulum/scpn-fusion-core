# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Multi-Layer UPDE Engine (Paper 27)
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Unified Phase Dynamics Equation — multi-layer evolution parameterised
by the Knm coupling matrix from Paper 27.

Per-layer equation:
    dθ_{m,i}/dt = ω_{m,i}
                + K_{mm} · R_m · sin(ψ_m − θ_{m,i} − α_{mm})
                + Σ_{n≠m} K_{nm} · R_n · sin(ψ_n − θ_{m,i} − α_{nm})
                + ζ_m · sin(Ψ − θ_{m,i})

K_{mm} (diagonal):     intra-layer synchronisation
K_{nm} (off-diagonal): inter-layer bidirectional causality
ζ_m sin(Ψ − θ):       global field driver (reviewer request)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.phase.knm import KnmSpec
from scpn_fusion.phase.kuramoto import (
    lyapunov_exponent,
    lyapunov_v,
    order_parameter,
    wrap_phase,
)

# Try to import Rust UPDE fast-path
try:
    from scpn_fusion_rs import upde_tick as _rust_upde_tick  # pragma: no cover

    HAS_RUST_UPDE = True  # pragma: no cover
except ImportError:
    HAS_RUST_UPDE = False

FloatArray = NDArray[np.float64]


@dataclass
class UPDESystem:
    """Multi-layer UPDE driven by a KnmSpec."""

    spec: KnmSpec
    dt: float = 1e-3
    psi_mode: str = "external"
    wrap: bool = True

    def step(
        self,
        theta_layers: Sequence[FloatArray],
        omega_layers: Sequence[FloatArray],
        *,
        psi_driver: float | None = None,
        actuation_gain: float = 1.0,
        pac_gamma: float = 0.0,
        K_override: FloatArray | None = None,
    ) -> dict:
        """Advance all L layers by one Euler step.

        Parameters
        ----------
        theta_layers : sequence of 1D arrays
            Phase vectors per layer.
        omega_layers : sequence of 1D arrays
            Natural frequencies per layer.
        psi_driver : float or None
            External global field phase Ψ (required if psi_mode="external").
        actuation_gain : float
            Multiplicative gain on all coupling terms.
        pac_gamma : float
            PAC-like gating: boost inter-layer coupling by
            (1 + pac_gamma·(1 − R_source)).
        K_override : array or None
            Per-tick replacement for spec.K (adaptive coupling).
        """
        K = np.asarray(K_override if K_override is not None else self.spec.K, dtype=np.float64)
        L = K.shape[0]
        if len(theta_layers) != L or len(omega_layers) != L:
            raise ValueError(f"Expected {L} layers, got {len(theta_layers)}")

        g = float(actuation_gain)

        # Per-layer order parameters
        Rm = np.empty(L)
        Psim = np.empty(L)
        for m in range(L):
            Rm[m], Psim[m] = order_parameter(theta_layers[m])

        # Resolve global Ψ
        if self.psi_mode == "external":
            if psi_driver is None:
                raise ValueError("psi_driver required when psi_mode='external'")
            Psi_global = float(psi_driver)
        elif self.psi_mode == "global_mean_field":
            z = np.sum(Rm * np.exp(1j * Psim))
            Psi_global = float(np.angle(z))
        else:
            raise ValueError(f"Unknown psi_mode: {self.psi_mode}")
        alpha = (
            np.zeros_like(K)
            if self.spec.alpha is None
            else np.asarray(self.spec.alpha, dtype=np.float64)
        )
        zeta = (
            np.zeros(L) if self.spec.zeta is None else np.asarray(self.spec.zeta, dtype=np.float64)
        )

        # Attempt Rust fast-path if L > 1 and all layers have same N
        if HAS_RUST_UPDE and L > 0:
            n_per = len(theta_layers[0])
            if all(len(t) == n_per for t in theta_layers):
                theta_flat = np.concatenate(theta_layers).astype(np.float64)
                omega_flat = np.concatenate(omega_layers).astype(np.float64)

                res = _rust_upde_tick(
                    theta_flat,
                    omega_flat,
                    K.ravel() * g,
                    alpha.ravel(),
                    zeta,
                    L,
                    n_per,
                    self.dt,
                    Psi_global,
                    float(pac_gamma),
                )

                # Reshape theta1 back to layers
                theta1_flat = np.asarray(res.theta_flat)
                theta1_rust = [theta1_flat[m * n_per : (m + 1) * n_per] for m in range(L)]

                return {
                    "theta1": theta1_rust,
                    "R_layer": np.asarray(res.r_layer),
                    "Psi_layer": np.asarray(res.v_layer),  # Rust v_layer is psi_layer
                    "R_global": res.r_global,
                    "Psi_global": Psi_global,
                    "V_layer": np.array([lyapunov_v(theta1_rust[m], Psi_global) for m in range(L)]),
                    "V_global": lyapunov_v(theta1_flat, Psi_global),
                }

        # Python fallback (supports non-uniform N)
        theta1: list[FloatArray] = []
        dtheta_all: list[FloatArray] = []

        for m in range(L):
            th = np.asarray(theta_layers[m], dtype=np.float64).ravel()
            om = np.asarray(omega_layers[m], dtype=np.float64).ravel()

            # Intra-layer: K_{mm} R_m sin(ψ_m − θ − α_{mm})
            dth = om + g * K[m, m] * Rm[m] * np.sin(Psim[m] - th - alpha[m, m])

            # Inter-layer: Σ_{n≠m} K_{nm} R_n sin(ψ_n − θ − α_{nm})
            for n in range(L):
                if n == m:
                    continue
                pac_gate = 1.0 + pac_gamma * (1.0 - Rm[n])
                dth += g * pac_gate * K[n, m] * Rm[n] * np.sin(Psim[n] - th - alpha[n, m])

            # Global driver: ζ_m sin(Ψ − θ)
            if zeta[m] != 0.0:
                dth += zeta[m] * np.sin(Psi_global - th)

            th_next = th + self.dt * dth
            if self.wrap:
                th_next = wrap_phase(th_next)

            theta1.append(th_next)
            dtheta_all.append(dth)

        R_global, Psi_r_global = order_parameter(
            np.concatenate([np.asarray(t).ravel() for t in theta_layers])
        )

        # Per-layer Lyapunov V_m(t) = (1/N_m) Σ (1 − cos(θ_{m,i} − Ψ))
        V_layer = np.array([lyapunov_v(theta1[m], Psi_global) for m in range(L)])
        V_global = lyapunov_v(
            np.concatenate([np.asarray(t).ravel() for t in theta1]),
            Psi_global,
        )

        return {
            "theta1": theta1,
            "dtheta": dtheta_all,
            "R_layer": Rm.copy(),
            "Psi_layer": Psim.copy(),
            "R_global": R_global,
            "Psi_global": Psi_global,
            "V_layer": V_layer,
            "V_global": V_global,
        }

    def run(
        self,
        n_steps: int,
        theta_layers: Sequence[FloatArray],
        omega_layers: Sequence[FloatArray],
        *,
        psi_driver: float | None = None,
        actuation_gain: float = 1.0,
        pac_gamma: float = 0.0,
        K_override: FloatArray | None = None,
    ) -> dict:
        """Run n_steps and return trajectory of per-layer R and global R."""
        R_layer_hist = []
        R_global_hist = []
        current = [np.asarray(t, dtype=np.float64).copy() for t in theta_layers]

        for _ in range(n_steps):
            out = self.step(
                current,
                omega_layers,
                psi_driver=psi_driver,
                actuation_gain=actuation_gain,
                pac_gamma=pac_gamma,
                K_override=K_override,
            )
            current = out["theta1"]
            R_layer_hist.append(out["R_layer"].copy())
            R_global_hist.append(out["R_global"])

        return {
            "theta_final": current,
            "R_layer_hist": np.array(R_layer_hist),
            "R_global_hist": np.array(R_global_hist),
        }

    def run_lyapunov(
        self,
        n_steps: int,
        theta_layers: Sequence[FloatArray],
        omega_layers: Sequence[FloatArray],
        *,
        psi_driver: float | None = None,
        actuation_gain: float = 1.0,
        pac_gamma: float = 0.0,
        K_override: FloatArray | None = None,
    ) -> dict:
        """Run n_steps with Lyapunov tracking.

        Returns R histories, V histories, and per-layer + global λ.
        λ < 0 ⟹ stable convergence toward Ψ.
        """
        R_layer_hist = []
        R_global_hist = []
        V_layer_hist = []
        V_global_hist = []
        current = [np.asarray(t, dtype=np.float64).copy() for t in theta_layers]

        for _ in range(n_steps):
            out = self.step(
                current,
                omega_layers,
                psi_driver=psi_driver,
                actuation_gain=actuation_gain,
                pac_gamma=pac_gamma,
                K_override=K_override,
            )
            current = out["theta1"]
            R_layer_hist.append(out["R_layer"].copy())
            R_global_hist.append(out["R_global"])
            V_layer_hist.append(out["V_layer"].copy())
            V_global_hist.append(out["V_global"])

        V_layer_arr = np.array(V_layer_hist)  # (n_steps, L)
        V_global_arr = np.array(V_global_hist)  # (n_steps,)

        L = V_layer_arr.shape[1]
        lambda_layer = np.array(
            [lyapunov_exponent(V_layer_arr[:, m].tolist(), self.dt) for m in range(L)]
        )
        lambda_global = lyapunov_exponent(V_global_arr.tolist(), self.dt)

        return {
            "theta_final": current,
            "R_layer_hist": np.array(R_layer_hist),
            "R_global_hist": np.array(R_global_hist),
            "V_layer_hist": V_layer_arr,
            "V_global_hist": V_global_arr,
            "lambda_layer": lambda_layer,
            "lambda_global": lambda_global,
        }
