# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Multi-Layer UPDE Engine (Paper 27)
"""Unified Phase Dynamics Equation — multi-layer evolution parameterised by the Knm coupling matrix from Paper 27.

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
from typing import Any, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.phase.knm import KnmSpec
from scpn_fusion.phase.kuramoto import (
    lyapunov_exponent,
    lyapunov_v,
    order_parameter,
    wrap_phase,
)

FloatArray = NDArray[np.float64]


def _upde_tick_numpy(
    theta_flat: FloatArray,
    omega_flat: FloatArray,
    offsets: NDArray[np.intp],
    K: FloatArray,
    alpha: FloatArray,
    zeta: FloatArray,
    *,
    dt: float,
    psi_global: float,
    actuation_gain: float = 1.0,
    pac_gamma: float = 0.0,
    wrap: bool = True,
) -> dict[str, Any]:
    """NumPy tier of the flat multi-layer UPDE tick kernel.

    Shares its contract with the Rust kernel (``fusion-phase``): layers are
    delimited by *offsets* inside the flat vectors, ``K``/``alpha`` are L×L
    (source row, target column), and *psi_global* is the already-resolved Ψ.
    """
    L = K.shape[0]
    g = float(actuation_gain)

    Rm = np.empty(L)
    Psim = np.empty(L)
    for m in range(L):
        Rm[m], Psim[m] = order_parameter(theta_flat[offsets[m] : offsets[m + 1]])
    R_global, Psi_r_global = order_parameter(theta_flat)

    theta1 = np.empty_like(theta_flat)
    dtheta = np.empty_like(theta_flat)
    for m in range(L):
        th = theta_flat[offsets[m] : offsets[m + 1]]
        om = omega_flat[offsets[m] : offsets[m + 1]]

        # Intra-layer: K_{mm} R_m sin(ψ_m − θ − α_{mm})
        dth = om + g * K[m, m] * Rm[m] * np.sin(Psim[m] - th - alpha[m, m])

        # Inter-layer: Σ_{n≠m} (1 + γ_pac (1 − R_n)) K_{nm} R_n sin(ψ_n − θ − α_{nm})
        for n in range(L):
            if n == m:
                continue
            pac_gate = 1.0 + pac_gamma * (1.0 - Rm[n])
            dth += g * pac_gate * K[n, m] * Rm[n] * np.sin(Psim[n] - th - alpha[n, m])

        # Global driver: ζ_m sin(Ψ − θ)
        if zeta[m] != 0.0:
            dth += zeta[m] * np.sin(psi_global - th)

        th_next = th + dt * dth
        if wrap:
            th_next = wrap_phase(th_next)

        theta1[offsets[m] : offsets[m + 1]] = th_next
        dtheta[offsets[m] : offsets[m + 1]] = dth

    V_layer = np.array(
        [lyapunov_v(theta1[offsets[m] : offsets[m + 1]], psi_global) for m in range(L)]
    )
    V_global = lyapunov_v(theta1, psi_global)

    return {
        "theta1": theta1,
        "dtheta": dtheta,
        "R_layer": Rm,
        "Psi_layer": Psim,
        "R_global": R_global,
        "Psi_r_global": Psi_r_global,
        "V_layer": V_layer,
        "V_global": V_global,
    }


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
    ) -> dict[str, Any]:
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

        # Flatten layers (supports non-uniform per-layer N) and execute the
        # tick on the fastest available dispatcher tier (Rust fusion-phase
        # when built, NumPy floor always). Tiers agree to floating-point
        # summation order (~1e-14 relative).
        from scpn_fusion.core._multi_compat import dispatch

        theta_flat = np.concatenate([np.asarray(t, dtype=np.float64).ravel() for t in theta_layers])
        omega_flat = np.concatenate([np.asarray(o, dtype=np.float64).ravel() for o in omega_layers])
        offsets = np.zeros(L + 1, dtype=np.intp)
        np.cumsum([np.asarray(t).ravel().size for t in theta_layers], out=offsets[1:])

        tick = dispatch("upde_tick")
        out = tick(
            theta_flat,
            omega_flat,
            offsets,
            K,
            alpha,
            zeta,
            dt=self.dt,
            psi_global=Psi_global,
            actuation_gain=g,
            pac_gamma=pac_gamma,
            wrap=self.wrap,
        )

        theta1_flat = np.asarray(out["theta1"], dtype=np.float64)
        dtheta_flat = np.asarray(out["dtheta"], dtype=np.float64)
        theta1 = [theta1_flat[offsets[m] : offsets[m + 1]] for m in range(L)]
        dtheta_all = [dtheta_flat[offsets[m] : offsets[m + 1]] for m in range(L)]

        return {
            "theta1": theta1,
            "dtheta": dtheta_all,
            "R_layer": np.asarray(out["R_layer"], dtype=np.float64),
            "Psi_layer": np.asarray(out["Psi_layer"], dtype=np.float64),
            "R_global": float(out["R_global"]),
            "Psi_global": Psi_global,
            "V_layer": np.asarray(out["V_layer"], dtype=np.float64),
            "V_global": float(out["V_global"]),
        }

    def _run_batched(
        self,
        n_steps: int,
        theta_layers: Sequence[FloatArray],
        omega_layers: Sequence[FloatArray],
        *,
        psi_driver: float,
        actuation_gain: float,
        pac_gamma: float,
        K_override: FloatArray | None,
    ) -> dict[str, Any]:
        """Run the constant-driver loop on the batched ``upde_run`` kernel."""
        from scpn_fusion.core._multi_compat import dispatch

        K = np.asarray(K_override if K_override is not None else self.spec.K, dtype=np.float64)
        L = K.shape[0]
        if len(theta_layers) != L or len(omega_layers) != L:
            raise ValueError(f"Expected {L} layers, got {len(theta_layers)}")
        alpha = (
            np.zeros_like(K)
            if self.spec.alpha is None
            else np.asarray(self.spec.alpha, dtype=np.float64)
        )
        zeta = (
            np.zeros(L) if self.spec.zeta is None else np.asarray(self.spec.zeta, dtype=np.float64)
        )
        theta_flat = np.concatenate([np.asarray(t, dtype=np.float64).ravel() for t in theta_layers])
        omega_flat = np.concatenate([np.asarray(o, dtype=np.float64).ravel() for o in omega_layers])
        offsets = np.zeros(L + 1, dtype=np.intp)
        np.cumsum([np.asarray(t).ravel().size for t in theta_layers], out=offsets[1:])

        run = dispatch("upde_run")
        out = run(
            theta_flat,
            omega_flat,
            offsets,
            K,
            alpha,
            zeta,
            n_steps=n_steps,
            dt=self.dt,
            psi_global=float(psi_driver),
            actuation_gain=actuation_gain,
            pac_gamma=pac_gamma,
            wrap=self.wrap,
        )
        theta_final_flat = np.asarray(out["theta_final"], dtype=np.float64)
        out["theta_final"] = [theta_final_flat[offsets[m] : offsets[m + 1]] for m in range(L)]
        return cast("dict[str, Any]", out)

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
    ) -> dict[str, Any]:
        """Run n_steps and return trajectory of per-layer R and global R.

        With ``psi_mode="external"`` (constant driver) the whole loop runs on
        the batched ``upde_run`` dispatcher kernel — one boundary crossing
        instead of one per tick. The mean-field mode keeps the per-step path
        because Ψ then depends on the evolving state.
        """
        if self.psi_mode == "external":
            if psi_driver is None:
                raise ValueError("psi_driver required when psi_mode='external'")
            out = self._run_batched(
                n_steps,
                theta_layers,
                omega_layers,
                psi_driver=float(psi_driver),
                actuation_gain=actuation_gain,
                pac_gamma=pac_gamma,
                K_override=K_override,
            )
            return {
                "theta_final": out["theta_final"],
                "R_layer_hist": out["R_layer_hist"],
                "R_global_hist": out["R_global_hist"],
            }

        R_layer_hist = []
        R_global_hist = []
        current = [np.asarray(t, dtype=np.float64).copy() for t in theta_layers]

        for _ in range(n_steps):
            out_step = self.step(
                current,
                omega_layers,
                psi_driver=psi_driver,
                actuation_gain=actuation_gain,
                pac_gamma=pac_gamma,
                K_override=K_override,
            )
            current = out_step["theta1"]
            R_layer_hist.append(out_step["R_layer"].copy())
            R_global_hist.append(out_step["R_global"])

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
    ) -> dict[str, Any]:
        """Run n_steps with Lyapunov tracking.

        Returns R histories, V histories, and per-layer + global λ.
        λ < 0 ⟹ stable convergence toward Ψ.
        """
        if self.psi_mode == "external":
            if psi_driver is None:
                raise ValueError("psi_driver required when psi_mode='external'")
            out_run = self._run_batched(
                n_steps,
                theta_layers,
                omega_layers,
                psi_driver=float(psi_driver),
                actuation_gain=actuation_gain,
                pac_gamma=pac_gamma,
                K_override=K_override,
            )
            current = out_run["theta_final"]
            R_layer_arr = np.asarray(out_run["R_layer_hist"], dtype=np.float64)
            R_global_arr = np.asarray(out_run["R_global_hist"], dtype=np.float64)
            V_layer_arr = np.asarray(out_run["V_layer_hist"], dtype=np.float64)
            V_global_arr = np.asarray(out_run["V_global_hist"], dtype=np.float64)
        else:
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

            R_layer_arr = np.array(R_layer_hist)
            R_global_arr = np.array(R_global_hist)
            V_layer_arr = np.array(V_layer_hist)  # (n_steps, L)
            V_global_arr = np.array(V_global_hist)  # (n_steps,)

        L = V_layer_arr.shape[1]
        lambda_layer = np.array(
            [lyapunov_exponent(V_layer_arr[:, m].tolist(), self.dt) for m in range(L)]
        )
        lambda_global = lyapunov_exponent(V_global_arr.tolist(), self.dt)

        return {
            "theta_final": current,
            "R_layer_hist": R_layer_arr,
            "R_global_hist": R_global_arr,
            "V_layer_hist": V_layer_arr,
            "V_global_hist": V_global_arr,
            "lambda_layer": lambda_layer,
            "lambda_global": lambda_global,
        }
