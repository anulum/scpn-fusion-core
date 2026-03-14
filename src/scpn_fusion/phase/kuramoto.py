# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Kuramoto-Sakaguchi + Global Field Driver
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Mean-field Kuramoto-Sakaguchi with exogenous global driver.

Equation:
    dθ_i/dt = ω_i + K·R·sin(ψ_r − θ_i − α) + ζ·sin(Ψ − θ_i)

The ζ sin(Ψ−θ) term implements the reviewer's requested "intention as
carrier" injection.  Ψ is a Lagrangian pull parameter with no own
dynamics (no dotΨ equation) — it is resolved either from an external
value or from the mean-field phase.

Reference: arXiv:2004.06344 (generalized Kuramoto-Sakaguchi finite-size)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]

# Rust fast-path (sub-ms for N > 1000)
try:
    from scpn_fusion_rs import kuramoto_step as _rust_step  # pragma: no cover

    RUST_KURAMOTO = True  # pragma: no cover
except ImportError:
    RUST_KURAMOTO = False


def wrap_phase(x: FloatArray) -> FloatArray:
    """Map phases to (-π, π]."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def order_parameter(
    theta: FloatArray,
    weights: FloatArray | None = None,
) -> tuple[float, float]:
    """Kuramoto order parameter R·exp(i·ψ_r) = <w·exp(i·θ)> / W.

    Returns (R, ψ_r).
    """
    th = np.asarray(theta, dtype=np.float64).ravel()

    if weights is None:
        z = np.mean(np.exp(1j * th))
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        W = float(np.sum(w))
        z = np.sum(w * np.exp(1j * th)) / max(W, 1e-15)

    return float(np.abs(z)), float(np.angle(z))


@dataclass(frozen=True)
class GlobalPsiDriver:
    """Resolve the global field phase Ψ.

    mode="external"    : Ψ supplied by caller (intention/carrier, no dotΨ).
    mode="mean_field"  : Ψ = arg(<exp(iθ)>) from the oscillator population.
    """

    mode: str = "external"

    def resolve(self, theta: FloatArray, psi_external: float | None) -> float:
        if self.mode == "external":
            if psi_external is None:
                raise ValueError("psi_external required when mode='external'")
            return float(psi_external)
        if self.mode == "mean_field":
            _, psi = order_parameter(theta)
            return psi
        raise ValueError(f"Unknown mode: {self.mode}")


def lyapunov_v(theta: FloatArray, psi: float) -> float:
    """Lyapunov candidate V(t) = (1/N) Σ (1 − cos(θ_i − Ψ)).

    V=0 at perfect sync (all θ_i = Ψ), V=2 at maximal desync.
    Range: [0, 2].  Mirror of control-math/kuramoto.rs::lyapunov_v.
    """
    th = np.asarray(theta, dtype=np.float64).ravel()
    if th.size == 0:
        return 0.0
    return float(np.mean(1.0 - np.cos(th - psi)))


def lyapunov_exponent(v_hist: Sequence[float], dt: float) -> float:
    """λ = (1/T) · ln(V_final / V_initial).  λ < 0 ⟹ stable."""
    if len(v_hist) < 2:
        return 0.0
    v0 = max(v_hist[0], 1e-15)
    vf = max(v_hist[-1], 1e-15)
    T = len(v_hist) * dt
    return float(np.log(vf / v0) / T)


def kuramoto_sakaguchi_step(
    theta: FloatArray,
    omega: FloatArray,
    *,
    dt: float,
    K: float,
    alpha: float = 0.0,
    zeta: float = 0.0,
    psi_driver: float | None = None,
    psi_mode: str = "external",
    wrap: bool = True,
) -> dict:
    """Single Euler step of mean-field Kuramoto-Sakaguchi + global driver.

    dθ_i/dt = ω_i + K·R·sin(ψ_r − θ_i − α) + ζ·sin(Ψ − θ_i)

    Uses Rust backend when available (rayon-parallelised, sub-ms for N>1000).
    """
    th = np.asarray(theta, dtype=np.float64).ravel()
    om = np.asarray(omega, dtype=np.float64).ravel()

    # Resolve Ψ before dispatching (Rust kernel needs resolved value)
    Psi = GlobalPsiDriver(mode=psi_mode).resolve(th, psi_driver)

    if RUST_KURAMOTO and wrap and alpha == 0.0:
        th1, R, psi_r, psi_g = _rust_step(
            th,
            om,
            dt,
            K,
            0.0,
            zeta,
            Psi,
        )
        return {
            "theta1": th1,
            "dtheta": th1 - th,  # approximate (post-wrap)
            "R": R,
            "Psi_r": psi_r,
            "Psi": psi_g,
        }

    R, psi_r = order_parameter(th)

    dtheta = om + (K * R) * np.sin(psi_r - th - alpha)
    if zeta != 0.0:
        dtheta += zeta * np.sin(Psi - th)

    th1 = th + dt * dtheta
    if wrap:
        th1 = wrap_phase(th1)

    return {
        "theta1": th1,
        "dtheta": dtheta,
        "R": R,
        "Psi_r": psi_r,
        "Psi": Psi,
    }
