# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — TEMHD Peltier
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


class TEMHD_Stabilizer:
    """
    Implicit 1D heat solver for TEMHD-stabilized liquid-metal divertors.
    """

    def __init__(self, layer_thickness_mm: float = 5.0, B_field: float = 10.0):
        self.L = float(layer_thickness_mm) / 1000.0
        self.B0 = float(B_field)
        self.N = 50
        self.z = np.linspace(0, self.L, self.N)
        self.dz = float(self.z[1] - self.z[0])
        self.rho = 500.0
        self.cp = 4200.0
        self.k_thermal = 50.0
        self.Seebeck = 20e-6
        self.sigma = 3e6
        self.viscosity = 1e-3
        self.T = np.ones(self.N, dtype=float) * 300.0
        self.T_wall = 300.0

    def solve_tridiagonal(self, a: object, b: object, c: object, d: object) -> NDArray[np.float64]:
        """Solve tridiagonal system Ax=d via Thomas algorithm."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        c = np.asarray(c, dtype=float)
        d = np.asarray(d, dtype=float)

        n = int(d.size)
        if b.size != n:
            raise ValueError(f"b length {b.size} must equal d length {n}")
        if n == 0:
            return np.array([], dtype=float)
        if a.size != max(n - 1, 0) or c.size != max(n - 1, 0):
            raise ValueError(
                f"Invalid tridiagonal sizes: len(a)={a.size}, len(b)={b.size}, "
                f"len(c)={c.size}, len(d)={n}"
            )

        if n == 1:
            if abs(b[0]) < 1e-14:
                raise ValueError("Singular diagonal encountered in tridiagonal solve.")
            return np.array([d[0] / b[0]], dtype=float)

        c_prime = np.zeros(n - 1, dtype=float)
        d_prime = np.zeros(n, dtype=float)

        den = b[0]
        if abs(den) < 1e-14:
            raise ValueError("Singular diagonal encountered in tridiagonal solve.")
        c_prime[0] = c[0] / den
        d_prime[0] = d[0] / den

        for i in range(1, n):
            den = b[i] - a[i - 1] * c_prime[i - 1]
            if abs(den) < 1e-14:
                raise ValueError("Singular diagonal encountered in tridiagonal solve.")
            if i < n - 1:
                c_prime[i] = c[i] / den
            d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / den

        res = np.zeros(n, dtype=float)
        res[-1] = d_prime[-1]
        for i in range(n - 2, -1, -1):
            res[i] = d_prime[i] - c_prime[i] * res[i + 1]
        return res

    def step(self, heat_flux_MW_m2: float, dt: float = 0.1) -> tuple[float, float]:
        dt = float(dt)
        heat_flux_MW_m2 = float(heat_flux_MW_m2)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a finite positive value.")
        if not np.isfinite(heat_flux_MW_m2) or heat_flux_MW_m2 < 0.0:
            raise ValueError("heat_flux_MW_m2 must be a finite non-negative value.")
        if not np.isfinite(self.dz) or self.dz <= 0.0:
            raise ValueError("Invalid grid spacing dz in TEMHD solver.")
        if not np.all(np.isfinite(self.T)):
            raise ValueError("Temperature state contains non-finite values.")

        grad_T = np.gradient(self.T, self.dz)
        J_te = -self.sigma * self.Seebeck * grad_T
        F_lorentz = np.abs(J_te * self.B0)
        v_conv = (F_lorentz * self.dz**2) / (self.viscosity + 1e-9)
        alpha = self.k_thermal / (self.rho * self.cp)
        Pe = np.clip(v_conv * self.dz / alpha, 0, 200.0)
        k_eff = np.maximum(self.k_thermal * (1.0 + 0.2 * Pe), 1e-9)

        r = (k_eff * dt) / (self.rho * self.cp * self.dz**2)
        if not np.all(np.isfinite(r)):
            raise ValueError("Non-finite diffusion coefficients encountered.")
        b = 1.0 + 2.0 * r[1:]
        a = -r[2:]
        c = -r[1:-1]
        d = self.T[1:].copy()

        d[0] += r[1] * self.T_wall
        b[-1] = 1.0 + r[-1]
        q_in = heat_flux_MW_m2 * 1e6
        d[-1] += r[-1] * (q_in * self.dz / k_eff[-1])

        self.T[1:] = self.solve_tridiagonal(a, b, c, d)
        return float(self.T[-1]), float(np.max(k_eff))


def run_temhd_experiment(
    *,
    layer_thickness_mm: float = 5.0,
    B_field: float = 10.0,
    flux_min_MW_m2: float = 0.0,
    flux_max_MW_m2: float = 100.0,
    flux_points: int = 20,
    settle_steps_per_flux: int = 20,
    dt_s: float = 0.5,
    save_plot: bool = True,
    output_path: str = "TEMHD_Corrected.png",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run deterministic TEMHD flux-ramp experiment and return summary metrics.
    """
    f_lo = float(flux_min_MW_m2)
    f_hi = float(flux_max_MW_m2)
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo >= f_hi:
        raise ValueError(
            "flux_min_MW_m2/flux_max_MW_m2 must be finite with flux_min_MW_m2 < flux_max_MW_m2."
        )
    n_flux = int(flux_points)
    if n_flux < 2:
        raise ValueError("flux_points must be >= 2.")
    settle_steps = int(settle_steps_per_flux)
    if settle_steps < 1:
        raise ValueError("settle_steps_per_flux must be >= 1.")
    dt = float(dt_s)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and > 0.")

    sim = TEMHD_Stabilizer(layer_thickness_mm=layer_thickness_mm, B_field=B_field)
    flux_ramp = np.linspace(f_lo, f_hi, n_flux, dtype=np.float64)

    res_T: list[float] = []
    res_k: list[float] = []
    if verbose:
        print(f"{'Flux':<10} | {'T_surf':<10}")
    cadence = max(n_flux // 5, 1)
    for i, q in enumerate(flux_ramp):
        t_surf = 0.0
        k_eff = 0.0
        for _ in range(settle_steps):
            t_surf, k_eff = sim.step(float(q), dt=dt)
        res_T.append(float(t_surf))
        res_k.append(float(k_eff))
        if verbose and (i % cadence == 0 or i == n_flux - 1):
            print(f"{float(q):<10.1f} | {float(t_surf):<10.1f}")

    t_arr = np.asarray(res_T, dtype=np.float64)
    k_arr = np.asarray(res_k, dtype=np.float64)

    plot_saved = False
    plot_error: Optional[str] = None
    if save_plot:
        try:
            fig, ax = plt.subplots()
            ax.plot(flux_ramp, t_arr, "r-")
            ax.axhline(1342.0, color="k", ls="--")
            fig.savefig(output_path)
            plt.close(fig)
            plot_saved = True
            if verbose:
                print(f"Saved: {output_path}")
        except Exception as exc:
            plot_error = f"{exc.__class__.__name__}: {exc}"

    return {
        "layer_thickness_mm": float(layer_thickness_mm),
        "B_field": float(B_field),
        "flux_min_MW_m2": float(f_lo),
        "flux_max_MW_m2": float(f_hi),
        "flux_points": int(n_flux),
        "settle_steps_per_flux": int(settle_steps),
        "dt_s": float(dt),
        "min_surface_temp_K": float(np.min(t_arr)) if t_arr.size else 0.0,
        "max_surface_temp_K": float(np.max(t_arr)) if t_arr.size else 0.0,
        "max_k_eff": float(np.max(k_arr)) if k_arr.size else 0.0,
        "plot_saved": bool(plot_saved),
        "plot_error": plot_error,
    }


if __name__ == "__main__":
    run_temhd_experiment()
