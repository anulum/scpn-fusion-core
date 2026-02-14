# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — PWI Erosion
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np


class SputteringPhysics:
    """
    Simulates plasma-wall interaction sputtering and macroscopic erosion.
    """

    def __init__(self, material: str = "Tungsten", redeposition_factor: float = 0.95):
        self.material = str(material)
        if self.material == "Tungsten":
            self.E_th = 200.0
            self.Q = 0.03
            self.atomic_mass = 183.84
            self.density = 19.25
        else:
            self.E_th = 30.0
            self.Q = 0.1
            self.atomic_mass = 12.0
            self.density = 2.2
        self.redeposition_factor = float(np.clip(redeposition_factor, 0.0, 0.999))

    def calculate_yield(self, E_ion_eV: float, angle_deg: float = 45.0) -> float:
        """
        Calculate sputtering yield (ejected atoms / incident ion).
        """
        E = float(E_ion_eV)
        if not np.isfinite(E):
            raise ValueError("E_ion_eV must be finite.")
        if E <= self.E_th:
            return 0.0

        eps = E / self.E_th
        eth_ratio = self.E_th / E
        s_n = np.log1p(1.2288 * eps) / (1.0 + np.sqrt(eps))
        threshold_term = (1.0 - eth_ratio ** (2.0 / 3.0)) * (1.0 - eth_ratio) ** 2
        threshold_term = max(0.0, threshold_term)

        ang = float(angle_deg)
        if not np.isfinite(ang):
            raise ValueError("angle_deg must be finite.")
        theta = np.deg2rad(np.clip(ang, 0.0, 89.0))
        f_alpha = min(5.0, 1.0 / max(np.cos(theta), 1e-3))

        Y = self.Q * s_n * threshold_term * f_alpha
        return float(max(0.0, Y))

    def calculate_erosion_rate(
        self,
        flux_particles_m2_s: float,
        T_ion_eV: float,
        angle_deg: float = 45.0,
    ) -> dict[str, float]:
        """
        Calculate erosion metrics and net impurity source.
        """
        flux = float(flux_particles_m2_s)
        temp = float(T_ion_eV)
        if not np.isfinite(flux) or flux < 0.0:
            raise ValueError("flux_particles_m2_s must be finite and >= 0.")
        if not np.isfinite(temp) or temp < 0.0:
            raise ValueError("T_ion_eV must be finite and >= 0.")

        E_impact = 5.0 * temp
        Y = self.calculate_yield(E_impact, angle_deg=angle_deg)
        flux_erosion = flux * Y
        flux_net = flux_erosion * (1.0 - self.redeposition_factor)

        recession_speed = (flux_net * (self.atomic_mass * 1.66e-27)) / (self.density * 1000.0)
        seconds_per_year = 3600.0 * 24.0 * 365.0
        mm_year = recession_speed * 1000.0 * seconds_per_year

        return {
            "Yield": float(Y),
            "E_impact": float(E_impact),
            "Net_Flux": float(flux_net),
            "Redeposition": float(self.redeposition_factor),
            "Erosion_mm_year": float(mm_year),
            "Impurity_Source": float(flux_net),
        }


def run_pwi_demo(
    *,
    material: str = "Tungsten",
    redeposition_factor: float = 0.95,
    flux_particles_m2_s: float = 1e24,
    temp_min_eV: float = 10.0,
    temp_max_eV: float = 100.0,
    num_points: int = 50,
    angle_deg: float = 45.0,
    save_plot: bool = True,
    output_path: str = "PWI_Erosion_Result.png",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run deterministic PWI erosion scan and return summary.
    """
    t_lo = float(temp_min_eV)
    t_hi = float(temp_max_eV)
    if not np.isfinite(t_lo) or not np.isfinite(t_hi) or t_lo >= t_hi:
        raise ValueError("temp_min_eV/temp_max_eV must be finite with temp_min_eV < temp_max_eV.")
    n = max(int(num_points), 3)

    pwi = SputteringPhysics(material=material, redeposition_factor=redeposition_factor)
    flux = float(flux_particles_m2_s)
    temps = np.linspace(t_lo, t_hi, n, dtype=np.float64)
    erosion_rates: list[float] = []
    yields: list[float] = []

    if verbose:
        print("--- SCPN PLASMA-WALL INTERACTION: SPUTTERING ---")
        print(f"{'T_ion (eV)':<10} | {'Impact (eV)':<12} | {'Yield':<10} | {'Erosion (mm/y)':<15}")
        print("-" * 55)

    cadence = max(n // 5, 1)
    for i, T in enumerate(temps):
        res = pwi.calculate_erosion_rate(flux, float(T), angle_deg=angle_deg)
        erosion_rates.append(float(res["Erosion_mm_year"]))
        yields.append(float(res["Yield"]))
        if verbose and (i % cadence == 0 or i == n - 1):
            print(
                f"{float(T):<10.1f} | {float(res['E_impact']):<12.1f} | "
                f"{float(res['Yield']):<10.4f} | {float(res['Erosion_mm_year']):<15.2f}"
            )

    er_arr = np.asarray(erosion_rates, dtype=np.float64)
    y_arr = np.asarray(yields, dtype=np.float64)

    plot_saved = False
    plot_error: Optional[str] = None
    if save_plot:
        try:
            fig, ax1 = plt.subplots(figsize=(10, 6))

            ax1.set_xlabel("Divertor Ion Temperature (eV)")
            ax1.set_ylabel("Sputtering Yield (Y)", color="tab:red")
            ax1.plot(temps, y_arr, color="tab:red", linewidth=2)
            ax1.tick_params(axis="y", labelcolor="tab:red")

            ax2 = ax1.twinx()
            ax2.set_ylabel("Erosion Rate (mm/year)", color="tab:blue")
            ax2.plot(temps, er_arr, color="tab:blue", linestyle="--", linewidth=2)
            ax2.tick_params(axis="y", labelcolor="tab:blue")
            ax2.axhline(5.0, color="k", linestyle=":", label="Max limit (5mm/y)")

            plt.title("Tungsten Divertor Erosion vs Plasma Temperature")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close(fig)
            plot_saved = True
            if verbose:
                print(f"Saved: {output_path}")
        except Exception as exc:
            plot_error = f"{exc.__class__.__name__}: {exc}"

    return {
        "material": str(material),
        "redep_factor": float(pwi.redeposition_factor),
        "flux_particles_m2_s": float(flux),
        "angle_deg": float(angle_deg),
        "points": int(n),
        "temp_min_eV": float(t_lo),
        "temp_max_eV": float(t_hi),
        "min_yield": float(np.min(y_arr)) if y_arr.size else 0.0,
        "max_yield": float(np.max(y_arr)) if y_arr.size else 0.0,
        "min_erosion_mm_year": float(np.min(er_arr)) if er_arr.size else 0.0,
        "max_erosion_mm_year": float(np.max(er_arr)) if er_arr.size else 0.0,
        "plot_saved": bool(plot_saved),
        "plot_error": plot_error,
    }


if __name__ == "__main__":
    run_pwi_demo()
