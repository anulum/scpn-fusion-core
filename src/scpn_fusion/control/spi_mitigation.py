# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — SPI Mitigation
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np


class ShatteredPelletInjection:
    """
    Reduced SPI mitigation model for thermal/current quench campaigns.
    """

    def __init__(self, Plasma_Energy_MJ: float = 300.0, Plasma_Current_MA: float = 15.0):
        w_mj = float(Plasma_Energy_MJ)
        ip_ma = float(Plasma_Current_MA)
        if not np.isfinite(w_mj) or w_mj <= 0.0:
            raise ValueError("Plasma_Energy_MJ must be finite and > 0.")
        if not np.isfinite(ip_ma) or ip_ma <= 0.0:
            raise ValueError("Plasma_Current_MA must be finite and > 0.")
        self.W_th = w_mj * 1e6
        self.Ip = ip_ma * 1e6
        self.Te = 20.0
        self.Z_eff = 1.0
        self.last_tau_cq_s = 0.02

    @staticmethod
    def estimate_z_eff(neon_quantity_mol: float) -> float:
        neon = max(float(neon_quantity_mol), 0.0)
        impurity_fraction = np.clip((neon / 0.12) * 0.015, 0.0, 0.12)
        z_imp = 10.0
        zeff = (1.0 - impurity_fraction) * 1.0 + impurity_fraction * (z_imp**2)
        return float(np.clip(zeff, 1.0, 12.0))

    @staticmethod
    def estimate_tau_cq(te_keV: float, z_eff: float) -> float:
        te = max(float(te_keV), 0.01)
        zeff = max(float(z_eff), 1.0)
        tau = 0.02 * (2.0 / zeff) * ((te / 0.1) ** 0.25)
        return float(np.clip(tau, 0.002, 0.05))

    def trigger_mitigation(
        self,
        neon_quantity_mol: float = 0.1,
        return_diagnostics: bool = False,
        *,
        duration_s: float = 0.05,
        dt_s: float = 1e-5,
        verbose: bool = True,
    ):
        neon = float(neon_quantity_mol)
        if not np.isfinite(neon) or neon < 0.0:
            raise ValueError("neon_quantity_mol must be finite and >= 0.")
        duration = float(duration_s)
        dt = float(dt_s)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration_s must be finite and > 0.")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be finite and > 0.")

        if verbose:
            print(f"--- DISRUPTION DETECTED! TRIGGERING SPI ({neon} mol Neon) ---")

        t_mix = 0.002
        time_axis: list[float] = []
        history_W: list[float] = []
        history_I: list[float] = []
        history_T: list[float] = []
        history_tau_cq: list[float] = []

        t = 0.0
        phase = "Thermal Quench"
        self.Z_eff = 1.0
        self.last_tau_cq_s = 0.02

        while t < duration:
            if t > t_mix:
                self.Z_eff = self.estimate_z_eff(neon)
                P_rad = 1e9 * (self.Z_eff**0.5) * (self.Te / 1.0) ** 0.5
                dW = -P_rad * dt
                prev_W = self.W_th
                self.W_th += dW
                denom = max(prev_W - dW, 1e-12)
                self.Te = max(0.01, self.Te * (self.W_th / denom))

                if self.Te < 5.0 and phase == "Thermal Quench":
                    phase = "Current Quench"
                    if verbose:
                        print(
                            f"  [t={t*1000:.1f}ms] Thermal Quench Complete. "
                            "Entering Current Quench."
                        )

            if phase == "Current Quench":
                tau_cq_s = self.estimate_tau_cq(self.Te, self.Z_eff)
                self.last_tau_cq_s = tau_cq_s
                dI = -(self.Ip / tau_cq_s) * dt
                self.Ip += dI
                history_tau_cq.append(tau_cq_s * 1000.0)
            else:
                history_tau_cq.append(self.last_tau_cq_s * 1000.0)

            history_W.append(self.W_th / 1e6)
            history_I.append(self.Ip / 1e6)
            history_T.append(self.Te)
            time_axis.append(t * 1000.0)
            t += dt

        if return_diagnostics:
            diagnostics = {
                "z_eff": float(self.Z_eff),
                "tau_cq_ms_mean": float(np.mean(history_tau_cq)) if history_tau_cq else 0.0,
                "tau_cq_ms_p95": float(np.percentile(history_tau_cq, 95)) if history_tau_cq else 0.0,
                "final_current_MA": float(self.Ip / 1e6),
                "final_temperature_keV": float(self.Te),
                "duration_ms": float(duration * 1000.0),
            }
            return time_axis, history_W, history_I, diagnostics
        return time_axis, history_W, history_I


def run_spi_mitigation(
    *,
    plasma_energy_mj: float = 300.0,
    plasma_current_ma: float = 15.0,
    neon_quantity_mol: float = 0.1,
    duration_s: float = 0.05,
    dt_s: float = 1e-5,
    save_plot: bool = True,
    output_path: str = "SPI_Mitigation_Result.png",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run SPI mitigation simulation and return deterministic summary metrics.
    """
    spi = ShatteredPelletInjection(
        Plasma_Energy_MJ=plasma_energy_mj,
        Plasma_Current_MA=plasma_current_ma,
    )
    t, w, i, diag = spi.trigger_mitigation(
        neon_quantity_mol=neon_quantity_mol,
        return_diagnostics=True,
        duration_s=duration_s,
        dt_s=dt_s,
        verbose=verbose,
    )

    t_arr = np.asarray(t, dtype=np.float64)
    w_arr = np.asarray(w, dtype=np.float64)
    i_arr = np.asarray(i, dtype=np.float64)

    plot_saved = False
    plot_error: Optional[str] = None
    if save_plot:
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            ax1.plot(t_arr, w_arr, "r-", linewidth=2)
            ax1.set_title("Thermal Energy (Thermal Quench)")
            ax1.set_ylabel("Energy (MJ)")
            ax1.grid(True)

            ax2.plot(t_arr, i_arr, "b-", linewidth=2)
            ax2.set_title("Plasma Current (Current Quench)")
            ax2.set_xlabel("Time (ms)")
            ax2.set_ylabel("Current (MA)")
            ax2.grid(True)

            plt.tight_layout()
            plt.savefig(output_path)
            plt.close(fig)
            plot_saved = True
            if verbose:
                print(f"Saved: {output_path}")
        except Exception as exc:
            plot_error = f"{exc.__class__.__name__}: {exc}"

    return {
        "plasma_energy_mj": float(plasma_energy_mj),
        "plasma_current_ma": float(plasma_current_ma),
        "neon_quantity_mol": float(neon_quantity_mol),
        "duration_s": float(duration_s),
        "dt_s": float(dt_s),
        "samples": int(t_arr.size),
        "initial_energy_mj": float(w_arr[0]) if w_arr.size else 0.0,
        "final_energy_mj": float(w_arr[-1]) if w_arr.size else 0.0,
        "initial_current_ma": float(i_arr[0]) if i_arr.size else 0.0,
        "final_current_ma": float(i_arr[-1]) if i_arr.size else 0.0,
        "z_eff": float(diag["z_eff"]),
        "tau_cq_ms_mean": float(diag["tau_cq_ms_mean"]),
        "tau_cq_ms_p95": float(diag["tau_cq_ms_p95"]),
        "plot_saved": bool(plot_saved),
        "plot_error": plot_error,
    }


def run_spi_test() -> dict[str, Any]:
    return run_spi_mitigation(save_plot=True, verbose=True)


if __name__ == "__main__":
    run_spi_test()
