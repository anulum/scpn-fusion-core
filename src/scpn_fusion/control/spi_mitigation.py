# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — SPI Mitigation
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class ShatteredPelletInjection:
    """
    Reduced SPI mitigation model for thermal/current quench campaigns.
    """

    def __init__(
        self, Plasma_Energy_MJ: float = 300.0, Plasma_Current_MA: float = 15.0
    ):
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
    def _require_non_negative(name: str, value: float) -> float:
        out = float(value)
        if not np.isfinite(out) or out < 0.0:
            raise ValueError(f"{name} must be finite and >= 0.")
        return out

    @staticmethod
    def estimate_z_eff(neon_quantity_mol: float) -> float:
        return ShatteredPelletInjection.estimate_z_eff_cocktail(
            neon_quantity_mol=neon_quantity_mol,
            argon_quantity_mol=0.0,
            xenon_quantity_mol=0.0,
        )

    @staticmethod
    def estimate_z_eff_cocktail(
        *,
        neon_quantity_mol: float = 0.0,
        argon_quantity_mol: float = 0.0,
        xenon_quantity_mol: float = 0.0,
    ) -> float:
        neon = ShatteredPelletInjection._require_non_negative(
            "neon_quantity_mol", neon_quantity_mol
        )
        argon = ShatteredPelletInjection._require_non_negative(
            "argon_quantity_mol", argon_quantity_mol
        )
        xenon = ShatteredPelletInjection._require_non_negative(
            "xenon_quantity_mol", xenon_quantity_mol
        )

        # Empirical weighting: higher-Z gases radiate/ionize more efficiently per mol.
        weighted_moles = 1.00 * neon + 1.35 * argon + 1.90 * xenon
        impurity_fraction = np.clip((weighted_moles / 0.12) * 0.015, 0.0, 0.12)

        species_weight = np.array([neon, argon, xenon], dtype=np.float64)
        z2 = np.array([10.0**2, 18.0**2, 36.0**2], dtype=np.float64)
        denom = float(np.sum(species_weight))
        z2_eff = float(np.dot(species_weight, z2) / denom) if denom > 0.0 else 1.0
        zeff = (1.0 - impurity_fraction) * 1.0 + impurity_fraction * z2_eff
        return float(np.clip(zeff, 1.0, 12.0))

    @staticmethod
    def estimate_mitigation_cocktail(
        *,
        risk_score: float,
        disturbance: float,
        action_bias: float = 0.0,
    ) -> dict[str, float]:
        risk_raw = float(risk_score)
        dist_raw = float(disturbance)
        action_raw = float(action_bias)
        if not np.isfinite(risk_raw):
            raise ValueError("risk_score must be finite.")
        if not np.isfinite(dist_raw):
            raise ValueError("disturbance must be finite.")
        if not np.isfinite(action_raw):
            raise ValueError("action_bias must be finite.")

        risk = float(np.clip(risk_raw, 0.0, 1.0))
        dist = float(np.clip(dist_raw, 0.0, 1.0))
        action = float(np.clip(action_raw, -1.0, 1.0))

        total_mol = float(
            np.clip(
                0.05 + 0.15 * risk + 0.07 * dist + 0.02 * action,
                0.03,
                0.24,
            )
        )
        heavy_drive = float(
            np.clip(0.55 * risk + 0.45 * dist + 0.20 * max(action, 0.0), 0.0, 1.0)
        )
        xenon_frac = 0.05 + 0.30 * heavy_drive
        argon_frac = 0.15 + 0.25 * heavy_drive
        neon_frac = max(0.0, 1.0 - xenon_frac - argon_frac)
        frac_sum = neon_frac + argon_frac + xenon_frac
        neon_frac /= frac_sum
        argon_frac /= frac_sum
        xenon_frac /= frac_sum

        return {
            "neon_quantity_mol": float(total_mol * neon_frac),
            "argon_quantity_mol": float(total_mol * argon_frac),
            "xenon_quantity_mol": float(total_mol * xenon_frac),
            "total_quantity_mol": float(total_mol),
        }

    @staticmethod
    def estimate_tau_cq(te_keV: float, z_eff: float) -> float:
        te = max(float(te_keV), 0.01)
        zeff = max(float(z_eff), 1.0)
        tau = 0.02 * (2.0 / zeff) * ((te / 0.1) ** 0.25)
        return float(np.clip(tau, 0.002, 0.05))

    def trigger_mitigation(
        self,
        neon_quantity_mol: float = 0.1,
        argon_quantity_mol: float = 0.0,
        xenon_quantity_mol: float = 0.0,
        return_diagnostics: bool = False,
        *,
        duration_s: float = 0.05,
        dt_s: float = 1e-5,
        verbose: bool = True,
    ):
        neon = self._require_non_negative("neon_quantity_mol", neon_quantity_mol)
        argon = self._require_non_negative("argon_quantity_mol", argon_quantity_mol)
        xenon = self._require_non_negative("xenon_quantity_mol", xenon_quantity_mol)
        total_impurity = neon + argon + xenon
        duration = float(duration_s)
        dt = float(dt_s)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration_s must be finite and > 0.")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be finite and > 0.")

        if verbose:
            logger.warning(
                "DISRUPTION DETECTED! TRIGGERING SPI (Ne=%.3f mol, Ar=%.3f mol, Xe=%.3f mol)",
                neon, argon, xenon,
            )

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
        
        # NGPS Ablation Parameters
        # Modeled as an ensemble of fragments
        n_fragments = 50
        pellet_radius_m = 0.002 
        # Total Moles -> Atoms
        total_atoms = total_impurity * 6.022e23
        atoms_per_fragment = total_atoms / n_fragments
        
        # Current plasma state (local to SPI)
        ne_local = 1.0e20 # Initial
        
        while t < duration:
            if t > t_mix:
                # NGPS Ablation Rate: dN/dt ~ n_e^(1/3) * T_e^(1.64) * r_p^(4/3)
                # Ref: Parks et al., Phys. Plasmas 5, 1024 (1998)
                ablation_rate = 1.2e16 * (ne_local**0.33) * (self.Te**1.64) * (pellet_radius_m**1.33)
                
                # Atoms released
                d_atoms = ablation_rate * dt * n_fragments
                
                # Self-consistent density increase
                vol_plasma = 800.0 # ITER-like
                ne_local += d_atoms / vol_plasma
                
                self.Z_eff = self.estimate_z_eff_cocktail(
                    neon_quantity_mol=neon,
                    argon_quantity_mol=argon,
                    xenon_quantity_mol=xenon,
                )
                
                # Enhanced radiation from density increase
                P_rad = 1e9 * (self.Z_eff**0.5) * (self.Te / 1.0)**0.5 * (ne_local / 1e20)
                dW = -P_rad * dt
                prev_W = self.W_th
                self.W_th += dW
                denom = max(prev_W - dW, 1e-12)
                self.Te = max(0.01, self.Te * (self.W_th / denom))

                if self.Te < 5.0 and phase == "Thermal Quench":
                    phase = "Current Quench"
                    if verbose:
                        logger.info(
                            "[t=%.1fms] Thermal Quench Complete. Entering Current Quench.",
                            t * 1000,
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
                "tau_cq_ms_mean": (
                    float(np.mean(history_tau_cq)) if history_tau_cq else 0.0
                ),
                "tau_cq_ms_p95": (
                    float(np.percentile(history_tau_cq, 95)) if history_tau_cq else 0.0
                ),
                "final_current_MA": float(self.Ip / 1e6),
                "final_temperature_keV": float(self.Te),
                "neon_quantity_mol": float(neon),
                "argon_quantity_mol": float(argon),
                "xenon_quantity_mol": float(xenon),
                "total_impurity_mol": float(total_impurity),
                "duration_ms": float(duration * 1000.0),
            }
            return time_axis, history_W, history_I, diagnostics
        return time_axis, history_W, history_I


def run_spi_mitigation(
    *,
    plasma_energy_mj: float = 300.0,
    plasma_current_ma: float = 15.0,
    neon_quantity_mol: float = 0.1,
    argon_quantity_mol: float = 0.0,
    xenon_quantity_mol: float = 0.0,
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
        argon_quantity_mol=argon_quantity_mol,
        xenon_quantity_mol=xenon_quantity_mol,
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
                logger.info("Saved: %s", output_path)
        except Exception as exc:
            plot_error = f"{exc.__class__.__name__}: {exc}"

    return {
        "plasma_energy_mj": float(plasma_energy_mj),
        "plasma_current_ma": float(plasma_current_ma),
        "neon_quantity_mol": float(neon_quantity_mol),
        "argon_quantity_mol": float(argon_quantity_mol),
        "xenon_quantity_mol": float(xenon_quantity_mol),
        "total_impurity_mol": float(diag["total_impurity_mol"]),
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
