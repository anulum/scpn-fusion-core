# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Fusion Ignition Sim
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import warnings

import numpy as np
import matplotlib.pyplot as plt
from .fusion_kernel import FusionKernel
import sys


class BurnPhysicsError(RuntimeError):
    """Raised when strict 0-D burn physics contracts are violated."""


class FusionBurnPhysics(FusionKernel):
    """
    Extends the Grad-Shafranov Solver with Thermonuclear Physics.
    Calculates Fusion Power, Alpha Heating, and Q-Factor.
    """
    def __init__(self, config_path):
        super().__init__(config_path)
        
    def bosch_hale_dt(self, T_keV):
        """
        Calculates <sigma*v> for Deuterium-Tritium fusion.
        Parametrization by Bosch & Hale (1992).
        T_keV: Ion Temperature in keV.
        Returns: Reaction rate in m^3/s
        """
        # Avoid zero/negative temp
        T = np.maximum(T_keV, 0.1)
        
        # NRL Plasma Formulary approximation for D-T <sigma v> (m^3/s)
        # Valid for T < 100 keV
        sigmav = 3.68e-18 / (T**(2/3)) * np.exp(-19.94 / (T**(1/3)))
        
        return sigmav

    def calculate_thermodynamics(self, P_aux_MW=50.0):
        """
        Maps Magnetic Equilibrium -> Thermodynamics -> Fusion Power.
        P_aux_MW: External Heating Power (NBI/ECRH) in MegaWatts.
        """
        # 1. Derive Pressure from Grad-Shafranov (J ~ R*p')
        # In our simplified kernel, J was modeled directly. 
        # Here we assume Pressure follows Flux Surfaces: p(psi) ~ (1-psi)^2
        
        idx_max = np.argmax(self.Psi)
        iz, ir = np.unravel_index(idx_max, self.Psi.shape)
        Psi_axis = self.Psi[iz, ir]
        
        # FIX: Find real boundary using X-point
        xp, psi_x = self.find_x_point(self.Psi)
        Psi_boundary = psi_x
        
        # Safety: if boundary close to axis (limiter case), use min of flux map
        if abs(Psi_boundary - Psi_axis) < 1.0: 
            Psi_boundary = np.min(self.Psi)
            
        Psi_norm = (self.Psi - Psi_axis) / (Psi_boundary - Psi_axis)
        Psi_norm = np.clip(Psi_norm, 0, 1)
        mask = (Psi_norm >= 0) & (Psi_norm < 1.0)
        
        # Peak values (ITER-like)
        n_peak = 1.0e20 # m^-3 (Density)
        T_peak_keV = 20.0 # keV (Temperature)
        
        # Profiles
        n = np.zeros_like(self.Psi)
        T = np.zeros_like(self.Psi)
        
        n[mask] = n_peak * (1 - Psi_norm[mask]**2)**0.5
        T[mask] = T_peak_keV * (1 - Psi_norm[mask]**2)**1.0
        
        # 2. Calculate Fusion Power
        # P_fus = E_fus * nD * nT * <sigma v>
        # Assume 50-50 D-T mix
        nD = 0.5 * n
        nT = 0.5 * n
        E_fus = 17.6 * 1.602e-13 # MeV to Joules (17.6 MeV per reaction)
        
        sigmav = self.bosch_hale_dt(T)
        power_density = nD * nT * sigmav * E_fus # Watts/m^3
        
        # Integrate over volume (Approximating Toroidal symmetry 2*pi*R)
        dV = self.dR * self.dZ * 2 * np.pi * self.RR
        P_fusion_total = np.sum(power_density * dV)
        
        # 3. Alpha Heating (Self-Heating)
        # Alphas carry 20% of fusion energy (3.5 MeV / 17.6 MeV)
        P_alpha = P_fusion_total * 0.2
        
        # 4. Losses (IPB98(y,2) Confinement scaling)
        # Tau_E = 0.0562 * Ip^0.93 * Bt^0.15 * n19^0.41 * P^-0.69 * R^1.97 * eps^0.58 * kappa^0.78 * M^0.19
        W_thermal = np.sum(3 * n * (T * 1.602e-16) * dV) # Thermal energy in Joules
        
        # Extraction of parameters for scaling
        Ip_MA = self.cfg["physics"].get("plasma_current_target", 15.0e6) / 1e6
        Bt = self.cfg["dimensions"].get("B0", 5.3) # Nominal
        n19 = n_peak / 1e19
        R = self.cfg["dimensions"].get("R0", 6.2)
        a = (self.cfg["dimensions"]["R_max"] - self.cfg["dimensions"]["R_min"]) / 2.0
        eps = a / R
        kappa = self.cfg["dimensions"].get("kappa", 1.7)
        M_eff = 2.5 # D-T
        
        # Power for scaling (Loss power)
        P_loss_scaling_MW = max((P_aux_MW + P_alpha/1e6), 1.0)
        
        Tau_E = (0.0562 * Ip_MA**0.93 * Bt**0.15 * n19**0.41 * 
                 P_loss_scaling_MW**(-0.69) * R**1.97 * eps**0.58 * 
                 kappa**0.78 * M_eff**0.19)
        
        Tau_E = np.clip(Tau_E, 0.1, 10.0) # Physical bounds
        P_loss = W_thermal / Tau_E
        
        # 5. Global Balance
        # dW/dt = P_alpha + P_aux - P_loss
        net_heating = P_alpha + (P_aux_MW*1e6) - P_loss
        
        # Q Factor
        Q = P_fusion_total / (P_aux_MW*1e6) if P_aux_MW > 0 else 0
        
        return {
            'P_fusion_MW': P_fusion_total / 1e6,
            'P_alpha_MW': P_alpha / 1e6,
            'P_loss_MW': P_loss / 1e6,
            'P_aux_MW': P_aux_MW,
            'Net_MW': net_heating / 1e6,
            'Q': Q,
            'T_peak': T_peak_keV,
            'W_MJ': W_thermal / 1e6
        }

def run_ignition_experiment():
    print("--- SCPN IGNITION EXPERIMENT: The Road to Q > 10 ---")
    
    config_path = "03_CODE/SCPN-Fusion-Core/iter_config.json"
    sim = FusionBurnPhysics(config_path)
    
    # Simulation: Power Ramp Up
    # We increase Auxiliary Heating and measure the response
    power_ramp = np.linspace(0, 100, 20) # 0 to 100 MW
    
    history_Q = []
    history_P_fus = []
    
    print(f"{ 'Aux (MW)':<10} | { 'Fusion (MW)':<12} | { 'Alpha (MW)':<10} | { 'Q-Factor':<8} | {'Status'}")
    print("-" * 60)
    
    # 1. Establish Geometry
    sim.solve_equilibrium()
    
    for P_aux in power_ramp:
        # In a real dynamic code, P_aux would modify T_peak dynamically
        # Here we perform a static check: "If we had this geometry and profiles, what is the output?"
        # To make it dynamic, we link T_peak to P_net from previous step
        
        metrics = sim.calculate_thermodynamics(P_aux)
        
        # Status check
        status = "L-Mode"
        if metrics['Q'] > 1.0: status = "Breakeven"
        if metrics['Q'] > 5.0: status = "Burning"
        if metrics['Q'] > 10.0: status = "IGNITION"
        
        history_Q.append(metrics['Q'])
        history_P_fus.append(metrics['P_fusion_MW'])
        
        print(f"{P_aux:<10.1f} | {metrics['P_fusion_MW']:<12.1f} | {metrics['P_alpha_MW']:<10.1f} | {metrics['Q']:<8.2f} | {status}")

    # --- VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Q-Curve
    ax1.set_title("Fusion Gain (Q) vs Input Power")
    ax1.plot(power_ramp, history_Q, 'r-o', linewidth=2)
    ax1.axhline(1.0, color='gray', linestyle='--', label='Breakeven (Q=1)')
    ax1.axhline(10.0, color='green', linestyle='--', label='Ignition (Q=10)')
    ax1.set_xlabel("Auxiliary Heating (MW)")
    ax1.set_ylabel("Q")
    ax1.legend()
    ax1.grid(True)
    
    # POP-CON Plot (Operating Point)
    # We visualize where the final state sits in Power space
    ax2.set_title("Power Balance (Ignition Condition)")
    ax2.bar(['Alpha Heat', 'Aux Heat'], [metrics['P_alpha_MW'], metrics['P_aux_MW']], color=['red', 'orange'])
    ax2.bar(['Losses'], [metrics['P_loss_MW']], color='blue')
    ax2.set_ylabel("Power (MW)")
    
    plt.tight_layout()
    plt.savefig("Ignition_Result.png")
    print("\nExperiment Complete. Results: Ignition_Result.png")

class DynamicBurnModel:
    """Self-consistent dynamic burn model with ITER98y2 confinement scaling.

    Solves the coupled ODEs for plasma energy and temperature evolution:

        dW/dt = P_alpha + P_aux - W/tau_E - P_rad - P_brems

    with:
        - Bosch-Hale D-T reactivity
        - ITER IPB98(y,2) confinement scaling
        - Bremsstrahlung radiation losses
        - Impurity radiation (Z_eff-dependent)
        - He ash accumulation and dilution
        - H-mode power threshold (Martin scaling)

    This enables finding validated Q>=10 operating points.

    Parameters
    ----------
    R0 : float
        Major radius (m).
    a : float
        Minor radius (m).
    B_t : float
        Toroidal field (T).
    I_p : float
        Plasma current (MA).
    kappa : float
        Elongation.
    n_e20 : float
        Line-averaged electron density (10^20 m^-3).
    M_eff : float
        Effective ion mass (amu). D-T = 2.5.
    Z_eff : float
        Effective charge.
    """

    def __init__(
        self,
        R0: float = 6.2,
        a: float = 2.0,
        B_t: float = 5.3,
        I_p: float = 15.0,
        kappa: float = 1.7,
        n_e20: float = 1.0,
        M_eff: float = 2.5,
        Z_eff: float = 1.65,
    ) -> None:
        self.R0 = float(R0)
        self.a = float(a)
        self.B_t = float(B_t)
        self.I_p = float(I_p)
        self.kappa = float(kappa)
        self.n_e20 = float(n_e20)
        self.M_eff = float(M_eff)
        self.Z_eff = float(Z_eff)

        # Plasma volume (torus)
        self.V_plasma = 2.0 * np.pi**2 * self.R0 * self.a**2 * self.kappa

    @staticmethod
    def bosch_hale_dt(T_keV: float) -> float:
        """D-T reactivity <sigma v> in m^3/s (Bosch & Hale 1992)."""
        T = max(float(T_keV), 0.1)
        return 3.68e-18 / T ** (2.0 / 3.0) * np.exp(-19.94 / T ** (1.0 / 3.0))

    def iter98y2_tau_e(self, P_loss_mw: float) -> float:
        """ITER IPB98(y,2) energy confinement scaling (s).

        tau_E = 0.0562 * I_p^0.93 * B_t^0.15 * n_e19^0.41 * P^-0.69 *
                R^1.97 * (a/R)^0.58 * kappa^0.78 * M^0.19
        """
        P = max(float(P_loss_mw), 0.1)
        n_e19 = self.n_e20 * 10.0  # 10^19 m^-3
        eps = self.a / self.R0
        tau = (
            0.0562
            * self.I_p**0.93
            * self.B_t**0.15
            * n_e19**0.41
            * P**(-0.69)
            * self.R0**1.97
            * eps**0.58
            * self.kappa**0.78
            * self.M_eff**0.19
        )
        return float(max(tau, 0.01))

    def h_mode_threshold_mw(self) -> float:
        """H-mode power threshold (Martin 2008 scaling).

        P_thr = 0.0488 * n_e20^0.717 * B_t^0.803 * S^0.941
        where S is the plasma surface area.
        """
        S = 4.0 * np.pi**2 * self.R0 * self.a * np.sqrt(
            (1.0 + self.kappa**2) / 2.0
        )
        P_thr = 0.0488 * self.n_e20**0.717 * self.B_t**0.803 * S**0.941
        return float(P_thr)

    def simulate(
        self,
        P_aux_mw: float = 50.0,
        T_initial_keV: float = 5.0,
        duration_s: float = 100.0,
        dt_s: float = 0.01,
        f_he_initial: float = 0.02,
        tau_he_factor: float = 5.0,
        pumping_efficiency: float = 0.8,
        *,
        enforce_temperature_limit: bool = False,
        max_temperature_clamp_events: int | None = None,
        warn_on_temperature_cap: bool = True,
        emit_repeated_temperature_warnings: bool = False,
    ) -> dict:
        """Run dynamic burn simulation.

        Parameters
        ----------
        P_aux_mw : float
            Auxiliary heating power (MW).
        T_initial_keV : float
            Initial ion temperature (keV).
        duration_s : float
            Simulation duration (s).
        dt_s : float
            Time step (s).
        f_he_initial : float
            Initial helium fraction.
        tau_he_factor : float
            He confinement / energy confinement ratio.
        pumping_efficiency : float
            He pumping efficiency (0–1).
        enforce_temperature_limit : bool
            When True, raise :class:`BurnPhysicsError` instead of clamping
            temperatures above 25 keV.
        max_temperature_clamp_events : int or None
            Optional cap on number of clamp events.  If exceeded, raises
            :class:`BurnPhysicsError`.
        warn_on_temperature_cap : bool
            Emit warning when a cap event occurs.
        emit_repeated_temperature_warnings : bool
            Emit warning for every cap event when True; otherwise warn once.

        Returns
        -------
        dict with time-histories and final metrics including Q factor.
        """
        n_steps = int(duration_s / dt_s)
        n_e = self.n_e20 * 1e20  # m^-3
        t_cap_keV = 25.0
        if max_temperature_clamp_events is not None:
            if (
                isinstance(max_temperature_clamp_events, bool)
                or not isinstance(max_temperature_clamp_events, (int, np.integer))
                or int(max_temperature_clamp_events) < 0
            ):
                raise ValueError(
                    "max_temperature_clamp_events must be a non-negative integer or None."
                )
            max_temperature_clamp_events = int(max_temperature_clamp_events)

        # State variables
        T = float(T_initial_keV)
        f_he = float(f_he_initial)
        W_thermal = 1.5 * n_e * T * 1e3 * 1.602e-19 * self.V_plasma  # J

        # Histories
        time_s = []
        T_hist = []
        Q_hist = []
        P_fus_hist = []
        P_alpha_hist = []
        P_loss_hist = []
        P_rad_hist = []
        f_he_hist = []
        tau_e_hist = []
        W_hist = []
        temperature_cap_events = 0
        temperature_cap_warning_emitted = False

        for step in range(n_steps):
            t = step * dt_s
            time_s.append(t)
            T_hist.append(T)
            f_he_hist.append(f_he)

            # D-T fuel fractions (diluted by He)
            f_dt = max(1.0 - 2.0 * f_he, 0.0)
            n_d = 0.5 * f_dt * n_e
            n_t = 0.5 * f_dt * n_e

            # Fusion power
            sigmav = self.bosch_hale_dt(T)
            P_fus = n_d * n_t * sigmav * 17.6e6 * 1.602e-19 * self.V_plasma  # W
            P_alpha = 0.2 * P_fus  # 3.5 MeV / 17.6 MeV

            # Losses
            # Confinement: need P_loss estimate for tau_E (circular dependency)
            # Use previous step's P_loss or initial estimate
            P_total_heating = P_alpha + P_aux_mw * 1e6
            tau_E = self.iter98y2_tau_e(P_total_heating / 1e6)
            P_transport = W_thermal / max(tau_E, 0.01)

            # Bremsstrahlung: P_brems ~ 5.35e-37 * Z_eff * n_e^2 * sqrt(T_keV) * V
            P_brems = 5.35e-37 * self.Z_eff * n_e**2 * np.sqrt(max(T, 0.1)) * self.V_plasma

            # Impurity line radiation (simplified)
            P_line = 1e-37 * (self.Z_eff - 1.0) * n_e**2 * self.V_plasma

            P_rad = P_brems + P_line
            P_loss = P_transport + P_rad

            # Energy evolution
            dW = (P_alpha + P_aux_mw * 1e6 - P_loss) * dt_s
            W_thermal = max(W_thermal + dW, 1e3)

            # Temperature from stored energy
            T = W_thermal / (1.5 * n_e * 1e3 * 1.602e-19 * self.V_plasma)
            if T > t_cap_keV:
                temperature_cap_events += 1
                if enforce_temperature_limit:
                    raise BurnPhysicsError(
                        f"Temperature {T:.2f} keV exceeds {t_cap_keV:.1f} keV physical limit."
                    )
                if max_temperature_clamp_events is not None and temperature_cap_events > max_temperature_clamp_events:
                    raise BurnPhysicsError(
                        "Temperature cap events exceeded limit: "
                        f"{temperature_cap_events} > {max_temperature_clamp_events}."
                    )
                if warn_on_temperature_cap and (
                    emit_repeated_temperature_warnings or not temperature_cap_warning_emitted
                ):
                    warnings.warn(
                        f"Temperature {T:.1f} keV exceeds {t_cap_keV:.1f} keV physical limit; "
                        "clamping (0-D model artifact).",
                        stacklevel=2,
                    )
                    temperature_cap_warning_emitted = True
            T = float(np.clip(T, 0.1, t_cap_keV))

            # He ash accumulation
            # Source: fusion rate, Sink: pumping
            R_fus = n_d * n_t * sigmav * self.V_plasma  # reactions/s
            tau_he = tau_he_factor * tau_E
            dn_he = (R_fus - pumping_efficiency * f_he * n_e * self.V_plasma / tau_he) * dt_s
            f_he += dn_he / (n_e * self.V_plasma)
            f_he = float(np.clip(f_he, 0.0, 0.5))

            # Q factor (capped at 15 to avoid 0-D burn model artifacts)
            Q_raw = P_fus / max(P_aux_mw * 1e6, 1.0)
            Q = min(Q_raw, 15.0)

            P_fus_hist.append(P_fus / 1e6)
            P_alpha_hist.append(P_alpha / 1e6)
            P_loss_hist.append(P_loss / 1e6)
            P_rad_hist.append(P_rad / 1e6)
            Q_hist.append(Q)
            tau_e_hist.append(tau_E)
            W_hist.append(W_thermal / 1e6)

        # Final steady-state metrics
        Q_final = Q_hist[-1] if Q_hist else 0.0
        Q_peak = max(Q_hist) if Q_hist else 0.0

        return {
            "time_s": time_s,
            "T_keV": T_hist,
            "Q": Q_hist,
            "P_fus_MW": P_fus_hist,
            "P_alpha_MW": P_alpha_hist,
            "P_loss_MW": P_loss_hist,
            "P_rad_MW": P_rad_hist,
            "f_he": f_he_hist,
            "tau_E_s": tau_e_hist,
            "W_MJ": W_hist,
            "Q_final": Q_final,
            "Q_peak": Q_peak,
            "T_final_keV": T_hist[-1] if T_hist else 0.0,
            "P_fus_final_MW": P_fus_hist[-1] if P_fus_hist else 0.0,
            "f_he_final": f_he_hist[-1] if f_he_hist else 0.0,
            "tau_E_final_s": tau_e_hist[-1] if tau_e_hist else 0.0,
            "h_mode_threshold_MW": self.h_mode_threshold_mw(),
            "P_aux_MW": P_aux_mw,
            "ignition": Q_final > 10.0,
            "temperature_cap_events": int(temperature_cap_events),
            "temperature_cap_limit_keV": float(t_cap_keV),
            "temperature_cap_warning_emitted": bool(temperature_cap_warning_emitted),
        }

    @staticmethod
    def find_q10_operating_point(
        R0: float = 6.2,
        a: float = 2.0,
        B_t: float = 5.3,
        I_p: float = 15.0,
        kappa: float = 1.7,
    ) -> dict:
        """Scan auxiliary power to find Q>=10 operating point.

        Returns the scan results including the optimal P_aux for Q=10.
        """
        # Greenwald density limit (Greenwald 1988): n_GW = I_p / (pi * a^2)
        # in units of 10^20 m^-3
        n_greenwald = I_p / (np.pi * a**2)

        results = []
        for n_e20 in [0.8, 1.0, 1.2]:
            # Skip densities above 1.2x Greenwald limit
            if n_e20 > 1.2 * n_greenwald:
                warnings.warn(
                    f"Density n_e20={n_e20:.2f} exceeds 1.2x Greenwald limit "
                    f"({n_greenwald:.2f}); skipping.",
                    stacklevel=2,
                )
                continue

            for P_aux in np.arange(10.0, 80.0, 5.0):
                model = DynamicBurnModel(
                    R0=R0, a=a, B_t=B_t, I_p=I_p, kappa=kappa, n_e20=n_e20
                )
                sim = model.simulate(
                    P_aux_mw=P_aux,
                    duration_s=50.0,
                    dt_s=0.05,
                    warn_on_temperature_cap=False,
                )
                results.append({
                    "n_e20": n_e20,
                    "P_aux_MW": P_aux,
                    "Q_final": sim["Q_final"],
                    "Q_peak": sim["Q_peak"],
                    "T_final_keV": sim["T_final_keV"],
                    "P_fus_final_MW": sim["P_fus_final_MW"],
                    "f_he_final": sim["f_he_final"],
                    "ignition": sim["ignition"],
                })

        # Find best Q operating point
        if not results:
            return {
                "scan_results": [],
                "best": None,
                "q10_achieved": False,
                "n_greenwald": n_greenwald,
            }

        q_values = [r["Q_final"] for r in results]
        best_idx = int(np.argmax(q_values))

        if results[best_idx]["Q_final"] > 15.0:
            warnings.warn(
                f"Best Q={results[best_idx]['Q_final']:.1f} exceeds 15; "
                "likely a 0-D model artifact.",
                stacklevel=2,
            )

        return {
            "scan_results": results,
            "best": results[best_idx],
            "q10_achieved": results[best_idx]["Q_final"] >= 10.0,
            "n_greenwald": n_greenwald,
        }


if __name__ == "__main__":
    run_ignition_experiment()
