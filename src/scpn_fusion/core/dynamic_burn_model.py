# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Dynamic Burn Model
"""Self-consistent dynamic zero-dimensional burn model.

Integrates the coupled plasma energy and temperature ODEs with ITER IPB98(y,2)
confinement scaling, bremsstrahlung and impurity radiation, He-ash dilution, and
the Martin H-mode threshold, and scans for validated Q>=10 operating points.
"""

from __future__ import annotations

import warnings
from typing import cast

import numpy as np

from .uncertainty import _dt_reactivity

from scpn_fusion.exceptions import FusionCoreError as _FusionCoreError


class BurnPhysicsError(RuntimeError, _FusionCoreError):
    """Raised when strict 0-D burn physics contracts are violated."""


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
        for name, value in {
            "R0": self.R0,
            "a": self.a,
            "B_t": self.B_t,
            "I_p": self.I_p,
            "kappa": self.kappa,
            "n_e20": self.n_e20,
            "M_eff": self.M_eff,
            "Z_eff": self.Z_eff,
        }.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0.")

        # Plasma volume (torus)
        self.V_plasma = 2.0 * np.pi**2 * self.R0 * self.a**2 * self.kappa

    @staticmethod
    def bosch_hale_dt(T_keV: float) -> float:
        """D-T <sigma v> [m^3/s]. Bosch & Hale, NF 32 (1992) 611."""
        return float(_dt_reactivity(T_keV))

    def iter98y2_tau_e(self, P_loss_mw: float) -> float:
        """Compute the ITER IPB98(y,2) energy confinement scaling (s).

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
            * P ** (-0.69)
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
        S = 4.0 * np.pi**2 * self.R0 * self.a * np.sqrt((1.0 + self.kappa**2) / 2.0)
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
    ) -> dict[str, object]:
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
        # Total plasma thermal energy W = (3/2)(n_e T_e + n_i T_i) = 3 n_e T for a
        # quasineutral D-T plasma with T_e = T_i (electron + ion heat capacity),
        # consistent with the IPB98(y,2) tau_E definition, FusionBurnPhysics, and
        # the Rust ignition kernel. An electron-only 1.5 n_e T halves the heat
        # capacity and overstates the heating rate.
        W_thermal = 3.0 * n_e * T * 1e3 * 1.602e-19 * self.V_plasma  # J

        # Delayed alpha heating from collisional slowing down.
        # The deposited channel follows dP_dep/dt = (P_born - P_dep) / tau_s.

        # Histories
        time_s = []
        T_hist = []
        Q_hist = []
        P_fus_hist = []
        P_alpha_hist: list[float] = []
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
            P_alpha_born = 0.2 * P_fus  # 3.5 MeV / 17.6 MeV

            # Alpha slowing-down time: tau_s ~ 0.012 * Te^1.5 / (ne/1e19)
            tau_s_alpha = 0.012 * (max(T, 0.1) ** 1.5) / (self.n_e20 * 10.0)
            tau_s_alpha = np.clip(tau_s_alpha, 0.01, 2.0)

            # Exact first-order relaxation for P_alpha_deposited.  This preserves
            # positivity and boundedness even when dt_s exceeds tau_s_alpha.
            if step == 0:
                P_alpha_dep = P_alpha_born
            else:
                prev_alpha_dep = P_alpha_hist[-1] * 1e6
                relaxation = 1.0 - np.exp(-dt_s / tau_s_alpha)
                P_alpha_dep = prev_alpha_dep + relaxation * (P_alpha_born - prev_alpha_dep)

            # Losses
            P_total_heating = P_alpha_dep + P_aux_mw * 1e6
            tau_E = self.iter98y2_tau_e(P_total_heating / 1e6)
            P_transport = W_thermal / max(tau_E, 0.01)

            # Bremsstrahlung: P_brems ~ 5.35e-37 * Z_eff * n_e^2 * sqrt(T_keV) * V
            P_brems = 5.35e-37 * self.Z_eff * n_e**2 * np.sqrt(max(T, 0.1)) * self.V_plasma

            # Impurity line radiation (reduced-order closure)
            P_line = 1e-37 * (self.Z_eff - 1.0) * n_e**2 * self.V_plasma

            P_rad = P_brems + P_line
            P_loss = P_transport + P_rad

            # Energy evolution
            dW = (P_alpha_dep + P_aux_mw * 1e6 - P_loss) * dt_s
            W_thermal = max(W_thermal + dW, 1e3)

            # Temperature from stored energy (total electron + ion heat capacity)
            T = W_thermal / (3.0 * n_e * 1e3 * 1.602e-19 * self.V_plasma)
            if t_cap_keV < T:
                temperature_cap_events += 1
                if enforce_temperature_limit:
                    raise BurnPhysicsError(
                        f"Temperature {T:.2f} keV exceeds {t_cap_keV:.1f} keV physical limit."
                    )
                if (
                    max_temperature_clamp_events is not None
                    and temperature_cap_events > max_temperature_clamp_events
                ):
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
            R_fus = n_d * n_t * sigmav * self.V_plasma  # reactions/s
            tau_he = tau_he_factor * tau_E
            dn_he = (R_fus - pumping_efficiency * f_he * n_e * self.V_plasma / tau_he) * dt_s
            f_he += dn_he / (n_e * self.V_plasma)
            f_he = float(np.clip(f_he, 0.0, 0.5))

            # Q factor (capped at 15 to avoid 0-D burn model artifacts)
            Q_raw = P_fus / max(P_aux_mw * 1e6, 1.0)
            Q = min(Q_raw, 15.0)

            P_fus_hist.append(P_fus / 1e6)
            P_alpha_hist.append(P_alpha_dep / 1e6)
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
    ) -> dict[str, object]:
        """Scan auxiliary power to find Q>=10 operating point.

        Returns the scan results including the optimal P_aux for Q=10.
        """
        # Greenwald density limit (Greenwald 1988): n_GW = I_p / (pi * a^2)
        # in units of 10^20 m^-3
        n_greenwald = I_p / (np.pi * a**2)

        results: list[dict[str, float | bool]] = []
        for n_e20 in [0.8, 1.0, 1.2]:
            # Skip densities above 1.2x Greenwald limit
            if n_e20 > 1.2 * n_greenwald:
                warnings.warn(
                    f"Density n_e20={n_e20:.2f} exceeds 1.2x Greenwald limit "
                    f"({n_greenwald:.2f}); skipping.",
                    stacklevel=2,
                )
                continue

            for P_aux in [float(x) for x in np.arange(10.0, 80.0, 5.0)]:
                model = DynamicBurnModel(R0=R0, a=a, B_t=B_t, I_p=I_p, kappa=kappa, n_e20=n_e20)
                sim = model.simulate(
                    P_aux_mw=P_aux,
                    duration_s=50.0,
                    dt_s=0.05,
                    warn_on_temperature_cap=False,
                )
                results.append(
                    {
                        "n_e20": n_e20,
                        "P_aux_MW": P_aux,
                        "Q_final": cast(float, sim["Q_final"]),
                        "Q_peak": cast(float, sim["Q_peak"]),
                        "T_final_keV": cast(float, sim["T_final_keV"]),
                        "P_fus_final_MW": cast(float, sim["P_fus_final_MW"]),
                        "f_he_final": cast(float, sim["f_he_final"]),
                        "ignition": cast(bool, sim["ignition"]),
                    }
                )

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

        if (
            results[best_idx]["Q_final"] > 15.0
        ):  # pragma: no cover - defensive: simulate caps Q_final at 15
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
