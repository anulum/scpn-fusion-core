# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Integrated Transport Solver
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List, cast

try:
    from numpy.typing import NDArray
except ImportError:
    NDArray = Any # type: ignore

from scpn_fusion.core._rust_compat import FusionKernel

def chang_hinton_chi_profile(rho: NDArray[Any], T_i: NDArray[Any], n_e_19: NDArray[Any], q: NDArray[Any], R0: float, a: float, B0: float, A_ion: float = 2.0, Z_eff: float = 1.5) -> NDArray[Any]:

    """
    Chang-Hinton (1982) neoclassical ion thermal diffusivity profile [m²/s].

    Parameters
    ----------
    rho : array  — normalised radius [0,1]
    T_i : array  — ion temperature [keV]
    n_e_19 : array  — electron density [10^19 m^-3]
    q : array  — safety factor profile
    R0 : float  — major radius [m]
    a : float  — minor radius [m]
    B0 : float  — toroidal field [T]
    A_ion : float  — ion mass number (default 2 = deuterium)
    Z_eff : float  — effective charge

    Returns
    -------
    chi_nc : array  — neoclassical chi_i [m²/s]
    """
    e_charge = 1.602176634e-19
    eps0 = 8.854187812e-12
    m_p = 1.672621924e-27
    m_e = 9.10938370e-31
    m_i = A_ion * m_p

    chi_nc = np.zeros_like(rho)
    for i in range(len(rho)):
        r = rho[i]
        if r <= 0.0 or T_i[i] <= 0.0 or n_e_19[i] <= 0.0 or q[i] <= 0.0:
            chi_nc[i] = 0.01
            continue

        epsilon = r * a / R0
        if epsilon < 1e-6:
            chi_nc[i] = 0.01
            continue

        T_J = T_i[i] * 1.602176634e-16  # keV -> J
        v_ti = np.sqrt(2.0 * T_J / m_i)
        rho_i = m_i * v_ti / (e_charge * B0)

        # ion-ion collision frequency
        n_e = n_e_19[i] * 1e19
        ln_lambda = 17.0
        nu_ii = (n_e * Z_eff**2 * e_charge**4 * ln_lambda
                 / (12.0 * np.pi**1.5 * eps0**2 * m_i**0.5 * T_J**1.5))

        eps32 = epsilon**1.5
        nu_star = nu_ii * q[i] * R0 / (eps32 * v_ti)

        alpha_sh = epsilon
        chi_val = (0.66 * (1.0 + 1.54 * alpha_sh) * q[i]**2
                   * rho_i**2 * nu_ii
                   / (eps32 * (1.0 + 0.74 * nu_star**(2.0/3.0))))

        chi_nc[i] = max(chi_val, 0.01) if np.isfinite(chi_val) else 0.01

    return chi_nc


class TransportSolver(FusionKernel):  # type: ignore[misc]
    """
    1.5D Integrated Transport Code.
    Solves Heat and Particle diffusion equations on flux surfaces,
    coupled self-consistently with the 2D Grad-Shafranov equilibrium.
    """
    def __init__(self, config_path: Union[str, Path]) -> None:
        super().__init__(config_path)
        self.external_profile_mode = True # Tell Kernel to respect our calculated profiles
        self.nr = 50 # Radial grid points (normalized radius rho)
        self.rho: NDArray[Any] = np.linspace(0, 1, self.nr)
        self.drho = 1.0 / (self.nr - 1)
        
        # PROFILES (Evolving state variables)
        # Te = Electron Temp (keV), Ti = Ion Temp (keV), ne = Density (10^19 m-3)
        self.Te: NDArray[Any] = 1.0 * (1 - self.rho**2) # Initial guess
        self.Ti: NDArray[Any] = 1.0 * (1 - self.rho**2)
        self.ne: NDArray[Any] = 5.0 * (1 - self.rho**2)**0.5
        
        # Transport Coefficients (Anomalous Transport Models)
        self.chi_e: NDArray[Any] = np.ones(self.nr) # Electron diffusivity
        self.chi_i: NDArray[Any] = np.ones(self.nr) # Ion diffusivity
        self.D_n: NDArray[Any] = np.ones(self.nr)   # Particle diffusivity
        
        # Impurity Profile (Tungsten density)
        self.n_impurity: NDArray[Any] = np.zeros(self.nr)

        # 2D attributes
        self.Pressure_2D: NDArray[Any] = np.zeros((self.NZ, self.NR))
        self.J_phi: NDArray[Any] = np.zeros((self.NZ, self.NR))

        # Neoclassical transport configuration (None = constant chi_base=0.5)
        self.neoclassical_params: Optional[Dict[str, Any]] = None

    def set_neoclassical(self, R0: float, a: float, B0: float, A_ion: float = 2.0, Z_eff: float = 1.5, q0: float = 1.0, q_edge: float = 3.0) -> None:
        """Configure Chang-Hinton neoclassical transport model.

        When set, update_transport_model uses the Chang-Hinton formula instead
        of the constant chi_base = 0.5.
        """
        q_profile = q0 + (q_edge - q0) * self.rho**2
        self.neoclassical_params = {
            'R0': R0, 'a': a, 'B0': B0,
            'A_ion': A_ion, 'Z_eff': Z_eff,
            'q_profile': q_profile,
        }

    def inject_impurities(self, flux_from_wall_per_sec: float, dt: float) -> None:
        """
        Models impurity influx from PWI erosion.
        Simple diffusion model: Source at edge, diffuses inward.
        """
        # Source at edge (last grid point)
        # Flux is total particles. Volume of edge shell approx 20 m3.
        # Delta_n = Flux * dt / Vol_edge
        # Scaling factor adjusted for simulation stability
        d_n_edge = (flux_from_wall_per_sec * dt) / 20.0 * 1e-18 
        
        # Add to edge
        self.n_impurity[-1] += d_n_edge
        
        # Diffuse inward (Explicit step)
        D_imp = 1.0 # m2/s
        new_imp = self.n_impurity.copy()
        
        grad = np.gradient(self.n_impurity, self.drho)
        flux = -D_imp * grad
        div = np.gradient(flux, self.drho) / (self.rho + 1e-6)
        
        new_imp += (-div) * dt
        
        # Boundary
        new_imp[0] = new_imp[1] # Axis symmetry
        
        self.n_impurity = np.maximum(0, new_imp)

    def calculate_bootstrap_current(self, R0: float, B_pol: float) -> NDArray[Any]:
        """
        Calculates the neoclassical bootstrap current density [A/m2]
        using a simplified Sauter model.
        J_bs = - (R/B_pol) * [ L31 * (dP/dpsi) + ... ]
        """
        # Simplified Sauter model coefficients
        # In a real model, these depend on collisionality and trapped fraction
        f_trapped = 1.46 * np.sqrt(self.rho * (self.cfg["dimensions"]["R_max"] - self.cfg["dimensions"]["R_min"]) / (2 * R0))
        
        # Pressure gradient in SI
        P = self.ne * 1e19 * (self.Ti + self.Te) * 1.602e-16 # J/m3
        dP_drho = np.gradient(P, self.drho)
        
        # Scaling constant for J_bs
        # J_bs ~ f_trapped / B_pol * dP/dr
        B_pol_val = max(B_pol, 0.1) # Avoid div by zero at axis
        
        J_bs = 1.2 * (f_trapped / B_pol_val) * dP_drho / (self.cfg["dimensions"]["R_max"] - self.cfg["dimensions"]["R_min"])
        
        # Ensure it's zero at axis and edge
        J_bs[0] = 0.0
        J_bs[-1] = 0.0
        
        return J_bs

    def update_transport_model(self, P_aux: float) -> None:
        """
        Bohm / Gyro-Bohm Transport Model.
        """
        # 1. Critical Gradient Model
        grad_T = np.gradient(self.Ti, self.drho)
        threshold = 2.0
        
        # Base Level — use Chang-Hinton if configured, else constant
        chi_base: Union[float, NDArray[Any]]
        if self.neoclassical_params is not None:
            p = self.neoclassical_params
            chi_base = chang_hinton_chi_profile(
                self.rho, self.Ti, self.ne,
                p['q_profile'], p['R0'], p['a'], p['B0'],
                p['A_ion'], p['Z_eff']
            )
        else:
            chi_base = 0.5

        # Turbulent Level
        chi_turb = 5.0 * np.maximum(0, -grad_T - threshold)
        
        # H-Mode Barrier Logic
        is_H_mode = P_aux > 30.0 # MW
        
        if is_H_mode:
            # Suppress turbulence at edge (rho > 0.9)
            edge_mask = self.rho > 0.9
            chi_turb[edge_mask] *= 0.1 # Transport Barrier
            
        self.chi_e = chi_base + chi_turb
        self.chi_i = chi_base + chi_turb
        self.D_n = 0.1 * self.chi_e

    # ── Tridiagonal (Thomas) solver ─────────────────────────────────

    @staticmethod
    def _thomas_solve(a, b, c, d):
        """O(n) tridiagonal solver (Thomas algorithm).

        Solves  A x = d  where A is tridiagonal with sub-diagonal *a*,
        main diagonal *b*, and super-diagonal *c*.

        Parameters
        ----------
        a : array, length n-1  — sub-diagonal
        b : array, length n    — main diagonal
        c : array, length n-1  — super-diagonal
        d : array, length n    — right-hand side

        Returns
        -------
        x : array, length n
        """
        n = len(d)
        # Work on copies to avoid mutating input
        cp = np.empty(n - 1)
        dp = np.empty(n)

        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]

        for i in range(1, n):
            m = b[i] - a[i - 1] * (cp[i - 1] if i - 1 < len(cp) else 0.0)
            if abs(m) < 1e-30:
                m = 1e-30
            dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / m
            if i < n - 1:
                cp[i] = c[i] / m

        x = np.empty(n)
        x[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]

        return x

    # ── Crank-Nicolson helpers ───────────────────────────────────────

    def _explicit_diffusion_rhs(self, T: NDArray[Any], chi: NDArray[Any]) -> NDArray[Any]:
        """Compute explicit diffusion operator L_h(T) = (1/r) d/dr(r chi dT/dr).

        Uses half-grid diffusivities and central differences on the
        interior, returning an array of the same length as *T*.
        """
        n = len(T)
        Lh = np.zeros(n)
        dr = self.drho

        for i in range(1, n - 1):
            r = self.rho[i]
            # half-grid chi
            chi_ip = 0.5 * (chi[i] + chi[i + 1])
            chi_im = 0.5 * (chi[i] + chi[i - 1])
            r_ip = r + 0.5 * dr
            r_im = r - 0.5 * dr

            flux_ip = chi_ip * r_ip * (T[i + 1] - T[i]) / dr
            flux_im = chi_im * r_im * (T[i] - T[i - 1]) / dr

            Lh[i] = (flux_ip - flux_im) / (r * dr)

        return Lh

    def _build_cn_tridiag(self, chi: NDArray[Any], dt: float) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Build tridiagonal coefficients for the Crank-Nicolson LHS.

        The implicit system is:
            (I - 0.5*dt*L_h) T^{n+1} = (I + 0.5*dt*L_h) T^n + dt*(S - Sink)

        Returns (a, b, c) sub/main/super diagonals for the interior points,
        padded to full grid size (BCs applied separately).
        """
        n = len(self.rho)
        dr = self.drho
        a = np.zeros(n - 1)  # sub-diagonal
        b = np.ones(n)       # main diagonal
        c = np.zeros(n - 1)  # super-diagonal

        for i in range(1, n - 1):
            r = self.rho[i]
            chi_ip = 0.5 * (chi[i] + chi[i + 1])
            chi_im = 0.5 * (chi[i] + chi[i - 1])
            r_ip = r + 0.5 * dr
            r_im = r - 0.5 * dr

            coeff_ip = chi_ip * r_ip / (r * dr * dr)
            coeff_im = chi_im * r_im / (r * dr * dr)

            # LHS: (I - 0.5*dt*L_h) => diag entries are *subtracted*
            b[i] = 1.0 + 0.5 * dt * (coeff_ip + coeff_im)
            c[i] = -0.5 * dt * coeff_ip       # T_{i+1} coefficient
            a[i - 1] = -0.5 * dt * coeff_im   # T_{i-1} coefficient

        return a, b, c

    # ── Main evolution (Crank-Nicolson) ──────────────────────────────

    def evolve_profiles(self, dt: float, P_aux: float) -> Tuple[float, float]:
        """Advance Ti by one time step using Crank-Nicolson implicit diffusion.

        The scheme is unconditionally stable, allowing dt up to ~1.0 s
        without NaN.  The full equation solved is:

            (T^{n+1} - T^n)/dt = 0.5*[L_h(T^{n+1}) + L_h(T^n)] + S - Sink
        """
        # ── Sources ──
        heating_profile = np.exp(-self.rho**2 / 0.1)
        S_heat = (P_aux / np.sum(heating_profile)) * heating_profile

        # ── Sinks (radiation) ──
        cooling_factor = 5.0
        S_rad = cooling_factor * self.ne * self.n_impurity * np.sqrt(self.Te + 0.1)

        net_source = S_heat - S_rad

        # ── Explicit half: RHS = T^n + 0.5*dt*L_h(T^n) + dt*net_source ──
        Lh_explicit = self._explicit_diffusion_rhs(self.Ti, self.chi_i)
        rhs = self.Ti + 0.5 * dt * Lh_explicit + dt * net_source

        # ── Implicit half: build tridiagonal for (I - 0.5*dt*L_h) ──
        a, b, c = self._build_cn_tridiag(self.chi_i, dt)

        # ── Solve ──
        new_Ti = self._thomas_solve(a, b, c, rhs)

        # ── Boundary Conditions ──
        new_Ti[0] = new_Ti[1]    # Neumann at core
        new_Ti[-1] = 0.1         # Dirichlet at edge

        self.Ti = np.maximum(0.01, new_Ti)
        self.Te = self.Ti  # Assume equilibrated

        return float(np.mean(self.Ti)), float(self.Ti[0])

    def map_profiles_to_2d(self) -> None:
        """
        Projects the 1D radial profiles back onto the 2D Grad-Shafranov grid,
        including neoclassical bootstrap current.
        """
        # 1. Get Flux Topology
        idx_max = np.argmax(self.Psi)
        iz_ax, ir_ax = np.unravel_index(idx_max, self.Psi.shape)
        Psi_axis = self.Psi[iz_ax, ir_ax]
        xp, psi_x = self.find_x_point(self.Psi)
        Psi_edge = psi_x
        if abs(Psi_edge - Psi_axis) < 1.0: Psi_edge = float(np.min(self.Psi))
        
        # 2. Calculate Rho for every 2D point
        denom = Psi_edge - Psi_axis
        if abs(denom) < 1e-9: denom = 1e-9
        Psi_norm = (self.Psi - Psi_axis) / denom
        Psi_norm = np.clip(Psi_norm, 0, 1)
        Rho_2D = np.sqrt(Psi_norm)
        
        # 3. Calculate 1D Bootstrap Current
        R0 = cast(float, (self.cfg["dimensions"]["R_min"] + self.cfg["dimensions"]["R_max"]) / 2.0)
        # Estimate B_pol from Ip
        I_target = self.cfg['physics']['plasma_current_target']
        B_pol_est = cast(float, (1.256e-6 * I_target) / (2 * np.pi * 0.5 * (self.cfg["dimensions"]["R_max"] - self.cfg["dimensions"]["R_min"])))
        J_bs_1d = self.calculate_bootstrap_current(R0, B_pol_est)
        
        # 4. Interpolate 1D profiles to 2D
        self.Pressure_2D = np.interp(Rho_2D.flatten(), self.rho, self.ne * (self.Ti + self.Te))
        self.Pressure_2D = self.Pressure_2D.reshape(self.Psi.shape)
        
        J_bs_2D = np.interp(Rho_2D.flatten(), self.rho, J_bs_1d)
        J_bs_2D = J_bs_2D.reshape(self.Psi.shape)
        
        # 5. Update J_phi (Pressure driven + Bootstrap)
        # J_phi = R p' + J_bs
        self.J_phi = (self.Pressure_2D * self.RR) + J_bs_2D
        
        # Normalize to target current
        I_curr = np.sum(self.J_phi) * self.dR * self.dZ
        if I_curr > 1e-9:
            self.J_phi *= (I_target / I_curr)

    # ── Confinement time ───────────────────────────────────────────────

    def compute_confinement_time(self, P_loss_MW: float) -> float:
        """Compute the energy confinement time from stored energy.

        τ_E = W_stored / P_loss, where W_stored = ∫ 3/2 n (Ti+Te) dV
        and the volume element is estimated from the 1D radial profiles
        using cylindrical approximation.

        Parameters
        ----------
        P_loss_MW : float
            Total loss power [MW].  Must be > 0.

        Returns
        -------
        float
            Energy confinement time [s].
        """
        if P_loss_MW <= 0:
            return float("inf")

        # Stored energy: W = ∫ 3/2 n_e (T_i + T_e) dV
        # In 1D with cylindrical approx: dV ≈ 2πR₀ · 2π · r · a² · dρ
        # Units: n_e is in 10^19 m^-3, T in keV → W in MJ
        e_keV = 1.602176634e-16  # J per keV
        dims = self.cfg["dimensions"]
        R0 = (dims["R_min"] + dims["R_max"]) / 2.0
        a = (dims["R_max"] - dims["R_min"]) / 2.0

        # Volume element per rho bin: dV = 2π R₀ · 2π ρ a² dρ
        rho_mid = self.rho
        dV = 2.0 * np.pi * R0 * 2.0 * np.pi * rho_mid * a**2 * self.drho

        # Energy density: 3/2 * n_e * (Ti + Te) [10^19 m^-3 * keV]
        energy_density = 1.5 * (self.ne * 1e19) * (self.Ti + self.Te) * e_keV
        W_stored_J = float(np.sum(energy_density * dV))
        W_stored_MW = W_stored_J / 1e6  # J → MJ → MW·s

        return W_stored_MW / P_loss_MW

    def run_to_steady_state(
        self,
        P_aux: float,
        n_steps: int = 500,
        dt: float = 0.01,
        adaptive: bool = False,
        tol: float = 1e-3,
    ) -> Dict[str, Any]:
        """Run transport evolution until approximate steady state.

        Parameters
        ----------
        P_aux : float
            Auxiliary heating power [MW].
        n_steps : int
            Number of evolution steps.
        dt : float
            Time step [s] (initial value when adaptive=True).
        adaptive : bool
            Use Richardson-extrapolation adaptive time stepping.
        tol : float
            Error tolerance for adaptive stepping.

        Returns
        -------
        dict
            ``{"T_avg": float, "T_core": float, "tau_e": float,
            "n_steps": int, "Ti_profile": ndarray,
            "ne_profile": ndarray}``
            When adaptive=True, also includes ``dt_final``,
            ``dt_history``, ``error_history``.
        """
        if not adaptive:
            for _ in range(n_steps):
                self.update_transport_model(P_aux)
                T_avg, T_core = self.evolve_profiles(dt, P_aux)

            tau_e = self.compute_confinement_time(P_aux)
            return {
                "T_avg": float(T_avg),
                "T_core": float(T_core),
                "tau_e": tau_e,
                "n_steps": n_steps,
                "Ti_profile": self.Ti.copy(),
                "ne_profile": self.ne.copy(),
            }

        # ── Adaptive time stepping ──
        atc = AdaptiveTimeController(dt_init=dt, tol=tol)

        for step in range(n_steps):
            self.update_transport_model(P_aux)
            error = atc.estimate_error(self, P_aux)
            atc.adapt_dt(error)

            # Take the accepted step (full step already applied inside estimate_error)
            T_avg = float(np.mean(self.Ti))
            T_core = float(self.Ti[0])

        tau_e = self.compute_confinement_time(P_aux)
        return {
            "T_avg": float(T_avg),
            "T_core": float(T_core),
            "tau_e": tau_e,
            "n_steps": n_steps,
            "Ti_profile": self.Ti.copy(),
            "ne_profile": self.ne.copy(),
            "dt_final": atc.dt,
            "dt_history": atc.dt_history.copy(),
            "error_history": atc.error_history.copy(),
        }


class AdaptiveTimeController:
    """Richardson-extrapolation adaptive time controller for CN transport.

    Compares one full CN step vs. two half-steps to estimate the local
    truncation error, then uses a PI controller to adjust dt.

    Parameters
    ----------
    dt_init : float — initial time step [s]
    dt_min : float — minimum allowed dt
    dt_max : float — maximum allowed dt
    tol : float — target local error tolerance
    safety : float — safety factor (< 1) for step adjustment
    """

    def __init__(
        self,
        dt_init: float = 0.01,
        dt_min: float = 1e-5,
        dt_max: float = 1.0,
        tol: float = 1e-3,
        safety: float = 0.9,
    ):
        self.dt = dt_init
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.tol = tol
        self.safety = safety
        self.p = 2  # CN is second-order

        self.dt_history: list[float] = []
        self.error_history: list[float] = []
        self._err_prev: float = tol  # initialise for PI controller

    def estimate_error(self, solver: "TransportSolver", P_aux: float) -> float:
        """Estimate local error via Richardson extrapolation.

        Takes one full CN step of size dt and two half-steps of size dt/2,
        then compares.  The solver state is advanced by the *half-step*
        result (more accurate).

        Returns the estimated error norm.
        """
        Ti_save = solver.Ti.copy()
        Te_save = solver.Te.copy()

        # One full step
        solver.Ti = Ti_save.copy()
        solver.Te = Te_save.copy()
        solver.evolve_profiles(self.dt, P_aux)
        T_full = solver.Ti.copy()

        # Two half steps
        solver.Ti = Ti_save.copy()
        solver.Te = Te_save.copy()
        solver.evolve_profiles(self.dt / 2.0, P_aux)
        solver.evolve_profiles(self.dt / 2.0, P_aux)
        T_half = solver.Ti.copy()

        # Richardson error estimate: ||T_full - T_half|| / (2^p - 1)
        error = float(np.linalg.norm(T_full - T_half)) / (2**self.p - 1)
        error = max(error, 1e-15)

        # Accept the half-step result (more accurate)
        solver.Ti = T_half
        solver.Te = T_half.copy()

        return error

    def adapt_dt(self, error: float) -> None:
        """Adjust dt using a PI controller.

        dt *= min(2, safety * (tol/err)^(0.7/p) * (err_prev/err)^(0.4/p))
        """
        self.error_history.append(error)
        self.dt_history.append(self.dt)

        ratio_i = (self.tol / error) ** (0.7 / self.p)
        ratio_p = (self._err_prev / error) ** (0.4 / self.p)
        factor = self.safety * ratio_i * ratio_p
        factor = min(factor, 2.0)
        factor = max(factor, 0.1)  # don't shrink too aggressively

        self.dt *= factor
        self.dt = max(self.dt, self.dt_min)
        self.dt = min(self.dt, self.dt_max)

        self._err_prev = error
