# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Integrated Transport Solver
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
from pathlib import Path

try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel  # type: ignore[assignment]

_logger = logging.getLogger(__name__)


class PhysicsError(RuntimeError):
    """Raised when a physics constraint is violated."""

# ── Gyro-Bohm coefficient loader ─────────────────────────────────────

_GYRO_BOHM_COEFF_PATH = (
    Path(__file__).resolve().parents[3]
    / "validation"
    / "reference_data"
    / "itpa"
    / "gyro_bohm_coefficients.json"
)

_GYRO_BOHM_DEFAULT = 0.1  # Fallback if JSON not found


def _load_gyro_bohm_coefficient(
    path: Path | str | None = None,
) -> float:
    """Load the calibrated gyro-Bohm coefficient c_gB from JSON.

    Parameters
    ----------
    path : Path or str, optional
        Override path.  Defaults to the file shipped in
        ``validation/reference_data/itpa/gyro_bohm_coefficients.json``.

    Returns
    -------
    float
        The calibrated c_gB value, or 0.1 if the file is not found.
    """
    p = Path(path) if path else _GYRO_BOHM_COEFF_PATH
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        c_gB = float(data["c_gB"])
        _logger.debug("Loaded c_gB = %.6f from %s", c_gB, p)
        return c_gB
    except (FileNotFoundError, KeyError, json.JSONDecodeError, TypeError) as exc:
        _logger.warning(
            "Could not load c_gB from %s (%s), using default %.4f",
            p, exc, _GYRO_BOHM_DEFAULT,
        )
        return _GYRO_BOHM_DEFAULT


def chang_hinton_chi_profile(rho, T_i, n_e_19, q, R0, a, B0, A_ion=2.0, Z_eff=1.5):
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


def calculate_sauter_bootstrap_current_full(rho, Te, Ti, ne, q, R0, a, B0, Z_eff=1.5):
    """Full Sauter bootstrap current model (Sauter et al., Phys. Plasmas 6, 1999).

    Parameters
    ----------
    rho : array — normalised radius [0,1]
    Te : array — electron temperature [keV]
    Ti : array — ion temperature [keV]
    ne : array — electron density [10^19 m^-3]
    q : array — safety factor profile
    R0 : float — major radius [m]
    a : float — minor radius [m]
    B0 : float — toroidal field [T]
    Z_eff : float — effective charge

    Returns
    -------
    j_bs : array — bootstrap current density [A/m^2]
    """
    n = len(rho)
    j_bs = np.zeros(n)
    e_charge = 1.602176634e-19
    m_e = 9.10938370e-31
    eps0 = 8.854187812e-12

    for i in range(1, n - 1):
        eps = rho[i] * a / R0
        if eps < 1e-6 or Te[i] <= 0 or ne[i] <= 0 or q[i] <= 0:
            continue

        # Trapped fraction (Sauter formula)
        f_t = 1.0 - (1.0 - eps)**2 / (np.sqrt(1.0 - eps**2) * (1.0 + 1.46 * np.sqrt(eps)))
        f_t = max(0.0, min(f_t, 1.0))

        # Electron thermal velocity
        T_e_J = Te[i] * 1e3 * e_charge
        v_te = np.sqrt(2.0 * T_e_J / m_e)

        # Collision frequency
        n_e = ne[i] * 1e19
        ln_lambda = 17.0
        nu_ei = n_e * Z_eff * e_charge**4 * ln_lambda / (
            12.0 * np.pi**1.5 * eps0**2 * m_e**0.5 * T_e_J**1.5
        )

        # Collisionality
        nu_star_e = nu_ei * q[i] * R0 / (eps**1.5 * v_te) if v_te > 0 else 1e6

        # Sauter L31 coefficient
        alpha_31 = 1.0 / (1.0 + 0.36 / Z_eff)
        L31 = f_t * (1.0 + (1.0 - 0.1 * f_t) * np.sqrt(nu_star_e) +
               0.5 * (1.0 - f_t) * nu_star_e / Z_eff)
        L31 = f_t * alpha_31 / (1.0 + alpha_31 * np.sqrt(nu_star_e) +
               0.25 * nu_star_e * (1.0 - f_t)**2)

        # Sauter L32 coefficient
        L32 = f_t * (0.05 + 0.62 * Z_eff) / (Z_eff * (1.0 + 0.44 * Z_eff))
        L32 /= (1.0 + 0.22 * np.sqrt(nu_star_e) + 0.19 * nu_star_e * (1.0 - f_t))

        # Sauter L34 coefficient (ion contribution)
        L34 = L31 * Ti[i] / max(Te[i], 0.01)

        # Gradients (central differences)
        dr = (rho[i+1] - rho[i-1]) * a
        if abs(dr) < 1e-12:
            continue
        dn_dr = (ne[i+1] - ne[i-1]) * 1e19 / dr
        dTe_dr = (Te[i+1] - Te[i-1]) * 1e3 * e_charge / dr
        dTi_dr = (Ti[i+1] - Ti[i-1]) * 1e3 * e_charge / dr

        # Poloidal field
        B_pol = B0 * eps / max(q[i], 0.1)
        if B_pol < 1e-10:
            continue

        # Bootstrap current
        p_e = n_e * T_e_J
        j_bs[i] = -(p_e / B_pol) * (
            L31 * dn_dr / max(n_e, 1e10) +
            L32 * dTe_dr / max(T_e_J, 1e-30) +
            L34 * dTi_dr / max(Ti[i] * 1e3 * e_charge, 1e-30)
        )

    return j_bs


class TransportSolver(FusionKernel):
    """
    1.5D Integrated Transport Code.
    Solves Heat and Particle diffusion equations on flux surfaces,
    coupled self-consistently with the 2D Grad-Shafranov equilibrium.
    """
    def __init__(self, config_path):
        super().__init__(config_path)
        self.external_profile_mode = True # Tell Kernel to respect our calculated profiles
        self.nr = 50 # Radial grid points (normalized radius rho)
        self.rho = np.linspace(0, 1, self.nr)
        self.drho = 1.0 / (self.nr - 1)
        
        # PROFILES (Evolving state variables)
        # Te = Electron Temp (keV), Ti = Ion Temp (keV), ne = Density (10^19 m-3)
        self.Te = 1.0 * (1 - self.rho**2) # Initial guess
        self.Ti = 1.0 * (1 - self.rho**2)
        self.ne = 5.0 * (1 - self.rho**2)**0.5
        
        # Transport Coefficients (Anomalous Transport Models)
        self.chi_e = np.ones(self.nr) # Electron diffusivity
        self.chi_i = np.ones(self.nr) # Ion diffusivity
        self.D_n = np.ones(self.nr)   # Particle diffusivity
        
        # Impurity Profile (Tungsten density)
        self.n_impurity = np.zeros(self.nr)

        # Neoclassical transport configuration (None = constant chi_base=0.5)
        self.neoclassical_params = None

        # Energy conservation diagnostic (updated each evolve_profiles call)
        self._last_conservation_error: float = 0.0

    def set_neoclassical(self, R0, a, B0, A_ion=2.0, Z_eff=1.5, q0=1.0, q_edge=3.0):
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

    def inject_impurities(self, flux_from_wall_per_sec, dt):
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

    def calculate_bootstrap_current_simple(self, R0, B_pol):
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
        B_pol = np.maximum(B_pol, 0.1) # Avoid div by zero at axis

        J_bs = 1.2 * (f_trapped / B_pol) * dP_drho / (self.cfg["dimensions"]["R_max"] - self.cfg["dimensions"]["R_min"])

        # Ensure it's zero at axis and edge
        J_bs[0] = 0
        J_bs[-1] = 0

        return J_bs

    def calculate_bootstrap_current(self, R0, B_pol):
        """Calculate bootstrap current. Uses full Sauter if neoclassical params set."""
        if hasattr(self, 'neoclassical_params') and self.neoclassical_params is not None:
            return calculate_sauter_bootstrap_current_full(
                self.rho, self.Te, self.Ti, self.ne,
                self.neoclassical_params.get('q_profile', np.linspace(1, 4, len(self.rho))),
                R0, self.neoclassical_params.get('a', 2.0),
                self.neoclassical_params.get('B0', 5.3),
                self.neoclassical_params.get('Z_eff', 1.5),
            )
        return self.calculate_bootstrap_current_simple(R0, B_pol)

    def _gyro_bohm_chi(self) -> np.ndarray:
        """Gyro-Bohm anomalous transport diffusivity [m^2/s].

        chi_gB = c_gB * rho_s^2 * c_s / (a * q * R)

        where rho_s = sqrt(T_i m_i) / (e B), c_s = sqrt(T_e / m_i).

        The calibration coefficient c_gB is loaded from
        ``validation/reference_data/itpa/gyro_bohm_coefficients.json``
        if available (calibrated against the ITPA H-mode confinement
        database by ``tools/calibrate_gyro_bohm.py``).  Falls back to
        the value in ``neoclassical_params['c_gB']`` if explicitly set,
        or to the module-level default (0.1) otherwise.
        """
        if self.neoclassical_params is None:
            return np.full_like(self.rho, 0.5)

        p = self.neoclassical_params
        R0 = p['R0']
        a = p['a']
        B0 = p['B0']
        A_ion = p.get('A_ion', 2.0)
        q = p['q_profile']

        # Load c_gB: explicit param > JSON file > default
        if 'c_gB' in p:
            c_gB = p['c_gB']
        else:
            c_gB = _load_gyro_bohm_coefficient()

        e_charge = 1.602176634e-19
        m_i = A_ion * 1.672621924e-27

        chi_gB = np.zeros_like(self.rho)
        for i in range(len(self.rho)):
            Ti_keV = max(self.Ti[i], 0.01)
            Te_keV = max(self.Te[i], 0.01)
            qi = max(q[i], 0.5)

            T_i_J = Ti_keV * 1e3 * e_charge
            T_e_J = Te_keV * 1e3 * e_charge

            rho_s = np.sqrt(T_i_J * m_i) / (e_charge * B0)
            c_s = np.sqrt(T_e_J / m_i)

            chi_val = c_gB * rho_s**2 * c_s / max(a * qi * R0, 1e-6)
            chi_gB[i] = max(chi_val, 0.01) if np.isfinite(chi_val) else 0.01

        return chi_gB

    def update_transport_model(self, P_aux):
        """
        Gyro-Bohm + neoclassical transport model with EPED-like pedestal.

        When neoclassical params are set, uses:
        - Chang-Hinton neoclassical chi as additive floor
        - Gyro-Bohm anomalous transport (calibrated c_gB)
        - EPED-like pedestal model for H-mode boundary condition

        Falls back to constant chi_base=0.5 when neoclassical is not configured.
        """
        # 1. Critical Gradient Model
        grad_T = np.gradient(self.Ti, self.drho)
        threshold = 2.0

        # Base Level: neoclassical + gyro-Bohm, or constant fallback
        if self.neoclassical_params is not None:
            p = self.neoclassical_params
            chi_nc = chang_hinton_chi_profile(
                self.rho, self.Ti, self.ne,
                p['q_profile'], p['R0'], p['a'], p['B0'],
                p['A_ion'], p['Z_eff']
            )
            chi_gB = self._gyro_bohm_chi()
            chi_base = chi_nc + chi_gB
        else:
            chi_base = np.full_like(self.rho, 0.5)

        # Turbulent Level (critical gradient excess)
        chi_turb = 5.0 * np.maximum(0, -grad_T - threshold)

        # H-Mode detection and EPED-like pedestal model
        is_H_mode = P_aux > 30.0  # MW

        if is_H_mode and self.neoclassical_params is not None:
            try:
                from scpn_fusion.core.eped_pedestal import EpedPedestalModel

                p = self.neoclassical_params
                eped = EpedPedestalModel(
                    R0=p['R0'], a=p['a'], B0=p['B0'],
                    Ip_MA=p.get('Ip_MA', 15.0),
                    kappa=p.get('kappa', 1.7),
                    A_ion=p.get('A_ion', 2.0),
                    Z_eff=p.get('Z_eff', 1.5),
                )
                # Use current edge density for pedestal prediction
                n_ped = max(float(self.ne[-5]), 1.0)
                ped = eped.predict(n_ped)

                # Apply pedestal: suppress transport inside pedestal region
                ped_start = 1.0 - ped.Delta_ped
                edge_mask = self.rho > ped_start
                chi_turb[edge_mask] *= 0.05  # Strong transport barrier

                # Set pedestal boundary conditions on profiles
                ped_idx = np.searchsorted(self.rho, ped_start)
                if ped_idx < len(self.Te):
                    self.Te[ped_idx:] = np.minimum(
                        self.Te[ped_idx:],
                        ped.T_ped_keV * np.linspace(1.0, 0.1, len(self.Te[ped_idx:]))
                    )
                    self.Ti[ped_idx:] = np.minimum(
                        self.Ti[ped_idx:],
                        ped.T_ped_keV * np.linspace(1.0, 0.1, len(self.Ti[ped_idx:]))
                    )
            except Exception:
                # Fallback: simple edge suppression
                edge_mask = self.rho > 0.9
                chi_turb[edge_mask] *= 0.1
        elif is_H_mode:
            # No neoclassical params — simple suppression
            edge_mask = self.rho > 0.9
            chi_turb[edge_mask] *= 0.1

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

    def _explicit_diffusion_rhs(self, T, chi):
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

    def _build_cn_tridiag(self, chi, dt):
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

    def evolve_profiles(self, dt: float, P_aux: float, enforce_conservation: bool = False) -> tuple[float, float]:
        """Advance Ti by one time step using Crank-Nicolson implicit diffusion.

        The scheme is unconditionally stable, allowing dt up to ~1.0 s
        without NaN.  The full equation solved is:

            (T^{n+1} - T^n)/dt = 0.5*[L_h(T^{n+1}) + L_h(T^n)] + S - Sink

        Parameters
        ----------
        dt : float
            Time step [s].
        P_aux : float
            Auxiliary heating power [MW].
        enforce_conservation : bool
            When True, raise :class:`PhysicsError` if the per-step energy
            conservation error exceeds 1%.
        """
        Ti_old = self.Ti.copy()

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

        # ── Energy conservation diagnostic ──
        e_keV_J = 1.602176634e-16  # J per keV
        dims = self.cfg["dimensions"]
        R0 = (dims["R_min"] + dims["R_max"]) / 2.0
        a_minor = (dims["R_max"] - dims["R_min"]) / 2.0
        dV = 2.0 * np.pi * R0 * 2.0 * np.pi * self.rho * a_minor**2 * self.drho

        W_before = 1.5 * np.sum(self.ne * 1e19 * Ti_old * e_keV_J * dV)
        W_after = 1.5 * np.sum(self.ne * 1e19 * self.Ti * e_keV_J * dV)
        dW_source = dt * 1.5 * np.sum(self.ne * 1e19 * net_source * e_keV_J * dV)

        dW_actual = W_after - W_before
        self._last_conservation_error = (
            abs(dW_actual - dW_source) / max(abs(W_before), 1e-10)
        )

        if enforce_conservation and self._last_conservation_error > 0.01:
            raise PhysicsError(
                f"Energy conservation violated: relative error "
                f"{self._last_conservation_error:.4e} > 1% threshold. "
                f"W_before={W_before:.4e} J, W_after={W_after:.4e} J, "
                f"dW_source={dW_source:.4e} J."
            )

        return float(np.mean(self.Ti)), float(self.Ti[0])

    def map_profiles_to_2d(self):
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
        if abs(Psi_edge - Psi_axis) < 1.0: Psi_edge = np.min(self.Psi)
        
        # 2. Calculate Rho for every 2D point
        denom = Psi_edge - Psi_axis
        if abs(denom) < 1e-9: denom = 1e-9
        Psi_norm = (self.Psi - Psi_axis) / denom
        Psi_norm = np.clip(Psi_norm, 0, 1)
        Rho_2D = np.sqrt(Psi_norm)
        
        # 3. Calculate 1D Bootstrap Current
        R0 = (self.cfg["dimensions"]["R_min"] + self.cfg["dimensions"]["R_max"]) / 2.0
        # Estimate B_pol from Ip
        I_target = self.cfg['physics']['plasma_current_target']
        B_pol_est = (1.256e-6 * I_target) / (2 * np.pi * 0.5 * (self.cfg["dimensions"]["R_max"] - self.cfg["dimensions"]["R_min"]))
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

    # ── GS ↔ transport self-consistency loop ─────────────────────────

    def run_self_consistent(
        self,
        P_aux: float,
        n_inner: int = 100,
        n_outer: int = 10,
        dt: float = 0.01,
        psi_tol: float = 1e-3,
    ) -> dict:
        """Run self-consistent GS <-> transport iteration.

        This implements the standard integrated-modelling loop used by
        codes such as ASTRA and JINTRAC: evolve the 1D transport for
        *n_inner* steps, project profiles onto the 2D grid, re-solve the
        Grad-Shafranov equilibrium, and repeat until the poloidal-flux
        change drops below *psi_tol*.

        Algorithm
        ---------
        1. Run transport for *n_inner* steps (evolve Ti/Te/ne).
        2. Call :meth:`map_profiles_to_2d` to update ``J_phi`` on the 2D grid.
        3. Re-solve the Grad-Shafranov equilibrium with the updated source.
        4. Check psi convergence:
           ``||Psi_new - Psi_old|| / ||Psi_old|| < psi_tol``.
        5. Repeat until converged or *n_outer* iterations exhausted.

        Parameters
        ----------
        P_aux : float
            Auxiliary heating power [MW].
        n_inner : int
            Number of transport evolution steps per outer iteration.
        n_outer : int
            Maximum number of outer (GS re-solve) iterations.
        dt : float
            Transport time step [s].
        psi_tol : float
            Relative psi convergence tolerance.

        Returns
        -------
        dict
            ``{"T_avg": float, "T_core": float, "tau_e": float,
            "n_outer_converged": int, "psi_residuals": list[float],
            "Ti_profile": ndarray, "ne_profile": ndarray,
            "converged": bool}``
        """
        psi_residuals: list[float] = []
        converged = False
        n_outer_converged = 0

        for outer in range(n_outer):
            # Save Psi before this outer iteration
            Psi_old = self.Psi.copy()
            psi_old_norm = float(np.linalg.norm(Psi_old))
            if psi_old_norm < 1e-30:
                psi_old_norm = 1.0  # avoid division by zero on first call

            # 1. Run n_inner transport steps
            for _ in range(n_inner):
                self.update_transport_model(P_aux)
                self.evolve_profiles(dt, P_aux)

            # 2. Project 1D profiles onto 2D GS grid (updates self.J_phi)
            self.map_profiles_to_2d()

            # 3. Re-solve Grad-Shafranov equilibrium
            #    external_profile_mode=True ensures solve_equilibrium uses
            #    the J_phi we just set (no internal source update).
            self.solve_equilibrium()

            # 4. Compute psi convergence metric
            psi_residual = float(
                np.linalg.norm(self.Psi - Psi_old) / psi_old_norm
            )
            psi_residuals.append(psi_residual)
            n_outer_converged = outer + 1

            _logger.info(
                "GS-transport outer iter %d/%d: psi_residual=%.4e",
                outer + 1, n_outer, psi_residual,
            )

            # 5. Convergence check
            if psi_residual < psi_tol:
                converged = True
                _logger.info(
                    "GS-transport converged after %d outer iterations "
                    "(residual %.4e < tol %.4e).",
                    outer + 1, psi_residual, psi_tol,
                )
                break

        T_avg = float(np.mean(self.Ti))
        T_core = float(self.Ti[0])
        tau_e = self.compute_confinement_time(P_aux)

        return {
            "T_avg": T_avg,
            "T_core": T_core,
            "tau_e": tau_e,
            "n_outer_converged": n_outer_converged,
            "psi_residuals": psi_residuals,
            "Ti_profile": self.Ti.copy(),
            "ne_profile": self.ne.copy(),
            "converged": converged,
        }

    # ── Fast one-shot transport path ──────────────────────────────────

    def run_to_steady_state(
        self,
        P_aux: float,
        n_steps: int = 500,
        dt: float = 0.01,
        adaptive: bool = False,
        tol: float = 1e-3,
        self_consistent: bool = False,
        sc_n_inner: int = 100,
        sc_n_outer: int = 10,
        sc_psi_tol: float = 1e-3,
    ) -> dict:
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
        self_consistent : bool
            When True, delegate to :meth:`run_self_consistent` which
            iterates GS <-> transport to convergence.  The remaining
            ``sc_*`` parameters are forwarded.
        sc_n_inner : int
            Transport steps per outer GS iteration (self-consistent mode).
        sc_n_outer : int
            Maximum outer GS iterations (self-consistent mode).
        sc_psi_tol : float
            Relative psi convergence tolerance (self-consistent mode).

        Returns
        -------
        dict
            ``{"T_avg": float, "T_core": float, "tau_e": float,
            "n_steps": int, "Ti_profile": ndarray,
            "ne_profile": ndarray}``
            When adaptive=True, also includes ``dt_final``,
            ``dt_history``, ``error_history``.
            When self_consistent=True, returns the
            :meth:`run_self_consistent` dict instead.
        """
        # ── Self-consistent GS↔transport mode ──
        if self_consistent:
            return self.run_self_consistent(
                P_aux=P_aux,
                n_inner=sc_n_inner,
                n_outer=sc_n_outer,
                dt=dt,
                psi_tol=sc_psi_tol,
            )

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
