# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Alpha Particle Guiding-Center Orbit Following
"""Guiding-center orbit tools and fast-ion collisional slowing-down.

The guiding-centre integrator advances ``(R, Z, phi, v_par)`` with the
combined parallel motion plus grad-B and curvature drift, using the conserved
magnetic moment ``mu = m v_perp^2 / (2 B)`` as the perpendicular invariant and
the mirror force ``m dv_par/dt = -mu grad_par B``.

The collisional helpers in :class:`SlowingDown` implement the Spitzer / NRL
Plasma Formulary fast-ion drag model (critical velocity, electron drag
slowing-down time, full velocity slowing-down time, and the ion/electron
heating split) rather than order-of-magnitude heuristics. References:

- NRL Plasma Formulary (2019), "Collisions and Transport" / fast-ion relaxation.
- T. H. Stix, *Plasma Physics* 14, 367 (1972).
- Helander & Sigmar, *Collisional Transport in Magnetized Plasmas* (2002).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BField = Callable[[float, float], tuple[float, float, float]]

# Physical constants (SI, CODATA 2018).
ELEMENTARY_CHARGE_C = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837015e-31
ATOMIC_MASS_KG = 1.66053906660e-27
VACUUM_PERMITTIVITY = 8.8541878128e-12
VACUUM_PERMEABILITY = 1.25663706212e-6

# Default deuterium-tritium background mean ion mass (amu) for the critical
# velocity, and the 3.52 MeV D-T fusion alpha birth energy (keV).
DT_MEAN_ION_AMU = 2.5
ALPHA_BIRTH_ENERGY_KEV = 3520.0
ALPHA_MASS_AMU = 4.002602
ALPHA_CHARGE_NUMBER = 2


@dataclass
class EnsembleResult:
    """Summary metrics from a Monte-Carlo orbit-following ensemble."""

    loss_fraction: float
    heating_profile: FloatArray
    current_drive: float
    n_passing: int
    n_trapped: int
    n_lost: int


class GuidingCenterOrbit:
    """Guiding-centre orbit ``(R, Z, phi, v_par)`` advanced with RK4.

    The perpendicular dynamics are carried by the adiabatic invariant
    ``mu = m v_perp^2 / (2 B)``; the grad-B plus curvature drift uses the
    low-beta combined coefficient ``(v_par^2 + v_perp^2/2) / (omega_c B^2)``
    with ``v_perp^2/2 = mu B / m``.
    """

    def __init__(
        self, m_amu: float, Z: int, E_keV: float, pitch_angle: float, R0_init: float, Z0_init: float
    ):
        if not math.isfinite(m_amu) or m_amu <= 0.0:
            raise ValueError("m_amu must be a positive finite mass number.")
        if not math.isfinite(Z) or Z == 0:
            raise ValueError("Z must be a non-zero charge number.")
        if not math.isfinite(E_keV) or E_keV <= 0.0:
            raise ValueError("E_keV must be a positive finite energy.")
        if not math.isfinite(pitch_angle) or not 0.0 <= pitch_angle <= math.pi:
            raise ValueError("pitch_angle must lie in [0, pi].")
        if not math.isfinite(R0_init) or R0_init <= 0.0:
            raise ValueError("R0_init must be a positive finite major radius.")
        if not math.isfinite(Z0_init):
            raise ValueError("Z0_init must be finite.")

        self.m = m_amu * ATOMIC_MASS_KG
        self.Z_charge = Z * ELEMENTARY_CHARGE_C
        self.E_J = E_keV * 1e3 * ELEMENTARY_CHARGE_C

        self.v_tot = math.sqrt(2.0 * self.E_J / self.m)
        self.v_par = self.v_tot * math.cos(pitch_angle)
        v_perp = self.v_tot * math.sin(pitch_angle)

        self.R = R0_init
        self.Z = Z0_init
        self.phi = 0.0

        # mu is computed lazily from the first sampled |B| (B is only known
        # inside the equation of motion); -1 flags "not yet initialised".
        self.mu = -1.0
        self.v_perp_0 = v_perp

    def _eom(self, state: FloatArray, B_field: BField) -> FloatArray:
        R, Z, phi, v_par = state

        B_R, B_Z, B_phi = B_field(R, Z)
        B_mag = math.sqrt(B_R**2 + B_Z**2 + B_phi**2)
        if B_mag <= 0.0:
            raise ValueError("B_field magnitude must be positive along the orbit.")

        if self.mu < 0:
            self.mu = self.m * self.v_perp_0**2 / (2.0 * B_mag)

        omega_c = self.Z_charge * B_mag / self.m

        # |B| gradient by forward finite differences in (R, Z).
        eps = 1e-4
        B_R_plus, B_Z_plus, B_phi_plus = B_field(R + eps, Z)
        B_mag_R_plus = math.sqrt(B_R_plus**2 + B_Z_plus**2 + B_phi_plus**2)
        dB_dR = (B_mag_R_plus - B_mag) / eps

        B_R_zplus, B_Z_zplus, B_phi_zplus = B_field(R, Z + eps)
        B_mag_Z_plus = math.sqrt(B_R_zplus**2 + B_Z_zplus**2 + B_phi_zplus**2)
        dB_dZ = (B_mag_Z_plus - B_mag) / eps

        # B x grad B with B = (B_R, B_phi, B_Z) and grad|B| = (dB_dR, 0, dB_dZ).
        bxg_R = B_phi * dB_dZ
        bxg_phi = B_Z * dB_dR - B_R * dB_dZ
        bxg_Z = -B_phi * dB_dR

        # Combined grad-B + curvature drift coefficient. The perpendicular
        # contribution is v_perp^2/2 = mu B / m (NOT mu B): mu carries the
        # particle mass, so the energy term must be divided by m to recover a
        # squared velocity consistent with v_par^2.
        drift_coeff = (v_par**2 + self.mu * B_mag / self.m) / (omega_c * B_mag**2)

        dR_dt = v_par * B_R / B_mag + drift_coeff * bxg_R
        dZ_dt = v_par * B_Z / B_mag + drift_coeff * bxg_Z
        dphi_dt = v_par * B_phi / (R * B_mag) + drift_coeff * bxg_phi / R

        # Mirror force: m dv_par/dt = -mu (B . grad B) / B.
        b_dot_grad_b = B_R * dB_dR + B_Z * dB_dZ
        dv_par_dt = -(self.mu / self.m) * b_dot_grad_b / B_mag

        return np.array([dR_dt, dZ_dt, dphi_dt, dv_par_dt])

    def step(self, B_field: BField, dt: float) -> tuple[float, float, float, float]:
        """Advance one RK4 step and return the updated phase-space state."""
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a positive finite time step.")
        state = np.array([self.R, self.Z, self.phi, self.v_par])

        k1 = self._eom(state, B_field)
        k2 = self._eom(state + 0.5 * dt * k1, B_field)
        k3 = self._eom(state + 0.5 * dt * k2, B_field)
        k4 = self._eom(state + dt * k3, B_field)

        state_new = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        self.R, self.Z, self.phi, self.v_par = state_new
        return float(self.R), float(self.Z), float(self.phi), float(self.v_par)


class OrbitClassifier:
    """Classify orbit geometry from sampled traces."""

    @staticmethod
    def classify(
        orbit_R: FloatArray,
        orbit_Z: FloatArray,
        v_par: FloatArray,
        R_wall: float,
        Z_wall_upper: float,
    ) -> str:
        """Classify an orbit as ``lost``, ``trapped``, or ``passing``.

        ``lost`` if the trace crosses the wall envelope, ``trapped`` if the
        parallel velocity reverses sign (bounce point), otherwise ``passing``.
        """
        if orbit_R.size == 0 or v_par.size == 0:
            raise ValueError("orbit traces must be non-empty.")
        if not math.isfinite(R_wall) or R_wall <= 0.0:
            raise ValueError("R_wall must be a positive finite radius.")
        if not math.isfinite(Z_wall_upper) or Z_wall_upper <= 0.0:
            raise ValueError("Z_wall_upper must be a positive finite height.")

        if (
            np.any(orbit_R > R_wall)
            or np.any(np.abs(orbit_Z) > Z_wall_upper)
            or np.any(orbit_R < 0.1)
        ):
            return "lost"

        v_par_signs = np.sign(v_par)
        if np.any(v_par_signs != v_par_signs[0]):
            return "trapped"

        return "passing"


class MonteCarloEnsemble:
    """Monte-Carlo wrapper over many guiding-centre alpha particles."""

    def __init__(self, n_particles: int, E_birth_keV: float, R0: float, a: float, B0: float):
        if not isinstance(n_particles, int | np.integer) or n_particles < 1:
            raise ValueError("n_particles must be a positive integer.")
        for name, value in (("E_birth_keV", E_birth_keV), ("R0", R0), ("a", a), ("B0", B0)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        self.n_particles = int(n_particles)
        self.E_birth_keV = E_birth_keV
        self.R0 = R0
        self.a = a
        self.B0 = B0
        self.particles: list[GuidingCenterOrbit] = []

    def initialize(self, *, seed: int = 0) -> None:
        """Sample initial alpha guiding-centres from a centrally peaked prior.

        The birth minor radius follows a ``Beta(2, 5)`` profile (centrally
        peaked, vanishing at the edge) and the pitch angle is isotropic. The
        sampler is seeded for reproducibility; pass a different ``seed`` for an
        independent ensemble.
        """
        rng = np.random.default_rng(seed)
        self.particles = []
        for _ in range(self.n_particles):
            r = rng.beta(2, 5) * self.a
            theta = rng.uniform(0, 2 * np.pi)
            pitch = rng.uniform(0, np.pi)

            R_init = self.R0 + r * math.cos(theta)
            Z_init = r * math.sin(theta)

            particle = GuidingCenterOrbit(
                ALPHA_MASS_AMU, ALPHA_CHARGE_NUMBER, self.E_birth_keV, pitch, R_init, Z_init
            )
            self.particles.append(particle)

    def follow(self, B_field: BField, n_steps: int = 100, dt: float = 1e-7) -> EnsembleResult:
        """Integrate every particle and return aggregate loss statistics.

        Parameters
        ----------
        B_field : Callable
            Magnetic field callback returning ``(B_R, B_Z, B_phi)``.
        n_steps : int
            Number of RK4 steps per particle.
        dt : float
            Integration step in seconds.
        """
        if not self.particles:
            raise ValueError("call initialize() before follow().")
        if not isinstance(n_steps, int | np.integer) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer.")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a positive finite time step.")

        n_pass = 0
        n_trap = 0
        n_lost = 0
        heating = np.zeros(50)

        for particle in self.particles:
            R_trace = []
            Z_trace = []
            v_trace = []
            for _ in range(int(n_steps)):
                particle.step(B_field, dt)
                R_trace.append(particle.R)
                Z_trace.append(particle.Z)
                v_trace.append(particle.v_par)

            classification = OrbitClassifier.classify(
                np.array(R_trace),
                np.array(Z_trace),
                np.array(v_trace),
                self.R0 + self.a + 0.5,
                self.a + 0.5,
            )
            if classification == "lost":
                n_lost += 1
            elif classification == "trapped":
                n_trap += 1
            else:
                n_pass += 1

        frac = n_lost / max(self.n_particles, 1)
        return EnsembleResult(frac, heating, 0.0, n_pass, n_trap, n_lost)


def first_orbit_loss(
    R0: float,
    a: float,
    Ip_MA: float,
    E_alpha_keV: float = ALPHA_BIRTH_ENERGY_KEV,
    fast_ion_amu: float = ALPHA_MASS_AMU,
    fast_ion_Z: int = ALPHA_CHARGE_NUMBER,
) -> float:
    """Prompt fast-ion loss-zone width as a fraction of the minor radius.

    Returns ``min(1, rho_pol / a)``, where ``rho_pol = m v / (Z e B_pol)`` is the
    poloidal (banana-scale) gyroradius and ``B_pol = mu_0 I_p / (2 pi a)`` is the
    edge poloidal field. Fast ions born within one poloidal gyroradius of the
    last closed flux surface are promptly lost, so ``rho_pol / a`` is the natural
    width of the prompt-loss zone and an order-of-magnitude **upper bound** on the
    prompt loss fraction. This is a confinement-scaling estimate, not a
    Lorentz-orbit Monte-Carlo: the actual lost fraction is lower because the alpha
    birth profile is centrally peaked. The poloidal field (set by ``I_p``), not
    the toroidal field, controls the banana width, which is why the loss falls
    with plasma current.
    """
    if not math.isfinite(R0) or R0 <= 0.0:
        raise ValueError("R0 must be a positive finite major radius.")
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError("a must be a positive finite minor radius.")
    if not math.isfinite(Ip_MA) or Ip_MA <= 0.0:
        raise ValueError("Ip_MA must be a positive finite plasma current.")
    if not math.isfinite(E_alpha_keV) or E_alpha_keV <= 0.0:
        raise ValueError("E_alpha_keV must be a positive finite energy.")
    if not math.isfinite(fast_ion_amu) or fast_ion_amu <= 0.0:
        raise ValueError("fast_ion_amu must be positive and finite.")
    if not math.isfinite(fast_ion_Z) or fast_ion_Z == 0:
        raise ValueError("fast_ion_Z must be a non-zero charge number.")

    mass_kg = fast_ion_amu * ATOMIC_MASS_KG
    energy_j = E_alpha_keV * 1e3 * ELEMENTARY_CHARGE_C
    v_alpha = math.sqrt(2.0 * energy_j / mass_kg)
    b_poloidal = VACUUM_PERMEABILITY * (Ip_MA * 1e6) / (2.0 * math.pi * a)
    rho_poloidal = mass_kg * v_alpha / (abs(fast_ion_Z) * ELEMENTARY_CHARGE_C * b_poloidal)
    return float(min(1.0, rho_poloidal / a))


class SlowingDown:
    """Spitzer / NRL fast-ion collisional slowing-down on a Maxwellian background.

    A fast ion of mass ``m_f`` and charge ``Z_f`` drags on background electrons
    (friction ~ v for v << v_te) and background ions (friction ~ 1/v^2), with the
    transition at the critical velocity ``v_c``. Above ``v_c`` the ion heats
    electrons preferentially; below ``v_c`` it heats bulk ions.
    """

    @staticmethod
    def coulomb_logarithm_ei(Te_keV: float, ne_20: float) -> float:
        """NRL electron-ion Coulomb logarithm for ``T_e > 10`` eV.

        ``ln Lambda_ei = 24 - ln( sqrt(n_e[cm^-3]) / T_e[eV] )``.
        """
        if not math.isfinite(Te_keV) or Te_keV <= 0.0:
            raise ValueError("Te_keV must be positive and finite.")
        if not math.isfinite(ne_20) or ne_20 <= 0.0:
            raise ValueError("ne_20 must be positive and finite.")
        te_ev = Te_keV * 1e3
        ne_cm3 = ne_20 * 1e20 * 1e-6
        return 24.0 - math.log(math.sqrt(ne_cm3) / te_ev)

    @staticmethod
    def critical_velocity(Te_keV: float, background_ion_amu: float = DT_MEAN_ION_AMU) -> float:
        """Critical velocity where electron and ion drag are equal (m/s).

        ``v_c = (3 sqrt(pi)/4 * m_e/m_i)^(1/3) * sqrt(2 T_e / m_e)``
        (NRL Plasma Formulary; Stix 1972). Independent of fast-ion mass/charge
        and of density.
        """
        if not math.isfinite(Te_keV) or Te_keV <= 0.0:
            raise ValueError("Te_keV must be positive and finite.")
        if not math.isfinite(background_ion_amu) or background_ion_amu <= 0.0:
            raise ValueError("background_ion_amu must be positive and finite.")
        te_j = Te_keV * 1e3 * ELEMENTARY_CHARGE_C
        v_te = math.sqrt(2.0 * te_j / ELECTRON_MASS_KG)
        mass_ratio = ELECTRON_MASS_KG / (background_ion_amu * ATOMIC_MASS_KG)
        return float((3.0 * math.sqrt(math.pi) / 4.0 * mass_ratio) ** (1.0 / 3.0) * v_te)

    @staticmethod
    def electron_slowing_down_time(
        Te_keV: float,
        ne_20: float,
        fast_ion_amu: float = ALPHA_MASS_AMU,
        fast_ion_Z: int = ALPHA_CHARGE_NUMBER,
        coulomb_log: float | None = None,
    ) -> float:
        """Electron-drag (Spitzer) slowing-down time ``tau_se`` in seconds.

        ``tau_se = 3 (2 pi)^(1/2) eps0^2 m_f T_e^(3/2)
                   / (m_e^(1/2) n_e Z_f^2 e^4 ln Lambda)``,
        the inverse of the NRL electron drag rate ``nu_se``. For an ITER-like
        D-T alpha (``T_e ~ 20`` keV, ``n_e ~ 1e20`` m^-3) this is ~0.3 s.
        """
        if not math.isfinite(Te_keV) or Te_keV <= 0.0:
            raise ValueError("Te_keV must be positive and finite.")
        if not math.isfinite(ne_20) or ne_20 <= 0.0:
            raise ValueError("ne_20 must be positive and finite.")
        if not math.isfinite(fast_ion_amu) or fast_ion_amu <= 0.0:
            raise ValueError("fast_ion_amu must be positive and finite.")
        if not math.isfinite(fast_ion_Z) or fast_ion_Z == 0:
            raise ValueError("fast_ion_Z must be a non-zero charge number.")
        ln_lambda = (
            SlowingDown.coulomb_logarithm_ei(Te_keV, ne_20)
            if coulomb_log is None
            else float(coulomb_log)
        )
        if not math.isfinite(ln_lambda) or ln_lambda <= 0.0:
            raise ValueError("coulomb_log must be positive and finite.")
        te_j = Te_keV * 1e3 * ELEMENTARY_CHARGE_C
        m_f = fast_ion_amu * ATOMIC_MASS_KG
        n_e = ne_20 * 1e20
        numerator = 3.0 * 2.0**1.5 * math.pi**1.5 * VACUUM_PERMITTIVITY**2 * m_f * te_j**1.5
        denominator = (
            math.sqrt(ELECTRON_MASS_KG) * n_e * fast_ion_Z**2 * ELEMENTARY_CHARGE_C**4 * ln_lambda
        )
        return float(numerator / denominator)

    @staticmethod
    def slowing_down_time(v1: float, v2: float, v_c: float, tau_se: float) -> float:
        """Time to slow from ``v1`` to ``v2`` (``v1 > v2``) in seconds.

        ``t = (tau_se / 3) ln[ (v1^3 + v_c^3) / (v2^3 + v_c^3) ]`` (NRL).
        """
        for name, value in (("v1", v1), ("v2", v2), ("v_c", v_c), ("tau_se", tau_se)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if v2 > v1:
            raise ValueError("v2 must not exceed v1 (slowing down only).")
        ratio = (v1**3 + v_c**3) / (v2**3 + v_c**3)
        return float((tau_se / 3.0) * math.log(ratio))

    @staticmethod
    def heating_partition(v: float, v_c: float) -> tuple[float, float]:
        """Return the (ion, electron) heating fractions at fast-ion speed ``v``.

        ``f_ion = v_c^3 / (v^3 + v_c^3)``: above ``v_c`` the ion heats electrons,
        below ``v_c`` it heats bulk ions.
        """
        if not math.isfinite(v) or v < 0.0:
            raise ValueError("v must be non-negative and finite.")
        if not math.isfinite(v_c) or v_c <= 0.0:
            raise ValueError("v_c must be positive and finite.")
        f_ion = (v_c**3) / (v**3 + v_c**3)
        return float(f_ion), float(1.0 - f_ion)
