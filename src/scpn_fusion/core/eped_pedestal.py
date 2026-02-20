# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — EPED-like Pedestal Model
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
EPED-like simplified pedestal scaling model for H-mode tokamak plasmas.

Based on Snyder et al., Physics of Plasmas 16, 056118 (2009) and
Snyder et al., Physics of Plasmas 18, 056115 (2011).

This is NOT a full EPED code (which requires coupled peeling-ballooning
stability + kinetic ballooning mode analysis). Instead, it implements
the published width-height scaling:

    Delta_ped ~ 0.076 * beta_p_ped^{0.5} * nu_star_ped^{-0.2}

combined with pressure-balance constraints to predict (p_ped, T_ped,
n_ped, Delta_ped).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Physical constants
_E_CHARGE = 1.602176634e-19  # C
_M_P = 1.672621924e-27       # kg
_EPS0 = 8.854187812e-12      # F/m

_EPED_DOMAIN_BOUNDS: Dict[str, Tuple[float, float, str]] = {
    "R0": (1.0, 10.0, "m"),
    "a": (0.3, 3.5, "m"),
    "B0": (1.5, 12.0, "T"),
    "Ip_MA": (0.5, 25.0, "MA"),
    "kappa": (1.2, 2.4, "-"),
    "A_ion": (1.0, 3.5, "-"),
    "Z_eff": (1.0, 3.5, "-"),
    "epsilon": (0.15, 0.50, "-"),
    "n_ped_1e19": (2.0, 15.0, "1e19 m^-3"),
    "T_ped_guess_keV": (0.2, 8.0, "keV"),
}
_EXTRAPOLATION_PENALTY_SLOPE = 0.35
_MIN_EXTRAPOLATION_PENALTY = 0.65


def _require_positive_finite(name: str, value: float) -> float:
    """Parse and validate a finite, strictly positive scalar input."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return parsed


def _normalized_domain_violation(value: float, *, lower: float, upper: float) -> float:
    """Return zero in-domain, else normalized distance outside [lower, upper]."""
    span = max(upper - lower, 1e-12)
    if value < lower:
        return float((lower - value) / span)
    if value > upper:
        return float((value - upper) / span)
    return 0.0


@dataclass(frozen=True)
class PedestalDomainAssessment:
    """Domain-validity check for one pedestal prediction request."""

    in_domain: bool
    max_violation: float
    extrapolation_penalty: float
    violations: tuple[str, ...]


@dataclass
class PedestalResult:
    """Result of EPED-like pedestal prediction."""
    p_ped_kPa: float       # pedestal-top pressure [kPa]
    T_ped_keV: float       # pedestal-top temperature [keV]
    n_ped_1e19: float      # pedestal-top density [10^19 m^-3]
    Delta_ped: float       # normalised pedestal width (in psi_N)
    beta_p_ped: float      # poloidal beta at pedestal
    nu_star_ped: float     # pedestal collisionality
    in_domain: bool = True
    extrapolation_score: float = 0.0
    extrapolation_penalty: float = 1.0
    domain_violations: tuple[str, ...] = ()


class EpedPedestalModel:
    """Simplified EPED-like pedestal model.

    Parameters
    ----------
    R0 : float
        Major radius [m].
    a : float
        Minor radius [m].
    B0 : float
        Toroidal field on axis [T].
    Ip_MA : float
        Plasma current [MA].
    kappa : float
        Elongation.
    A_ion : float
        Ion mass number (default 2 = deuterium).
    Z_eff : float
        Effective charge.
    """

    def __init__(
        self,
        R0: float,
        a: float,
        B0: float,
        Ip_MA: float,
        kappa: float = 1.7,
        A_ion: float = 2.0,
        Z_eff: float = 1.5,
    ) -> None:
        self.R0 = _require_positive_finite("R0", R0)
        self.a = _require_positive_finite("a", a)
        self.B0 = _require_positive_finite("B0", B0)
        self.Ip_MA = _require_positive_finite("Ip_MA", Ip_MA)
        self.kappa = _require_positive_finite("kappa", kappa)
        self.A_ion = _require_positive_finite("A_ion", A_ion)
        self.Z_eff = _require_positive_finite("Z_eff", Z_eff)

        # Derived
        self.epsilon = self.a / self.R0
        mu0 = 4.0 * np.pi * 1e-7
        self.B_pol = mu0 * self.Ip_MA * 1e6 / (2.0 * np.pi * self.a * np.sqrt(
            (1.0 + self.kappa**2) / 2.0
        ))

    @classmethod
    def domain_metadata(cls) -> dict[str, dict[str, object]]:
        """Return applicability bounds for the EPED-like surrogate."""
        return {
            name: {"min": lo, "max": hi, "units": units}
            for name, (lo, hi, units) in _EPED_DOMAIN_BOUNDS.items()
        }

    def assess_domain(
        self,
        *,
        n_ped_1e19: float,
        T_ped_guess_keV: float,
    ) -> PedestalDomainAssessment:
        """Assess whether a query lies inside the calibrated EPED-like domain."""
        values = {
            "R0": self.R0,
            "a": self.a,
            "B0": self.B0,
            "Ip_MA": self.Ip_MA,
            "kappa": self.kappa,
            "A_ion": self.A_ion,
            "Z_eff": self.Z_eff,
            "epsilon": self.epsilon,
            "n_ped_1e19": n_ped_1e19,
            "T_ped_guess_keV": T_ped_guess_keV,
        }
        max_violation = 0.0
        violations: list[str] = []
        for name, value in values.items():
            lo, hi, units = _EPED_DOMAIN_BOUNDS[name]
            violation = _normalized_domain_violation(value, lower=lo, upper=hi)
            if violation > 0.0:
                violations.append(
                    f"{name}={value:.4g} outside [{lo:.4g}, {hi:.4g}] {units}"
                )
            max_violation = max(max_violation, violation)
        penalty = float(
            np.clip(
                1.0 - _EXTRAPOLATION_PENALTY_SLOPE * max_violation,
                _MIN_EXTRAPOLATION_PENALTY,
                1.0,
            )
        )
        return PedestalDomainAssessment(
            in_domain=bool(max_violation <= 0.0),
            max_violation=float(max_violation),
            extrapolation_penalty=penalty,
            violations=tuple(violations),
        )

    def predict(
        self,
        n_ped_1e19: float,
        T_ped_guess_keV: float = 3.0,
        *,
        domain_mode: str = "warn",
    ) -> PedestalResult:
        """Predict pedestal parameters for given pedestal density.

        Parameters
        ----------
        n_ped_1e19 : float
            Pedestal-top electron density [10^19 m^-3].
        T_ped_guess_keV : float
            Initial guess for pedestal temperature [keV].

        Returns
        -------
        PedestalResult
            Pedestal predictions.
        """
        n_ped_1e19 = _require_positive_finite("n_ped_1e19", n_ped_1e19)
        T_ped_guess_keV = _require_positive_finite("T_ped_guess_keV", T_ped_guess_keV)
        domain_mode_norm = domain_mode.strip().lower()
        if domain_mode_norm not in {"warn", "raise", "ignore"}:
            raise ValueError("domain_mode must be 'warn', 'raise', or 'ignore'.")

        domain = self.assess_domain(
            n_ped_1e19=n_ped_1e19,
            T_ped_guess_keV=T_ped_guess_keV,
        )
        if not domain.in_domain:
            msg = (
                "EPED-like pedestal query outside calibrated domain; "
                + "; ".join(domain.violations)
                + f". Applying extrapolation_penalty={domain.extrapolation_penalty:.3f}."
            )
            if domain_mode_norm == "raise":
                raise ValueError(msg)
            if domain_mode_norm == "warn":
                logger.warning(msg)

        m_i = self.A_ion * _M_P
        n_e = n_ped_1e19 * 1e19  # m^-3
        mu0 = 4.0 * np.pi * 1e-7

        # Iterative solve: T_ped determines beta_p_ped and nu_star,
        # which determine Delta_ped, which constrains T_ped via
        # pressure gradient limit.
        T_ped = T_ped_guess_keV
        for _ in range(20):
            T_J = T_ped * 1e3 * _E_CHARGE  # keV → J

            # Pedestal pressure
            p_ped = n_e * 2.0 * T_J  # p = n_e (T_e + T_i) ≈ 2 n_e T

            # Poloidal beta at pedestal
            beta_p_ped = 2.0 * mu0 * p_ped / self.B_pol**2

            # Collisionality (electron-ion)
            v_te = np.sqrt(2.0 * T_J / (9.109e-31))
            ln_lambda = 17.0
            nu_ei = (n_e * self.Z_eff * _E_CHARGE**4 * ln_lambda /
                     (12.0 * np.pi**1.5 * _EPS0**2 * (9.109e-31)**0.5 * T_J**1.5))

            # Safety factor at pedestal (≈ edge q)
            q_ped = 5.0 * self.a**2 * self.B0 / (self.R0 * self.Ip_MA * mu0 / (2 * np.pi))
            q_ped = max(q_ped, 2.0)

            eps_ped = 0.95 * self.epsilon  # pedestal at rho~0.95
            eps_ped = max(eps_ped, 0.01)
            nu_star_ped = nu_ei * q_ped * self.R0 / (eps_ped**1.5 * max(v_te, 1.0))

            # Snyder scaling: Delta_ped = 0.076 * beta_p_ped^0.5 * nu_star_ped^{-0.2}
            # Clamp nu_star to avoid blowup at very low collisionality
            nu_star_safe = max(nu_star_ped, 0.001)
            Delta_ped = 0.076 * np.sqrt(max(beta_p_ped, 0.001)) * nu_star_safe**(-0.2)
            Delta_ped *= domain.extrapolation_penalty
            Delta_ped = np.clip(Delta_ped, 0.01, 0.15)

            # Pressure gradient constraint (simplified KBM limit):
            # alpha_MHD ~ -R0 q^2 dp/dr / B^2 <= alpha_crit ~ 2.5
            # dp/dr ~ p_ped / (Delta_ped * a)
            # => T_ped constrained by: alpha_crit * B^2 * Delta_ped * a / (R0 * q^2 * 2 n_e)
            alpha_crit = 2.5
            T_ped_max = (alpha_crit * self.B0**2 * Delta_ped * self.a /
                         (mu0 * self.R0 * q_ped**2 * 2.0 * n_e))
            T_ped_max *= domain.extrapolation_penalty
            T_ped_max_keV = T_ped_max / (1e3 * _E_CHARGE)

            T_ped_new = min(T_ped, T_ped_max_keV)
            T_ped_new = max(T_ped_new, 0.1)

            if abs(T_ped_new - T_ped) / max(T_ped, 1e-9) < 1e-3:
                T_ped = T_ped_new
                break
            T_ped = 0.5 * T_ped + 0.5 * T_ped_new

        p_ped_kPa = n_e * 2.0 * T_ped * 1e3 * _E_CHARGE / 1e3  # kPa

        return PedestalResult(
            p_ped_kPa=float(p_ped_kPa),
            T_ped_keV=float(T_ped),
            n_ped_1e19=float(n_ped_1e19),
            Delta_ped=float(Delta_ped),
            beta_p_ped=float(beta_p_ped),
            nu_star_ped=float(nu_star_ped),
            in_domain=domain.in_domain,
            extrapolation_score=domain.max_violation,
            extrapolation_penalty=domain.extrapolation_penalty,
            domain_violations=domain.violations,
        )
