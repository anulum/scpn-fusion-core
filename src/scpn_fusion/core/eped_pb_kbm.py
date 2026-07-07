# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Peeling-Ballooning + KBM Constraint-Loop Pedestal Model
"""Two-constraint pedestal prediction following the EPED1 methodology.

Implements the constraint-loop structure of Snyder et al., Phys. Plasmas 16,
056118 (2009): the pedestal width follows the kinetic-ballooning-mode (KBM)
relation and the pedestal height is set by the onset of the coupled
peeling-ballooning (P-B) instability — not by a parametric width-height fit.

Constraints:

1. **KBM width** — ``Delta_psiN = 0.076 * sqrt(beta_p_ped)`` (the EPED1
   width relation).
2. **P-B height** — the ballooning-critical normalised pressure gradient
   ``alpha_crit`` is computed numerically from the infinite-n ideal-MHD
   ballooning equation in s-alpha geometry
   (:func:`scpn_fusion.core.ballooning_solver.find_marginal_stability`),
   and the peeling drive enters through the edge bootstrap current from the
   full Sauter model
   (:func:`scpn_fusion.core.integrated_transport_solver.calculate_sauter_bootstrap_current_full`)
   against the elliptical coupled boundary
   ``(j/j_crit)^2 + (alpha/alpha_crit)^2 = 1`` of Snyder et al., Phys.
   Plasmas 9, 2037 (2002).

The predicted pedestal is the marginally P-B-stable point along the
KBM-width-consistent scan of pedestal temperature.

Honest scope boundary: the ballooning boundary is evaluated in reduced
s-alpha geometry and the peeling coupling uses the published elliptical
proxy — this is NOT an ELITE/EPED production-stability calculation, and no
parity with the EPED code is claimed. The width-height scaling tier
(:class:`scpn_fusion.core.eped_pedestal.EpedPedestalModel`) remains available
as the fast tier via :func:`predict_pedestal`.

References
----------
Snyder P. B. et al. (2009) *Phys. Plasmas* 16, 056118 (EPED1 methodology).
Snyder P. B. et al. (2002) *Phys. Plasmas* 9, 2037 (P-B boundary shape).
Sauter O. et al. (1999) *Phys. Plasmas* 6, 2834 (bootstrap current).
Connor J. W., Hastie R. J., Taylor J. B. (1978) *Phys. Rev. Lett.* 40, 396
(ballooning equation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.ballooning_solver import find_marginal_stability
from scpn_fusion.core.eped_pedestal import EpedPedestalModel, PedestalResult
from scpn_fusion.core.integrated_transport_solver import (
    calculate_sauter_bootstrap_current_full,
)

FloatArray = NDArray[np.float64]

logger = logging.getLogger(__name__)

_E_CHARGE = 1.602176634e-19  # C
_MU0 = 4.0 * np.pi * 1e-7  # H/m

KBM_WIDTH_COEFFICIENT = 0.076  # EPED1: Delta_psiN = 0.076 sqrt(beta_p_ped)


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


@dataclass(frozen=True)
class PBKBMCandidate:
    """One point of the KBM-consistent pedestal-height scan."""

    T_ped_keV: float
    p_ped_kPa: float
    beta_p_ped: float
    Delta_ped: float
    alpha_ped: float
    j_ped_MA_m2: float
    pb_boundary_radius: float
    stable: bool


@dataclass(frozen=True)
class PBKBMPedestalResult:
    """Marginally peeling-ballooning-stable pedestal prediction.

    ``method`` names the physics chain so downstream consumers can
    distinguish this tier from the width-height-scaling fast tier.
    """

    p_ped_kPa: float
    T_ped_keV: float
    n_ped_1e19: float
    Delta_ped: float
    beta_p_ped: float
    alpha_ped: float
    alpha_crit: float
    s_ped: float
    q_ped: float
    j_ped_MA_m2: float
    j_crit_MA_m2: float
    pb_boundary_radius: float
    converged: bool
    candidates_evaluated: int
    method: str = "pb_kbm_salpha"


class PBKBMPedestalModel:
    """Peeling-ballooning + KBM constraint-loop pedestal model.

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
    delta : float
        Triangularity (|delta| < 1).
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
        delta: float = 0.3,
        A_ion: float = 2.0,
        Z_eff: float = 1.5,
    ) -> None:
        self.R0 = _require_positive_finite("R0", R0)
        self.a = _require_positive_finite("a", a)
        self.B0 = _require_positive_finite("B0", B0)
        self.Ip_MA = _require_positive_finite("Ip_MA", Ip_MA)
        self.kappa = _require_positive_finite("kappa", kappa)
        if not np.isfinite(float(delta)) or abs(float(delta)) >= 1.0:
            raise ValueError(f"delta must be finite with |delta| < 1, got {delta!r}")
        self.delta = float(delta)
        self.A_ion = _require_positive_finite("A_ion", A_ion)
        self.Z_eff = _require_positive_finite("Z_eff", Z_eff)

        self.epsilon = self.a / self.R0
        self.B_pol = (
            _MU0 * self.Ip_MA * 1e6 / (2.0 * np.pi * self.a * np.sqrt((1.0 + self.kappa**2) / 2.0))
        )
        # Cylindrical edge safety factor with elongation — the same
        # approximation used by the fast tier so the two tiers differ only
        # in the stability physics, not in the geometry proxy.
        q_ped = (
            (self.B0 / self.R0) * (self.a**2 / (0.2 * self.Ip_MA)) * ((1.0 + self.kappa**2) / 2.0)
        )
        self.q_ped = float(max(q_ped, 2.0))

    def pedestal_shear(self) -> float:
        """Magnetic shear ``s = (rho/q) dq/drho`` at the pedestal location.

        Uses the parabolic-current cylindrical q-profile
        ``q(rho) = q0 + (q_ped - q0) rho^2`` with ``q0 = 1`` evaluated at
        ``rho = 0.95``, matching the pedestal-top convention of the fast
        tier.

        Returns
        -------
        float
            Magnetic shear at the pedestal.
        """
        rho_ped = 0.95
        q0 = 1.0
        q_at = q0 + (self.q_ped - q0) * rho_ped**2
        dq_drho = 2.0 * (self.q_ped - q0) * rho_ped
        return float(rho_ped * dq_drho / max(q_at, 1e-9))

    def ballooning_alpha_crit(self) -> float:
        """Numerical ballooning-critical alpha at the pedestal shear.

        Solves the infinite-n ideal-MHD ballooning equation in s-alpha
        geometry by bisection over alpha at fixed pedestal shear. This
        replaces the parametric ``alpha_crit(s)`` fit of the fast tier.

        Returns
        -------
        float
            First-stability critical normalised pressure gradient.
        """
        s_ped = self.pedestal_shear()
        alpha_crit = find_marginal_stability(s_ped, alpha_max=3.0)
        return float(max(alpha_crit, 1e-3))

    def peeling_current_limit(self) -> float:
        """Critical edge current density of the elliptical P-B boundary.

        Snyder et al. (2002) shaping calibration:
        ``j_crit = 2 B_pol f_shape / (mu0 q_ped^2 R0)`` with
        ``f_shape = (1 + 0.5 (kappa - 1)) (1 + 0.8 delta)``.

        Returns
        -------
        float
            Critical edge parallel current density [A/m^2].
        """
        f_shape = (1.0 + 0.5 * (self.kappa - 1.0)) * (1.0 + 0.8 * self.delta)
        j_crit = 2.0 * self.B_pol * f_shape / (_MU0 * self.q_ped**2 * self.R0)
        return float(max(j_crit, 1e-6))

    def _pedestal_profiles(
        self, n_ped_1e19: float, T_ped_keV: float, delta_psin: float
    ) -> dict[str, FloatArray]:
        """Model core+pedestal profiles for the edge bootstrap evaluation.

        A tanh-like pedestal of width ``delta_psin`` carries the pedestal-top
        values; the core carries a mild parabolic peaking on top. The q
        profile is the parabolic-current cylindrical model shared with
        :meth:`pedestal_shear`.
        """
        rho = np.linspace(0.0, 1.0, 201)
        width = max(delta_psin, 0.01)
        ped_centre = 1.0 - width / 2.0
        shape = 0.5 * (1.0 - np.tanh((rho - ped_centre) / (width / 2.0)))
        Te = T_ped_keV * (shape + 1.0) + 2.0 * T_ped_keV * (1.0 - rho**2) ** 1.5
        ne = n_ped_1e19 * (shape + 1.0) / 2.0 + n_ped_1e19 * 0.3 * (1.0 - rho**2)
        q = 1.0 + (self.q_ped - 1.0) * rho**2
        return {
            "rho": rho,
            "Te": np.maximum(Te, 0.05),
            "Ti": np.maximum(Te, 0.05),
            "ne": np.maximum(ne, 0.1),
            "q": q,
        }

    def edge_bootstrap_current(
        self, n_ped_1e19: float, T_ped_keV: float, delta_psin: float
    ) -> float:
        """Peak Sauter bootstrap current density across the pedestal [A/m^2].

        Parameters
        ----------
        n_ped_1e19 : float
            Pedestal-top electron density [1e19 m^-3].
        T_ped_keV : float
            Pedestal-top temperature [keV].
        delta_psin : float
            Pedestal width in normalised flux.

        Returns
        -------
        float
            Maximum bootstrap current density over the pedestal region.
        """
        profiles = self._pedestal_profiles(n_ped_1e19, T_ped_keV, delta_psin)
        j_bs = calculate_sauter_bootstrap_current_full(
            profiles["rho"],
            profiles["Te"],
            profiles["Ti"],
            profiles["ne"],
            profiles["q"],
            self.R0,
            self.a,
            self.B0,
            Z_eff=self.Z_eff,
        )
        pedestal_mask = profiles["rho"] >= 1.0 - 2.0 * max(delta_psin, 0.01)
        return float(np.max(np.abs(j_bs[pedestal_mask])))

    def _evaluate_candidate(
        self, n_ped_1e19: float, T_ped_keV: float, alpha_crit: float, j_crit: float
    ) -> PBKBMCandidate:
        """Evaluate the P-B boundary for one KBM-consistent pedestal height."""
        n_e = n_ped_1e19 * 1e19
        T_J = T_ped_keV * 1e3 * _E_CHARGE
        p_ped = 2.0 * n_e * T_J
        beta_p_ped = 2.0 * _MU0 * p_ped / self.B_pol**2

        delta_psin = float(
            np.clip(KBM_WIDTH_COEFFICIENT * np.sqrt(max(beta_p_ped, 1e-12)), 0.01, 0.15)
        )
        dp_dr = p_ped / max(delta_psin * self.a, 1e-6)
        alpha_ped = 2.0 * _MU0 * self.R0 * self.q_ped**2 / self.B0**2 * dp_dr

        j_ped = self.edge_bootstrap_current(n_ped_1e19, T_ped_keV, delta_psin)
        radius = float(np.sqrt((j_ped / j_crit) ** 2 + (alpha_ped / alpha_crit) ** 2))
        return PBKBMCandidate(
            T_ped_keV=float(T_ped_keV),
            p_ped_kPa=float(p_ped / 1e3),
            beta_p_ped=float(beta_p_ped),
            Delta_ped=delta_psin,
            alpha_ped=float(alpha_ped),
            j_ped_MA_m2=float(j_ped / 1e6),
            pb_boundary_radius=radius,
            stable=radius < 1.0,
        )

    def predict(
        self,
        n_ped_1e19: float,
        *,
        T_min_keV: float = 0.1,
        T_max_keV: float = 12.0,
        coarse_points: int = 25,
        refine_iterations: int = 20,
    ) -> PBKBMPedestalResult:
        """Predict the pedestal as the marginally P-B-stable KBM point.

        Scans pedestal temperature upward along the KBM-width-consistent
        family, locates the first peeling-ballooning-unstable candidate, and
        bisects to the marginal-stability boundary.

        Parameters
        ----------
        n_ped_1e19 : float
            Pedestal-top electron density [1e19 m^-3].
        T_min_keV : float
            Lower bound of the pedestal-temperature scan.
        T_max_keV : float
            Upper bound of the pedestal-temperature scan.
        coarse_points : int
            Number of coarse scan points before bisection.
        refine_iterations : int
            Bisection iterations refining the marginal point.

        Returns
        -------
        PBKBMPedestalResult
            Marginal-stability pedestal with full P-B diagnostics.
            ``converged`` is ``False`` when the whole scan stays stable
            (prediction saturates at ``T_max_keV``) or the lowest candidate
            is already unstable (prediction collapses to ``T_min_keV``).
        """
        n_ped = _require_positive_finite("n_ped_1e19", n_ped_1e19)
        t_lo = _require_positive_finite("T_min_keV", T_min_keV)
        t_hi = _require_positive_finite("T_max_keV", T_max_keV)
        if t_hi <= t_lo:
            raise ValueError("T_max_keV must exceed T_min_keV")
        if int(coarse_points) < 3:
            raise ValueError("coarse_points must be at least 3")
        if int(refine_iterations) < 1:
            raise ValueError("refine_iterations must be at least 1")

        s_ped = self.pedestal_shear()
        alpha_crit = self.ballooning_alpha_crit()
        j_crit = self.peeling_current_limit()

        evaluated = 0
        last_stable: PBKBMCandidate | None = None
        first_unstable: PBKBMCandidate | None = None
        for T_ped in np.linspace(t_lo, t_hi, int(coarse_points)):
            candidate = self._evaluate_candidate(n_ped, float(T_ped), alpha_crit, j_crit)
            evaluated += 1
            if candidate.stable:
                last_stable = candidate
            else:
                first_unstable = candidate
                break

        converged = True
        if first_unstable is None:
            # Entire scan stable: the P-B boundary was not reached.
            assert last_stable is not None
            marginal = last_stable
            converged = False
            logger.warning(
                "PB-KBM scan stayed stable up to T_max=%.2f keV; prediction saturates.",
                t_hi,
            )
        elif last_stable is None:
            # Already unstable at the lowest candidate.
            marginal = first_unstable
            converged = False
            logger.warning(
                "PB-KBM scan unstable at T_min=%.2f keV; prediction collapses to the floor.",
                t_lo,
            )
        else:
            lo = last_stable.T_ped_keV
            hi = first_unstable.T_ped_keV
            marginal = last_stable
            for _ in range(int(refine_iterations)):
                mid = 0.5 * (lo + hi)
                candidate = self._evaluate_candidate(n_ped, mid, alpha_crit, j_crit)
                evaluated += 1
                if candidate.stable:
                    marginal = candidate
                    lo = mid
                else:
                    hi = mid

        return PBKBMPedestalResult(
            p_ped_kPa=marginal.p_ped_kPa,
            T_ped_keV=marginal.T_ped_keV,
            n_ped_1e19=float(n_ped),
            Delta_ped=marginal.Delta_ped,
            beta_p_ped=marginal.beta_p_ped,
            alpha_ped=marginal.alpha_ped,
            alpha_crit=alpha_crit,
            s_ped=s_ped,
            q_ped=self.q_ped,
            j_ped_MA_m2=marginal.j_ped_MA_m2,
            j_crit_MA_m2=float(j_crit / 1e6),
            pb_boundary_radius=marginal.pb_boundary_radius,
            converged=converged,
            candidates_evaluated=evaluated,
        )


def predict_pedestal(
    *,
    R0: float,
    a: float,
    B0: float,
    Ip_MA: float,
    n_ped_1e19: float,
    kappa: float = 1.7,
    delta: float = 0.3,
    A_ion: float = 2.0,
    Z_eff: float = 1.5,
    tier: str = "fast",
) -> PedestalResult | PBKBMPedestalResult:
    """Predict pedestal parameters through the selected model tier.

    Parameters
    ----------
    R0, a, B0, Ip_MA : float
        Machine geometry and current.
    n_ped_1e19 : float
        Pedestal-top electron density [1e19 m^-3].
    kappa, delta, A_ion, Z_eff : float
        Shaping and composition parameters (``delta`` is ignored by the
        fast tier, which carries no triangularity dependence).
    tier : str
        ``"fast"`` selects the width-height-scaling
        :class:`~scpn_fusion.core.eped_pedestal.EpedPedestalModel`;
        ``"pb_kbm"`` selects the constraint-loop
        :class:`PBKBMPedestalModel`.

    Returns
    -------
    PedestalResult | PBKBMPedestalResult
        Tier-specific pedestal prediction.
    """
    tier_norm = tier.strip().lower()
    if tier_norm == "fast":
        fast = EpedPedestalModel(
            R0=R0, a=a, B0=B0, Ip_MA=Ip_MA, kappa=kappa, A_ion=A_ion, Z_eff=Z_eff
        )
        return fast.predict(n_ped_1e19)
    if tier_norm == "pb_kbm":
        model = PBKBMPedestalModel(
            R0=R0,
            a=a,
            B0=B0,
            Ip_MA=Ip_MA,
            kappa=kappa,
            delta=delta,
            A_ion=A_ion,
            Z_eff=Z_eff,
        )
        return model.predict(n_ped_1e19)
    raise ValueError("tier must be 'fast' or 'pb_kbm'")
