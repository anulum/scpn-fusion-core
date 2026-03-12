# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Bayesian Uncertainty Quantification
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Bayesian uncertainty quantification for fusion performance predictions.

Provides error bars on confinement time, fusion power, and Q-factor by
Monte Carlo sampling over scaling-law parameter uncertainties. Uses the
IPB98(y,2) H-mode confinement scaling as the baseline model.

Full-chain propagation (equilibrium -> transport -> fusion) lives in
``uncertainty_full_chain``.

References
----------
- ITER Physics Basis, Nucl. Fusion 39 (1999) 2175
- Verdoolaege et al., Nucl. Fusion 61 (2021) 076006
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from scpn_fusion.core.scaling_laws import load_ipb98y2_coefficients


_SAFE_LOG_MAX = 700.0
_SAFE_LOG_MIN = -745.0


# IPB98(y,2) scaling: τ_E = C · I_p^α_I · B^α_B · P^α_P · n^α_n · R^α_R · A^α_A · κ^α_κ · M^α_M
_COEFFS = load_ipb98y2_coefficients()
_UNC = _COEFFS.get("uncertainties_1sigma", _COEFFS.get("exponent_uncertainties", {}))
IPB98_CENTRAL = {
    "C": _COEFFS["C"],
    "alpha_I": _COEFFS["exponents"]["Ip_MA"],
    "alpha_B": _COEFFS["exponents"]["BT_T"],
    "alpha_P": _COEFFS["exponents"]["Ploss_MW"],
    "alpha_n": _COEFFS["exponents"]["ne19_1e19m3"],
    "alpha_R": _COEFFS["exponents"]["R_m"],
    "alpha_A": -_COEFFS["exponents"]["epsilon"],  # ε = a/R = 1/A
    "alpha_kappa": _COEFFS["exponents"]["kappa"],
    "alpha_M": _COEFFS["exponents"]["M_AMU"],
}

# 1-sigma uncertainties (from Verdoolaege 2021 Bayesian regression)
IPB98_SIGMA = {
    "C": float(_UNC.get("C", 0.012)),
    "alpha_I": float(_UNC.get("Ip_MA", 0.03)),
    "alpha_B": float(_UNC.get("BT_T", 0.05)),
    "alpha_P": float(_UNC.get("Ploss_MW", 0.02)),
    "alpha_n": float(_UNC.get("ne19_1e19m3", 0.04)),
    "alpha_R": float(_UNC.get("R_m", 0.08)),
    "alpha_A": float(_UNC.get("epsilon", 0.06)),
    "alpha_kappa": float(_UNC.get("kappa", 0.07)),
    "alpha_M": float(_UNC.get("M_AMU", 0.04)),
}


def _validate_n_samples(n_samples: int) -> int:
    """Validate Monte Carlo sample count and normalise to int."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise ValueError("n_samples must be an integer >= 1")
    parsed = int(n_samples)
    if parsed < 1:
        raise ValueError("n_samples must be an integer >= 1")
    return parsed


def _validate_seed(seed: Optional[int]) -> Optional[int]:
    """Validate optional RNG seed."""
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer >= 0 or None")
    parsed = int(seed)
    if parsed < 0:
        raise ValueError("seed must be an integer >= 0 or None")
    return parsed


def _require_positive_finite(name: str, value: float) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return parsed


def _validate_scenario(scenario: PlasmaScenario) -> PlasmaScenario:
    """Validate that all scenario parameters are finite and physically positive."""
    scenario.I_p = _require_positive_finite("scenario.I_p", scenario.I_p)
    scenario.B_t = _require_positive_finite("scenario.B_t", scenario.B_t)
    scenario.P_heat = _require_positive_finite("scenario.P_heat", scenario.P_heat)
    scenario.n_e = _require_positive_finite("scenario.n_e", scenario.n_e)
    scenario.R = _require_positive_finite("scenario.R", scenario.R)
    scenario.A = _require_positive_finite("scenario.A", scenario.A)
    scenario.kappa = _require_positive_finite("scenario.kappa", scenario.kappa)
    scenario.M = _require_positive_finite("scenario.M", scenario.M)
    return scenario


def _safe_exp_from_log(log_value: float, *, name: str) -> float:
    """Exponentiate bounded log value; raise when outside stable float range."""
    if not np.isfinite(log_value):
        raise ValueError(f"{name} became non-finite in log space")
    if log_value > _SAFE_LOG_MAX or log_value < _SAFE_LOG_MIN:
        raise ValueError(f"{name} outside numerically stable range (log={log_value:.2f})")
    out = float(np.exp(log_value))
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} is non-finite after exponentiation")
    return out


@dataclass
class PlasmaScenario:
    """Input plasma parameters for a confinement prediction."""

    I_p: float  # Plasma current (MA)
    B_t: float  # Toroidal field (T)
    P_heat: float  # Total heating power (MW)
    n_e: float  # Line-average electron density (10^19 m^-3)
    R: float  # Major radius (m)
    A: float  # Aspect ratio R/a
    kappa: float  # Elongation
    M: float = 2.5  # Effective ion mass (AMU, 2.5 for D-T)


@dataclass
class UQResult:
    """Uncertainty-quantified prediction result."""

    tau_E: float  # Confinement time (s)
    P_fusion: float  # Fusion power (MW)
    Q: float  # Fusion gain Q = P_fus / P_heat

    tau_E_sigma: float
    P_fusion_sigma: float
    Q_sigma: float

    # Percentiles [5%, 25%, 50%, 75%, 95%]
    tau_E_percentiles: np.ndarray = field(default_factory=lambda: np.zeros(5))
    P_fusion_percentiles: np.ndarray = field(default_factory=lambda: np.zeros(5))
    Q_percentiles: np.ndarray = field(default_factory=lambda: np.zeros(5))

    n_samples: int = 0


def ipb98_tau_e(scenario: PlasmaScenario, params: Optional[dict] = None) -> float:
    """
    Compute IPB98(y,2) confinement time for given plasma parameters.

    Parameters
    ----------
    scenario : PlasmaScenario
        Plasma parameters.
    params : dict, optional
        Scaling law coefficients. Defaults to IPB98 central values.

    Returns
    -------
    float — confinement time in seconds.
    """
    scenario = _validate_scenario(scenario)
    p = params or IPB98_CENTRAL
    required = (
        "C",
        "alpha_I",
        "alpha_B",
        "alpha_P",
        "alpha_n",
        "alpha_R",
        "alpha_A",
        "alpha_kappa",
        "alpha_M",
    )
    for key in required:
        if key not in p:
            raise ValueError(f"params missing required key '{key}'")
        if not np.isfinite(float(p[key])):
            raise ValueError(f"params.{key} must be finite")
    c = float(p["C"])
    if c <= 0.0:
        raise ValueError("params.C must be finite and > 0")

    log_tau = (
        np.log(c)
        + float(p["alpha_I"]) * np.log(scenario.I_p)
        + float(p["alpha_B"]) * np.log(scenario.B_t)
        + float(p["alpha_P"]) * np.log(scenario.P_heat)
        + float(p["alpha_n"]) * np.log(scenario.n_e)
        + float(p["alpha_R"]) * np.log(scenario.R)
        + float(p["alpha_A"]) * np.log(scenario.A)
        + float(p["alpha_kappa"]) * np.log(scenario.kappa)
        + float(p["alpha_M"]) * np.log(scenario.M)
    )
    return _safe_exp_from_log(float(log_tau), name="ipb98_tau_e")


def _dt_reactivity(Ti_keV):
    """D-T fusion reactivity <sigma v> [m^3/s].

    Bosch & Hale, Nuclear Fusion 32 (1992) 611, Table IV.
    Valid for 0.2 <= T <= 100 keV.  Accepts scalar or array.
    """
    T = np.clip(np.asarray(Ti_keV, dtype=np.float64), 0.2, 100.0)
    # Bosch-Hale D-T coefficients (Table IV, 0.2-100 keV range)
    _BG2 = 34.3827**2  # keV
    _MRC2 = 1124656.0  # keV
    _C1 = 1.17302e-9
    _C2, _C3 = 1.51361e-2, 7.51886e-2
    _C4, _C5 = 4.60643e-3, 1.35000e-2
    _C6, _C7 = -1.06750e-4, 1.36600e-5
    theta = T / (1.0 - T * (_C2 + T * (_C4 + T * _C6)) / (1.0 + T * (_C3 + T * (_C5 + T * _C7))))
    xi = (_BG2 / (4.0 * theta)) ** (1.0 / 3.0)
    # Formula yields cm^3/s; convert to m^3/s
    sv = _C1 * theta * np.sqrt(xi / (_MRC2 * T**3)) * np.exp(-3.0 * xi) * 1e-6
    sv = np.maximum(sv, 0.0)
    return float(sv) if sv.ndim == 0 else sv


def fusion_power_from_tau(scenario: PlasmaScenario, tau_E: float) -> float:
    """
    Estimate fusion power from cross-section integrated thermal reactivity.
    P_fus = n_D * n_T * <sigma v> * V * E_fusion

    Includes alpha self-heating iteration: in a burning plasma, 20% of
    fusion energy (3.5 MeV alphas out of 17.6 MeV total) heats the
    plasma, raising Ti and thus reactivity.  Three fixed-point iterations
    converge for ITER-class Q~10 scenarios.
    """
    scenario = _validate_scenario(scenario)
    tau_E = _require_positive_finite("tau_E", tau_E)

    a = scenario.R / scenario.A
    V = 2.0 * np.pi**2 * scenario.R * a**2 * scenario.kappa
    ne = scenario.n_e * 1e19  # m^-3

    k_B_keV = 1.602176634e-16  # J per keV
    E_fus_J = 17.6e6 * 1.602176634e-19  # J per D-T reaction
    f_alpha = 3.5 / 17.6  # fraction of fusion energy to alphas

    def _pfus_at_Ptotal(P_tot: float) -> float:
        W_MJ = P_tot * tau_E
        Ti = (W_MJ * 1e6) / (3.0 * ne * k_B_keV * V)
        Ti = float(np.clip(Ti, 0.5, 100.0))
        sv = _dt_reactivity(Ti)
        return (0.25 * ne**2 * sv * V * E_fus_J) / 1e6

    pfus_0 = _pfus_at_Ptotal(scenario.P_heat)
    pfus_mw = _pfus_at_Ptotal(scenario.P_heat + f_alpha * pfus_0)

    return float(pfus_mw)


def quantify_uncertainty(
    scenario: PlasmaScenario, n_samples: int = 10000, seed: Optional[int] = None
) -> UQResult:
    """
    Monte Carlo uncertainty quantification for fusion performance.

    Samples scaling-law coefficients from their Gaussian posteriors and
    propagates through the confinement and fusion power models.

    Parameters
    ----------
    scenario : PlasmaScenario
        Plasma parameters (held fixed).
    n_samples : int
        Number of Monte Carlo samples (default 10,000).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    UQResult — central estimates + error bars + percentiles.
    """
    scenario = _validate_scenario(scenario)
    n_samples = _validate_n_samples(n_samples)
    seed = _validate_seed(seed)
    rng = np.random.default_rng(seed)

    tau_samples = np.zeros(n_samples)
    pfus_samples = np.zeros(n_samples)
    q_samples = np.zeros(n_samples)

    for i in range(n_samples):
        params = {}
        for key in IPB98_CENTRAL:
            params[key] = rng.normal(IPB98_CENTRAL[key], IPB98_SIGMA[key])

        params["C"] = max(params["C"], 1e-4)
        params["alpha_P"] = min(params["alpha_P"], -0.1)

        tau = ipb98_tau_e(scenario, params)
        tau = max(tau, 1e-6)

        pfus = fusion_power_from_tau(scenario, tau)
        pfus = max(pfus, 0.0)

        q = pfus / scenario.P_heat if scenario.P_heat > 0 else 0.0
        if not np.isfinite(q):
            q = 0.0

        tau_samples[i] = tau
        pfus_samples[i] = pfus
        q_samples[i] = q

    pcts = [5, 25, 50, 75, 95]

    return UQResult(
        tau_E=float(np.median(tau_samples)),
        P_fusion=float(np.median(pfus_samples)),
        Q=float(np.median(q_samples)),
        tau_E_sigma=float(np.std(tau_samples)),
        P_fusion_sigma=float(np.std(pfus_samples)),
        Q_sigma=float(np.std(q_samples)),
        tau_E_percentiles=np.asarray(np.percentile(tau_samples, pcts), dtype=float),
        P_fusion_percentiles=np.asarray(
            np.percentile(pfus_samples, pcts),
            dtype=float,
        ),
        Q_percentiles=np.asarray(np.percentile(q_samples, pcts), dtype=float),
        n_samples=n_samples,
    )


# Re-exports from uncertainty_full_chain for backward compatibility.
# All existing `from scpn_fusion.core.uncertainty import X` statements
# continue to work unchanged.
from scpn_fusion.core.uncertainty_full_chain import (  # noqa: E402, F401
    EquilibriumUncertainty,
    FullChainUQResult,
    TransportUncertainty,
    quantify_full_chain,
    summarize_uq,
)
