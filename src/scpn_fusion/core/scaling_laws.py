# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Confinement Scaling Laws
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Energy confinement time scaling laws for tokamak H-mode plasmas.

Implements the IPB98(y,2) empirical scaling law from the ITER Physics
Basis (Nuclear Fusion 39, 1999, 2175) with updated coefficients from
Verdoolaege et al. (Nuclear Fusion 61, 2021, 076006).

The IPB98(y,2) law is a multi-variate power-law regression fitted to
5920 H-mode data points from 18 tokamaks in the ITPA global confinement
database.  It predicts the thermal energy confinement time τ_E as:

    τ_E = C · Ip^α_I · B_T^α_B · n̄_e19^α_n · P_loss^α_P
              · R^α_R · κ^α_κ · ε^α_ε · M^α_M

Coefficients are loaded from the JSON file shipped in
``validation/reference_data/itpa/ipb98y2_coefficients.json``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import warnings

import numpy as np

logger = logging.getLogger(__name__)

# ── Default coefficient path ──────────────────────────────────────────

_DEFAULT_COEFF_PATH = (
    Path(__file__).resolve().parents[3]
    / "validation"
    / "reference_data"
    / "itpa"
    / "ipb98y2_coefficients.json"
)

_REQUIRED_EXPONENT_KEYS = (
    "Ip_MA",
    "BT_T",
    "ne19_1e19m3",
    "Ploss_MW",
    "R_m",
    "kappa",
    "epsilon",
    "M_AMU",
)

_IPB98Y2_TRAINING_DOMAIN = {
    "Ip": (0.4, 22.0),
    "BT": (0.3, 14.0),
    "ne19": (0.5, 25.0),
    "Ploss": (0.5, 220.0),
    "R": (0.45, 8.5),
    "kappa": (1.0, 3.0),
    "epsilon": (0.08, 0.9),
    "M": (1.0, 3.0),
}


# ── Data container ────────────────────────────────────────────────────

@dataclass
class TransportBenchmarkResult:
    """Result of an IPB98(y,2) benchmark comparison."""

    machine: str
    shot: str
    tau_e_measured: float
    tau_e_predicted: float
    h_factor: float
    relative_error: float


# ── Core functions ────────────────────────────────────────────────────


def _require_positive_finite(name: str, value: float) -> float:
    """Validate scalar inputs for confinement scaling calculations."""
    value_f = float(value)
    if not np.isfinite(value_f) or value_f <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return value_f


def _require_finite_number(name: str, value: Any) -> float:
    """Validate generic numeric metadata loaded from coefficient files."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return parsed


def _validate_ipb98y2_coefficients(raw: Any) -> dict[str, Any]:
    """Validate and normalize an IPB98(y,2) coefficient mapping."""
    if not isinstance(raw, dict):
        raise ValueError("IPB98(y,2) coefficients must be a JSON object")

    if "C" not in raw:
        raise ValueError("IPB98(y,2) coefficients missing required key 'C'")
    C = _require_finite_number("C", raw["C"])
    if C <= 0.0:
        raise ValueError(f"C must be > 0, got {C!r}")

    exp_raw = raw.get("exponents")
    if not isinstance(exp_raw, dict):
        raise ValueError("IPB98(y,2) coefficients key 'exponents' must be an object")

    exponents: dict[str, float] = {}
    for key in _REQUIRED_EXPONENT_KEYS:
        if key not in exp_raw:
            raise ValueError(f"IPB98(y,2) coefficients missing exponent key '{key}'")
        exponents[key] = _require_finite_number(f"exponents.{key}", exp_raw[key])

    normalized: dict[str, Any] = dict(raw)
    normalized["C"] = C
    normalized["exponents"] = exponents

    if "sigma_lnC" in normalized:
        sigma_lnc = _require_finite_number("sigma_lnC", normalized["sigma_lnC"])
        if sigma_lnc < 0.0:
            raise ValueError(f"sigma_lnC must be >= 0, got {sigma_lnc!r}")
        normalized["sigma_lnC"] = sigma_lnc

    if "exponent_uncertainties" in normalized:
        unc_raw = normalized["exponent_uncertainties"]
        if not isinstance(unc_raw, dict):
            raise ValueError("exponent_uncertainties must be an object")
        uncertainties: dict[str, float] = {}
        for key, value in unc_raw.items():
            sigma = _require_finite_number(f"exponent_uncertainties.{key}", value)
            if sigma < 0.0:
                raise ValueError(
                    f"exponent_uncertainties.{key} must be >= 0, got {sigma!r}"
                )
            uncertainties[key] = sigma
        normalized["exponent_uncertainties"] = uncertainties

    return normalized


def load_ipb98y2_coefficients(
    path: Optional[str | Path] = None,
) -> dict:
    """Load IPB98(y,2) coefficients from the JSON reference file.

    Parameters
    ----------
    path : str or Path, optional
        Override path.  Defaults to the file shipped with this package.

    Returns
    -------
    dict
        Parsed JSON with keys ``"C"``, ``"exponents"``, etc.
    """
    p = Path(path) if path else _DEFAULT_COEFF_PATH
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    return _validate_ipb98y2_coefficients(raw)


def assess_ipb98y2_domain(
    *,
    Ip: float,
    BT: float,
    ne19: float,
    Ploss: float,
    R: float,
    kappa: float,
    epsilon: float,
    M: float = 2.5,
) -> dict[str, Any]:
    """Assess whether parameters lie within IPB98(y,2) training domain."""
    values = {
        "Ip": _require_positive_finite("Ip", Ip),
        "BT": _require_positive_finite("BT", BT),
        "ne19": _require_positive_finite("ne19", ne19),
        "Ploss": _require_positive_finite("Ploss", Ploss),
        "R": _require_positive_finite("R", R),
        "kappa": _require_positive_finite("kappa", kappa),
        "epsilon": _require_positive_finite("epsilon", epsilon),
        "M": _require_positive_finite("M", M),
    }
    extrapolated_dimensions: list[str] = []
    frac_extrapolation: dict[str, float] = {}
    for key, val in values.items():
        lo, hi = _IPB98Y2_TRAINING_DOMAIN[key]
        if val < lo:
            extrapolated_dimensions.append(key)
            frac_extrapolation[key] = float((lo - val) / max(lo, 1e-12))
        elif val > hi:
            extrapolated_dimensions.append(key)
            frac_extrapolation[key] = float((val - hi) / max(hi, 1e-12))
    return {
        "in_training_domain": len(extrapolated_dimensions) == 0,
        "extrapolated_dimensions": extrapolated_dimensions,
        "fractional_extrapolation": frac_extrapolation,
        "max_fractional_extrapolation": float(
            max(frac_extrapolation.values(), default=0.0)
        ),
    }


def ipb98y2_tau_e(
    Ip: float,
    BT: float,
    ne19: float,
    Ploss: float,
    R: float,
    kappa: float,
    epsilon: float,
    M: float = 2.5,
    *,
    coefficients: Optional[dict] = None,
    enforce_training_domain: bool = False,
    warn_if_extrapolated: bool = False,
) -> float:
    """Evaluate the IPB98(y,2) confinement time scaling law.

    Parameters
    ----------
    Ip : float
        Plasma current [MA].
    BT : float
        Toroidal magnetic field [T].
    ne19 : float
        Line-averaged electron density [10^19 m^-3].
    Ploss : float
        Loss power [MW].  Must be > 0.
    R : float
        Major radius [m].
    kappa : float
        Elongation (κ).
    epsilon : float
        Inverse aspect ratio a/R.
    M : float
        Effective ion mass [AMU].  Default 2.5 (D-T).
    coefficients : dict, optional
        Pre-loaded coefficient dict.  If *None*, loaded from disk.
    enforce_training_domain : bool
        When True, raises if any input is outside the supported training
        envelope of the empirical fit.
    warn_if_extrapolated : bool
        When True, emits a warning when any input is outside the training
        envelope.

    Returns
    -------
    float
        Predicted thermal energy confinement time τ_E [s].

    Raises
    ------
    ValueError
        If any input is non-finite or non-positive.
    """
    Ip = _require_positive_finite("Ip", Ip)
    BT = _require_positive_finite("BT", BT)
    ne19 = _require_positive_finite("ne19", ne19)
    Ploss = _require_positive_finite("Ploss", Ploss)
    R = _require_positive_finite("R", R)
    kappa = _require_positive_finite("kappa", kappa)
    epsilon = _require_positive_finite("epsilon", epsilon)
    M = _require_positive_finite("M", M)
    domain = assess_ipb98y2_domain(
        Ip=Ip, BT=BT, ne19=ne19, Ploss=Ploss, R=R, kappa=kappa, epsilon=epsilon, M=M
    )
    if enforce_training_domain and not domain["in_training_domain"]:
        dims = ", ".join(domain["extrapolated_dimensions"])
        raise ValueError(
            "IPB98(y,2) inputs outside training domain: " f"{dims}"
        )
    if warn_if_extrapolated and not domain["in_training_domain"]:
        warnings.warn(
            "IPB98(y,2) extrapolation beyond training domain for: "
            + ", ".join(domain["extrapolated_dimensions"]),
            stacklevel=2,
        )

    if coefficients is None:
        coefficients = load_ipb98y2_coefficients()
    else:
        coefficients = _validate_ipb98y2_coefficients(coefficients)

    C = coefficients["C"]
    exp = coefficients["exponents"]

    tau = (
        C
        * Ip ** exp["Ip_MA"]
        * BT ** exp["BT_T"]
        * ne19 ** exp["ne19_1e19m3"]
        * Ploss ** exp["Ploss_MW"]
        * R ** exp["R_m"]
        * kappa ** exp["kappa"]
        * epsilon ** exp["epsilon"]
        * M ** exp["M_AMU"]
    )
    tau_f = float(tau)
    if not np.isfinite(tau_f) or tau_f <= 0:
        raise ValueError(
            "Computed IPB98(y,2) confinement time is invalid "
            f"(tau={tau_f!r}) for supplied inputs."
        )
    return tau_f


def ipb98y2_tau_e_with_metadata(
    Ip: float,
    BT: float,
    ne19: float,
    Ploss: float,
    R: float,
    kappa: float,
    epsilon: float,
    M: float = 2.5,
    *,
    coefficients: Optional[dict] = None,
    enforce_training_domain: bool = False,
    warn_if_extrapolated: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Evaluate confinement time and return training-domain metadata."""
    metadata = assess_ipb98y2_domain(
        Ip=Ip, BT=BT, ne19=ne19, Ploss=Ploss, R=R, kappa=kappa, epsilon=epsilon, M=M
    )
    tau = ipb98y2_tau_e(
        Ip=Ip,
        BT=BT,
        ne19=ne19,
        Ploss=Ploss,
        R=R,
        kappa=kappa,
        epsilon=epsilon,
        M=M,
        coefficients=coefficients,
        enforce_training_domain=enforce_training_domain,
        warn_if_extrapolated=warn_if_extrapolated,
    )
    metadata = dict(metadata)
    metadata["tau_e_s"] = float(tau)
    return float(tau), metadata


def ipb98y2_with_uncertainty(
    Ip: float,
    BT: float,
    ne19: float,
    Ploss: float,
    R: float,
    kappa: float,
    epsilon: float,
    M: float = 2.5,
    *,
    coefficients: Optional[dict] = None,
) -> tuple[float, float]:
    """Evaluate IPB98(y,2) with log-linear error propagation.

    Uses published exponent uncertainties from Verdoolaege et al.,
    Nuclear Fusion 61, 076006 (2021) for 95% confidence interval
    estimation via log-linear error propagation.

    Parameters
    ----------
    (same as ipb98y2_tau_e)

    Returns
    -------
    tuple[float, float]
        (tau_E, sigma_tau_E) — predicted confinement time and its
        1-sigma uncertainty [s].
    """
    if coefficients is None:
        coeff = load_ipb98y2_coefficients()
    else:
        coeff = _validate_ipb98y2_coefficients(coefficients)

    tau = ipb98y2_tau_e(
        Ip,
        BT,
        ne19,
        Ploss,
        R,
        kappa,
        epsilon,
        M,
        coefficients=coeff,
    )

    # Published exponent uncertainties (Verdoolaege et al. NF 2021)
    # These are 1-sigma uncertainties on the log-space exponents.
    exp_unc = coeff.get("exponent_uncertainties", {
        "Ip_MA": 0.02,
        "BT_T": 0.04,
        "ne19_1e19m3": 0.03,
        "Ploss_MW": 0.02,
        "R_m": 0.09,
        "kappa": 0.08,
        "epsilon": 0.07,
        "M_AMU": 0.04,
    })

    # Log-linear error propagation:
    # ln(tau) = ln(C) + sum_k alpha_k * ln(x_k)
    # sigma_ln_tau^2 = sum_k (ln(x_k))^2 * sigma_alpha_k^2 + sigma_ln_C^2
    sigma_lnC = float(coeff.get("sigma_lnC", 0.14))

    inputs = {
        "Ip_MA": Ip, "BT_T": BT, "ne19_1e19m3": ne19,
        "Ploss_MW": Ploss, "R_m": R, "kappa": kappa,
        "epsilon": epsilon, "M_AMU": M,
    }

    var_ln_tau = sigma_lnC ** 2
    max_f64 = np.finfo(np.float64).max
    for key, val in inputs.items():
        if val > 0 and key in exp_unc:
            log_val = abs(float(np.log(val)))
            sigma_alpha = abs(float(exp_unc[key]))
            if not np.isfinite(log_val) or not np.isfinite(sigma_alpha):
                var_ln_tau = float("inf")
                break

            # Avoid runtime overflow warnings in extreme uncertainty metadata.
            if log_val > 0.0 and sigma_alpha > np.sqrt(max_f64) / log_val:
                var_ln_tau = float("inf")
                break

            var_ln_tau += (log_val * sigma_alpha) ** 2

    sigma_ln_tau = np.sqrt(var_ln_tau)
    # Convert from log-space: sigma_tau ≈ tau * sigma_ln_tau
    sigma_tau = float(tau * sigma_ln_tau)
    if not np.isfinite(sigma_tau) or sigma_tau < 0.0:
        raise ValueError(
            "Computed IPB98(y,2) uncertainty is invalid "
            f"(sigma_tau={sigma_tau!r}) for supplied inputs."
        )

    return float(tau), sigma_tau


def compute_h_factor(tau_actual: float, tau_predicted: float) -> float:
    """Compute the H-factor (enhancement factor over scaling law).

    Parameters
    ----------
    tau_actual : float
        Measured or simulated confinement time [s].
    tau_predicted : float
        IPB98(y,2) predicted confinement time [s].

    Returns
    -------
    float
        H98(y,2) = tau_actual / tau_predicted.
    """
    tau_actual_f = _require_finite_number("tau_actual", tau_actual)
    tau_predicted_f = _require_finite_number("tau_predicted", tau_predicted)

    if tau_predicted_f <= 0:
        return float("inf")
    return tau_actual_f / tau_predicted_f
