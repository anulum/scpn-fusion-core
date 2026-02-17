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
from typing import Optional

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
    with open(p) as f:
        return json.load(f)


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

    Returns
    -------
    float
        Predicted thermal energy confinement time τ_E [s].

    Raises
    ------
    ValueError
        If any input is non-positive.
    """
    if Ploss <= 0:
        raise ValueError(f"Ploss must be > 0, got {Ploss}")
    if any(v <= 0 for v in (Ip, BT, ne19, R, kappa, epsilon, M)):
        raise ValueError(
            "All IPB98(y,2) inputs must be positive: "
            f"Ip={Ip}, BT={BT}, ne19={ne19}, R={R}, "
            f"kappa={kappa}, epsilon={epsilon}, M={M}"
        )

    if coefficients is None:
        coefficients = load_ipb98y2_coefficients()

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
    return float(tau)


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
    tau = ipb98y2_tau_e(Ip, BT, ne19, Ploss, R, kappa, epsilon, M,
                        coefficients=coefficients)

    if coefficients is None:
        coefficients = load_ipb98y2_coefficients()

    # Published exponent uncertainties (Verdoolaege et al. NF 2021)
    # These are 1-sigma uncertainties on the log-space exponents.
    exp_unc = coefficients.get("exponent_uncertainties", {
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
    sigma_lnC = float(coefficients.get("sigma_lnC", 0.14))

    inputs = {
        "Ip_MA": Ip, "BT_T": BT, "ne19_1e19m3": ne19,
        "Ploss_MW": Ploss, "R_m": R, "kappa": kappa,
        "epsilon": epsilon, "M_AMU": M,
    }

    var_ln_tau = sigma_lnC ** 2
    for key, val in inputs.items():
        if val > 0 and key in exp_unc:
            var_ln_tau += (np.log(val) * exp_unc[key]) ** 2

    sigma_ln_tau = np.sqrt(var_ln_tau)
    # Convert from log-space: sigma_tau ≈ tau * sigma_ln_tau
    sigma_tau = float(tau * sigma_ln_tau)

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
    if tau_predicted <= 0:
        return float("inf")
    return tau_actual / tau_predicted
