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
