# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Validation Regression Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Regression checks against published ITER/SPARC confinement references."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "validation" / "reference_data"


def load_reference(filename: str) -> dict:
    with (REFERENCE_DIR / filename).open("r", encoding="utf-8") as f:
        return json.load(f)


def ipb98_tau_e(
    ip_ma: float,
    b_t: float,
    n_e19: float,
    p_loss_mw: float,
    r_m: float,
    kappa: float,
    epsilon: float,
    a_eff_amu: float = 2.5,
) -> float:
    """IPB98(y,2) scaling law [s]."""
    return (
        0.0562
        * (ip_ma ** 0.93)
        * (b_t ** 0.15)
        * (n_e19 ** 0.41)
        * (p_loss_mw ** -0.69)
        * (r_m ** 1.97)
        * (kappa ** 0.78)
        * (epsilon ** 0.58)
        * (a_eff_amu ** 0.19)
    )


def toroidal_volume(r_m: float, a_m: float, kappa: float) -> float:
    """Simple shaped-tokamak volume estimate [m^3]."""
    return 2.0 * math.pi * math.pi * r_m * a_m * a_m * kappa


def test_iter_tau_e_within_20pct() -> None:
    ref = load_reference("iter_reference.json")
    tau_ipb98 = ipb98_tau_e(
        ip_ma=ref["I_p_MA"],
        b_t=ref["B_t_T"],
        n_e19=ref["n_e_1e19"],
        p_loss_mw=ref["P_loss_MW"],
        r_m=ref["R_m"],
        kappa=ref["kappa"],
        epsilon=ref["a_m"] / ref["R_m"],
        a_eff_amu=ref["A_eff_amu"],
    )

    assert 2.9 <= tau_ipb98 <= 4.4
    rel_err = abs(tau_ipb98 - ref["tau_E_s"]) / ref["tau_E_s"]
    assert rel_err <= 0.20


def test_sparc_high_field_advantage() -> None:
    iter_ref = load_reference("iter_reference.json")
    sparc_ref = load_reference("sparc_reference.json")

    tau_iter = ipb98_tau_e(
        ip_ma=iter_ref["I_p_MA"],
        b_t=iter_ref["B_t_T"],
        n_e19=iter_ref["n_e_1e19"],
        p_loss_mw=iter_ref["P_loss_MW"],
        r_m=iter_ref["R_m"],
        kappa=iter_ref["kappa"],
        epsilon=iter_ref["a_m"] / iter_ref["R_m"],
        a_eff_amu=iter_ref["A_eff_amu"],
    )
    tau_sparc = ipb98_tau_e(
        ip_ma=sparc_ref["I_p_MA"],
        b_t=sparc_ref["B_t_T"],
        n_e19=sparc_ref["n_e_1e19"],
        p_loss_mw=sparc_ref["P_loss_MW"],
        r_m=sparc_ref["R_m"],
        kappa=sparc_ref["kappa"],
        epsilon=sparc_ref["a_m"] / sparc_ref["R_m"],
        a_eff_amu=sparc_ref["A_eff_amu"],
    )

    v_iter = toroidal_volume(iter_ref["R_m"], iter_ref["a_m"], iter_ref["kappa"])
    v_sparc = toroidal_volume(sparc_ref["R_m"], sparc_ref["a_m"], sparc_ref["kappa"])

    confinement_density_iter = tau_iter / v_iter
    confinement_density_sparc = tau_sparc / v_sparc

    assert confinement_density_sparc > confinement_density_iter


def test_ipb98_scaling_sanity() -> None:
    # Fixed regression point for deterministic sanity checking.
    tau = ipb98_tau_e(
        ip_ma=15.0,
        b_t=5.3,
        n_e19=10.1,
        p_loss_mw=85.0,
        r_m=6.2,
        kappa=1.7,
        epsilon=2.0 / 6.2,
        a_eff_amu=2.5,
    )
    assert abs(tau - 3.6643409641578857) < 1e-9


def test_diiid_lmode_baseline() -> None:
    # Typical DIII-D L-mode operating point for sanity validation.
    tau = ipb98_tau_e(
        ip_ma=1.5,
        b_t=2.1,
        n_e19=5.0,
        p_loss_mw=5.0,
        r_m=1.67,
        kappa=1.8,
        epsilon=0.67 / 1.67,
        a_eff_amu=2.5,
    )
    assert 0.12 <= tau <= 0.25
