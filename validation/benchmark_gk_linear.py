# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Linear GK Benchmark Suite
"""
Benchmark suite for the native linear GK solver.

Runs the Cyclone Base Case, GA-standard case, and parameter scans,
comparing against published results from GENE/GS2/GYRO.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from scpn_fusion.core.gk_eigenvalue import solve_linear_gk
from scpn_fusion.core.gk_quasilinear import (
    critical_gradient_scan,
    quasilinear_fluxes_from_spectrum,
)
from scpn_fusion.core.gk_species import deuterium_ion, electron

_REF_DIR = Path(__file__).parent / "reference_data" / "cyclone_base"
_REPORT_DIR = Path(__file__).parent / "reports"


def run_cyclone_base_case(n_ky: int = 12, n_theta: int = 32) -> dict:
    """Cyclone Base Case: Dimits et al. 2000."""
    species = [
        deuterium_ion(T_keV=2.0, n_19=5.0, R_L_T=6.9, R_L_n=2.2),
        electron(T_keV=2.0, n_19=5.0, R_L_T=6.9, R_L_n=2.2, adiabatic=True),
    ]
    t0 = time.perf_counter()
    result = solve_linear_gk(
        species_list=species,
        R0=2.78,
        a=1.0,
        B0=2.0,
        q=1.4,
        s_hat=0.78,
        n_ky_ion=n_ky,
        n_ky_etg=0,
        n_theta=n_theta,
        n_period=2,
    )
    elapsed = time.perf_counter() - t0

    ion = deuterium_ion(T_keV=2.0, R_L_T=6.9, R_L_n=2.2)
    fluxes = quasilinear_fluxes_from_spectrum(result, ion, R0=2.78, a=1.0, B0=2.0)

    return {
        "case": "cyclone_base",
        "gamma_max": float(result.gamma_max),
        "k_y_max": float(result.k_y_max),
        "dominant_mode": (
            result.mode_type[int(np.argmax(result.gamma))] if result.gamma_max > 0 else "stable"
        ),
        "chi_i_m2s": fluxes.chi_i,
        "chi_e_m2s": fluxes.chi_e,
        "n_ky": n_ky,
        "n_theta": n_theta,
        "elapsed_s": round(elapsed, 3),
        "k_y": result.k_y.tolist(),
        "gamma": result.gamma.tolist(),
        "omega_r": result.omega_r.tolist(),
    }


def run_critical_gradient_scan() -> dict:
    """Scan R/L_Ti to find critical gradient."""
    rlt = np.linspace(1.0, 12.0, 12)
    _, gamma = critical_gradient_scan(rlt, R0=2.78, a=1.0, B0=2.0, q=1.4, s_hat=0.78, n_ky=4)
    return {
        "case": "critical_gradient_scan",
        "R_L_Ti": rlt.tolist(),
        "gamma_max": gamma.tolist(),
    }


def run_multi_code_comparison() -> dict:
    """Compare native GK vs existing quasilinear model at CBC parameters."""
    from scpn_fusion.core.gyrokinetic_transport import (
        GyrokineticsParams,
        compute_spectrum,
        quasilinear_fluxes,
    )

    # Native GK
    species = [
        deuterium_ion(T_keV=2.0, R_L_T=6.9, R_L_n=2.2),
        electron(T_keV=2.0, R_L_T=6.9, R_L_n=2.2),
    ]
    gk_result = solve_linear_gk(
        species_list=species,
        R0=2.78,
        a=1.0,
        B0=2.0,
        q=1.4,
        s_hat=0.78,
        n_ky_ion=8,
        n_theta=32,
        n_period=1,
    )
    gk_fluxes = quasilinear_fluxes_from_spectrum(gk_result, species[0], R0=2.78, a=1.0, B0=2.0)

    # Existing quasilinear dispersion
    params = GyrokineticsParams(
        R_L_Ti=6.9,
        R_L_Te=6.9,
        R_L_ne=2.2,
        q=1.4,
        s_hat=0.78,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.0,
        nu_star=0.01,
        beta_e=0.0,
        epsilon=0.18,
    )
    ql_spec = compute_spectrum(params, n_modes=16)
    ql_fluxes = quasilinear_fluxes(params, ql_spec)

    return {
        "case": "multi_code_comparison",
        "native_gk": {
            "gamma_max": float(gk_result.gamma_max),
            "chi_i": gk_fluxes.chi_i,
            "chi_e": gk_fluxes.chi_e,
        },
        "quasilinear": {
            "gamma_max": float(np.max(ql_spec.gamma_linear)),
            "chi_i": ql_fluxes.chi_i,
            "chi_e": ql_fluxes.chi_e,
        },
    }


def run_sparc_iter_scans() -> dict:
    """Parameter scans at SPARC and ITER-like conditions."""
    results = {}
    for name, R0, a, B0, q, s_hat, T_keV in [
        ("SPARC", 1.85, 0.57, 12.2, 1.8, 1.0, 10.0),
        ("ITER", 6.2, 2.0, 5.3, 1.5, 0.8, 8.0),
    ]:
        species = [
            deuterium_ion(T_keV=T_keV, R_L_T=6.0, R_L_n=2.0),
            electron(T_keV=T_keV, R_L_T=6.0, R_L_n=2.0),
        ]
        result = solve_linear_gk(
            species_list=species,
            R0=R0,
            a=a,
            B0=B0,
            q=q,
            s_hat=s_hat,
            n_ky_ion=6,
            n_theta=32,
            n_period=1,
        )
        fluxes = quasilinear_fluxes_from_spectrum(result, species[0], R0=R0, a=a, B0=B0)
        results[name] = {
            "gamma_max": float(result.gamma_max),
            "chi_i_m2s": fluxes.chi_i,
            "chi_e_m2s": fluxes.chi_e,
        }
    return {"case": "sparc_iter_scans", "results": results}


def run_all() -> dict:
    """Run full benchmark suite and save report."""
    report = {
        "cyclone_base": run_cyclone_base_case(),
        "critical_gradient": run_critical_gradient_scan(),
        "multi_code": run_multi_code_comparison(),
        "sparc_iter": run_sparc_iter_scans(),
    }
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = _REPORT_DIR / "gk_linear_benchmark.json"
    out.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    report = run_all()
    print(json.dumps(report, indent=2))
