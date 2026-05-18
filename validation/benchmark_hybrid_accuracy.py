# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Hybrid Transport Accuracy Benchmark
"""
End-to-end benchmark of hybrid surrogate+GK transport accuracy.

Compares the correction layer output against pure GK and pure surrogate,
measuring the improvement from spot-check validation and online learning.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scpn_fusion.core.gk_corrector import CorrectionRecord, CorrectorConfig, GKCorrector
from scpn_fusion.core.gk_eigenvalue import solve_linear_gk
from scpn_fusion.core.gk_quasilinear import quasilinear_fluxes_from_spectrum
from scpn_fusion.core.gk_scheduler import GKScheduler, SchedulerConfig
from scpn_fusion.core.gk_species import deuterium_ion, electron
from scpn_fusion.core.gk_verification_report import VerificationReport

_REPORT_DIR = Path(__file__).parent / "reports"
_GK_CALIBRATION_R_L_TI = np.array([3.0, 4.5, 6.0, 7.5, 9.0], dtype=np.float64)
_GK_TRANSPORT_CALIBRATION_CACHE: dict[
    tuple[float, ...], tuple[np.ndarray, np.ndarray, np.ndarray]
] = {}


def _gk_chi(R_L_Ti: float) -> tuple[float, float, float]:
    """GK ground truth at mid-radius CBC-like params."""
    species = [deuterium_ion(R_L_T=R_L_Ti, R_L_n=2.2), electron(R_L_T=R_L_Ti, R_L_n=2.2)]
    result = solve_linear_gk(
        species_list=species,
        R0=2.78,
        a=1.0,
        B0=2.0,
        q=1.4,
        s_hat=0.78,
        n_ky_ion=4,
        n_theta=16,
        n_period=1,
    )
    fluxes = quasilinear_fluxes_from_spectrum(result, species[0], R0=2.78, a=1.0, B0=2.0)
    return fluxes.chi_i, fluxes.chi_e, fluxes.D_e


def _gk_transport_calibration(
    calibration_points: np.ndarray = _GK_CALIBRATION_R_L_TI,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = tuple(float(x) for x in np.asarray(calibration_points, dtype=np.float64))
    cached = _GK_TRANSPORT_CALIBRATION_CACHE.get(key)
    if cached is not None:
        return cached

    records = np.asarray([_gk_chi(point) for point in key], dtype=np.float64)
    if records.shape != (len(key), 3) or not np.all(np.isfinite(records)):
        raise ValueError("GK calibration produced non-finite transport coefficients.")
    if np.any(records < 0.0):
        raise ValueError("GK calibration produced negative transport coefficients.")

    chi_i = records[:, 0]
    chi_e = records[:, 1]
    d_e = records[:, 2]
    _GK_TRANSPORT_CALIBRATION_CACHE[key] = (chi_i, chi_e, d_e)
    return chi_i, chi_e, d_e


def _gk_calibrated_transport_estimator(R_L_Ti: float) -> tuple[float, float, float]:
    """Deterministic reduced-order transport estimate interpolated from GK calibration."""
    rlt = float(R_L_Ti)
    if not np.isfinite(rlt):
        raise ValueError("R_L_Ti must be finite.")
    chi_i_ref, chi_e_ref, d_e_ref = _gk_transport_calibration()
    points = _GK_CALIBRATION_R_L_TI
    chi_i = float(np.interp(rlt, points, chi_i_ref))
    chi_e = float(np.interp(rlt, points, chi_e_ref))
    d_e = float(np.interp(rlt, points, d_e_ref))
    return chi_i, chi_e, d_e


def run_hybrid_accuracy_benchmark(n_steps: int = 20) -> dict:
    """Simulate n_steps of hybrid transport with correction."""
    nr = 20
    rho = np.linspace(0.05, 0.95, nr)
    R_L_Ti_profile = 3.0 + 6.0 * rho  # gradient increases toward edge

    scheduler = GKScheduler(SchedulerConfig(strategy="periodic", period=3, budget=3))
    corrector = GKCorrector(nr=nr, config=CorrectorConfig(smoothing_alpha=0.5))
    report = VerificationReport()

    surrogate_errors = []
    corrected_errors = []

    for step in range(n_steps):
        transport_estimates = np.asarray(
            [_gk_calibrated_transport_estimator(rlt) for rlt in R_L_Ti_profile],
            dtype=np.float64,
        )
        chi_i_surr = transport_estimates[:, 0]
        chi_e_surr = transport_estimates[:, 1]
        D_e_surr = transport_estimates[:, 2]

        req = scheduler.step(rho, chi_i_surr)

        if req is not None:
            records = []
            for idx in req.rho_indices:
                gk_i, gk_e, gk_d = _gk_chi(R_L_Ti_profile[idx])
                records.append(
                    CorrectionRecord(
                        rho_idx=idx,
                        rho=rho[idx],
                        chi_i_surrogate=chi_i_surr[idx],
                        chi_i_gk=max(gk_i, 0.01),
                        chi_e_surrogate=chi_e_surr[idx],
                        chi_e_gk=max(gk_e, 0.01),
                        D_e_surrogate=D_e_surr[idx],
                        D_e_gk=max(gk_d, 0.001),
                    )
                )
            corrector.update(records, rho)
            report.add_step(verified=True, n_spot_checks=len(records))
            report.add_records(records)
        else:
            report.add_step(verified=False)

        chi_i_corr, chi_e_corr, D_e_corr = corrector.correct(chi_i_surr, chi_e_surr, D_e_surr)

        # Measure error at a reference point (mid-radius)
        ref_idx = nr // 2
        gk_ref_i, gk_ref_e, _ = _gk_chi(R_L_Ti_profile[ref_idx])
        gk_ref_i = max(gk_ref_i, 0.01)

        surr_err = abs(chi_i_surr[ref_idx] - gk_ref_i) / gk_ref_i if gk_ref_i > 0 else 0.0
        corr_err = abs(chi_i_corr[ref_idx] - gk_ref_i) / gk_ref_i if gk_ref_i > 0 else 0.0

        surrogate_errors.append(surr_err)
        corrected_errors.append(corr_err)
        report.add_correction_factor(corrector.mean_correction_factor)

    result = {
        "n_steps": n_steps,
        "mean_surrogate_error": float(np.mean(surrogate_errors)),
        "mean_corrected_error": float(np.mean(corrected_errors)),
        "final_surrogate_error": float(surrogate_errors[-1]),
        "final_corrected_error": float(corrected_errors[-1]),
        "transport_estimator": "gk_calibrated_reference_interpolation",
        "calibration_R_L_Ti": [float(x) for x in _GK_CALIBRATION_R_L_TI],
        "verification_report": report.to_dict(),
    }

    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = _REPORT_DIR / "hybrid_accuracy_benchmark.json"
    out.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    result = run_hybrid_accuracy_benchmark()
    print(json.dumps(result, indent=2))
