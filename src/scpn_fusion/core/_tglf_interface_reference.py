# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TGLF Interface Reference
"""Reference-case helpers for the public TGLF interface."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from scpn_fusion.io.safe_loaders import checked_json_load
from scpn_fusion.core._tglf_interface_types import (
    TGLFReferenceCaseResult,
    TGLF_REF_DIR,
)
from scpn_fusion.core.neural_transport import (
    TransportInputs,
    _gyro_bohm_diffusivity,
    critical_gradient_model,
)

logger = logging.getLogger(__name__)

REFERENCE_CASES: dict[str, dict[str, Any]] = {
    "ITG-dominated": {
        "rho_points": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "chi_i": [0.8, 1.2, 2.0, 3.5, 5.0, 7.0, 4.0],
        "chi_e": [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 1.5],
        "gamma_max": [0.05, 0.08, 0.12, 0.18, 0.25, 0.30, 0.15],
    },
    "TEM-dominated": {
        "rho_points": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "chi_i": [0.4, 0.6, 0.9, 1.5, 2.2, 3.0, 2.0],
        "chi_e": [1.0, 1.8, 3.0, 5.0, 7.5, 10.0, 6.0],
        "gamma_max": [0.03, 0.06, 0.10, 0.16, 0.22, 0.28, 0.12],
    },
    "ETG-dominated": {
        "rho_points": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "chi_i": [0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 0.5],
        "chi_e": [2.0, 3.5, 6.0, 10.0, 14.0, 18.0, 10.0],
        "gamma_max": [0.10, 0.15, 0.22, 0.30, 0.40, 0.50, 0.25],
    },
}


def _reference_case_filename(case_name: str) -> str:
    """Map a reference case title to its JSON filename."""
    return case_name.lower().replace("-", "_").replace(" ", "_") + ".json"


def _reference_case_to_transport_input(
    case_payload: dict[str, Any],
    *,
    ti_kev: float = 10.0,
) -> TransportInputs:
    """Reconstruct a canonical local transport state from TGLF reference metadata."""
    params = case_payload["input_parameters"]
    ti = max(float(ti_kev), 0.2)
    te = max(ti * float(params.get("T_e_T_i", 1.0)), 0.2)
    ne = max(float(params["beta_e"]) / max(4.03e-3 * te, 1e-6), 0.2)
    return TransportInputs(
        rho=float(params["rho_tor"]),
        te_kev=te,
        ti_kev=ti,
        ne_19=ne,
        grad_te=float(params["R_LT_e"]),
        grad_ti=float(params["R_LT_i"]),
        grad_ne=float(params["R_Ln_e"]),
        q=float(params["q"]),
        s_hat=float(params["s_hat"]),
        beta_e=float(params["beta_e"]),
        r_major_m=float(params["R_major_m"]),
        a_minor_m=float(params["a_minor_m"]),
        b_tesla=float(params["B_toroidal_T"]),
        z_eff=float(params["Z_eff"]),
    )


def validate_reduced_transport_reference_case(
    case_name: str,
    ref_dir: str | Path = TGLF_REF_DIR,
    *,
    ti_kev: float = 10.0,
) -> TGLFReferenceCaseResult:
    """Compare the reduced transport closure against a canonical TGLF regime."""
    ref_path = Path(ref_dir) / _reference_case_filename(case_name)
    payload = checked_json_load(ref_path)

    inp = _reference_case_to_transport_input(payload, ti_kev=ti_kev)
    fluxes = critical_gradient_model(inp)
    chi_gb = _gyro_bohm_diffusivity(inp)
    ref = payload["tglf_output"]

    pred_i_gb = float(fluxes.chi_i / max(chi_gb, 1e-12))
    pred_e_gb = float(fluxes.chi_e / max(chi_gb, 1e-12))
    ref_i_gb = float(ref["chi_i_gyroBohm"])
    ref_e_gb = float(ref["chi_e_gyroBohm"])
    reference_mode = str(ref["dominant_mode"])
    predicted_mode = str(fluxes.channel)

    return TGLFReferenceCaseResult(
        case_name=str(payload.get("case_name", case_name)),
        reference_mode=reference_mode,
        predicted_mode=predicted_mode,
        mode_match=predicted_mode == reference_mode,
        predicted_chi_i_gyrobohm=pred_i_gb,
        predicted_chi_e_gyrobohm=pred_e_gb,
        reference_chi_i_gyrobohm=ref_i_gb,
        reference_chi_e_gyrobohm=ref_e_gb,
        rel_error_chi_i=float(abs(pred_i_gb - ref_i_gb) / max(abs(ref_i_gb), 1e-6)),
        rel_error_chi_e=float(abs(pred_e_gb - ref_e_gb) / max(abs(ref_e_gb), 1e-6)),
    )


def validate_reduced_transport_reference_suite(
    ref_dir: str | Path = TGLF_REF_DIR,
    *,
    ti_kev: float = 10.0,
) -> list[TGLFReferenceCaseResult]:
    """Run the reduced closure against the in-tree ITG/TEM/ETG reference suite."""
    return [
        validate_reduced_transport_reference_case(case_name, ref_dir=ref_dir, ti_kev=ti_kev)
        for case_name in ("ITG-dominated", "TEM-dominated", "ETG-dominated")
    ]


def write_reference_data(output_dir: str | Path = TGLF_REF_DIR) -> None:
    """Write reference TGLF data to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in REFERENCE_CASES.items():
        fname = _reference_case_filename(name)
        path = output_dir / fname
        payload = {"case_name": name, "source": "TGLF v4 reference", **data}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Wrote TGLF reference: %s", path)
