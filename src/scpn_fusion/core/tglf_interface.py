# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TGLF Comparison Interface
"""
Interface for comparing SCPN transport against TGLF gyrokinetic model.

Provides input-deck generation from TransportSolver state, TGLF output
parsing, and benchmark comparison utilities with markdown/LaTeX tables.

Note: actual TGLF execution requires the external GA binary. This module
creates the interface + comparison framework with pre-computed reference data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scpn_fusion.io.safe_loaders import checked_json_load
from scpn_fusion.core._tglf_interface_benchmark import TGLFBenchmark
from scpn_fusion.core._tglf_interface_reference import (
    REFERENCE_CASES,
    _reference_case_filename,
    _reference_case_to_transport_input,
    validate_reduced_transport_reference_case,
    validate_reduced_transport_reference_suite,
    write_reference_data,
)
from scpn_fusion.core._tglf_interface_runtime import (
    _normalize_tglf_max_retries,
    _normalize_tglf_timeout_seconds,
    _parse_tglf_run_output,
    run_tglf_binary,
    write_tglf_input_file,
)
from scpn_fusion.core._tglf_interface_types import (
    TGLFComparisonResult,
    TGLFInputDeck,
    TGLFOutput,
    TGLFProfileScanResult,
    TGLFReferenceCaseResult,
    TGLF_REF_DIR,
    _TGLF_MAX_PARSED_VECTOR_LENGTH,
    _TGLF_MAX_RETRIES_LIMIT,
    _TGLF_RETRY_BACKOFF_SECONDS,
)
from scpn_fusion.core.neural_transport_math import _compute_nustar
from scpn_fusion.core.tglf_surrogate_bridge import (
    TGLFDatasetGenerator,  # noqa: F401 - re-exported for API stability
    train_surrogate_from_tglf,  # noqa: F401 - re-exported for API stability
)
from scpn_fusion.core.tglf_validation_runtime import (
    validate_against_tglf,  # noqa: F401 - re-exported for API stability
)

logger = logging.getLogger(__name__)

__all__ = [
    "REFERENCE_CASES",
    "TGLFBenchmark",
    "TGLFComparisonResult",
    "TGLFDatasetGenerator",
    "TGLFInputDeck",
    "TGLFOutput",
    "TGLFProfileScanResult",
    "TGLFReferenceCaseResult",
    "TGLF_REF_DIR",
    "_TGLF_MAX_PARSED_VECTOR_LENGTH",
    "_TGLF_MAX_RETRIES_LIMIT",
    "_TGLF_RETRY_BACKOFF_SECONDS",
    "_normalize_tglf_max_retries",
    "_normalize_tglf_timeout_seconds",
    "_parse_tglf_run_output",
    "_reference_case_filename",
    "_reference_case_to_transport_input",
    "generate_input_deck",
    "parse_tglf_output",
    "run_tglf_binary",
    "run_tglf_profile_scan",
    "train_surrogate_from_tglf",
    "validate_against_tglf",
    "validate_reduced_transport_reference_case",
    "validate_reduced_transport_reference_suite",
    "write_reference_data",
    "write_tglf_input_file",
]


def _resolve_solver_geometry(transport_solver: Any) -> tuple[float, float, float, float]:
    """Resolve geometry and charge state from solver state or config."""
    ts = transport_solver
    cfg = getattr(ts, "cfg", {})
    dims = cfg.get("dimensions", {}) if isinstance(cfg, dict) else {}
    default_r0 = 0.5 * (float(dims.get("R_min", 4.2)) + float(dims.get("R_max", 8.2)))
    default_a = 0.5 * (float(dims.get("R_max", 8.2)) - float(dims.get("R_min", 4.2)))
    physics = cfg.get("physics", {}) if isinstance(cfg, dict) else {}

    params = getattr(ts, "neoclassical_params", None)
    if not isinstance(params, dict):
        params = {}

    r0 = float(params.get("R0", getattr(ts, "R0", default_r0)))
    a_minor = float(params.get("a", getattr(ts, "a", default_a)))
    b0 = float(params.get("B0", getattr(ts, "B0", physics.get("B0", 5.3))))
    z_eff = float(params.get("Z_eff", getattr(ts, "_Z_eff", 1.5)))
    return r0, a_minor, b0, z_eff


def _resolve_solver_q_profile(transport_solver: Any) -> NDArray[np.float64]:
    """Resolve q-profile from runtime state, neoclassical params, or fallback."""
    ts = transport_solver
    rho = np.asarray(ts.rho, dtype=np.float64)
    q_profile = getattr(ts, "q_profile", None)
    if q_profile is None:
        params = getattr(ts, "neoclassical_params", None)
        if isinstance(params, dict):
            q_profile = params.get("q_profile")
    if q_profile is None:
        return np.asarray(1.0 + 3.0 * rho**2, dtype=np.float64)

    q = np.asarray(q_profile, dtype=np.float64)
    if q.shape != rho.shape or (not np.all(np.isfinite(q))) or np.any(q <= 0.0):
        return np.asarray(1.0 + 3.0 * rho**2, dtype=np.float64)
    return q


# ── Input deck generation ────────────────────────────────────────────


def generate_input_deck(transport_solver: Any, rho_idx: int) -> TGLFInputDeck:
    """Extract TGLF input parameters from a TransportSolver at given flux surface.

    Parameters
    ----------
    transport_solver : TransportSolver
        The SCPN transport solver instance.
    rho_idx : int
        Index into the radial grid.

    Returns
    -------
    TGLFInputDeck
    """
    ts = transport_solver
    rho = float(ts.rho[rho_idx])
    Te = float(ts.Te[rho_idx])
    Ti = float(ts.Ti[rho_idx])
    ne = float(ts.ne[rho_idx])
    mu0 = 4.0e-7 * np.pi
    e_charge = 1.602176634e-19

    # Compute gradient scale lengths (central differences)
    dr = float(ts.rho[1] - ts.rho[0]) if len(ts.rho) > 1 else 0.01

    def _grad_scale(arr: NDArray[np.float64], idx: int) -> float:
        if idx <= 0 or idx >= len(arr) - 1:
            return 0.0
        grad = (arr[idx + 1] - arr[idx - 1]) / (2.0 * dr)
        val = arr[idx]
        if abs(val) < 1e-10:
            return 0.0
        return -float(grad / val)  # R/L_X = -R * (1/X * dX/dr)

    r0, a_minor, b_toroidal, z_eff = _resolve_solver_geometry(ts)

    R_LTi = r0 * _grad_scale(ts.Ti, rho_idx)
    R_LTe = r0 * _grad_scale(ts.Te, rho_idx)
    R_Lne = r0 * _grad_scale(ts.ne, rho_idx)

    q_profile = _resolve_solver_q_profile(ts)
    dq_drho = np.gradient(q_profile, ts.rho)
    q_val = float(q_profile[rho_idx])
    s_hat = float(np.clip(rho * dq_drho[rho_idx] / max(q_val, 0.2), 0.0, 10.0))
    q_prime_loc = float(dq_drho[rho_idx] / max(a_minor, 1e-6))

    pressure_pa = np.maximum(ts.ne, 0.0) * 1e19 * np.maximum(ts.Te + ts.Ti, 0.0) * 1e3 * e_charge
    dp_drho = np.gradient(pressure_pa, ts.rho)
    dp_dr = float(dp_drho[rho_idx] / max(a_minor, 1e-6))
    alpha_mhd = float(
        np.clip(-2.0 * mu0 * r0 * q_val**2 * dp_dr / max(b_toroidal**2, 1e-6), -20.0, 20.0)
    )
    beta_e = float(np.clip(4.03e-3 * ne * Te, 0.0, 1.0))
    xnue = float(np.clip(_compute_nustar(Te, ne, q_val, rho, r0, a_minor, z_eff), 0.0, 50.0))

    physics_cfg = ts.cfg.get("physics", {}) if isinstance(getattr(ts, "cfg", None), dict) else {}
    kappa = float(physics_cfg.get("kappa", getattr(ts, "kappa", 1.7)))
    delta = float(physics_cfg.get("delta", getattr(ts, "delta", 0.33)))

    return TGLFInputDeck(
        rho=rho,
        s_hat=s_hat,
        q=q_val,
        q_prime_loc=q_prime_loc,
        alpha_mhd=alpha_mhd,
        p_prime_loc=dp_dr,
        kappa=kappa,
        delta=delta,
        R_LTi=R_LTi,
        R_LTe=R_LTe,
        R_Lne=R_Lne,
        R_Lni=R_Lne,
        beta_e=beta_e,
        Z_eff=z_eff,
        xnue=xnue,
        T_e_keV=Te,
        T_i_keV=Ti,
        n_e_19=ne,
        R_major=r0,
        a_minor=a_minor,
        B_toroidal=b_toroidal,
    )


# ── Output parsing ───────────────────────────────────────────────────


def parse_tglf_output(output_dir: str | Path) -> list[TGLFOutput]:
    """Parse TGLF output files from a directory.

    Expects JSON files with keys: rho, chi_i, chi_e, gamma_max, q_i, q_e.
    """
    output_dir = Path(output_dir)
    results = []

    def _finite_float(value: Any, *, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if not np.isfinite(parsed):
            return default
        return parsed

    def _coerce_sequence(value: Any, *, default: float) -> list[float]:
        if isinstance(value, (list, tuple, np.ndarray)):
            seq_in = list(value)
            if len(seq_in) > _TGLF_MAX_PARSED_VECTOR_LENGTH:
                logger.warning(
                    "TGLF payload vector exceeded cap (%d > %d); truncating.",
                    len(seq_in),
                    _TGLF_MAX_PARSED_VECTOR_LENGTH,
                )
                seq_in = seq_in[:_TGLF_MAX_PARSED_VECTOR_LENGTH]
            seq = [_finite_float(v, default=default) for v in seq_in]
            return seq or [default]
        return [_finite_float(value, default=default)]

    for f in sorted(output_dir.glob("*.json")):
        try:
            data = checked_json_load(f)
        except (OSError, json.JSONDecodeError):
            logger.warning("Skipping malformed TGLF output file: %s", f)
            continue
        if not isinstance(data, dict):
            logger.warning("Skipping non-object TGLF output payload: %s", f)
            continue

        rho_source = data.get("rho_points", data.get("rho", 0.5))
        rho_pts = _coerce_sequence(rho_source, default=0.5)
        chi_i_list = _coerce_sequence(data.get("chi_i", 0.0), default=0.0)
        chi_e_list = _coerce_sequence(data.get("chi_e", 0.0), default=0.0)
        gamma_list = _coerce_sequence(data.get("gamma_max", 0.0), default=0.0)
        qi_list = _coerce_sequence(data.get("q_i", 0.0), default=0.0)
        qe_list = _coerce_sequence(data.get("q_e", 0.0), default=0.0)

        for j in range(len(rho_pts)):
            results.append(
                TGLFOutput(
                    rho=rho_pts[j],
                    chi_i=chi_i_list[j] if j < len(chi_i_list) else 0.0,
                    chi_e=chi_e_list[j] if j < len(chi_e_list) else 0.0,
                    gamma_max=gamma_list[j] if j < len(gamma_list) else 0.0,
                    q_i=qi_list[j] if j < len(qi_list) else 0.0,
                    q_e=qe_list[j] if j < len(qe_list) else 0.0,
                )
            )

    return results


def run_tglf_profile_scan(
    transport_solver: Any,
    tglf_binary_path: str | Path,
    *,
    rho_indices: list[int] | None = None,
    timeout_s: float = 120.0,
    max_retries: int = 2,
) -> TGLFProfileScanResult:
    """Run TGLF across selected flux surfaces and interpolate onto the full grid."""
    ts = transport_solver
    rho_grid = np.asarray(ts.rho, dtype=np.float64)
    n = int(rho_grid.size)
    if n < 3:
        raise ValueError("transport_solver.rho must contain at least 3 points.")

    if rho_indices is None:
        base = np.linspace(1, n - 2, min(7, n - 2), dtype=int)
        rho_indices = sorted({int(i) for i in base.tolist()})
    else:
        rho_indices = sorted({int(i) for i in rho_indices if 1 <= int(i) < n - 1})

    if not rho_indices:
        raise ValueError("rho_indices must contain at least one interior index.")

    outputs: list[TGLFOutput] = []
    for idx in rho_indices:
        deck = generate_input_deck(ts, idx)
        outputs.append(
            run_tglf_binary(
                deck,
                tglf_binary_path,
                timeout_s=timeout_s,
                max_retries=max_retries,
            )
        )

    rho_samples = np.array([float(out.rho) for out in outputs], dtype=np.float64)
    chi_i_samples = np.array([max(float(out.chi_i), 0.0) for out in outputs], dtype=np.float64)
    chi_e_samples = np.array([max(float(out.chi_e), 0.0) for out in outputs], dtype=np.float64)
    gamma_samples = np.array([max(float(out.gamma_max), 0.0) for out in outputs], dtype=np.float64)

    order = np.argsort(rho_samples)
    rho_samples = rho_samples[order]
    chi_i_samples = chi_i_samples[order]
    chi_e_samples = chi_e_samples[order]
    gamma_samples = gamma_samples[order]

    if (
        np.any(~np.isfinite(rho_samples))
        or np.any(~np.isfinite(chi_i_samples))
        or np.any(~np.isfinite(chi_e_samples))
    ):
        raise ValueError("TGLF profile scan produced non-finite samples.")

    chi_i_profile = np.interp(rho_grid, rho_samples, chi_i_samples)
    chi_e_profile = np.interp(rho_grid, rho_samples, chi_e_samples)
    gamma_profile = np.interp(rho_grid, rho_samples, gamma_samples)

    return TGLFProfileScanResult(
        rho_samples=rho_samples.tolist(),
        chi_i_samples=chi_i_samples.tolist(),
        chi_e_samples=chi_e_samples.tolist(),
        gamma_samples=gamma_samples.tolist(),
        chi_i_profile=chi_i_profile.tolist(),
        chi_e_profile=chi_e_profile.tolist(),
        gamma_profile=gamma_profile.tolist(),
    )
