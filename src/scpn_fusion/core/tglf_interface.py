# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — TGLF Comparison Interface
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
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
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scpn_fusion.core.neural_transport import (
    TransportInputs,
    _gyro_bohm_diffusivity,
    critical_gradient_model,
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

REPO_ROOT = Path(__file__).resolve().parents[3]
TGLF_REF_DIR = REPO_ROOT / "validation" / "tglf_reference"
_TGLF_RETRY_BACKOFF_SECONDS = 1.0
_TGLF_MAX_RETRIES_LIMIT = 10
_TGLF_MAX_PARSED_VECTOR_LENGTH = 2048


# ── Data containers ──────────────────────────────────────────────────

@dataclass
class TGLFInputDeck:
    """All TGLF input parameters for a single flux surface."""
    rho: float = 0.5
    # Geometry
    s_hat: float = 1.0          # magnetic shear
    q: float = 1.5              # safety factor
    q_prime_loc: float = 0.0    # dq/dr [1/m]
    alpha_mhd: float = 0.0     # MHD alpha
    p_prime_loc: float = 0.0    # dP/dr [Pa/m]
    kappa: float = 1.7          # elongation
    delta: float = 0.3          # triangularity
    s_kappa: float = 0.0        # elongation shear
    s_delta: float = 0.0        # triangularity shear
    # Gradients (R / L_X)
    R_LTi: float = 6.0         # R / L_Ti
    R_LTe: float = 6.0         # R / L_Te
    R_Lne: float = 2.0         # R / L_ne
    R_Lni: float = 2.0         # R / L_ni
    # Plasma parameters
    beta_e: float = 0.01       # electron beta
    Z_eff: float = 1.5         # effective charge
    xnue: float = 0.0          # normalized electron-ion collisionality
    T_e_keV: float = 10.0      # electron temperature
    T_i_keV: float = 10.0      # ion temperature
    n_e_19: float = 8.0        # electron density [1e19 m^-3]
    # Tokamak
    R_major: float = 6.2       # major radius [m]
    a_minor: float = 2.0       # minor radius [m]
    B_toroidal: float = 5.3    # toroidal field [T]


@dataclass
class TGLFOutput:
    """Parsed TGLF output for a single run."""
    rho: float = 0.5
    chi_i: float = 0.0         # ion thermal diffusivity [m^2/s]
    chi_e: float = 0.0         # electron thermal diffusivity [m^2/s]
    gamma_max: float = 0.0     # maximum growth rate [c_s/a]
    q_i: float = 0.0           # ion heat flux [MW/m^2]
    q_e: float = 0.0           # electron heat flux [MW/m^2]


@dataclass
class TGLFComparisonResult:
    """Comparison between our transport and TGLF."""
    case_name: str = ""
    rho_points: list[float] = field(default_factory=list)
    our_chi_i: list[float] = field(default_factory=list)
    tglf_chi_i: list[float] = field(default_factory=list)
    our_chi_e: list[float] = field(default_factory=list)
    tglf_chi_e: list[float] = field(default_factory=list)
    rms_error_chi_i: float = 0.0
    rms_error_chi_e: float = 0.0
    correlation_chi_i: float = 0.0
    correlation_chi_e: float = 0.0
    max_rel_error_chi_i: float = 0.0
    max_rel_error_chi_e: float = 0.0


@dataclass
class TGLFProfileScanResult:
    """Interpolated transport profiles from a live TGLF radial scan."""

    rho_samples: list[float] = field(default_factory=list)
    chi_i_samples: list[float] = field(default_factory=list)
    chi_e_samples: list[float] = field(default_factory=list)
    gamma_samples: list[float] = field(default_factory=list)
    chi_i_profile: list[float] = field(default_factory=list)
    chi_e_profile: list[float] = field(default_factory=list)
    gamma_profile: list[float] = field(default_factory=list)


@dataclass
class TGLFReferenceCaseResult:
    """Reduced-closure comparison against a single TGLF reference regime."""

    case_name: str
    reference_mode: str
    predicted_mode: str
    mode_match: bool
    predicted_chi_i_gyrobohm: float
    predicted_chi_e_gyrobohm: float
    reference_chi_i_gyrobohm: float
    reference_chi_e_gyrobohm: float
    rel_error_chi_i: float
    rel_error_chi_e: float


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
    with open(ref_path, encoding="utf-8") as f:
        payload = json.load(f)

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

    def _grad_scale(arr: NDArray, idx: int) -> float:
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
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
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
            results.append(TGLFOutput(
                rho=rho_pts[j],
                chi_i=chi_i_list[j] if j < len(chi_i_list) else 0.0,
                chi_e=chi_e_list[j] if j < len(chi_e_list) else 0.0,
                gamma_max=gamma_list[j] if j < len(gamma_list) else 0.0,
                q_i=qi_list[j] if j < len(qi_list) else 0.0,
                q_e=qe_list[j] if j < len(qe_list) else 0.0,
            ))

    return results


# ── Benchmark comparison ─────────────────────────────────────────────

class TGLFBenchmark:
    """Compare our transport model against TGLF reference data."""

    def __init__(self, ref_dir: str | Path = TGLF_REF_DIR) -> None:
        self.ref_dir = Path(ref_dir)

    def compare(
        self,
        our_chi_i: NDArray,
        our_chi_e: NDArray,
        rho_grid: NDArray,
        tglf_outputs: list[TGLFOutput],
    ) -> TGLFComparisonResult:
        """Compare our chi profiles against TGLF outputs.

        Parameters
        ----------
        our_chi_i, our_chi_e : NDArray
            Our transport coefficients on rho_grid.
        rho_grid : NDArray
            Normalised radius grid.
        tglf_outputs : list[TGLFOutput]
            TGLF reference outputs.
        """
        result = TGLFComparisonResult()
        if not tglf_outputs:
            return result
        tglf_rho = np.array([o.rho for o in tglf_outputs])
        tglf_chi_i = np.array([o.chi_i for o in tglf_outputs])
        tglf_chi_e = np.array([o.chi_e for o in tglf_outputs])

        # Interpolate our values onto TGLF rho points
        our_i_interp = np.interp(tglf_rho, rho_grid, our_chi_i)
        our_e_interp = np.interp(tglf_rho, rho_grid, our_chi_e)

        result.rho_points = tglf_rho.tolist()
        result.our_chi_i = our_i_interp.tolist()
        result.tglf_chi_i = tglf_chi_i.tolist()
        result.our_chi_e = our_e_interp.tolist()
        result.tglf_chi_e = tglf_chi_e.tolist()

        # RMS error
        result.rms_error_chi_i = float(np.sqrt(np.mean((our_i_interp - tglf_chi_i) ** 2)))
        result.rms_error_chi_e = float(np.sqrt(np.mean((our_e_interp - tglf_chi_e) ** 2)))

        # Correlation
        if len(tglf_rho) > 1:
            if np.std(our_i_interp) > 0 and np.std(tglf_chi_i) > 0:
                result.correlation_chi_i = float(np.corrcoef(our_i_interp, tglf_chi_i)[0, 1])
            if np.std(our_e_interp) > 0 and np.std(tglf_chi_e) > 0:
                result.correlation_chi_e = float(np.corrcoef(our_e_interp, tglf_chi_e)[0, 1])

        # Max relative error
        denom_i = np.maximum(np.abs(tglf_chi_i), 1e-10)
        denom_e = np.maximum(np.abs(tglf_chi_e), 1e-10)
        result.max_rel_error_chi_i = float(np.max(np.abs(our_i_interp - tglf_chi_i) / denom_i))
        result.max_rel_error_chi_e = float(np.max(np.abs(our_e_interp - tglf_chi_e) / denom_e))

        return result

    def generate_comparison_table(self, results: list[TGLFComparisonResult]) -> str:
        """Generate markdown comparison table."""
        lines = [
            "| Case | RMS chi_i | RMS chi_e | Corr chi_i | Corr chi_e | Max Rel Err chi_i | Max Rel Err chi_e |",
            "|------|-----------|-----------|------------|------------|-------------------|-------------------|",
        ]
        for r in results:
            lines.append(
                f"| {r.case_name} | {r.rms_error_chi_i:.3f} | {r.rms_error_chi_e:.3f} "
                f"| {r.correlation_chi_i:.3f} | {r.correlation_chi_e:.3f} "
                f"| {r.max_rel_error_chi_i:.3f} | {r.max_rel_error_chi_e:.3f} |"
            )
        return "\n".join(lines)

    def generate_latex_table(self, results: list[TGLFComparisonResult]) -> str:
        """Generate publication-ready LaTeX table."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{SCPN Transport vs TGLF Comparison}",
            r"\label{tab:tglf_comparison}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Case & RMS $\chi_i$ & RMS $\chi_e$ & $r(\chi_i)$ & $r(\chi_e)$ "
            r"& Max Rel $\chi_i$ & Max Rel $\chi_e$ \\",
            r"\midrule",
        ]
        for r in results:
            lines.append(
                f"  {r.case_name} & {r.rms_error_chi_i:.3f} & {r.rms_error_chi_e:.3f} "
                f"& {r.correlation_chi_i:.3f} & {r.correlation_chi_e:.3f} "
                f"& {r.max_rel_error_chi_i:.3f} & {r.max_rel_error_chi_e:.3f} \\\\"
            )
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)


# ── Reference data ───────────────────────────────────────────────────

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


def write_reference_data(output_dir: str | Path = TGLF_REF_DIR) -> None:
    """Write reference TGLF data to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in REFERENCE_CASES.items():
        fname = name.lower().replace("-", "_").replace(" ", "_") + ".json"
        path = output_dir / fname
        payload = {"case_name": name, "source": "TGLF v4 reference", **data}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Wrote TGLF reference: %s", path)


# ── TGLF subprocess execution ────────────────────────────────────────

import subprocess
import tempfile


def _normalize_tglf_timeout_seconds(timeout_s: float) -> float:
    timeout = float(timeout_s)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("timeout_s must be finite and > 0.")
    return timeout


def _normalize_tglf_max_retries(max_retries: int) -> int:
    if isinstance(max_retries, bool) or not isinstance(max_retries, int):
        raise ValueError(
            f"max_retries must be an integer in [0, {_TGLF_MAX_RETRIES_LIMIT}]."
        )
    retries = int(max_retries)
    if retries < 0 or retries > _TGLF_MAX_RETRIES_LIMIT:
        raise ValueError(
            f"max_retries must be an integer in [0, {_TGLF_MAX_RETRIES_LIMIT}]."
        )
    return retries


def write_tglf_input_file(deck: TGLFInputDeck, output_dir: str | Path) -> Path:
    """Write a TGLF input.tglf file from a TGLFInputDeck.

    Parameters
    ----------
    deck : TGLFInputDeck
    output_dir : directory to write the file in

    Returns
    -------
    Path to the written file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "input.tglf"

    lines = [
        f"# TGLF input deck generated by SCPN Fusion Core",
        f"# rho = {deck.rho:.4f}",
        f"SIGN_BT = 1.0",
        f"SIGN_IT = 1.0",
        f"NS = 2",
        f"MASS_1 = 2.0",
        f"MASS_2 = 1.0",
        f"ZS_1 = 1.0",
        f"ZS_2 = -1.0",
        f"RLNS_1 = {deck.R_Lni:.6f}",
        f"RLNS_2 = {deck.R_Lne:.6f}",
        f"RLTS_1 = {deck.R_LTi:.6f}",
        f"RLTS_2 = {deck.R_LTe:.6f}",
        f"TAUS_1 = 1.0",
        f"TAUS_2 = {deck.T_e_keV / max(deck.T_i_keV, 0.01):.6f}",
        f"AS_1 = 1.0",
        f"AS_2 = 1.0",
        f"Q_LOC = {deck.q:.6f}",
        f"Q_PRIME_LOC = {deck.q_prime_loc:.6f}",
        f"P_PRIME_LOC = {deck.p_prime_loc:.6f}",
        f"S_KAPPA_LOC = {deck.s_kappa:.6f}",
        f"S_DELTA_LOC = {deck.s_delta:.6f}",
        f"KAPPA_LOC = {deck.kappa:.6f}",
        f"DELTA_LOC = {deck.delta:.6f}",
        f"SHAT_LOC = {deck.s_hat:.6f}",
        f"ALPHA_LOC = {deck.alpha_mhd:.6f}",
        f"XNUE = {deck.xnue:.6f}",
        f"BETAE = {deck.beta_e:.6f}",
        f"ZEFF = {deck.Z_eff:.6f}",
        f"RMAJ_LOC = {deck.R_major:.6f}",
        f"RMIN_LOC = {deck.a_minor * deck.rho:.6f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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

    if np.any(~np.isfinite(rho_samples)) or np.any(~np.isfinite(chi_i_samples)) or np.any(
        ~np.isfinite(chi_e_samples)
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


def run_tglf_binary(
    deck: TGLFInputDeck,
    tglf_binary_path: str | Path,
    *,
    timeout_s: float = 120.0,
    work_dir: str | Path | None = None,
    max_retries: int = 2,
) -> TGLFOutput:
    """Execute the TGLF binary on a given input deck and parse output.
    Harden with retries and input conditioning.
    """
    timeout_s = _normalize_tglf_timeout_seconds(timeout_s)
    max_retries = _normalize_tglf_max_retries(max_retries)
    tglf_path = Path(tglf_binary_path)
    if not tglf_path.exists():
        raise FileNotFoundError(f"TGLF binary not found: {tglf_path}")

    # Condition Inputs (Finite/Sanity Check)
    for field_name, val in deck.__dict__.items():
        if isinstance(val, (float, int)) and not np.isfinite(val):
            logger.warning(f"TGLF input '{field_name}' is non-finite ({val}). Clipping.")
            setattr(deck, field_name, 0.0)

    cleanup = False
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="tglf_"))
        cleanup = True
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            # Write input file
            input_path = write_tglf_input_file(deck, work_dir)
            
            # Run TGLF
            result = subprocess.run(
                [str(tglf_path)],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"TGLF exited with code {result.returncode}: {result.stderr[:500]}"
                )

            # Parse output
            output_file = work_dir / "out.tglf.run"
            if output_file.exists():
                return _parse_tglf_run_output(output_file, deck.rho)

            # Fallback: check for JSON output
            json_out = work_dir / "output.json"
            if json_out.exists():
                outputs = parse_tglf_output(work_dir)
                if outputs:
                    return outputs[0]
            
            raise RuntimeError("TGLF produced no parseable output.")

        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            last_exc = exc
            if attempt < max_retries:
                logger.warning(f"TGLF attempt {attempt+1} failed: {exc}. Retrying...")
                import time
                time.sleep(_TGLF_RETRY_BACKOFF_SECONDS)
        finally:
            if cleanup and (attempt == max_retries or not last_exc):
                import shutil
                shutil.rmtree(work_dir, ignore_errors=True)
    
    if last_exc:
        logger.error(f"TGLF execution failed after {max_retries+1} attempts.")
        # Return empty output rather than crashing the whole transport loop
        return TGLFOutput(rho=deck.rho)
    
    return TGLFOutput(rho=deck.rho)


def _parse_tglf_run_output(path: Path, rho: float) -> TGLFOutput:
    """Parse TGLF's out.tglf.run text output file.

    The file format has key=value lines with transport coefficients.
    """
    chi_i = 0.0
    chi_e = 0.0
    gamma_max = 0.0

    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().upper()
        val = val.strip()
        try:
            parsed = float(val)
        except ValueError:
            continue
        if not np.isfinite(parsed):
            continue
        fval = float(parsed)
        if key == "CHI_I" or key == "CHIEFF_I":
            chi_i = fval
        elif key == "CHI_E" or key == "CHIEFF_E":
            chi_e = fval
        elif key == "GAMMA_MAX":
            gamma_max = fval

    return TGLFOutput(rho=rho, chi_i=chi_i, chi_e=chi_e, gamma_max=gamma_max)
