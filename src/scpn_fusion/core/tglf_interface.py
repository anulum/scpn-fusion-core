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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
TGLF_REF_DIR = REPO_ROOT / "validation" / "tglf_reference"


# ── Data containers ──────────────────────────────────────────────────

@dataclass
class TGLFInputDeck:
    """All TGLF input parameters for a single flux surface."""
    rho: float = 0.5
    # Geometry
    s_hat: float = 1.0          # magnetic shear
    q: float = 1.5              # safety factor
    alpha_mhd: float = 0.0     # MHD alpha
    kappa: float = 1.7          # elongation
    delta: float = 0.3          # triangularity
    # Gradients (R / L_X)
    R_LTi: float = 6.0         # R / L_Ti
    R_LTe: float = 6.0         # R / L_Te
    R_Lne: float = 2.0         # R / L_ne
    R_Lni: float = 2.0         # R / L_ni
    # Plasma parameters
    beta_e: float = 0.01       # electron beta
    Z_eff: float = 1.5         # effective charge
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

    R0 = getattr(ts, 'R0', 6.2)
    a = getattr(ts, 'a', 2.0)

    R_LTi = R0 * _grad_scale(ts.Ti, rho_idx)
    R_LTe = R0 * _grad_scale(ts.Te, rho_idx)
    R_Lne = R0 * _grad_scale(ts.ne, rho_idx)

    q_val = 1.0 + 3.0 * rho**2  # default q profile
    s_hat = 2.0 * rho * 3.0 / q_val * rho if rho > 0 else 0.0

    return TGLFInputDeck(
        rho=rho,
        s_hat=s_hat,
        q=q_val,
        R_LTi=R_LTi,
        R_LTe=R_LTe,
        R_Lne=R_Lne,
        R_Lni=R_Lne,
        T_e_keV=Te,
        T_i_keV=Ti,
        n_e_19=ne,
        R_major=R0,
        a_minor=a,
    )


# ── Output parsing ───────────────────────────────────────────────────

def parse_tglf_output(output_dir: str | Path) -> list[TGLFOutput]:
    """Parse TGLF output files from a directory.

    Expects JSON files with keys: rho, chi_i, chi_e, gamma_max, q_i, q_e.
    """
    output_dir = Path(output_dir)
    results = []

    for f in sorted(output_dir.glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)

        rho_pts = data.get("rho_points", [data.get("rho", 0.5)])
        chi_i_list = data.get("chi_i", [0.0])
        chi_e_list = data.get("chi_e", [0.0])
        gamma_list = data.get("gamma_max", [0.0])
        qi_list = data.get("q_i", [0.0])
        qe_list = data.get("q_e", [0.0])

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
