# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GENE External Gyrokinetic Solver
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
GENE (Gyrokinetic Electromagnetic Numerical Experiment) external solver.

Generates ``parameters_gene`` namelist, executes via subprocess, parses
``nrg_xxxx`` eigenvalue files.

Reference: Jenko et al., Phys. Plasmas 7 (2000) 1904.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from scpn_fusion.core.gk_interface import GKLocalParams, GKOutput, GKSolverBase

_logger = logging.getLogger(__name__)

_GENE_TEMPLATE = """\
&parallelization
 n_procs_s = 1
 n_procs_v = 1
 n_procs_w = 1
/

&box
 n_spec = 2
 nx0 = 16
 nky0 = 1
 nz0 = 32
 nv0 = 32
 nw0 = 8
 ky0_ind = 1
/

&general
 nonlinear = .false.
 comp_type = 'IV'
 timelim = 300
 simtimelim = 100.0
 calc_dt = .true.
/

&geometry
 magn_geometry = 'miller'
 q0 = {q:.6f}
 shat = {s_hat:.6f}
 trpeps = {epsilon:.6f}
 major_R = {R0_over_a:.6f}
 amhd = {alpha_MHD:.6f}
 kappa = {kappa:.6f}
 delta = {delta:.6f}
/

&species
 name = 'ions'
 mass = 1.0
 charge = 1
 temp = 1.0
 dens = 1.0
 omt = {R_L_Ti:.6f}
 omn = {R_L_ne:.6f}
/

&species
 name = 'electrons'
 mass = 2.7234e-4
 charge = -1
 temp = {Te_Ti:.6f}
 dens = 1.0
 omt = {R_L_Te:.6f}
 omn = {R_L_ne:.6f}
/
"""


def generate_gene_input(params: GKLocalParams) -> str:
    R0_over_a = params.R0 / max(params.a, 0.01)
    return _GENE_TEMPLATE.format(
        q=params.q,
        s_hat=params.s_hat,
        epsilon=params.epsilon,
        R0_over_a=R0_over_a,
        alpha_MHD=params.alpha_MHD,
        kappa=params.kappa,
        delta=params.delta,
        R_L_Ti=params.R_L_Ti,
        R_L_Te=params.R_L_Te,
        R_L_ne=params.R_L_ne,
        Te_Ti=params.Te_Ti,
    )


def parse_gene_output(run_dir: Path) -> GKOutput:
    """Parse GENE nrg output files."""
    nrg_files = sorted(run_dir.glob("nrg_*"))
    if not nrg_files:
        return GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)

    try:
        data = np.loadtxt(nrg_files[-1])
        if data.ndim == 1:
            data = data.reshape(1, -1)
        # GENE nrg format: time, gamma, omega_r, ...
        gamma = data[-1, 1] if data.shape[1] > 1 else 0.0
        omega_r = data[-1, 2] if data.shape[1] > 2 else 0.0
        return GKOutput(
            chi_i=max(float(gamma), 0.0),
            chi_e=max(float(gamma) * 0.8, 0.0),
            D_e=max(float(gamma) * 0.1, 0.0),
            gamma=np.array([gamma]),
            omega_r=np.array([omega_r]),
            k_y=np.array([0.3]),
            dominant_mode="ITG" if omega_r < 0 else "TEM",
            converged=True,
        )
    except (ValueError, IndexError, OSError) as exc:
        _logger.warning("GENE output parse error: %s", exc)
        return GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)


class GENESolver(GKSolverBase):
    """GENE external solver via ``gene`` binary."""

    def __init__(self, binary: str = "gene", work_dir: Path | None = None) -> None:
        self.binary = binary
        self.work_dir = work_dir

    def is_available(self) -> bool:
        return shutil.which(self.binary) is not None

    def prepare_input(self, params: GKLocalParams) -> Path:
        base = self.work_dir or Path(tempfile.mkdtemp(prefix="gene_"))
        base.mkdir(parents=True, exist_ok=True)
        (base / "parameters").write_text(generate_gene_input(params))
        return base

    def run(self, input_path: Path, *, timeout_s: float = 30.0) -> GKOutput:
        if not self.is_available():
            return GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)
        try:
            subprocess.run(
                [self.binary],
                cwd=str(input_path),
                capture_output=True,
                timeout=timeout_s,
                check=True,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            return GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)
        return parse_gene_output(input_path)
