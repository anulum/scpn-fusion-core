# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TGLF External Gyrokinetic Solver
"""
TGLF (Trapped Gyro-Landau Fluid) external solver interface.

Generates TGLF input namelists, executes the ``tglf`` binary via
subprocess, and parses growth-rate / flux output files.  Falls back
to the built-in quasilinear dispersion solver when the binary is
unavailable.

Reference: Staebler et al., Phys. Plasmas 14 (2007) 055909.
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

# TGLF namelist keys mapped from GKLocalParams
_TGLF_NAMELIST_TEMPLATE = """\
&tglf_namelist
 UNITS = 'GYRO'
 USE_TRANSPORT_MODEL = .true.
 GEOMETRY_FLAG = 1
 SIGN_BT = 1.0
 SIGN_IT = 1.0
 NS = 2
 MASS_1 = 1.0
 MASS_2 = 2.7234e-4
 RLNS_1 = {R_L_ne:.6f}
 RLNS_2 = {R_L_ne:.6f}
 RLTS_1 = {R_L_Ti:.6f}
 RLTS_2 = {R_L_Te:.6f}
 TAUS_1 = 1.0
 TAUS_2 = {Te_Ti:.6f}
 AS_1 = 1.0
 AS_2 = 1.0
 ZS_1 = 1.0
 ZS_2 = -1.0
 VPAR_1 = 0.0
 VPAR_2 = 0.0
 VEXB_SHEAR = 0.0
 BETAE = {beta_e:.6e}
 XNUE = {nu_star:.6e}
 ZEFF = {Z_eff:.4f}
 RMIN_LOC = {rho:.6f}
 RMAJ_LOC = {R0_over_a:.6f}
 Q_LOC = {q:.6f}
 Q_PRIME_LOC = 0.0
 P_PRIME_LOC = 0.0
 KAPPA_LOC = {kappa:.6f}
 S_KAPPA_LOC = 0.0
 DELTA_LOC = {delta:.6f}
 S_DELTA_LOC = 0.0
 DRMINDX_LOC = 1.0
 DRMAJDX_LOC = 0.0
 DZMAJDX_LOC = 0.0
 SHAT = {s_hat:.6f}
 ALPHA_MHD = {alpha_MHD:.6f}
 NKY = 12
 KY = 0.3
/
"""


def generate_tglf_input(params: GKLocalParams) -> str:
    """Render a TGLF namelist string from local plasma parameters."""
    R0_over_a = params.R0 / max(params.a, 0.01)
    return _TGLF_NAMELIST_TEMPLATE.format(
        R_L_ne=params.R_L_ne,
        R_L_Ti=params.R_L_Ti,
        R_L_Te=params.R_L_Te,
        Te_Ti=params.Te_Ti,
        beta_e=params.beta_e,
        nu_star=params.nu_star,
        Z_eff=params.Z_eff,
        rho=params.rho,
        R0_over_a=R0_over_a,
        q=params.q,
        kappa=params.kappa,
        delta=params.delta,
        s_hat=params.s_hat,
        alpha_MHD=params.alpha_MHD,
    )


def parse_tglf_output(run_dir: Path) -> GKOutput:
    """Parse TGLF output files from *run_dir*.

    Expects ``out.tglf.run`` with growth rates and ``out.tglf.transport``
    with quasilinear fluxes.  If files are missing, returns a zero-flux
    unconverged result.
    """
    transport_file = run_dir / "out.tglf.transport"
    eigenvalue_file = run_dir / "out.tglf.eigenvalue_spectrum"

    chi_i = 0.0
    chi_e = 0.0
    D_e = 0.0
    converged = False

    if transport_file.exists():
        try:
            lines = transport_file.read_text().strip().splitlines()
            for line in lines:
                tokens = line.split()
                if len(tokens) < 2:
                    continue
                key = tokens[0].lower()
                val = float(tokens[1])
                if key == "chi_i":
                    chi_i = val
                elif key == "chi_e":
                    chi_e = val
                elif key in ("d_e", "particle_flux"):
                    D_e = val
            converged = True
        except (ValueError, IndexError) as exc:
            _logger.warning("TGLF transport parse error: %s", exc)

    gamma = np.empty(0)
    omega_r = np.empty(0)
    k_y = np.empty(0)

    if eigenvalue_file.exists():
        try:
            data = np.loadtxt(eigenvalue_file, comments="#")
            if data.ndim == 2 and data.shape[1] >= 3:
                k_y = data[:, 0]
                gamma = data[:, 1]
                omega_r = data[:, 2]
        except (ValueError, OSError) as exc:
            _logger.warning("TGLF eigenvalue parse error: %s", exc)

    dominant = _classify_dominant_mode(gamma, omega_r)

    return GKOutput(
        chi_i=chi_i,
        chi_e=chi_e,
        D_e=D_e,
        gamma=gamma,
        omega_r=omega_r,
        k_y=k_y,
        dominant_mode=dominant,
        converged=converged,
    )


def _classify_dominant_mode(gamma: np.ndarray, omega_r: np.ndarray) -> str:
    """Identify dominant instability from growth rate spectrum."""
    if len(gamma) == 0 or np.all(gamma <= 0):
        return "stable"
    idx = int(np.argmax(gamma))
    if omega_r[idx] < 0:
        return "ITG"  # ion diamagnetic direction
    return "TEM"  # electron diamagnetic direction


class TGLFSolver(GKSolverBase):
    """TGLF external solver via GACODE ``tglf`` binary.

    Parameters
    ----------
    binary : str
        Path or name of the ``tglf`` executable.
    work_dir : Path or None
        Persistent working directory.  If None, uses a tempdir per call.
    """

    def __init__(
        self,
        binary: str = "tglf",
        work_dir: Path | None = None,
    ) -> None:
        self.binary = binary
        self.work_dir = work_dir

    def is_available(self) -> bool:
        return shutil.which(self.binary) is not None

    def prepare_input(self, params: GKLocalParams) -> Path:
        base = self.work_dir or Path(tempfile.mkdtemp(prefix="tglf_"))
        base.mkdir(parents=True, exist_ok=True)
        input_file = base / "input.tglf"
        input_file.write_text(generate_tglf_input(params))
        return base

    def run(self, input_path: Path, *, timeout_s: float = 30.0) -> GKOutput:
        if not self.is_available():
            _logger.warning("TGLF binary not found, returning fallback")
            return self._fallback()

        try:
            subprocess.run(
                [self.binary, "-i", str(input_path / "input.tglf")],
                cwd=str(input_path),
                capture_output=True,
                timeout=timeout_s,
                check=True,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as exc:
            _logger.warning("TGLF execution failed: %s", exc)
            return self._fallback()

        return parse_tglf_output(input_path)

    @staticmethod
    def _fallback() -> GKOutput:
        return GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)
