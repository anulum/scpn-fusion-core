# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — GS2 External Gyrokinetic Solver
"""
GS2 external solver interface.

Reference: Kotschenreuther et al., Comp. Phys. Comm. 88 (1995) 128.
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


def generate_gs2_input(params: GKLocalParams) -> str:
    R0_over_a = params.R0 / max(params.a, 0.01)
    return f"""\
&theta_grid_eik_knobs
 itor = 1
 iflux = 0
 irho = 2
 local_eq = .true.
 bishop = 4
 s_hat_input = {params.s_hat:.6f}
 beta_prime_input = 0.0
 ntheta = 32
 nperiod = 1
/
&theta_grid_parameters
 rhoc = {params.rho:.6f}
 qinp = {params.q:.6f}
 shat = {params.s_hat:.6f}
 akappa = {params.kappa:.6f}
 tri = {params.delta:.6f}
 rmaj = {R0_over_a:.6f}
 shift = 0.0
/
&species_knobs
 nspec = 2
/
&species_parameters_1
 z = 1
 mass = 1.0
 dens = 1.0
 temp = 1.0
 tprim = {params.R_L_Ti:.6f}
 fprim = {params.R_L_ne:.6f}
 type = 'ion'
/
&species_parameters_2
 z = -1
 mass = 2.7234e-4
 dens = 1.0
 temp = {params.Te_Ti:.6f}
 tprim = {params.R_L_Te:.6f}
 fprim = {params.R_L_ne:.6f}
 type = 'electron'
/
&kt_grids_knobs
 grid_option = 'single'
/
&kt_grids_single_parameters
 aky = 0.3
 theta0 = 0.0
/
"""


def parse_gs2_output(run_dir: Path) -> GKOutput:
    """Parse GS2 NetCDF or text output."""
    omega_file = run_dir / "gs2.omega"
    if omega_file.exists():
        try:
            data = np.loadtxt(omega_file)
            if data.ndim == 1 and len(data) >= 3:
                ky, gamma, omega_r = data[0], data[1], data[2]
                return GKOutput(
                    chi_i=max(float(gamma), 0.0),
                    chi_e=max(float(gamma) * 0.8, 0.0),
                    D_e=0.0,
                    gamma=np.array([gamma]),
                    omega_r=np.array([omega_r]),
                    k_y=np.array([ky]),
                    dominant_mode="ITG" if omega_r < 0 else "TEM",
                    converged=True,
                )
        except (ValueError, OSError) as exc:
            _logger.warning("GS2 parse error: %s", exc)

    return GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)


class GS2Solver(GKSolverBase):
    """GS2 external solver."""

    def __init__(self, binary: str = "gs2", work_dir: Path | None = None) -> None:
        self.binary = binary
        self.work_dir = work_dir

    def is_available(self) -> bool:
        return shutil.which(self.binary) is not None

    def prepare_input(self, params: GKLocalParams) -> Path:
        base = self.work_dir or Path(tempfile.mkdtemp(prefix="gs2_"))
        base.mkdir(parents=True, exist_ok=True)
        (base / "gs2.in").write_text(generate_gs2_input(params))
        return base

    def run(self, input_path: Path, *, timeout_s: float = 30.0) -> GKOutput:
        if not self.is_available():
            return GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)
        try:
            subprocess.run(
                [self.binary, str(input_path / "gs2.in")],
                cwd=str(input_path),
                capture_output=True,
                timeout=timeout_s,
                check=True,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            return GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)
        return parse_gs2_output(input_path)
