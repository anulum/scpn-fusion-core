# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — CGYRO External Gyrokinetic Solver
"""
CGYRO external solver interface.

Reference: Candy & Waltz, J. Comp. Phys. 186 (2003) 545.
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


def generate_cgyro_input(params: GKLocalParams) -> str:
    R0_over_a = params.R0 / max(params.a, 0.01)
    return f"""\
# CGYRO input.cgyro
EQUILIBRIUM_MODEL=2
RMIN={params.rho:.6f}
RMAJ={R0_over_a:.6f}
Q={params.q:.6f}
S={params.s_hat:.6f}
KAPPA={params.kappa:.6f}
DELTA={params.delta:.6f}
BETAE_UNIT={params.beta_e:.6e}
ZEFF={params.Z_eff:.4f}
NU_EE={params.nu_star:.6e}
N_SPECIES=2
MASS_1=1.0
MASS_2=2.7234e-4
Z_1=1
Z_2=-1
DENS_1=1.0
DENS_2=1.0
TEMP_1=1.0
TEMP_2={params.Te_Ti:.6f}
DLNTDR_1={params.R_L_Ti:.6f}
DLNTDR_2={params.R_L_Te:.6f}
DLNNDR_1={params.R_L_ne:.6f}
DLNNDR_2={params.R_L_ne:.6f}
KY=0.3
NONLINEAR_FLAG=0
N_RADIAL=1
"""


def parse_cgyro_output(run_dir: Path) -> GKOutput:
    out_file = run_dir / "out.cgyro.freq"
    if out_file.exists():
        try:
            data = np.loadtxt(out_file)
            if data.ndim == 1 and len(data) >= 2:
                gamma, omega_r = float(data[0]), float(data[1])
                return GKOutput(
                    chi_i=max(gamma, 0.0),
                    chi_e=max(gamma * 0.8, 0.0),
                    D_e=0.0,
                    gamma=np.array([gamma]),
                    omega_r=np.array([omega_r]),
                    k_y=np.array([0.3]),
                    dominant_mode="ITG" if omega_r < 0 else "TEM",
                    converged=True,
                )
        except (ValueError, OSError) as exc:
            _logger.warning("CGYRO parse error: %s", exc)

    return GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)


class CGYROSolver(GKSolverBase):
    """CGYRO external solver."""

    def __init__(self, binary: str = "cgyro", work_dir: Path | None = None) -> None:
        self.binary = binary
        self.work_dir = work_dir

    def is_available(self) -> bool:
        return shutil.which(self.binary) is not None

    def prepare_input(self, params: GKLocalParams) -> Path:
        base = self.work_dir or Path(tempfile.mkdtemp(prefix="cgyro_"))
        base.mkdir(parents=True, exist_ok=True)
        (base / "input.cgyro").write_text(generate_cgyro_input(params))
        return base

    def run(self, input_path: Path, *, timeout_s: float = 30.0) -> GKOutput:
        if not self.is_available():
            return GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)
        try:
            subprocess.run(
                [self.binary, "-i", str(input_path / "input.cgyro")],
                cwd=str(input_path),
                capture_output=True,
                timeout=timeout_s,
                check=True,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            return GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)
        return parse_cgyro_output(input_path)
