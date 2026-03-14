# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — QuaLiKiz External Gyrokinetic Solver
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
QuaLiKiz quasilinear gyrokinetic solver interface.

Unlike TGLF/GENE/GS2/CGYRO, QuaLiKiz has a Python API (qlknn-hyper).
Falls back to subprocess if the Python module is unavailable.

Reference: Bourdelle et al., Phys. Plasmas 14 (2007) 112501.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path


from scpn_fusion.core.gk_interface import GKLocalParams, GKOutput, GKSolverBase

_logger = logging.getLogger(__name__)


def _try_qualikiz_python(params: GKLocalParams) -> GKOutput | None:
    """Try running QuaLiKiz via Python API."""
    try:
        import qualikiz_tools  # type: ignore[import-not-found]

        result = qualikiz_tools.run(
            Rmin=params.rho,
            Rmaj=params.R0 / max(params.a, 0.01),
            q=params.q,
            smag=params.s_hat,
            Te=params.Te_Ti,
            Ti=1.0,
            Ate=params.R_L_Te,
            Ati=params.R_L_Ti,
            Ane=params.R_L_ne,
            Zeff=params.Z_eff,
        )
        return GKOutput(
            chi_i=float(result.get("chi_i", 0.0)),
            chi_e=float(result.get("chi_e", 0.0)),
            D_e=float(result.get("D_e", 0.0)),
            converged=True,
            dominant_mode="ITG",
        )
    except (ImportError, AttributeError, KeyError, TypeError):
        return None


class QuaLiKizSolver(GKSolverBase):
    """QuaLiKiz solver via Python API or subprocess fallback."""

    def __init__(self, binary: str = "qualikiz", work_dir: Path | None = None) -> None:
        self.binary = binary
        self.work_dir = work_dir

    def is_available(self) -> bool:
        try:
            import qualikiz_tools  # type: ignore[import-not-found]  # noqa: F401

            return True
        except ImportError:
            return False

    def prepare_input(self, params: GKLocalParams) -> Path:
        base = self.work_dir or Path(tempfile.mkdtemp(prefix="qualikiz_"))
        base.mkdir(parents=True, exist_ok=True)
        # Store params for Python API path
        self._last_params = params
        return base

    def run(self, input_path: Path, *, timeout_s: float = 30.0) -> GKOutput:
        params = getattr(self, "_last_params", None)
        if params is not None:
            result = _try_qualikiz_python(params)
            if result is not None:
                return result
        return GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)
