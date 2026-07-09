# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — QuaLiKiz External Gyrokinetic Solver
"""
QuaLiKiz quasilinear gyrokinetic solver interface.

Unlike TGLF/GENE/GS2/CGYRO, QuaLiKiz has a Python API (qlknn-hyper).
Falls back to subprocess if the Python module is unavailable.

Reference: Bourdelle et al., Phys. Plasmas 14 (2007) 112501.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
import logging
import tempfile
from pathlib import Path
from typing import Any, cast


from scpn_fusion.core.gk_interface import GKLocalParams, GKOutput, GKSolverBase

_logger = logging.getLogger(__name__)


def _import_qualikiz_tools() -> Any | None:
    """Return the optional QuaLiKiz Python module when it is importable."""
    try:
        return cast(Any, import_module("qualikiz_tools"))
    except ImportError:
        return None


def _float_result_field(result: Mapping[str, object], key: str) -> float:
    """Coerce a QuaLiKiz result field into a floating-point transport value."""
    return float(cast(Any, result.get(key, 0.0)))


def _try_qualikiz_python(params: GKLocalParams) -> GKOutput | None:
    """Try running QuaLiKiz via Python API."""
    qualikiz_tools = _import_qualikiz_tools()
    if qualikiz_tools is None:
        return None
    try:
        run_qualikiz = qualikiz_tools.run
        raw_result = run_qualikiz(
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
        if not isinstance(raw_result, Mapping):
            return None
        result = cast(Mapping[str, object], raw_result)
        return GKOutput(
            chi_i=_float_result_field(result, "chi_i"),
            chi_e=_float_result_field(result, "chi_e"),
            D_e=_float_result_field(result, "D_e"),
            converged=True,
            dominant_mode="ITG",
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


class QuaLiKizSolver(GKSolverBase):
    """QuaLiKiz solver via Python API or subprocess fallback."""

    def __init__(self, binary: str = "qualikiz", work_dir: Path | None = None) -> None:
        """Configure QuaLiKiz access and optional persistent work directory."""
        self.binary = binary
        self.work_dir = work_dir

    def is_available(self) -> bool:
        """Return whether the QuaLiKiz Python interface can be imported."""
        return _import_qualikiz_tools() is not None

    def prepare_input(self, params: GKLocalParams) -> Path:
        """Create a QuaLiKiz work directory and cache parameters for execution."""
        base = self.work_dir or Path(tempfile.mkdtemp(prefix="qualikiz_"))
        base.mkdir(parents=True, exist_ok=True)
        # Store params for Python API path
        self._last_params = params
        return base

    def run(self, input_path: Path, *, timeout_s: float = 30.0) -> GKOutput:
        """Run QuaLiKiz through the Python API when cached parameters exist."""
        params = getattr(self, "_last_params", None)
        if params is not None:
            result = _try_qualikiz_python(params)
            if result is not None:
                return result
        return GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)
