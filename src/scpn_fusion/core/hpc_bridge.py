# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — HPC Bridge Compatibility
"""Compatibility import path for the hardened HPC bridge implementation."""

from __future__ import annotations

from scpn_fusion.hpc.hpc_bridge import (
    HPCBridge,
    _as_contiguous_f64,
    _require_c_contiguous_f64,
    _sanitize_convergence_params,
    compile_cpp,
)

__all__ = [
    "HPCBridge",
    "_as_contiguous_f64",
    "_require_c_contiguous_f64",
    "_sanitize_convergence_params",
    "compile_cpp",
]
