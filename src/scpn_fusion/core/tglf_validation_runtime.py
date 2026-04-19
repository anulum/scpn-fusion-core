# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TGLF Validation Runtime
"""Validation runtime extracted from ``tglf_interface`` monolith."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _resolve_runtime_transport_profiles(transport_solver: Any) -> tuple[np.ndarray, np.ndarray]:
    """Resolve public transport profiles, falling back only when absent."""
    ts = transport_solver
    n = len(ts.rho)
    chi_i = getattr(ts, "chi_i", getattr(ts, "_chi_i_profile", np.ones(n)))
    chi_e = getattr(ts, "chi_e", getattr(ts, "_chi_e_profile", np.ones(n) * 0.5))
    return np.asarray(chi_i, dtype=np.float64), np.asarray(chi_e, dtype=np.float64)


def validate_against_tglf(
    transport_solver: Any,
    tglf_binary_path: str | Path,
    rho_indices: list[int] | None = None,
):
    """Run TGLF on multiple flux surfaces and compare against our transport."""
    from scpn_fusion.core import tglf_interface as tglf

    ts = transport_solver
    n = len(ts.rho)

    if rho_indices is None:
        rho_indices = [n // 5, n // 4, n // 3, n // 2, 2 * n // 3]
    rho_indices = [i for i in rho_indices if 1 <= i < n - 1]

    tglf_outputs = []
    for idx in rho_indices:
        deck = tglf.generate_input_deck(ts, idx)
        output = tglf.run_tglf_binary(deck, tglf_binary_path)
        tglf_outputs.append(output)

    chi_i_profile, chi_e_profile = _resolve_runtime_transport_profiles(ts)

    benchmark = tglf.TGLFBenchmark()
    result = benchmark.compare(chi_i_profile, chi_e_profile, ts.rho, tglf_outputs)
    result.case_name = "Live TGLF validation"
    return result


__all__ = ["validate_against_tglf"]
