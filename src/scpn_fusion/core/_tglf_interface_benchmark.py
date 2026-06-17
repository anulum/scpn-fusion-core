# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TGLF Interface Benchmark
"""Benchmark comparison helpers for the public TGLF interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core._tglf_interface_types import (
    TGLFComparisonResult,
    TGLFOutput,
    TGLF_REF_DIR,
)

FloatArray = NDArray[np.float64]


class TGLFBenchmark:
    """Compare local transport coefficients against TGLF reference data."""

    def __init__(self, ref_dir: str | Path = TGLF_REF_DIR) -> None:
        self.ref_dir = Path(ref_dir)

    def compare(
        self,
        our_chi_i: FloatArray,
        our_chi_e: FloatArray,
        rho_grid: FloatArray,
        tglf_outputs: list[TGLFOutput],
    ) -> TGLFComparisonResult:
        """Compare local chi profiles against TGLF outputs."""
        result = TGLFComparisonResult()
        if not tglf_outputs:
            return result
        tglf_rho = np.array([o.rho for o in tglf_outputs])
        tglf_chi_i = np.array([o.chi_i for o in tglf_outputs])
        tglf_chi_e = np.array([o.chi_e for o in tglf_outputs])

        our_i_interp = np.interp(tglf_rho, rho_grid, our_chi_i)
        our_e_interp = np.interp(tglf_rho, rho_grid, our_chi_e)

        result.rho_points = tglf_rho.tolist()
        result.our_chi_i = our_i_interp.tolist()
        result.tglf_chi_i = tglf_chi_i.tolist()
        result.our_chi_e = our_e_interp.tolist()
        result.tglf_chi_e = tglf_chi_e.tolist()

        result.rms_error_chi_i = float(np.sqrt(np.mean((our_i_interp - tglf_chi_i) ** 2)))
        result.rms_error_chi_e = float(np.sqrt(np.mean((our_e_interp - tglf_chi_e) ** 2)))

        if len(tglf_rho) > 1:
            if np.std(our_i_interp) > 0 and np.std(tglf_chi_i) > 0:
                result.correlation_chi_i = float(np.corrcoef(our_i_interp, tglf_chi_i)[0, 1])
            if np.std(our_e_interp) > 0 and np.std(tglf_chi_e) > 0:
                result.correlation_chi_e = float(np.corrcoef(our_e_interp, tglf_chi_e)[0, 1])

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
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )
        return "\n".join(lines)
