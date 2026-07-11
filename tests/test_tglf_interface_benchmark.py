# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the TGLF benchmark comparison helpers.

Covers the chi-profile comparison (empty and populated), including the
correlation branch, and the markdown and LaTeX comparison-table renderers that
the module-linkage import never exercises.
"""

from __future__ import annotations

import numpy as np

from scpn_fusion.core._tglf_interface_benchmark import TGLFBenchmark
from scpn_fusion.core._tglf_interface_types import TGLFComparisonResult, TGLFOutput


def _outputs() -> list[TGLFOutput]:
    """Three TGLF reference points with distinct chi values (non-zero variance)."""
    return [
        TGLFOutput(rho=0.3, chi_i=1.0, chi_e=0.5),
        TGLFOutput(rho=0.5, chi_i=2.0, chi_e=1.2),
        TGLFOutput(rho=0.7, chi_i=3.5, chi_e=2.0),
    ]


def test_compare_empty_outputs_returns_default_result() -> None:
    """With no TGLF outputs the comparison returns an empty default result."""
    result = TGLFBenchmark().compare(
        np.array([1.0, 2.0]), np.array([0.5, 1.0]), np.array([0.3, 0.7]), []
    )
    assert isinstance(result, TGLFComparisonResult)
    assert result.rho_points == []


def test_compare_populates_errors_and_correlation() -> None:
    """A populated comparison fills RMS, max-relative-error, and correlation fields."""
    rho_grid = np.linspace(0.2, 0.8, 7)
    our_chi_i = np.linspace(1.0, 3.0, 7)
    our_chi_e = np.linspace(0.5, 2.0, 7)
    result = TGLFBenchmark().compare(our_chi_i, our_chi_e, rho_grid, _outputs())

    assert len(result.rho_points) == 3
    assert result.rms_error_chi_i >= 0.0
    assert result.rms_error_chi_e >= 0.0
    assert result.max_rel_error_chi_i >= 0.0
    # Three monotonic points give a well-defined Pearson correlation.
    assert -1.0 <= result.correlation_chi_i <= 1.0
    assert -1.0 <= result.correlation_chi_e <= 1.0


def test_comparison_table_renders_markdown_row() -> None:
    """The markdown renderer emits a header plus one row per result."""
    result = TGLFBenchmark().compare(
        np.linspace(1.0, 3.0, 7),
        np.linspace(0.5, 2.0, 7),
        np.linspace(0.2, 0.8, 7),
        _outputs(),
    )
    result.case_name = "ITG-dominated"
    table = TGLFBenchmark().generate_comparison_table([result])
    assert table.startswith("| Case |")
    assert "ITG-dominated" in table
    assert table.count("\n") >= 2


def test_latex_table_renders_publication_block() -> None:
    """The LaTeX renderer wraps the rows in a full table environment."""
    result = TGLFComparisonResult(case_name="TEM-dominated")
    latex = TGLFBenchmark().generate_latex_table([result])
    assert latex.startswith(r"\begin{table}")
    assert r"\end{table}" in latex
    assert "TEM-dominated" in latex
