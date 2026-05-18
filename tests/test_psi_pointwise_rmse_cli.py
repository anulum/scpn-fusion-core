# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — CLI tests for point-wise ψ RMSE validation
"""
CLI-level regression tests for validation/psi_pointwise_rmse.py.

Ensures `main()` remains compatible with strict ASCII stdout encodings
that appear on some Windows terminals.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import validation.psi_pointwise_rmse as psi_rmse_mod
from validation.psi_pointwise_rmse import EfitNRMSEBenchmarkGate, PsiRMSESummary


class _AsciiStrictStdout:
    """stdout stub that rejects non-ASCII writes."""

    encoding = "ascii"

    def __init__(self) -> None:
        self._parts: list[str] = []

    def write(self, text: str) -> int:
        text.encode("ascii", errors="strict")
        self._parts.append(text)
        return len(text)

    def flush(self) -> None:  # pragma: no cover - no side-effects to assert
        return None

    def getvalue(self) -> str:
        return "".join(self._parts)


def test_main_is_ascii_stdout_safe(tmp_path: Path, monkeypatch) -> None:
    summary = PsiRMSESummary(
        count=1,
        mean_psi_rmse_norm=0.125,
        mean_psi_relative_l2=0.05,
        mean_gs_residual_l2=0.01,
        worst_psi_rmse_norm=0.125,
        worst_file="sample.geqdsk",
        rows=[
            {
                "file": "sample.geqdsk",
                "grid": "9x9",
                "psi_rmse_norm": 0.125,
                "psi_relative_l2": 0.05,
                "gs_residual_l2": 0.01,
                "sor_iterations": 12,
                "solve_time_ms": 3.4,
            }
        ],
    )

    monkeypatch.setattr(psi_rmse_mod, "validate_all_sparc", lambda: summary)
    gate = EfitNRMSEBenchmarkGate(
        schema_version="efit-nrmse-benchmark.v1",
        benchmark_id="efit-nrmse-benchmark",
        count=1,
        min_required_files=1,
        threshold=0.05,
        pass_count=0,
        passes=False,
        mean_psi_rmse_norm=0.125,
        worst_psi_rmse_norm=0.125,
        worst_file="sparc/sample.geqdsk",
        count_by_machine={"sparc": 1},
        provenance_by_machine={"sparc": "real_public_design_reference"},
        source_consistency_counts={"profile_source_mismatch": 1},
        worst_source_residual_l2=2.5,
        worst_source_alignment_file="sparc/sample.geqdsk",
        failure_reasons=["profile-source mismatch attribution in 1 rows"],
        rows=[
            {
                "file": "sparc/sample.geqdsk",
                "grid": "9x9",
                "machine": "sparc",
                "provenance": "real_public_design_reference",
                "gs_residual_l2": 0.01,
                "gs_residual_max": 0.02,
                "psi_rmse_wb": 0.001,
                "psi_rmse_norm": 0.125,
                "psi_rmse_plasma_wb": 0.001,
                "psi_max_error_wb": 0.002,
                "psi_relative_l2": 0.05,
                "sor_iterations": 12,
                "sor_residual": 1e-8,
                "solve_time_ms": 3.4,
                "operator_source_psi_rmse_norm": 1e-12,
                "operator_source_sor_iterations": 10,
                "operator_source_sor_residual": 1e-12,
                "source_consistency_class": "profile_source_mismatch",
                "operator_source_norm": 1.0,
                "profile_source_norm": 1.0,
                "source_residual_l2": 2.5,
                "source_correlation": -0.2,
                "source_best_fit_scale": -0.5,
                "source_best_fit_offset": 0.1,
                "source_best_fit_relative_l2": 0.9,
                "plasma_mask_fraction": 0.5,
                "pressure_source_norm": 0.7,
                "ffprime_source_norm": 0.3,
                "total_source_norm": 1.0,
                "pressure_source_fraction": 0.7,
                "ffprime_source_fraction": 0.3,
                "source_plasma_residual_l2": 2.0,
                "source_vacuum_residual_l2": 4.0,
                "source_plasma_operator_norm": 0.5,
                "source_vacuum_operator_norm": 0.5,
                "source_plasma_point_count": 4.0,
                "source_vacuum_point_count": 5.0,
                "best_source_candidate": "profile_source",
                "best_source_candidate_residual_l2": 2.5,
                "profile_source_candidate_rank": 1,
                "best_operator_candidate": "delta_star_psi",
                "best_operator_candidate_residual_l2": 2.5,
                "delta_star_psi_candidate_rank": 1,
                "threshold": 0.05,
                "passes_threshold": False,
            }
        ],
    )
    monkeypatch.setattr(psi_rmse_mod, "validate_efit_nrmse_benchmark", lambda: gate)
    monkeypatch.setattr(psi_rmse_mod, "ROOT", tmp_path)
    ascii_stdout = _AsciiStrictStdout()
    monkeypatch.setattr(sys, "stdout", ascii_stdout)

    rc = psi_rmse_mod.main()
    assert rc == 0
    stdout = ascii_stdout.getvalue()
    assert "psi_N RMSE" in stdout
    assert "Source classes:" in stdout
    assert "profile_source_mismatch=1" in stdout
    assert "Worst source residual:" in stdout
    assert "Worst source components:" in stdout
    assert "pressure_fraction=0.700000" in stdout
    assert "ffprime_fraction=0.300000" in stdout
    assert "Worst source masked residuals:" in stdout
    assert "plasma=2.000000" in stdout
    assert "vacuum=4.000000" in stdout
    assert "Best source candidate:" in stdout
    assert "profile_source" in stdout
    assert "Best operator candidate:" in stdout
    assert "delta_star_psi" in stdout

    report = tmp_path / "validation" / "reports" / "psi_pointwise_rmse.json"
    assert report.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["count"] == 1
    assert payload["worst_file"] == "sample.geqdsk"

    benchmark_report = tmp_path / "validation" / "reports" / "psi_efit_nrmse_benchmark.json"
    assert benchmark_report.exists()
    benchmark_payload = json.loads(benchmark_report.read_text(encoding="utf-8"))
    assert benchmark_payload["source_consistency_counts"] == {"profile_source_mismatch": 1}
