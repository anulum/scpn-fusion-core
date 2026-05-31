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
        benchmark_id="sparc-pointwise-rmse",
        benchmark_scope="profile_source_fixed_boundary_reconstruction",
        benchmark_contract="test contract",
        solver_mode="raw_geqdsk_profile_source_fixed_boundary",
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
                "benchmark_scope": "profile_source_fixed_boundary_reconstruction",
                "solver_mode": "raw_geqdsk_profile_source_fixed_boundary",
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
        schema_version="efit-nrmse-benchmark.v2",
        benchmark_id="efit-nrmse-benchmark",
        benchmark_scope="profile_source_fixed_boundary_reconstruction",
        benchmark_contract="test contract",
        raw_profile_solver_mode="raw_geqdsk_profile_source_fixed_boundary",
        operator_source_solver_mode="operator_source_fixed_boundary_delta_star_psi",
        adapted_profile_solver_mode="adapted_geqdsk_profile_source_fixed_boundary",
        count=1,
        min_required_files=1,
        threshold=0.05,
        pass_count=0,
        gate_row_count=1,
        gate_pass_count=0,
        gate_worst_psi_rmse_norm=0.125,
        gate_worst_file="sparc/sample.geqdsk",
        passes=False,
        mean_psi_rmse_norm=0.125,
        worst_psi_rmse_norm=0.125,
        worst_file="sparc/sample.geqdsk",
        count_by_machine={"sparc": 1},
        provenance_by_machine={"sparc": "real_public_design_reference"},
        reference_role_counts={"gate": 1, "diagnostic": 0},
        reference_class_counts={"public_efit_reference": 1, "synthetic_proxy_reference": 0},
        solver_mode_counts={
            "raw_geqdsk_profile_source_fixed_boundary": 1,
            "operator_source_fixed_boundary_delta_star_psi": 1,
            "adapted_geqdsk_profile_source_fixed_boundary": 1,
        },
        source_consistency_counts={"profile_source_mismatch": 1},
        source_domain_residual_class_counts={"plasma_and_vacuum_source_mismatch": 1},
        source_domain_required_solver_mode_counts={
            "profile_source_then_free_boundary_reconstruction_required": 1
        },
        source_domain_next_action_counts={
            "repair_plasma_profile_source_then_reconstruct_vacuum_boundary_response": 1
        },
        operator_source_threshold=1e-6,
        operator_source_pass_count=1,
        gate_operator_source_pass_count=1,
        gate_operator_source_worst_psi_rmse_norm=1e-12,
        gate_operator_source_worst_file="sparc/sample.geqdsk",
        operator_source_worst_psi_rmse_norm=1e-12,
        operator_source_worst_file="sparc/sample.geqdsk",
        source_convention_adapter_threshold=0.15,
        source_convention_adapter_pass_count=0,
        gate_source_convention_adapter_pass_count=0,
        source_convention_adapter_counts={"not_evaluated": 1},
        adapted_profile_threshold=0.05,
        adapted_profile_pass_count=0,
        gate_adapted_profile_pass_count=0,
        adapted_profile_worst_psi_rmse_norm=float("nan"),
        adapted_profile_worst_file="",
        worst_source_residual_l2=2.5,
        worst_source_alignment_file="sparc/sample.geqdsk",
        gate_worst_source_residual_l2=2.5,
        gate_worst_source_alignment_file="sparc/sample.geqdsk",
        source_sum_identity_max_abs_error=0.0,
        source_sum_identity_pass=True,
        operator_current_closure_pass_count=1,
        gate_operator_current_closure_pass_count=1,
        operator_current_best_closure_pass_count=1,
        gate_operator_current_best_closure_pass_count=1,
        operator_current_worst_relative_error=0.00115,
        operator_current_worst_file="sparc/sample.geqdsk",
        gate_operator_current_worst_relative_error=0.00115,
        gate_operator_current_worst_file="sparc/sample.geqdsk",
        operator_current_best_worst_relative_error=0.00115,
        operator_current_best_worst_file="sparc/sample.geqdsk",
        gate_operator_current_best_worst_relative_error=0.00115,
        gate_operator_current_best_worst_file="sparc/sample.geqdsk",
        profile_current_closure_threshold=0.05,
        profile_current_closure_pass_count=0,
        gate_profile_current_closure_pass_count=0,
        adapted_profile_current_closure_pass_count=0,
        gate_adapted_profile_current_closure_pass_count=0,
        current_limited_adapted_profile_pass_count=0,
        gate_current_limited_adapted_profile_pass_count=0,
        effective_profile_current_closure_pass_count=0,
        gate_effective_profile_current_closure_pass_count=0,
        profile_current_closure_failure_class_counts={
            "profile_current_under_closes_declared_current": 1
        },
        profile_current_worst_relative_error=0.0862,
        profile_current_worst_file="sparc/sample.geqdsk",
        failure_reasons=["public gate profile-source mismatch attribution in 1/1 rows"],
        rows=[
            {
                "file": "sparc/sample.geqdsk",
                "grid": "9x9",
                "machine": "sparc",
                "provenance": "real_public_design_reference",
                "raw_profile_solver_mode": "raw_geqdsk_profile_source_fixed_boundary",
                "operator_source_solver_mode": "operator_source_fixed_boundary_delta_star_psi",
                "adapted_profile_solver_mode": "adapted_geqdsk_profile_source_fixed_boundary",
                "reference_class": "public_efit_reference",
                "reference_role": "gate",
                "reference_dataset_id": "sparc_public_design_reference",
                "reference_expected_contract": "public_efit_profile_source_reconstruction_gate",
                "reference_expected_convention": "scaled_by_2pi_adapter_gate",
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
                "source_best_fit_convention": "unclassified_global_scale",
                "source_convention_adapter": "not_evaluated",
                "source_convention_adapter_residual_l2": 2.5,
                "source_convention_adapter_pass": False,
                "adapted_profile_psi_rmse_norm": float("nan"),
                "adapted_profile_sor_iterations": 0,
                "adapted_profile_sor_residual": float("nan"),
                "adapted_profile_axis_error_m": float("nan"),
                "adapted_profile_boundary_containment_fraction": float("nan"),
                "adapted_profile_boundary_psi_rmse_norm": float("nan"),
                "q_profile_sanity_pass": False,
                "q_profile_finite_fraction": 0.0,
                "q_profile_min_abs": float("nan"),
                "q_profile_sign_changes": 0,
                "q_profile_monotonic_fraction": 0.0,
                "adapted_profile_pass": False,
                "plasma_mask_fraction": 0.5,
                "pressure_source_norm": 0.7,
                "ffprime_source_norm": 0.3,
                "total_source_norm": 1.0,
                "pressure_source_sum": -0.7,
                "ffprime_source_sum": -0.3,
                "total_source_sum": -1.0,
                "pressure_source_fraction": 0.7,
                "ffprime_source_fraction": 0.3,
                "source_plasma_residual_l2": 2.0,
                "source_vacuum_residual_l2": 4.0,
                "source_plasma_operator_norm": 0.5,
                "source_vacuum_operator_norm": 0.5,
                "source_plasma_point_count": 4.0,
                "source_vacuum_point_count": 5.0,
                "adapted_source_plasma_residual_l2": float("nan"),
                "adapted_source_vacuum_residual_l2": float("nan"),
                "adapted_source_plasma_operator_norm": float("nan"),
                "adapted_source_vacuum_operator_norm": float("nan"),
                "source_domain_residual_class": "plasma_and_vacuum_source_mismatch",
                "source_domain_required_solver_mode": (
                    "profile_source_then_free_boundary_reconstruction_required"
                ),
                "source_domain_next_action": (
                    "repair_plasma_profile_source_then_reconstruct_vacuum_boundary_response"
                ),
                "best_source_candidate": "profile_source",
                "best_source_candidate_residual_l2": 2.5,
                "profile_source_candidate_rank": 1,
                "best_operator_candidate": "delta_star_psi",
                "best_operator_candidate_residual_l2": 2.5,
                "delta_star_psi_candidate_rank": 1,
                "declared_toroidal_current_A": -8.7e6,
                "operator_toroidal_current_A": -8.69e6,
                "operator_toroidal_current_full_domain_A": -8.69e6,
                "operator_toroidal_current_plasma_domain_A": -8.68e6,
                "profile_toroidal_current_A": -7.95e6,
                "adapted_profile_toroidal_current_A": float("nan"),
                "adapted_profile_current_ratio_to_declared": float("nan"),
                "adapted_profile_current_relative_error": float("nan"),
                "adapted_profile_current_closure_pass": False,
                "current_limited_adapted_profile_psi_rmse_norm": float("nan"),
                "current_limited_adapted_profile_scale": float("nan"),
                "current_limited_adapted_profile_current_ratio_to_declared": float("nan"),
                "current_limited_adapted_profile_current_relative_error": float("nan"),
                "current_limited_adapted_profile_pass": False,
                "effective_profile_current_closure_pass": False,
                "pressure_toroidal_current_A": -1.95e6,
                "ffprime_toroidal_current_A": -6.0e6,
                "operator_current_ratio_to_declared": 0.99885,
                "operator_current_full_domain_ratio_to_declared": 0.99885,
                "operator_current_plasma_domain_ratio_to_declared": 0.9977,
                "operator_current_best_domain": "full_domain",
                "operator_current_best_ratio_to_declared": 0.99885,
                "operator_current_best_relative_error": 0.00115,
                "operator_current_best_closure_pass": True,
                "profile_current_ratio_to_declared": 0.9138,
                "pressure_current_ratio_to_declared": 0.2241,
                "ffprime_current_ratio_to_declared": 0.6897,
                "operator_current_relative_error": 0.00115,
                "operator_current_full_domain_relative_error": 0.00115,
                "operator_current_plasma_domain_relative_error": 0.0023,
                "profile_current_relative_error": 0.0862,
                "operator_current_closure_pass": True,
                "operator_current_full_domain_closure_pass": True,
                "operator_current_plasma_domain_closure_pass": True,
                "profile_current_closure_pass": False,
                "profile_current_closure_failure_class": "profile_current_under_closes_declared_current",
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
    assert "Source-domain residual classes:" in stdout
    assert "profile_source_mismatch=1" in stdout
    assert "Worst source residual:" in stdout
    assert "Best-domain operator-current closure gate:" in stdout
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
