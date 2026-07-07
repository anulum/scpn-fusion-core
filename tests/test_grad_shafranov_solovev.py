# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Grad-Shafranov Solov'ev Validation Tests
"""Tests for the Solov'ev analytic-equilibrium validation lane (F-1)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "validate_grad_shafranov_solovev.py"
REPORT = ROOT / "validation" / "reports" / "grad_shafranov_solovev.json"

FAST_RESOLUTIONS = (17, 25)


@pytest.fixture(scope="module")
def solovev() -> Any:
    spec = importlib.util.spec_from_file_location("validate_grad_shafranov_solovev", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fast_result(solovev: Any) -> Any:
    return solovev.validate_grad_shafranov(
        resolutions=FAST_RESOLUTIONS,
        operator_error_gate=5e-3,
        reconstruction_nrmse_gate=1e-3,
    )


class TestAnalyticFields:
    """Exact field and source satisfy the Solov'ev identity."""

    def test_psi_matches_polynomial(self, solovev: Any) -> None:
        geometry = solovev.SolovevGeometry.from_aspect()
        rr = np.array([[1.0, 2.0]])
        zz = np.array([[0.5, -0.5]])
        psi = solovev.solovev_psi(rr, zz, geometry)
        expected = geometry.c1 * rr**4 / 8.0 + geometry.c2 * zz**2
        np.testing.assert_allclose(psi, expected)

    def test_source_is_analytic_gs_operator_of_psi(self, solovev: Any) -> None:
        """Central-difference Δ* of the exact ψ converges to the analytic source."""
        geometry = solovev.SolovevGeometry.from_aspect()
        n = 201
        r = np.linspace(geometry.r_min, geometry.r_max, n)
        z = np.linspace(geometry.z_min, geometry.z_max, n)
        rr, zz = np.meshgrid(r, z)
        psi = solovev.solovev_psi(rr, zz, geometry)
        dr = r[1] - r[0]
        dz = z[1] - z[0]
        d2r = (psi[1:-1, 2:] - 2 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / dr**2
        d1r = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * dr)
        d2z = (psi[2:, 1:-1] - 2 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dz**2
        delta_star = d2r - d1r / rr[1:-1, 1:-1] + d2z
        source = solovev.solovev_source(rr, geometry)[1:-1, 1:-1]
        assert float(np.max(np.abs(delta_star - source))) < 5e-3

    def test_geometry_rejects_non_positive_r_min(self, solovev: Any) -> None:
        with pytest.raises(ValueError, match="r_min must be positive"):
            solovev.SolovevGeometry(
                r0=1.0, a=0.5, r_min=-0.1, r_max=1.5, z_min=-0.5, z_max=0.5, c1=1.0, c2=0.5
            )

    def test_geometry_rejects_inverted_box(self, solovev: Any) -> None:
        with pytest.raises(ValueError, match="r_max must exceed r_min"):
            solovev.SolovevGeometry(
                r0=1.0, a=0.5, r_min=1.5, r_max=1.0, z_min=-0.5, z_max=0.5, c1=1.0, c2=0.5
            )


class TestProductionPathGates:
    """Operator, SOR, and dispatched multigrid reconstruct the exact field."""

    def test_operator_truncation_is_second_order(self, fast_result: Any) -> None:
        assert fast_result.operator_order >= 1.8
        assert fast_result.operator_passed

    def test_sor_reconstruction_is_second_order(self, fast_result: Any) -> None:
        assert fast_result.reconstruction_order >= 1.8
        assert fast_result.reconstruction_passed
        assert all(detail.converged for detail in fast_result.reconstruction_details)

    def test_multigrid_numpy_floor_reconstructs(self, fast_result: Any) -> None:
        record = fast_result.multigrid_numpy_record
        assert record.tier == "numpy"
        assert record.converged
        assert record.meets_analytic_tolerance

    def test_multigrid_fastest_tier_reconstructs(self, fast_result: Any) -> None:
        record = fast_result.multigrid_fastest_record
        assert record.converged
        assert record.meets_analytic_tolerance

    def test_overall_pass(self, fast_result: Any) -> None:
        assert fast_result.passed

    def test_multigrid_tiers_agree_with_each_other(self, fast_result: Any) -> None:
        """NumPy floor and fastest tier converge to the same discrete solution."""
        numpy_rec = fast_result.multigrid_numpy_record
        fastest_rec = fast_result.multigrid_fastest_record
        assert abs(numpy_rec.nrmse - fastest_rec.nrmse) < 1e-6

    def test_requires_two_distinct_resolutions(self, solovev: Any) -> None:
        with pytest.raises(ValueError, match="at least two distinct resolutions"):
            solovev.validate_grad_shafranov(resolutions=(17, 17))

    def test_dispatched_multigrid_rejects_unknown_tier(self, solovev: Any) -> None:
        geometry = solovev.SolovevGeometry.from_aspect()
        with pytest.raises(ValueError, match="tier must be"):
            solovev.dispatched_multigrid_reconstruction(
                geometry, 17, tier="cuda", analytic_tolerance=1e-3
            )


class TestEvidencePayload:
    """Evidence payloads are sealed and tamper-evident."""

    def test_evidence_round_trip(self, solovev: Any, fast_result: Any) -> None:
        evidence = solovev.build_evidence(fast_result, target_id="unit-test")
        assert solovev.validate_evidence_payload(evidence) is True

    def test_tampered_payload_rejected(self, solovev: Any, fast_result: Any) -> None:
        evidence = solovev.build_evidence(fast_result, target_id="unit-test")
        evidence["operator_order"] = 99.0
        with pytest.raises(ValueError, match="does not match payload"):
            solovev.validate_evidence_payload(evidence)

    def test_foreign_schema_rejected(self, solovev: Any, fast_result: Any) -> None:
        evidence = solovev.build_evidence(fast_result, target_id="unit-test")
        evidence["schema_version"] = "other.schema"
        with pytest.raises(ValueError, match="unsupported"):
            solovev.validate_evidence_payload(evidence)

    def test_empty_target_id_rejected(self, solovev: Any, fast_result: Any) -> None:
        with pytest.raises(ValueError, match="target_id must be non-empty"):
            solovev.build_evidence(fast_result, target_id="  ")

    def test_committed_report_is_sealed_and_passing(self, solovev: Any) -> None:
        payload = json.loads(REPORT.read_text(encoding="utf-8"))
        assert solovev.validate_evidence_payload(payload) is True
        assert payload["resolutions"] == [33, 49, 65, 97]

    def test_committed_report_multigrid_records_present(self, solovev: Any) -> None:
        payload = json.loads(REPORT.read_text(encoding="utf-8"))
        assert payload["multigrid_numpy_record"]["tier"] == "numpy"
        assert payload["multigrid_numpy_record"]["meets_analytic_tolerance"] is True
        assert payload["multigrid_fastest_record"]["meets_analytic_tolerance"] is True


class TestCli:
    """CLI writes reports and exits by gate outcome."""

    FAST_GATES = (
        "--operator-error-gate",
        "5e-3",
        "--reconstruction-nrmse-gate",
        "1e-3",
    )

    def test_cli_writes_report_and_passes(self, solovev: Any, tmp_path: Path) -> None:
        report = tmp_path / "solovev.json"
        rc = solovev.main(
            [
                "--resolutions",
                "17",
                "25",
                *self.FAST_GATES,
                "--report",
                str(report),
                "--target-id",
                "cli-test",
            ]
        )
        assert rc == 0
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert solovev.validate_evidence_payload(payload) is True
        assert report.with_suffix(".md").exists()

    def test_cli_json_out(self, solovev: Any, capsys: pytest.CaptureFixture[str]) -> None:
        rc = solovev.main(["--resolutions", "17", "25", *self.FAST_GATES, "--json-out"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["schema_version"] == solovev.GRAD_SHAFRANOV_SOLOVEV_SCHEMA_VERSION

    def test_cli_default_gates_fail_on_coarse_grids(self, solovev: Any) -> None:
        """Default gates are tuned to the 97² default grids and stay strict."""
        rc = solovev.main(["--resolutions", "17", "25", "--json-out"])
        assert rc == 1
