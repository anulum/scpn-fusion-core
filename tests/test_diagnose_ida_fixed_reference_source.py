# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the IDA fixed-reference current-source ablation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import validation.diagnose_ida_fixed_reference_source as diagnostic


def _distribution(tv: float) -> dict[str, Any]:
    return {
        "candidate_centroid_m": {"r": 1.5, "z": 0.1},
        "centroid_delta_m": {"r": -0.01, "z": 0.02},
        "cosine_similarity": 0.95,
        "l1_distance": 2.0 * tv,
        "reference_centroid_m": {"r": 1.51, "z": 0.08},
        "total_variation_distance": tv,
    }


def _support() -> dict[str, Any]:
    return {
        "absolute_current_inside_reference_fraction": 0.99,
        "absolute_current_outside_reference_fraction": 0.01,
        "candidate_mask_sha256": "1" * 64,
        "candidate_point_count": 10,
        "false_negative_fraction_of_reference": 0.0,
        "false_negative_point_count": 0,
        "false_positive_fraction_of_candidate": 0.0,
        "false_positive_point_count": 0,
        "intersection_point_count": 10,
        "iou": 1.0,
        "reference_mask_sha256": "2" * 64,
        "reference_point_count": 10,
        "relative_floor": 1.0e-6,
        "union_point_count": 10,
    }


def _report() -> dict[str, Any]:
    profile_fit = {
        name: {
            "exact_sha256": digest * 64,
            "reconstructed_sha256": digest * 64,
            "relative_l2_error": 1.0e-15,
            "relative_max_error": 2.0e-15,
            "sample_count": 129,
        }
        for name, digest in (("ffprime", "3"), ("pprime", "4"))
    }
    fixed = {
        name: {
            "current_density_sha256": digest * 64,
            "distribution": _distribution(tv),
            "support": _support(),
        }
        for name, digest, tv in (
            ("compact_bspline", "5", 0.004),
            ("exact_sampled", "6", 0.004),
        )
    }
    self_consistent = {
        "candidate_psi_sha256": "7" * 64,
        "distribution": _distribution(0.25),
        "source_report_payload_sha256": "8" * 64,
        "support": _support(),
    }
    return diagnostic.build_report(
        generated_at="2026-07-23T10:00:00Z",
        environment={
            "affinity_cpu_count": 4,
            "backend": "cpu",
            "devices": ["TFRT_CPU_0"],
            "freegs_version": "0.8.2",
            "host_load_1m_5m_15m": [0.1, 0.2, 0.3],
            "isolated_host": False,
            "jax_version": "0.7.2",
            "jaxlib_version": "0.7.2",
            "machine": "x86_64",
            "platform": "test",
            "python_version": "3.12.0",
            "x64_enabled": True,
        },
        source_artifacts={
            **{
                name: {"path": path, "sha256": digest * 64}
                for (name, path), digest in zip(
                    sorted(diagnostic._SOURCE_PATHS.items()),
                    ("9", "a", "b", "c", "d"),
                    strict=True,
                )
            },
            "freegs_public_example": {
                "path": "data/external/full_fidelity_public_sources/repos/freegs/16-DIIID.py",
                "sha256": "e" * 64,
            },
        },
        source_same_case={
            "case_id": diagnostic.EVALUATION_CASE_ID,
            "grid_shape": [129, 129],
            "path": "validation/reports/ida_same_case_evidence.json",
            "payload_sha256": "8" * 64,
            "source_commit": "d" * 40,
        },
        profile_fit=profile_fit,
        fixed_reference_sources=fixed,
        self_consistent_candidate=self_consistent,
        source_commit="f" * 40,
        source_worktree_clean=True,
    )


def test_profile_fit_metrics_and_current_comparison() -> None:
    """Exact profiles and identical current fields must produce zero error."""
    exact = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    metrics = diagnostic._profile_fit_metrics(exact, exact.copy())
    assert metrics["relative_l2_error"] == 0.0
    assert metrics["sample_count"] == 5

    r_grid = np.linspace(1.0, 1.2, 3, dtype=np.float64)
    z_grid = np.linspace(-0.1, 0.1, 3, dtype=np.float64)
    psi = np.arange(9, dtype=np.float64).reshape(3, 3)
    current = np.ones((3, 3), dtype=np.float64)
    comparison = diagnostic._current_comparison(
        candidate_current_density=current,
        reference_current_density=current,
        reference_current_mask=np.ones((3, 3), dtype=np.bool_),
        reference_psi=psi,
        reference_axis=0.0,
        reference_boundary=8.0,
        r_grid=r_grid,
        z_grid=z_grid,
    )
    assert comparison["distribution"]["total_variation_distance"] == pytest.approx(0.0)
    assert comparison["support"]["iou"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("exact", "reconstructed", "message"),
    [
        (np.ones(2), np.ones(3), "matching non-trivial"),
        (np.array([1.0, np.nan]), np.ones(2), "must be finite"),
    ],
)
def test_profile_fit_metrics_rejects_invalid_arrays(
    exact: np.ndarray[Any, Any],
    reconstructed: np.ndarray[Any, Any],
    message: str,
) -> None:
    """Profile diagnostics must reject mismatched and non-finite vectors."""
    with pytest.raises(ValueError, match=message):
        diagnostic._profile_fit_metrics(
            exact.astype(np.float64),
            reconstructed.astype(np.float64),
        )


def test_build_validate_render_and_write_report(tmp_path: Path) -> None:
    """A consistent report must validate, render, persist, and reload."""
    report = _report()
    diagnostic.validate_report(report)
    assert report["routing"]["profile_representation_excluded"] is True
    assert (
        report["routing"]["next_ratcheting_target"]
        == "self_consistent_equilibrium_geometry_and_flux_normalisation"
    )
    markdown = diagnostic.render_markdown(report)
    assert "Self-consistent candidate" in markdown
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    diagnostic.write_report(report, json_path=json_path, markdown_path=markdown_path)
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    assert markdown_path.read_text(encoding="utf-8") == markdown
    assert diagnostic.main(["--validate-report", str(json_path)]) == 0


def test_validate_report_rejects_tamper_and_overclaim() -> None:
    """Digest-valid routing tamper and claim promotion must still fail closed."""
    report = _report()
    tampered = copy.deepcopy(report)
    tampered["routing"]["compact_exact_tv_delta"] = 0.2
    tampered["payload_sha256"] = diagnostic._payload_sha256(tampered)
    with pytest.raises(ValueError, match="routing is inconsistent"):
        diagnostic.validate_report(tampered)

    overclaimed = copy.deepcopy(report)
    overclaimed["claim_boundary"]["scientific_validation"] = True
    overclaimed["payload_sha256"] = diagnostic._payload_sha256(overclaimed)
    with pytest.raises(ValueError, match="claim_boundary"):
        diagnostic.validate_report(overclaimed)


def test_build_report_records_dirty_source_and_unresolved_route() -> None:
    """Dirty source and a material compact-profile delta must remain blocked."""
    report = _report()
    fixed = copy.deepcopy(report["fixed_reference_sources"])
    fixed["compact_bspline"]["distribution"] = _distribution(0.08)
    rebuilt = diagnostic.build_report(
        generated_at=report["generated_at"],
        environment=report["environment"],
        source_artifacts={
            name: value
            for name, value in report["source_artifacts"].items()
            if name != "repository"
        },
        source_same_case=report["source_same_case"],
        profile_fit=report["profile_fit"],
        fixed_reference_sources=fixed,
        self_consistent_candidate=report["self_consistent_candidate"],
        source_commit="f" * 40,
        source_worktree_clean=False,
    )
    assert "source_worktree_not_clean" in rebuilt["blockers"]
    assert (
        rebuilt["routing"]["next_ratcheting_target"]
        == "source_convention_or_profile_representation_unresolved"
    )


def test_evaluation_case_requires_exact_single_candidate() -> None:
    """The source report must bind exactly one frozen DIII-D candidate."""
    candidate = {
        "case_id": diagnostic.EVALUATION_CASE_ID,
        "role": "evaluation_candidate",
    }
    assert diagnostic._evaluation_case({"cases": [candidate]}) == candidate
    with pytest.raises(ValueError, match="exactly one"):
        diagnostic._evaluation_case({"cases": []})
    with pytest.raises(ValueError, match="must be a list"):
        diagnostic._evaluation_case({"cases": {}})


@pytest.mark.parametrize("value", [True, "1.0", float("nan"), -1.0])
def test_require_number_rejects_invalid_values(value: object) -> None:
    """Metric validation must reject non-numeric, non-finite, and negative values."""
    with pytest.raises(ValueError, match="must be"):
        diagnostic._require_number(value, field="metric", minimum=0.0)


def test_main_requires_generated_at() -> None:
    """Execution mode must require an explicit evidence timestamp."""
    with pytest.raises(SystemExit) as exc:
        diagnostic.main([])
    assert exc.value.code == 2


@pytest.mark.experimental
def test_real_diiid_fixed_reference_ablation() -> None:
    """The public FreeGS DIII-D surface must reproduce the routing mechanism."""
    pytest.importorskip("freegs")
    report = diagnostic.run_ablation(generated_at="2026-07-23T10:00:00Z")
    diagnostic.validate_report(report)
    route = report["routing"]
    assert route["profile_representation_excluded"] is True
    assert route["fixed_reference_source_matches"] is True
    assert route["geometry_separation"] is True
    assert (
        route["next_ratcheting_target"]
        == "self_consistent_equilibrium_geometry_and_flux_normalisation"
    )
