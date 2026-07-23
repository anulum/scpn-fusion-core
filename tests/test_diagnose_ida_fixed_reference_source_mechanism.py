# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Focused tests for the IDA fixed-reference source-mechanism decomposition."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import validation.diagnose_ida_fixed_reference_source_mechanism as diagnostic
import validation.ida_fixed_reference_source_mechanism_contract as contract


def _vector(relative_l2: float, digest: str) -> dict[str, Any]:
    return {
        "field_sha256": digest * 64,
        "linf": relative_l2,
        "relative_l2_to_reference_scale": relative_l2,
        "rms": relative_l2 / 2.0,
    }


def _report() -> dict[str, Any]:
    current_fields = {
        name: {
            "absolute_current_outside_reference_fraction": outside,
            "candidate_support_point_count": 100 + index,
            "current_density_sha256": digest * 64,
            "rectangular_current_a": -1_533_632.0,
            "reference_support_point_count": 100,
            "relative_ip_error": 1.0e-12,
            "relative_l2_to_reference": relative_l2,
            "total_variation_distance": tv,
        }
        for index, (name, digest, relative_l2, tv, outside) in enumerate(
            (
                ("freegs_hard_romberg", "1", 0.0, 0.0, 0.0),
                ("freegs_hard_rectangular_normalised", "2", 1.0e-7, 1.0e-7, 0.0),
                ("fusion_smooth_unscaled", "3", 8.0e-3, 4.0e-3, 3.0e-3),
                (
                    "fusion_smooth_rectangular_normalised",
                    "4",
                    9.0e-3,
                    4.0e-3,
                    3.0e-3,
                ),
            )
        )
    }
    vectors = {
        "hard_rectangular_normalisation": _vector(1.0e-7, "5"),
        "smooth_cutoff": _vector(8.0e-3, "6"),
        "smooth_ip_normalisation": _vector(4.0e-3, "7"),
    }
    environment = {
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
    }
    source_artifacts: dict[str, dict[str, Any]] = {
        name: {"path": path, "sha256": digest * 64}
        for (name, path), digest in zip(
            sorted(contract.SOURCE_PATHS.items()),
            ("8", "9", "a", "b", "c"),
            strict=True,
        )
    }
    source_artifacts.update(
        {
            "control_repository": {
                "git_commit": "d" * 40,
                "path": contract.CONTROL_REPOSITORY_PATH,
                "worktree_clean": True,
            },
            "freegs_public_example": {
                "path": "data/external/full_fidelity_public_sources/repos/freegs/16-DIIID.py",
                "sha256": "e" * 64,
            },
            "fusion_repository": {
                "git_commit": "f" * 40,
                "path": ".",
                "worktree_clean": True,
            },
        }
    )
    bindings: dict[str, dict[str, Any]] = {
        "operator_decomposition": {
            "path": contract.OPERATOR_DECOMPOSITION_PATH,
            "payload_sha256": "c" * 64,
            "source_ablation_payload_sha256": "b" * 64,
            "source_commit": "c" * 40,
            "source_same_case_payload_sha256": "a" * 64,
        },
        "same_case": {
            "case_id": contract.EVALUATION_CASE_ID,
            "grid_shape": contract.GRID_SHAPE,
            "path": contract.SAME_CASE_PATH,
            "payload_sha256": "a" * 64,
            "source_commit": "a" * 40,
        },
        "source_ablation": {
            "path": contract.SOURCE_ABLATION_PATH,
            "payload_sha256": "b" * 64,
            "source_commit": "b" * 40,
            "source_same_case_payload_sha256": "a" * 64,
        },
    }
    return contract.build_report(
        generated_at="2026-07-24T00:00:00Z",
        environment=environment,
        source_artifacts=source_artifacts,
        bindings=bindings,
        current_fields=current_fields,
        current_vectors=copy.deepcopy(vectors),
        interior_source_vectors=copy.deepcopy(vectors),
        wall_response_vectors=copy.deepcopy(vectors),
        control_parity={
            "absolute_current_outside_freegs_support_fraction": 0.004,
            "actual_current_density_sha256": "2" * 64,
            "formula_current_density_sha256": "3" * 64,
            "hard_mask_sha256": "3" * 64,
            "max_abs_a_per_m2": 1.0e-9,
            "rectangular_current_a": -1_533_632.0,
            "relative_l2": 1.0e-17,
            "relative_l2_to_freegs_reference": 0.009,
            "total_variation_to_freegs_reference": 0.004,
        },
        closure={
            "current_max_abs_a_per_m2": 0.0,
            "interior_source_max_abs": 0.0,
            "wall_response_max_abs_wb": 0.0,
        },
        cutoff_width=diagnostic.DEFAULT_CUTOFF_WIDTH,
    )


def test_current_metrics_vector_metrics_and_exact_closure() -> None:
    reference = np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    mask = reference != 0.0
    metrics = diagnostic._current_metrics(
        reference.copy(),
        reference=reference,
        reference_mask=mask,
        target_current_a=3.0,
        d_area=1.0,
    )
    assert metrics["relative_l2_to_reference"] == 0.0
    assert metrics["total_variation_distance"] == 0.0
    assert metrics["rectangular_current_a"] == 3.0

    a = np.full((2, 2), 1.0, dtype=np.float64)
    b = np.full((2, 2), -0.25, dtype=np.float64)
    c = np.full((2, 2), 0.5, dtype=np.float64)
    assert diagnostic._closure_max_abs(a + b + c, (a, b, c)) == 0.0


def test_current_metrics_fail_closed_on_shape_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        diagnostic._current_metrics(
            np.ones((2, 2), dtype=np.float64),
            reference=np.ones((3, 3), dtype=np.float64),
            reference_mask=np.ones((3, 3), dtype=np.bool_),
            target_current_a=1.0,
            d_area=1.0,
        )
    invalid = np.ones((2, 2), dtype=np.float64)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        diagnostic._current_metrics(
            invalid,
            reference=np.ones((2, 2), dtype=np.float64),
            reference_mask=np.ones((2, 2), dtype=np.bool_),
            target_current_a=1.0,
            d_area=1.0,
        )


def test_build_validate_render_write_and_validate_cli(tmp_path: Path) -> None:
    report = _report()
    diagnostic.validate_report(report)
    assert report["routing"]["current_dominant_component"] == "smooth_cutoff"
    assert (
        report["routing"]["next_ratcheting_target"]
        == "soft_axis_connected_support_topology_and_unclipped_lcfs_distance"
    )
    markdown = diagnostic.render_markdown(report)
    assert "CONTROL hard" not in markdown
    assert "freegs_hard_rectangular_normalised" in markdown
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    diagnostic.write_report(report, json_path=json_path, markdown_path=markdown_path)
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    assert markdown_path.read_text(encoding="utf-8") == markdown
    assert diagnostic.main(["--validate-report", str(json_path)]) == 0


def test_contract_rejects_tamper_overclaim_and_binding_drift() -> None:
    report = _report()
    tampered = copy.deepcopy(report)
    tampered["routing"]["next_ratcheting_target"] = "declare_victory"
    tampered["payload_sha256"] = contract._payload_sha256(tampered)
    with pytest.raises(ValueError, match="routing is inconsistent"):
        diagnostic.validate_report(tampered)

    overclaimed = copy.deepcopy(report)
    overclaimed["claim_boundary"]["control_admission"] = True
    overclaimed["payload_sha256"] = contract._payload_sha256(overclaimed)
    with pytest.raises(ValueError, match="claim_boundary"):
        diagnostic.validate_report(overclaimed)

    drifted = copy.deepcopy(report)
    drifted["bindings"]["operator_decomposition"]["source_ablation_payload_sha256"] = "9" * 64
    drifted["payload_sha256"] = contract._payload_sha256(drifted)
    with pytest.raises(ValueError, match="payload bindings disagree"):
        diagnostic.validate_report(drifted)


def test_main_requires_generated_at() -> None:
    with pytest.raises(SystemExit) as exc:
        diagnostic.main([])
    assert exc.value.code == 2


@pytest.mark.experimental
def test_real_diiid_fixed_reference_source_mechanism() -> None:
    pytest.importorskip("freegs")
    report = diagnostic.run_diagnostic(generated_at="2026-07-24T00:00:00Z")
    diagnostic.validate_report(report)
    assert report["routing"]["current_dominant_component"] == "smooth_cutoff"
    assert report["control_parity"]["max_abs_a_per_m2"] <= 1.0e-8
    assert max(report["closure"].values()) <= contract.CLOSURE_MAX_ABS
