# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Fail-closed tests for the IDA geometry-feedback trace contract."""

from __future__ import annotations

import copy
import hashlib
from typing import Any

import pytest

import validation.ida_geometry_feedback_trace_contract as contract


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _current(label: str, tv: float) -> dict[str, Any]:
    return {
        "centroid_delta_m": {"r": -0.01, "z": 0.02},
        "cosine_similarity": 1.0 - tv,
        "current_density_sha256": _digest(label),
        "total_variation_distance": tv,
    }


def _checkpoint(
    *,
    run_name: str,
    index: int,
    tv: float,
    before_digest: str,
    after_digest: str,
    terminal: bool,
) -> dict[str, Any]:
    return {
        "converged": terminal,
        "fixed_point": {
            "accepted_update_relative_l2": 1.0e-4,
            "residual_linf_wb": 1.0e-5,
            "residual_relative_l2": 1.0e-4,
        },
        "geometry": {
            "active_boundary_wb": 0.2,
            "active_span_wb": 1.2,
            "axis_wb": -1.0,
            "physical_boundary_wb": 0.25,
            "physical_span_wb": 1.25,
        },
        "ip_fraction": 1.0,
        "ip_now_a": 1.0e6,
        "iteration_index": index,
        "phase": contract.phase_for_iteration(run_name, index),
        "physical_residual": {
            "interior_relative_rms": 1.0e-3,
            "wall_relative_l2": 2.0e-3,
        },
        "production_current": _current(f"{run_name}-{index}-production", tv),
        "psi_after_sha256": after_digest,
        "psi_before_sha256": before_digest,
        "reference_boundary_counterfactual": _current(
            f"{run_name}-{index}-counterfactual",
            max(tv - 0.02, 0.0),
        ),
        "separatrix_refinement": 1.0,
        "terminal": terminal,
    }


def _run(
    *,
    run_name: str,
    count: int,
    cap: int,
    indices: tuple[int, ...],
    first_before: str,
    terminal_after: str,
    tv_offset: float,
) -> dict[str, Any]:
    reached = sorted({index for index in indices if index < count} | {count - 1})
    rows = []
    for position, index in enumerate(reached):
        rows.append(
            _checkpoint(
                run_name=run_name,
                index=index,
                tv=tv_offset + 0.002 * position,
                before_digest=first_before if position == 0 else _digest(f"{run_name}-{index}-before"),
                after_digest=(
                    terminal_after if index == count - 1 else _digest(f"{run_name}-{index}-after")
                ),
                terminal=index == count - 1,
            )
        )
    return {
        "checkpoint_indices_requested": list(indices),
        "checkpoints": rows,
        "iteration_cap": cap,
        "iteration_count": count,
        "terminal_psi_sha256": terminal_after,
        "terminated_early": count < cap,
    }


def _report() -> dict[str, Any]:
    cold_terminal = _digest("cold-terminal")
    warm_terminal = _digest("warm-terminal")
    runs = {
        "cold": _run(
            run_name="cold",
            count=180,
            cap=180,
            indices=contract.COLD_CHECKPOINT_INDICES,
            first_before=_digest("cold-start"),
            terminal_after=cold_terminal,
            tv_offset=0.01,
        ),
        "warm": _run(
            run_name="warm",
            count=7,
            cap=20,
            indices=contract.WARM_CHECKPOINT_INDICES,
            first_before=cold_terminal,
            terminal_after=warm_terminal,
            tv_offset=0.2,
        ),
    }
    return contract.build_report(
        generated_at="2026-07-23T12:00:00Z",
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
                name: {"path": path, "sha256": _digest(name)}
                for name, path in contract.SOURCE_PATHS.items()
            },
            "freegs_public_example": {
                "path": contract.FREEGS_PUBLIC_EXAMPLE_PATH,
                "sha256": _digest("freegs"),
            },
        },
        source_reports={
            "same_case": {
                "path": contract.SAME_CASE_REPORT_PATH,
                "payload_sha256": _digest("same-case-report"),
                "source_commit": "a" * 40,
            },
            "source_ablation": {
                "path": contract.SOURCE_ABLATION_REPORT_PATH,
                "payload_sha256": _digest("source-ablation-report"),
                "source_commit": "b" * 40,
            },
        },
        runs=runs,
        terminal_parity={
            "expected_same_case_candidate_sha256": warm_terminal,
            "matches_same_case_candidate": True,
            "traced_candidate_sha256": warm_terminal,
        },
        source_commit="c" * 40,
        source_worktree_clean=True,
    )


def _resign(report: dict[str, Any]) -> None:
    report["payload_sha256"] = contract._payload_sha256(report)


def test_build_validate_render_and_route() -> None:
    """A coherent trace must validate and derive routing only from measurements."""
    report = _report()
    contract.validate_report(report)
    assert report["routing"]["trace_matches_same_case_candidate"] is True
    assert report["routing"]["next_ratcheting_target"].endswith(
        "_geometry_source_feedback"
    )
    markdown = contract.render_markdown(report)
    assert "Same-case terminal parity: `true`" in markdown
    assert "scientific admission: `false`" in markdown


def test_validate_report_rejects_tamper_overclaim_and_route_forgery() -> None:
    """Digest, claim, and derived routing mutations must all fail closed."""
    report = _report()
    payload_tamper = copy.deepcopy(report)
    payload_tamper["status"] = "promoted"
    with pytest.raises(ValueError, match="payload_sha256"):
        contract.validate_report(payload_tamper)

    overclaim = copy.deepcopy(report)
    overclaim["claim_boundary"]["scientific_validation"] = True
    _resign(overclaim)
    with pytest.raises(ValueError, match="claim_boundary"):
        contract.validate_report(overclaim)

    forged_route = copy.deepcopy(report)
    forged_route["routing"]["next_ratcheting_target"] = "manual_claim"
    _resign(forged_route)
    with pytest.raises(ValueError, match="routing is inconsistent"):
        contract.validate_report(forged_route)


def test_validate_report_rejects_phase_coverage_and_source_drift() -> None:
    """Checkpoint phase/coverage and bound source paths cannot drift."""
    report = _report()
    bad_phase = copy.deepcopy(report)
    bad_phase["runs"]["cold"]["checkpoints"][0]["phase"] = "warm_polish"
    _resign(bad_phase)
    with pytest.raises(ValueError, match="phase is inconsistent"):
        contract.validate_report(bad_phase)

    missing_checkpoint = copy.deepcopy(report)
    missing_checkpoint["runs"]["cold"]["checkpoints"].pop(1)
    _resign(missing_checkpoint)
    with pytest.raises(ValueError, match="checkpoint coverage"):
        contract.validate_report(missing_checkpoint)

    bad_path = copy.deepcopy(report)
    bad_path["source_artifacts"]["trace"]["path"] = "validation/other.py"
    _resign(bad_path)
    with pytest.raises(ValueError, match="path is invalid"):
        contract.validate_report(bad_path)


def test_validate_report_rejects_terminal_continuity_and_parity_forgery() -> None:
    """Terminal identities must connect cold, warm, and same-case evidence."""
    report = _report()
    broken_continuity = copy.deepcopy(report)
    broken_continuity["runs"]["warm"]["checkpoints"][0]["psi_before_sha256"] = _digest("other")
    _resign(broken_continuity)
    with pytest.raises(ValueError, match="cold-to-warm"):
        contract.validate_report(broken_continuity)

    bad_flag = copy.deepcopy(report)
    bad_flag["terminal_parity"]["matches_same_case_candidate"] = False
    _resign(bad_flag)
    with pytest.raises(ValueError, match="match flag"):
        contract.validate_report(bad_flag)

    unbound_terminal = copy.deepcopy(report)
    digest = _digest("different-traced-terminal")
    unbound_terminal["terminal_parity"]["expected_same_case_candidate_sha256"] = digest
    unbound_terminal["terminal_parity"]["traced_candidate_sha256"] = digest
    _resign(unbound_terminal)
    with pytest.raises(ValueError, match="traced digest"):
        contract.validate_report(unbound_terminal)
