# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Fail-closed tests for the IDA operator-response evidence contract."""

from __future__ import annotations

import copy
import hashlib
from typing import Any

import pytest

import validation.ida_operator_response_contract as contract


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _forcing(label: str, magnitude: float) -> dict[str, Any]:
    return {
        "field_sha256": _digest(label),
        "l2": magnitude,
        "linf": magnitude / 2.0,
        "relative_l2_to_exact_source_residual": magnitude / 10.0,
    }


def _response(label: str, magnitude: float) -> dict[str, Any]:
    return {
        "cosine_to_terminal_error": 0.75,
        "field_sha256": _digest(label),
        "l2_wb": magnitude,
        "linf_wb": magnitude / 2.0,
        "projection_on_terminal_error": magnitude / 5.0,
        "relative_l2_to_terminal_error": magnitude / 10.0,
    }


def _report() -> dict[str, Any]:
    same = _digest("same")
    operator = _digest("operator")
    mechanism = _digest("mechanism")
    fixed_point = _digest("fixed-point")
    return contract.build_report(
        generated_at="2026-07-24T16:00:00Z",
        environment={
            "affinity_cpu_count": 4,
            "backend": "cpu",
            "devices": ["TFRT_CPU_0"],
            "freegs_version": "0.8.2",
            "host_load_1m_5m_15m": [0.1, 0.2, 0.3],
            "isolated_host": False,
            "jax_version": "0.7.1",
            "jaxlib_version": "0.7.1",
            "machine": "x86_64",
            "platform": "test",
            "python_version": "3.12.3",
            "x64_enabled": True,
        },
        source_artifacts={
            **{
                name: {"path": path, "sha256": _digest(name)}
                for name, path in contract.SOURCE_PATHS.items()
            },
            "freegs_public_example": {
                "path": "data/external/full_fidelity_public_sources/repos/freegs/16-DIIID.py",
                "sha256": _digest("freegs"),
            },
            "repository": {
                "git_commit": "a" * 40,
                "path": ".",
                "worktree_clean": True,
            },
        },
        bindings={
            "same_case": {
                "candidate_psi_sha256": _digest("candidate"),
                "path": contract.SAME_CASE_PATH,
                "payload_sha256": same,
                "reference_psi_sha256": _digest("reference"),
                "source_commit": "b" * 40,
            },
            "operator_decomposition": {
                "path": contract.OPERATOR_DECOMPOSITION_PATH,
                "payload_sha256": operator,
                "same_case_payload_sha256": same,
                "source_commit": "c" * 40,
            },
            "source_mechanism": {
                "operator_payload_sha256": operator,
                "path": contract.SOURCE_MECHANISM_PATH,
                "payload_sha256": mechanism,
                "same_case_payload_sha256": same,
                "source_commit": "d" * 40,
            },
            "fixed_point": {
                "path": contract.FIXED_POINT_PATH,
                "payload_sha256": fixed_point,
                "same_case_payload_sha256": same,
                "source_commit": "e" * 40,
                "source_mechanism_payload_sha256": mechanism,
            },
        },
        forcing_decomposition={
            "components": {
                name: _forcing(name, float(index + 1))
                for index, name in enumerate(contract.COMPONENTS)
            },
            "exact_source_residual": _forcing("exact-total", 10.0),
            "native_operator_residual": _forcing("native-total", 6.0),
        },
        response_decomposition={
            "components": {
                name: _response(name, float(index + 1))
                for index, name in enumerate(contract.COMPONENTS)
            },
            "exact_source_total": _response("exact-total", 10.0),
            "native_operator_total": _response("native-total", 6.0),
        },
        closure={
            "exact_source_forcing_max_abs": 1.0e-15,
            "exact_source_response_max_abs_wb": 1.0e-15,
            "fixed_point_native_operator_max_abs_wb": 1.0e-15,
            "native_operator_forcing_max_abs": 1.0e-15,
            "native_operator_response_max_abs_wb": 1.0e-15,
        },
    )


def _resign(report: dict[str, Any]) -> None:
    report["payload_sha256"] = contract._payload_sha256(report)


def test_build_validate_render_and_route_without_promoting_claims() -> None:
    report = _report()
    contract.validate_report(report)
    assert report["routing"] == {
        "dominant_response_component": "exact_source_convention",
        "next_ratcheting_target": "current_support_and_source_convention",
        "solver_physics_changed": False,
    }
    assert set(report["claim_boundary"].values()) == {False}
    markdown = contract.render_markdown(report)
    assert "Solver physics changed: `false`" in markdown
    assert "held out validation: `false`" in markdown


def test_validate_rejects_tamper_overclaim_and_routing_forgery() -> None:
    report = _report()
    tamper = copy.deepcopy(report)
    tamper["status"] = "admitted"
    with pytest.raises(ValueError, match="status"):
        contract.validate_report(tamper)

    overclaim = copy.deepcopy(report)
    overclaim["claim_boundary"]["scientific_validation"] = True
    _resign(overclaim)
    with pytest.raises(ValueError, match="claim_boundary"):
        contract.validate_report(overclaim)

    routing = copy.deepcopy(report)
    routing["routing"]["next_ratcheting_target"] = "change_solver_physics"
    _resign(routing)
    with pytest.raises(ValueError, match="routing is inconsistent"):
        contract.validate_report(routing)


def test_validate_rejects_broken_chain_dirty_source_and_open_closure() -> None:
    report = _report()
    chain = copy.deepcopy(report)
    chain["bindings"]["fixed_point"]["source_mechanism_payload_sha256"] = _digest("other")
    _resign(chain)
    with pytest.raises(ValueError, match="fixed point does not bind"):
        contract.validate_report(chain)

    dirty = copy.deepcopy(report)
    dirty["source_artifacts"]["repository"]["worktree_clean"] = False
    _resign(dirty)
    with pytest.raises(ValueError, match="clean canonical repository"):
        contract.validate_report(dirty)

    closure = copy.deepcopy(report)
    closure["closure"]["native_operator_response_max_abs_wb"] = 1.0e-3
    _resign(closure)
    with pytest.raises(ValueError, match="exceeds the frozen threshold"):
        contract.validate_report(closure)


def test_validate_rejects_metric_shape_and_signature_drift() -> None:
    report = _report()
    metric = copy.deepcopy(report)
    del metric["response_decomposition"]["components"]["native_second_order_stencil"]["linf_wb"]
    _resign(metric)
    with pytest.raises(ValueError, match="fields are invalid"):
        contract.validate_report(metric)

    signature = copy.deepcopy(report)
    signature["generated_at"] = "2026-07-24T16:00:01Z"
    with pytest.raises(ValueError, match="payload_sha256"):
        contract.validate_report(signature)
