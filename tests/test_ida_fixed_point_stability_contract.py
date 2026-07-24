# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Fail-closed tests for the IDA fixed-point stability contract."""

from __future__ import annotations

import copy
import hashlib
from typing import Any

import pytest

import validation.ida_fixed_point_stability_contract as contract


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _vector(label: str, magnitude: float, projection: float) -> dict[str, Any]:
    return {
        "cosine_to_terminal_error": 0.5,
        "field_sha256": _digest(label),
        "l2_wb": magnitude,
        "linf_wb": magnitude / 2.0,
        "projection_on_terminal_error": projection,
        "relative_l2_to_terminal_error": magnitude / 10.0,
    }


def _report() -> dict[str, Any]:
    map_digest = _digest("map")
    return contract.build_report(
        generated_at="2026-07-24T04:00:00Z",
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
                "payload_sha256": _digest("same-case"),
                "reference_psi_sha256": _digest("reference"),
                "source_commit": "b" * 40,
            },
            "source_mechanism": {
                "path": contract.SOURCE_MECHANISM_PATH,
                "payload_sha256": _digest("mechanism"),
                "same_case_payload_sha256": _digest("same-case"),
                "source_commit": "c" * 40,
            },
        },
        decomposition={
            "closure_max_abs_wb": 1.0e-15,
            "components": {
                "native_operator_residual": _vector("operator", 1.0, 0.1),
                "boundary_anchor": _vector("boundary", 2.0, 0.2),
                "source_mechanism": _vector("source", 3.0, 0.3),
            },
            "total_forcing": _vector("total", 6.0, 0.6),
        },
        map_parity={
            "manual_map_sha256": map_digest,
            "max_abs_wb": 0.0,
            "production_map_sha256": map_digest,
            "relative_l2": 0.0,
            "sha256_match": True,
        },
        jvp_gains={
            "terminal_error": {
                "alignment_with_input": 0.8,
                "gain_l2": 0.9,
                "input_sha256": _digest("terminal-error"),
                "output_sha256": _digest("terminal-jvp"),
            },
            "source_mechanism": {
                "alignment_with_input": 0.7,
                "gain_l2": 0.6,
                "input_sha256": _digest("source"),
                "output_sha256": _digest("source-jvp"),
            },
        },
        trajectory=[
            {
                "distance_to_candidate_relative_to_terminal": 1.0 - 0.1 * index,
                "distance_to_reference_relative_to_terminal": 0.1 * index,
                "projection_on_terminal_error": 0.1 * index,
                "psi_sha256": _digest(f"trajectory-{index}"),
                "step": index,
            }
            for index in range(contract.TRAJECTORY_STEPS + 1)
        ],
        source_worktree_clean=True,
    )


def _resign(report: dict[str, Any]) -> None:
    report["payload_sha256"] = contract._payload_sha256(report)


def test_build_validate_render_and_route() -> None:
    report = _report()
    contract.validate_report(report)
    assert report["routing"] == {
        "dominant_forcing_component": "source_mechanism",
        "locally_amplifying_along_terminal_error": False,
        "next_ratcheting_target": "source_mechanism_reference_stationarity",
        "raw_picard_moves_toward_candidate": True,
        "stationary_map_parity_ok": True,
    }
    markdown = contract.render_markdown(report)
    assert "Terminal-error JVP gain: `0.9`" in markdown
    assert "scientific admission: `false`" in markdown


def test_validate_rejects_tamper_overclaim_and_routing_forgery() -> None:
    report = _report()
    tamper = copy.deepcopy(report)
    tamper["status"] = "admitted"
    with pytest.raises(ValueError, match="payload_sha256"):
        contract.validate_report(tamper)

    overclaim = copy.deepcopy(report)
    overclaim["claim_boundary"]["scientific_validation"] = True
    _resign(overclaim)
    with pytest.raises(ValueError, match="claim_boundary"):
        contract.validate_report(overclaim)

    forged = copy.deepcopy(report)
    forged["routing"]["next_ratcheting_target"] = "manual_claim"
    _resign(forged)
    with pytest.raises(ValueError, match="routing is inconsistent"):
        contract.validate_report(forged)


def test_validate_rejects_binding_closure_and_trajectory_drift() -> None:
    report = _report()
    binding = copy.deepcopy(report)
    binding["bindings"]["source_mechanism"]["same_case_payload_sha256"] = _digest("other")
    _resign(binding)
    with pytest.raises(ValueError, match="does not bind"):
        contract.validate_report(binding)

    closure = copy.deepcopy(report)
    closure["decomposition"]["closure_max_abs_wb"] = 1.0e-3
    _resign(closure)
    with pytest.raises(ValueError, match="closure exceeds"):
        contract.validate_report(closure)

    trajectory = copy.deepcopy(report)
    trajectory["trajectory"][2]["step"] = 4
    _resign(trajectory)
    with pytest.raises(ValueError, match="contiguous"):
        contract.validate_report(trajectory)


def test_validate_rejects_map_parity_flag_and_source_path_drift() -> None:
    report = _report()
    parity = copy.deepcopy(report)
    parity["map_parity"]["sha256_match"] = False
    _resign(parity)
    with pytest.raises(ValueError, match="sha256_match is inconsistent"):
        contract.validate_report(parity)

    source = copy.deepcopy(report)
    source["source_artifacts"]["diagnostic"]["path"] = "validation/other.py"
    _resign(source)
    with pytest.raises(ValueError, match="path is invalid"):
        contract.validate_report(source)
