# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Fixed-Physical Coil-Vacuum Contract Tests
"""Fail-closed tests for CVGC2 evidence, routing, and claim boundaries."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from validation import ida_coil_vacuum_fixed_physical_contract as contract
from validation import ida_coil_vacuum_grid_contract as grid_contract

ROOT = Path(__file__).resolve().parents[1]
_DIGEST = "1" * 64
_REAL_EXECUTION_SOURCE_ARTIFACTS = contract.execution_source_artifacts
_REAL_EXECUTION_SOURCE_SNAPSHOT = contract.execution_source_snapshot


def _field_metric(scale: float) -> dict[str, Any]:
    """Return one internally consistent positive field metric."""
    return {
        "area_weighted_l2": scale,
        "area_weighted_rms": 0.5 * scale,
        "field_sha256": _DIGEST,
        "l2": 2.0 * scale,
        "linf": scale,
    }


def _comparison_metric(scale: float) -> dict[str, float]:
    """Return one finite exact-restriction comparison metric."""
    return {
        "area_weighted_l2": scale,
        "area_weighted_rms": 0.5 * scale,
        "cosine": 1.0,
        "linf": scale,
        "projection": 0.25,
        "relative_l2": 0.75,
    }


def _grids(
    *,
    recovery: tuple[float, float, float, float] = (0.04, 0.01, 0.0025, 0.000625),
    localisation: float = 0.99,
    response_fraction: float = 0.01,
) -> list[dict[str, Any]]:
    """Return the exact four-grid measurement shape used by the contract."""
    rows: list[dict[str, Any]] = []
    for resolution, error in zip((33, 65, 129, 257), recovery, strict=True):
        rows.append(
            {
                "current_recovery_weighted_error": error,
                "fixed_source_l2_fraction": localisation,
                "forcing_closure_max_abs": 0.0,
                "forcing_partition": {
                    "source": _field_metric(1.0),
                    "source_free": _field_metric(0.01),
                    "total": _field_metric(1.01),
                },
                "inverse_timings_ms": {
                    "inverse_source": 1.0,
                    "inverse_source_free": 1.0,
                    "inverse_total": 1.0,
                },
                "resolution": resolution,
                "response_partition": {
                    "closure_max_abs_wb": 1.0e-14,
                    "source": _field_metric(1.0),
                    "source_free": _field_metric(response_fraction),
                    "total": _field_metric(1.0),
                },
                "source_free_response_fraction_of_total": response_fraction,
                "total_response_reproduction_max_abs_wb": 1.0e-14,
            }
        )
    return rows


def _convergence(*, forcing_order: float = 2.0, response_order: float = 2.0) -> dict[str, Any]:
    """Return both exact nested-grid order surfaces."""
    pairwise = {
        pair: {
            region: {
                "forcing": _comparison_metric(0.01),
                "response": _comparison_metric(0.001),
            }
            for region in (
                "fixed_physical_source_free",
                "full_interior",
                "plasma_support",
            )
        }
        for pair in ("33_65", "65_129", "129_257")
    }
    return {
        "pairwise": pairwise,
        "source_free_forcing_order": {
            triple: {
                "coarse_to_medium_rms": 0.04,
                "medium_to_fine_rms": 0.01,
                "observed_order": forcing_order,
            }
            for triple in contract.GRID_TRIPLES
        },
        "source_free_response_order": {
            triple: {
                "coarse_to_medium_rms": 0.004,
                "medium_to_fine_rms": 0.001,
                "observed_order": response_order,
            }
            for triple in contract.GRID_TRIPLES
        },
    }


def _source_artifacts() -> dict[str, dict[str, Any]]:
    """Return source provenance authenticated against the current commit."""
    head = contract._git_bytes(ROOT, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    artifacts = contract.source_artifacts_for_commit(ROOT, head)
    artifacts["repository"]["worktree_clean"] = True
    return artifacts


@pytest.fixture(autouse=True)
def _clean_execution_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep schema tests independent of the author's deliberately staged diff."""
    monkeypatch.setattr(contract, "execution_source_artifacts", lambda root: _source_artifacts())


def _report(
    *,
    grids: list[dict[str, Any]] | None = None,
    convergence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a self-validating report through the public contract."""
    upstream = contract.load_upstream_report(ROOT)
    source_artifacts = _source_artifacts()
    return contract.build_report(
        generated_at="2026-07-26T03:30:00Z",
        environment={"backend": "gpu", "x64_enabled": True},
        execution_binding=contract.build_execution_binding(
            anchor=upstream["anchor"],
            coil_manifest=upstream["coil_manifest"],
            source_artifacts=source_artifacts,
        ),
        source_artifacts=source_artifacts,
        upstream=contract.upstream_binding(ROOT, upstream),
        grids=_grids() if grids is None else grids,
        convergence=_convergence() if convergence is None else convergence,
    )


def _resign(report: dict[str, Any]) -> None:
    """Update the digest after a deliberate test mutation."""
    report["payload_sha256"] = contract._payload_sha256(report)


def test_build_validate_render_resolves_numerics_but_keeps_claims_false() -> None:
    """Passing fixed-physical numerics must never become scientific admission."""
    report = _report()
    contract.validate_report(report)

    assert report["routing"] == {
        "production_solver_physics_changed": False,
        "source_numerics_pass": True,
        "state": "fixed_physical_source_and_vacuum_numerics_resolved",
        "vacuum_numerics_pass": True,
    }
    assert set(report["claim_boundary"].values()) == {False}
    markdown = contract.render_markdown(report)
    assert "CVGC1 remains immutable" in markdown
    assert "Production solver physics changed: `false`" in markdown


@pytest.mark.parametrize(
    ("grids", "convergence", "state"),
    [
        (
            _grids(recovery=(0.04, 0.06, 0.07, 0.08)),
            _convergence(),
            "fixed_physical_source_numerics_unresolved",
        ),
        (
            _grids(response_fraction=0.06),
            _convergence(response_order=1.0),
            "fixed_physical_vacuum_numerics_unresolved",
        ),
        (
            _grids(recovery=(0.04, 0.06, 0.07, 0.08), response_fraction=0.06),
            _convergence(forcing_order=1.0, response_order=1.0),
            "fixed_physical_source_and_vacuum_numerics_unresolved",
        ),
    ],
)
def test_routing_separates_source_vacuum_and_mixed_failures(
    grids: list[dict[str, Any]],
    convergence: dict[str, Any],
    state: str,
) -> None:
    """Every numerical failure combination must retain its exact routing state."""
    assert _report(grids=grids, convergence=convergence)["routing"]["state"] == state


def test_validate_rejects_overclaim_upstream_and_measured_gate_tampering() -> None:
    """Claim, binding, and gate mutations must fail after digest recomputation."""
    report = _report()
    overclaim = copy.deepcopy(report)
    overclaim["claim_boundary"]["scientific_validation"] = True
    _resign(overclaim)
    with pytest.raises(ValueError, match="claim boundary"):
        contract.validate_report(overclaim)

    upstream = copy.deepcopy(report)
    upstream["upstream_binding"]["payload_sha256"] = "3" * 64
    _resign(upstream)
    with pytest.raises(ValueError, match="upstream binding drifted"):
        contract.validate_report(upstream)

    gates = copy.deepcopy(report)
    gates["gates"]["numerical"]["source_free_response_order"] = False
    _resign(gates)
    with pytest.raises(ValueError, match="gates do not match"):
        contract.validate_report(gates)


@pytest.mark.parametrize(
    "field",
    ["anchor_sha256", "coil_manifest_sha256", "source_artifacts_sha256"],
)
def test_validate_rejects_resigned_execution_binding_tampering(field: str) -> None:
    """Every execution digest remains derived after an attacker resigns the report."""
    report = _report()
    report["execution_binding"][field] = "9" * 64
    _resign(report)
    with pytest.raises(ValueError, match="execution binding drifted"):
        contract.validate_report(report)


def test_validate_rejects_resigned_source_manifest_and_response_fraction_tampering() -> None:
    """Derived provenance and response amplitude cannot be lowered by resigning."""
    source = _report()
    source["source_artifacts"]["grid_runtime"]["sha256"] = "8" * 64
    _resign(source)
    with pytest.raises(ValueError, match="committed blob"):
        contract.validate_report(source)

    amplitude = _report(grids=_grids(response_fraction=0.06))
    amplitude["grids"][2]["source_free_response_fraction_of_total"] = 0.01
    amplitude["grids"][3]["source_free_response_fraction_of_total"] = 0.01
    amplitude["gates"]["numerical"]["source_free_response_fine_amplitude"] = True
    amplitude["routing"] = contract._routing(True, amplitude["gates"]["numerical"])
    _resign(amplitude)
    with pytest.raises(ValueError, match="response fraction does not match"):
        contract.validate_report(amplitude)

    zero_total = _grids()
    zero_total[0]["response_partition"]["total"] = _field_metric(0.0)
    with pytest.raises(ValueError, match="area_weighted_l2 must be positive"):
        _report(grids=zero_total)


def test_validate_rejects_fully_resigned_source_manifest_tampering() -> None:
    """A forged blob digest fails even when every dependent digest is recomputed."""
    report = _report()
    report["source_artifacts"]["grid_runtime"]["sha256"] = "8" * 64
    report["execution_binding"] = contract.build_execution_binding(
        anchor=contract.load_upstream_report(ROOT)["anchor"],
        coil_manifest=contract.load_upstream_report(ROOT)["coil_manifest"],
        source_artifacts=report["source_artifacts"],
    )
    _resign(report)
    with pytest.raises(ValueError, match="committed blob"):
        contract.validate_report(report)


@pytest.mark.parametrize("commit", [123, "short", "f" * 64])
def test_source_manifest_rejects_noncommit_and_invalid_repository_oids(
    commit: object,
) -> None:
    """Repository provenance accepts only an inspectable 40- or 64-hex commit."""
    with pytest.raises(ValueError, match="repository commit is invalid|not inspectable"):
        contract.source_artifacts_for_commit(ROOT, cast(str, commit))


def test_source_manifest_rejects_blob_and_out_of_bundle_commit() -> None:
    """A valid Git object must be a commit with the exact frozen path delta."""
    blob = contract._git_bytes(ROOT, "rev-parse", "HEAD:README.md").decode().strip()
    with pytest.raises(ValueError, match="object is not a commit"):
        contract.source_artifacts_for_commit(ROOT, blob)
    with pytest.raises(ValueError, match="outside the frozen source bundle"):
        contract.source_artifacts_for_commit(ROOT, contract.SOURCE_BASE_COMMIT)


def test_source_manifest_rejects_unrelated_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A commit that fails the frozen-base ancestry check is rejected."""
    monkeypatch.setattr(contract, "_git_bytes", lambda root, *args: b"commit\n")

    def reject_ancestry(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr(
        "validation.ida_coil_vacuum_fixed_physical_contract.subprocess.run",
        reject_ancestry,
    )
    with pytest.raises(ValueError, match="does not descend from the frozen base"):
        contract.source_artifacts_for_commit(ROOT, "a" * 40)


def test_source_manifest_rejects_nonregular_committed_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A symlink or non-blob cannot impersonate an executed source file."""
    changed = ("\n".join(sorted(contract.SOURCE_CHANGE_PATHS)) + "\n").encode()

    def git_bytes(root: Path, *args: str) -> bytes:
        if args[0] == "cat-file":
            return b"commit\n"
        if args[0] == "diff":
            return changed
        if args[0] == "ls-tree":
            return b"120000 blob 1111111111111111111111111111111111111111\tpath\n"
        raise AssertionError(f"unexpected Git plumbing call: {args}")

    monkeypatch.setattr(contract, "_git_bytes", git_bytes)
    with pytest.raises(ValueError, match="not a regular committed file"):
        contract.source_artifacts_for_commit(ROOT, contract.SOURCE_BASE_COMMIT)


def test_execution_source_artifacts_requires_clean_matching_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only current HEAD bytes from a clean checkout can form execution provenance."""
    commit = "a" * 40
    content = b"authenticated source\n"
    digest = hashlib.sha256(content).hexdigest()
    historical: dict[str, dict[str, Any]] = {
        name: {"path": path, "sha256": digest} for name, path in contract.SOURCE_PATHS.items()
    }
    historical["repository"] = {"git_commit": commit, "path": "."}
    for path in contract.SOURCE_PATHS.values():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    def clean_git_bytes(root: Path, *args: str) -> bytes:
        if args[0] == "rev-parse":
            return f"{commit}\n".encode()
        if args[0] == "status":
            return b""
        raise AssertionError(f"unexpected Git plumbing call: {args}")

    monkeypatch.setattr(contract, "_git_bytes", clean_git_bytes)
    monkeypatch.setattr(contract, "source_artifacts_for_commit", lambda root, oid: historical)
    artifacts, signature = _REAL_EXECUTION_SOURCE_SNAPSHOT(tmp_path)
    assert artifacts["repository"] == {
        "git_commit": commit,
        "path": ".",
        "worktree_clean": True,
    }

    target = tmp_path / next(iter(contract.SOURCE_PATHS.values()))
    target.write_bytes(b"transient drift\n")
    target.write_bytes(content)
    restored_artifacts, restored_signature = _REAL_EXECUTION_SOURCE_SNAPSHOT(tmp_path)
    assert restored_artifacts == artifacts
    assert restored_signature != signature

    real_read_bytes = Path.read_bytes

    def mutating_read_bytes(path: Path) -> bytes:
        value = real_read_bytes(path)
        if path == target:
            path.write_bytes(b"capture-time drift\n")
            path.write_bytes(value)
        return value

    monkeypatch.setattr(Path, "read_bytes", mutating_read_bytes)
    with pytest.raises(ValueError, match="mutated while execution provenance was captured"):
        _REAL_EXECUTION_SOURCE_SNAPSHOT(tmp_path)
    monkeypatch.setattr(Path, "read_bytes", real_read_bytes)

    def invalid_head(root: Path, *args: str) -> bytes:
        return b"short\n" if args[0] == "rev-parse" else b""

    monkeypatch.setattr(contract, "_git_bytes", invalid_head)
    with pytest.raises(ValueError, match="executing HEAD is invalid"):
        _REAL_EXECUTION_SOURCE_ARTIFACTS(tmp_path)

    monkeypatch.setattr(contract, "_git_bytes", clean_git_bytes)
    target.write_bytes(b"drift\n")
    with pytest.raises(ValueError, match="executing bytes do not match"):
        _REAL_EXECUTION_SOURCE_ARTIFACTS(tmp_path)


@pytest.mark.parametrize(
    "status",
    [
        b"M  validation/ida_coil_vacuum_fixed_physical_contract.py\0",
        b" M validation/ida_coil_vacuum_fixed_physical_contract.py\0",
    ],
    ids=["staged", "unstaged"],
)
def test_validate_rejects_fully_resigned_dirty_execution(
    status: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-signing cannot hide staged or unstaged execution-source drift."""
    report = _report()
    report["generated_at"] = "2026-07-26T06:00:00Z"
    _resign(report)
    real_git_bytes = contract._git_bytes

    def dirty_git_bytes(root: Path, *args: str) -> bytes:
        if args[0] == "status":
            return status
        return real_git_bytes(root, *args)

    monkeypatch.setattr(contract, "_git_bytes", dirty_git_bytes)
    monkeypatch.setattr(contract, "execution_source_artifacts", _REAL_EXECUTION_SOURCE_ARTIFACTS)
    with pytest.raises(ValueError, match="staged or unstaged source drift"):
        contract.validate_report(report)


def test_validate_rejects_authenticated_repository_and_live_execution_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Historical and live repository bindings must both match the report."""
    report = _report()
    authenticated = _source_artifacts()
    authenticated["repository"].pop("worktree_clean")
    authenticated["repository"]["git_commit"] = "b" * 40
    monkeypatch.setattr(
        contract,
        "source_artifacts_for_commit",
        lambda root, commit: authenticated,
    )
    with pytest.raises(ValueError, match="repository provenance does not match"):
        contract.validate_report(report)

    monkeypatch.undo()
    live = _source_artifacts()
    live[next(iter(contract.SOURCE_PATHS))]["sha256"] = "9" * 64
    monkeypatch.setattr(contract, "execution_source_artifacts", lambda root: live)
    with pytest.raises(ValueError, match="clean executing checkout"):
        contract.validate_report(report)


@pytest.mark.parametrize("surface", ["anchor", "manifest"])
def test_build_execution_binding_rejects_upstream_object_drift(surface: str) -> None:
    """The builder cannot bind a different anchor or coil manifest under valid hex."""
    upstream = contract.load_upstream_report(ROOT)
    anchor = copy.deepcopy(upstream["anchor"])
    manifest = copy.deepcopy(upstream["coil_manifest"])
    if surface == "anchor":
        anchor["forcing_sha256"] = "9" * 64
    else:
        manifest["parent_count"] += 1
    with pytest.raises(ValueError, match="does not match the frozen CVGC1"):
        contract.build_execution_binding(
            anchor=anchor,
            coil_manifest=manifest,
            source_artifacts=_source_artifacts(),
        )


def test_validate_rejects_dirty_source_nonfinite_metric_and_digest_drift() -> None:
    """Source cleanliness, finiteness, and payload integrity are mandatory."""
    dirty = _report()
    dirty["source_artifacts"]["repository"]["worktree_clean"] = False
    _resign(dirty)
    with pytest.raises(ValueError, match="clean canonical source commit"):
        contract.validate_report(dirty)

    nonfinite = _report()
    nonfinite["grids"][0]["fixed_source_l2_fraction"] = float("nan")
    with pytest.raises(ValueError, match="valid range"):
        contract.validate_report(nonfinite)

    digest = _report()
    digest["payload_sha256"] = "4" * 64
    with pytest.raises(ValueError, match="payload digest"):
        contract.validate_report(digest)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("top", "top-level fields"),
        ("identity", "identity"),
        ("generated", "generated_at"),
        ("design", "frozen design"),
        ("blockers", "blockers"),
        ("upstream_fields", "upstream binding is invalid"),
        ("execution", "execution binding drifted"),
        ("environment", "environment"),
        ("source_set", "source artifact set"),
        ("repository", "repository provenance"),
        ("source_path", "source path"),
        ("source_digest", "source digest"),
        ("grid_count", "complete four-grid ladder"),
        ("grid_fields", "grid fields"),
        ("resolution_type", "resolution is invalid"),
        ("partition", "partition fields"),
        ("field_metric", "metric fields"),
        ("field_digest", "field_sha256"),
        ("timings", "timing fields"),
        ("ladder", "resolutions are not"),
        ("convergence", "convergence fields"),
        ("pairwise", "pairwise ladder"),
        ("regions", "pairwise regions"),
        ("pair_fields", "pairwise fields"),
        ("comparison", "comparison metric fields"),
        ("order_triples", "triples are invalid"),
        ("order_fields", "order fields are invalid"),
        ("routing", "routing does not match"),
        ("status", "status is inconsistent"),
    ],
)
def test_validate_rejects_each_schema_and_provenance_surface(
    case: str,
    message: str,
) -> None:
    """Every independent schema, provenance, and derived-state surface is fail-closed."""
    report = _report()
    if case == "top":
        report.pop("status")
    elif case == "identity":
        report["benchmark_id"] = "wrong"
    elif case == "generated":
        report["generated_at"] = "yesterday"
    elif case == "design":
        report["input_contract"]["fixed_physical_radius_m"] = 0.2
    elif case == "blockers":
        report["blockers"] = []
    elif case == "upstream_fields":
        report["upstream_binding"].pop("file_sha256")
    elif case == "execution":
        report["execution_binding"]["anchor_sha256"] = "short"
    elif case == "environment":
        report["environment"] = {}
    elif case == "source_set":
        report["source_artifacts"].pop("grid_runtime")
    elif case == "repository":
        report["source_artifacts"]["repository"].pop("path")
    elif case == "source_path":
        report["source_artifacts"]["grid_runtime"]["path"] = "wrong"
    elif case == "source_digest":
        report["source_artifacts"]["grid_runtime"]["sha256"] = "short"
    elif case == "grid_count":
        report["grids"].pop()
    elif case == "grid_fields":
        report["grids"][0].pop("fixed_source_l2_fraction")
    elif case == "resolution_type":
        report["grids"][0]["resolution"] = True
    elif case == "partition":
        report["grids"][0]["forcing_partition"].pop("source")
    elif case == "field_metric":
        report["grids"][0]["forcing_partition"]["source"].pop("l2")
    elif case == "field_digest":
        report["grids"][0]["forcing_partition"]["source"]["field_sha256"] = "short"
    elif case == "timings":
        report["grids"][0]["inverse_timings_ms"].pop("inverse_total")
    elif case == "ladder":
        report["grids"][0]["resolution"] = 65
    elif case == "convergence":
        report["convergence"].pop("pairwise")
    elif case == "pairwise":
        report["convergence"]["pairwise"].pop("33_65")
    elif case == "regions":
        report["convergence"]["pairwise"]["33_65"].pop("plasma_support")
    elif case == "pair_fields":
        report["convergence"]["pairwise"]["33_65"]["full_interior"].pop("forcing")
    elif case == "comparison":
        report["convergence"]["pairwise"]["33_65"]["full_interior"]["forcing"].pop("cosine")
    elif case == "order_triples":
        report["convergence"]["source_free_response_order"].pop("33_65_129")
    elif case == "order_fields":
        report["convergence"]["source_free_response_order"]["33_65_129"].pop("observed_order")
    elif case == "routing":
        report["routing"]["state"] = "wrong"
    elif case == "status":
        report["status"] = "wrong"
    else:
        raise AssertionError(f"unhandled case {case}")
    _resign(report)
    with pytest.raises(ValueError, match=message):
        contract.validate_report(report)


def test_internal_numeric_and_recursive_validators_reject_unsupported_values() -> None:
    """Booleans, non-finite numbers, and unsupported objects cannot enter evidence."""
    with pytest.raises(ValueError, match="must be numeric"):
        contract._number(True, field="value")
    with pytest.raises(ValueError, match="non-finite"):
        contract._walk_finite({"value": float("inf")})
    with pytest.raises(ValueError, match="unsupported"):
        contract._walk_finite(object())


def test_upstream_loader_rejects_nonobject_payload_and_byte_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The upstream CVGC1 file must be an object with the frozen payload."""
    path = tmp_path / contract.UPSTREAM_PATH
    path.parent.mkdir(parents=True)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        contract.load_upstream_report(tmp_path)

    upstream = contract.load_upstream_report(ROOT)
    upstream["payload_sha256"] = "5" * 64
    path.write_text(json.dumps(upstream), encoding="utf-8")
    monkeypatch.setattr(grid_contract, "validate_report", lambda report: None)
    with pytest.raises(ValueError, match="frozen binding"):
        contract.load_upstream_report(tmp_path)

    canonical = (ROOT / contract.UPSTREAM_PATH).read_text(encoding="utf-8")
    path.write_text(canonical.replace("\n", "\r\n"), encoding="utf-8", newline="")
    with pytest.raises(ValueError, match="frozen bytes"):
        contract.load_upstream_report(tmp_path)
    with pytest.raises(ValueError, match="frozen bytes"):
        contract.upstream_binding(tmp_path, upstream)


def test_upstream_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    """Duplicate JSON names fail before any semantic or digest validation."""
    path = tmp_path / contract.UPSTREAM_PATH
    path.parent.mkdir(parents=True)
    path.write_text('{"schema_version": "first", "schema_version": "second"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key: schema_version"):
        contract.load_upstream_report(tmp_path)


def test_partition_closure_failure_blocks_structurally_without_overclaim() -> None:
    """A measured closure failure must use the structural blocked route."""
    grids = _grids()
    grids[0]["forcing_closure_max_abs"] = 1.0e-6
    report = _report(grids=grids)
    assert report["status"] == "blocked_incomplete_or_drifted"
    assert report["routing"]["state"] == "blocked_upstream_or_partition_drift"
    assert set(report["claim_boundary"].values()) == {False}
