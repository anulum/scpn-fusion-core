# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Coil-Vacuum Grid Manifest Validation
"""Provenance, anchor, and coil-lineage validators for CVGC1 evidence."""

from __future__ import annotations

import hashlib
import json
import math

import validation.ida_coil_vacuum_grid_contract as contract


def validate_bindings(value: object) -> None:
    """Validate the exact six frozen prerequisite bindings."""
    if not isinstance(value, dict) or set(value) != set(contract.EXPECTED_PAYLOADS):
        raise ValueError("bindings fields are invalid")
    for name, expected_digest in contract.EXPECTED_PAYLOADS.items():
        row = value[name]
        if not isinstance(row, dict) or set(row) != {"path", "payload_sha256"}:
            raise ValueError(f"bindings.{name} fields are invalid")
        if row["path"] != contract.EXPECTED_BINDING_PATHS[name]:
            raise ValueError(f"bindings.{name}.path is invalid")
        if row["payload_sha256"] != expected_digest:
            raise ValueError(f"bindings.{name}.payload_sha256 is not the frozen payload")


def validate_source_artifacts(value: object) -> None:
    """Validate source hashes and clean committed-repository provenance."""
    expected = {*contract.SOURCE_PATHS, *contract.RUNTIME_SOURCE_NAMES, "repository"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("source_artifacts fields are invalid")
    for name in expected - {"repository"}:
        row = value[name]
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError(f"source_artifacts.{name} fields are invalid")
        if name in contract.SOURCE_PATHS and row["path"] != contract.SOURCE_PATHS[name]:
            raise ValueError(f"source_artifacts.{name}.path is invalid")
        if not isinstance(row["path"], str) or not row["path"]:
            raise ValueError(f"source_artifacts.{name}.path is invalid")
        contract._require_sha256(
            row["sha256"],
            field=f"source_artifacts.{name}.sha256",
        )
    repository = value["repository"]
    if not isinstance(repository, dict) or set(repository) != {
        "git_commit",
        "path",
        "worktree_clean",
    }:
        raise ValueError("source_artifacts.repository fields are invalid")
    if repository["path"] != "." or repository["worktree_clean"] is not True:
        raise ValueError("grid-convergence evidence requires a clean canonical repository")
    if (
        not isinstance(repository["git_commit"], str)
        or contract._GIT_OID_RE.fullmatch(repository["git_commit"]) is None
    ):
        raise ValueError("source_artifacts.repository.git_commit is invalid")


def validate_anchor(value: object) -> None:
    """Validate exact forcing/response digests and inherited response closure."""
    if not isinstance(value, dict) or set(value) != {
        "forcing_sha256",
        "response_sha256",
        "response_closure_max_abs_wb",
    }:
        raise ValueError("anchor fields are invalid")
    if value["forcing_sha256"] != contract.EXPECTED_ANCHOR_FORCING_SHA256:
        raise ValueError("129 forcing anchor does not match the frozen payload")
    if value["response_sha256"] != contract.EXPECTED_ANCHOR_RESPONSE_SHA256:
        raise ValueError("129 response anchor does not match the frozen payload")
    if (
        contract._require_number(
            value["response_closure_max_abs_wb"],
            field="anchor.response_closure_max_abs_wb",
            minimum=0.0,
        )
        > contract.PARTITION_CLOSURE_MAX_ABS
    ):
        raise ValueError("129 response anchor closure exceeds the frozen threshold")


def validate_manifest(value: object) -> None:
    """Validate exact parent/filament lineage, currents, domain, and digest."""
    if not isinstance(value, dict) or set(value) != {
        "filament_count",
        "manifest_sha256",
        "parent_count",
        "parents",
    }:
        raise ValueError("coil_manifest fields are invalid")
    if (
        not isinstance(value["parent_count"], int)
        or isinstance(value["parent_count"], bool)
        or not isinstance(value["filament_count"], int)
        or isinstance(value["filament_count"], bool)
        or value["parent_count"] != 18
        or value["filament_count"] != 216
    ):
        raise ValueError("coil_manifest cardinality is invalid")
    parents = value["parents"]
    if not isinstance(parents, list) or len(parents) != 18:
        raise ValueError("coil_manifest parents are invalid")
    identifiers: set[str] = set()
    names: set[str] = set()
    expected_parent_fields = {
        "coil_type",
        "current_a",
        "effective_current_a_turns",
        "filament_count",
        "filaments",
        "name",
        "parent_index",
        "turns",
    }
    expected_filament_fields = {
        "effective_current_a_turns",
        "filament_id",
        "filament_index",
        "parent_index",
        "parent_name",
        "r_m",
        "weight",
        "z_m",
    }
    total_filaments = 0
    for expected_parent_index, parent in enumerate(parents):
        if (
            not isinstance(parent, dict)
            or set(parent) != expected_parent_fields
            or not isinstance(parent["filament_count"], int)
            or isinstance(parent["filament_count"], bool)
            or parent["filament_count"] < 1
        ):
            raise ValueError("coil_manifest parent row is invalid")
        if (
            not isinstance(parent["parent_index"], int)
            or isinstance(parent["parent_index"], bool)
            or parent["parent_index"] != expected_parent_index
        ):
            raise ValueError("coil_manifest parent indices are invalid")
        name = parent["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("coil_manifest parent names are invalid")
        names.add(name)
        if not isinstance(parent["coil_type"], str) or not parent["coil_type"]:
            raise ValueError("coil_manifest parent coil_type is invalid")
        current = contract._require_number(
            parent["current_a"],
            field=f"coil_manifest.{name}.current_a",
        )
        turns = contract._require_number(
            parent["turns"],
            field=f"coil_manifest.{name}.turns",
            minimum=0.0,
        )
        if turns == 0.0:
            raise ValueError("coil_manifest parent turns must be positive")
        effective = contract._require_number(
            parent["effective_current_a_turns"],
            field=f"coil_manifest.{name}.effective_current_a_turns",
        )
        if not math.isclose(effective, current * turns, rel_tol=1.0e-12, abs_tol=1.0e-9):
            raise ValueError("coil_manifest parent effective current is inconsistent")
        filaments = parent["filaments"]
        if not isinstance(filaments, list) or len(filaments) != parent["filament_count"]:
            raise ValueError("coil_manifest filament rows are invalid")
        child_currents: list[float] = []
        for expected_filament_index, filament in enumerate(filaments):
            if not isinstance(filament, dict) or set(filament) != expected_filament_fields:
                raise ValueError("coil_manifest filament row is invalid")
            if (
                not isinstance(filament["parent_index"], int)
                or isinstance(filament["parent_index"], bool)
                or not isinstance(filament["filament_index"], int)
                or isinstance(filament["filament_index"], bool)
                or filament["parent_index"] != expected_parent_index
                or filament["parent_name"] != name
                or filament["filament_index"] != expected_filament_index
            ):
                raise ValueError("coil_manifest filament lineage is invalid")
            identifier = filament["filament_id"]
            expected_identifier = f"{name}:{expected_filament_index:03d}"
            if identifier != expected_identifier or identifier in identifiers:
                raise ValueError("coil_manifest filament identifiers are invalid")
            identifiers.add(identifier)
            r_m = contract._require_number(filament["r_m"], field=f"{identifier}.r_m")
            z_m = contract._require_number(filament["z_m"], field=f"{identifier}.z_m")
            if not (contract.R_BOUNDS_M[0] < r_m < contract.R_BOUNDS_M[1]) or not (
                contract.Z_BOUNDS_M[0] < z_m < contract.Z_BOUNDS_M[1]
            ):
                raise ValueError("coil_manifest filament lies outside the fixed domain")
            weight = contract._require_number(
                filament["weight"],
                field=f"{identifier}.weight",
            )
            child_current = contract._require_number(
                filament["effective_current_a_turns"],
                field=f"{identifier}.effective_current_a_turns",
            )
            if not math.isclose(
                child_current,
                effective * weight,
                rel_tol=1.0e-12,
                abs_tol=1.0e-9,
            ):
                raise ValueError("coil_manifest filament effective current is inconsistent")
            child_currents.append(child_current)
        if not math.isclose(
            math.fsum(child_currents),
            effective,
            rel_tol=1.0e-12,
            abs_tol=1.0e-9,
        ):
            raise ValueError("coil_manifest child currents do not close to the parent")
        total_filaments += len(filaments)
    if total_filaments != 216 or len(identifiers) != 216:
        raise ValueError("coil_manifest flattened filament cardinality is invalid")
    digest = value["manifest_sha256"]
    unsigned = {name: item for name, item in value.items() if name != "manifest_sha256"}
    if (
        digest
        != hashlib.sha256(
            json.dumps(
                unsigned,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
    ):
        raise ValueError("coil_manifest manifest_sha256 is invalid")
