# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — distributed CI ownership verification
"""Check the actual CI graph against its versioned ownership contract."""

from __future__ import annotations

import argparse
from collections.abc import Hashable
import json
import re
from pathlib import Path
from typing import Any, cast

import yaml


class _WorkflowLoader(yaml.BaseLoader):
    """Read workflow scalars as strings and reject ambiguous duplicate mapping keys."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        """Reject repeated keys before YAML construction can silently discard a value."""
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate workflow mapping key")
        return super().construct_mapping(node, deep=deep)


def _mapping(value: object, label: str) -> dict[str, Any]:
    """Require a structured mapping before reading the named contract surface."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label}: expected a mapping with string keys")
    return cast(dict[str, Any], value)


def _require(condition: bool, message: str) -> None:
    """Reject a violated graph invariant with its specific diagnostic."""
    if not condition:
        raise ValueError(message)


def _workflow(root: Path, relative: str, limits: dict[str, Any]) -> dict[str, Any]:
    """Read a repository-contained workflow within the declared byte and line limits."""
    path = (root / relative).resolve()
    _require(path.is_relative_to(root.resolve()), f"workflow outside repository: {relative}")
    raw = path.read_bytes()
    _require(len(raw) <= limits["bytes_per_file"], f"workflow byte limit: {relative}")
    _require(
        len(raw.splitlines()) <= limits["lines_per_file"],
        f"workflow line limit: {relative}",
    )
    return _mapping(yaml.load(raw.decode("utf-8"), Loader=_WorkflowLoader), relative)


def check_workflow_ownership(root: Path) -> dict[str, int]:
    """Validate declared jobs, routing, permissions and immutable action pins.

    Parameters
    ----------
    root : pathlib.Path
        Repository containing the ownership policy and actual workflow files.

    Returns
    -------
    dict[str, int]
        Counts of verified reusable workflows and executable jobs.

    Raises
    ------
    ValueError
        A workflow violates the declared ownership or aggregate contract.
    OSError
        A required policy or workflow file cannot be read.
    """
    policy = _mapping(json.loads((root / "tools/ci_workflow_ownership.json").read_text()), "policy")
    _require(policy["schema"] == "ci-workflow-ownership.v1", "unsupported ownership schema")
    groups = _mapping(policy["groups"], "groups")
    limits = _mapping(policy["limits"], "limits")
    _require(0 < len(groups) <= limits["reusable_workflows"], "reusable workflow count limit")
    coordinator = _workflow(root, policy["coordinator"], limits)
    jobs = _mapping(coordinator["jobs"], "coordinator jobs")
    gate = policy["gate"]
    _require(set(jobs) == set(groups) | {gate["job"]}, "coordinator job inventory mismatch")
    _require(jobs[gate["job"]] == gate["body"], "required aggregate body changed")
    _require(gate["body"]["name"] == gate["name"], "required check name mismatch")
    _require(
        set(gate["body"]["needs"]) == set(groups),
        "aggregate does not cover every group",
    )
    _require(len(gate["body"]["needs"]) == len(groups), "duplicate aggregate dependency")
    _require(gate["body"]["if"] == "${{ always() }}", "aggregate must run after failures")
    owned: set[str] = set()
    paths: set[str] = set()
    graph: dict[str, set[str]] = {}
    for name, specification in groups.items():
        owner = _mapping(specification, name)
        _require(
            type(owner["category"]) is int and 1 <= owner["category"] <= 11,
            f"category: {name}",
        )
        _require(owner["path"] not in paths, f"duplicate reusable path: {name}")
        paths.add(owner["path"])
        caller = jobs[name]
        _require(
            set(caller) <= {"uses", "needs", "if", "permissions", "secrets"},
            f"caller body: {name}",
        )
        _require(caller["uses"] == "./" + owner["path"], f"caller path: {name}")
        _require(caller.get("needs", []) == owner["needs"], f"caller dependencies: {name}")
        _require(caller.get("if") == owner["condition"], f"caller condition: {name}")
        _require(caller["permissions"] == owner["permissions"], f"caller permissions: {name}")
        _require(
            caller.get("secrets", {}) == _mapping(owner["secrets"], f"policy secrets: {name}"),
            f"caller secrets: {name}",
        )
        graph[name] = set(owner["needs"])
        _require(
            len(graph[name]) == len(owner["needs"]),
            f"duplicate group dependency: {name}",
        )
        _require(graph[name] <= set(groups), f"unknown dependency: {name}")
        child = _workflow(root, owner["path"], limits)
        _require(
            set(child["on"]) == {"workflow_call"},
            f"unexpected reusable trigger: {name}",
        )
        _require(
            child["on"]["workflow_call"]
            == _mapping(owner["workflow_call"], f"policy reusable interface: {name}"),
            f"reusable secret interface: {name}",
        )
        _require(child["permissions"] == {}, f"reusable default permissions: {name}")
        _require(child["env"] == policy["global_env"], f"reusable environment: {name}")
        actual = _mapping(child["jobs"], name)
        _require(set(actual) == set(owner["jobs"]), f"reusable job inventory: {name}")
        _require(len(owner["jobs"]) == len(actual), f"duplicate declared job: {name}")
        for job_id, body in actual.items():
            _require(job_id not in owned, f"job owned twice: {job_id}")
            owned.add(job_id)
            _require("needs" not in body and "if" not in body, f"unlifted routing: {job_id}")
            _require(
                body["permissions"] == owner["permissions"],
                f"job permissions: {job_id}",
            )
            for step in body["steps"]:
                if "uses" in step:
                    _require(
                        re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", step["uses"]) is not None,
                        f"action must use immutable SHA: {job_id}",
                    )
    while graph:
        ready = {name for name, dependencies in graph.items() if not dependencies}
        _require(bool(ready), "workflow dependency cycle")
        graph = {
            name: dependencies - ready for name, dependencies in graph.items() if name not in ready
        }
    return {"reusable_workflows": len(groups), "executable_jobs": len(owned) + 1}


def workflow_sources(root: Path) -> dict[str, str]:
    """Return source text for every workflow in the validated ownership graph.

    Parameters
    ----------
    root : pathlib.Path
        Repository containing the coordinator and its ownership policy.

    Returns
    -------
    dict[str, str]
        Repository-relative paths and UTF-8 source, ordered by path.

    Raises
    ------
    ValueError
        The declared workflow graph violates an ownership invariant.
    OSError
        A policy or workflow file is unavailable.
    """
    check_workflow_ownership(root)
    policy = json.loads((root / "tools/ci_workflow_ownership.json").read_text())
    paths = {policy["coordinator"], *(group["path"] for group in policy["groups"].values())}
    return {name: (root / name).read_text(encoding="utf-8") for name in sorted(paths)}


def main() -> int:
    """Run the ownership check and print its verified graph counts.

    Returns
    -------
    int
        Zero when the workflow graph satisfies the declared contract.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(check_workflow_ownership(args.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
