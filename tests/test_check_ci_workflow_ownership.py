# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — distributed workflow ownership tests
"""Exercise the ownership CLI against real workflow copies and explicit drift."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tools.check_ci_workflow_ownership import check_workflow_ownership

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/check_ci_workflow_ownership.py"


def test_current_distributed_workflows_satisfy_ownership() -> None:
    """The public API and CLI agree on the complete maintained graph and counts."""
    result = subprocess.run(
        [sys.executable, str(TOOL), "--root", str(ROOT)],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    assert (
        json.loads(result.stdout)
        == check_workflow_ownership(ROOT)
        == {
            "reusable_workflows": 17,
            "executable_jobs": 21,
        }
    )


@pytest.mark.parametrize(
    ("change", "diagnostic"),
    [
        ("missing_caller", "coordinator job inventory"),
        ("extra_caller", "coordinator job inventory"),
        ("aggregate_bypass", "aggregate body changed"),
        ("caller_steps", "caller body"),
        ("caller_path", "caller path"),
        ("caller_dependency", "caller dependencies"),
        ("caller_permission", "caller permissions"),
        ("caller_condition", "caller condition"),
        ("extra_trigger", "unexpected reusable trigger"),
        ("child_permission", "reusable default permissions"),
        ("child_environment", "reusable environment"),
        ("missing_job", "reusable job inventory"),
        ("child_dependency", "unlifted routing"),
        ("job_permission", "job permissions"),
        ("mutable_action", "immutable SHA"),
    ],
)
def test_workflow_drift_is_rejected(tmp_path: Path, change: str, diagnostic: str) -> None:
    """Real workflow drift fails before a changed graph can report acceptance."""
    shutil.copytree(ROOT / ".github", tmp_path / ".github")
    (tmp_path / "tools").mkdir()
    shutil.copyfile(
        ROOT / "tools/ci_workflow_ownership.json",
        tmp_path / "tools/ci_workflow_ownership.json",
    )
    coordinator_path = tmp_path / ".github/workflows/ci.yml"
    child_path = tmp_path / ".github/workflows/ci-lean-safety-proofs.yml"
    coordinator = yaml.load(coordinator_path.read_text(), Loader=yaml.BaseLoader)
    child = yaml.load(child_path.read_text(), Loader=yaml.BaseLoader)
    caller = coordinator["jobs"]["lean-safety-proofs"]
    job = child["jobs"]["lean-safety-proofs"]
    if change == "missing_caller":
        del coordinator["jobs"]["lean-safety-proofs"]
    elif change == "extra_caller":
        coordinator["jobs"]["undeclared"] = caller
    elif change == "aggregate_bypass":
        coordinator["jobs"]["required-gate"]["steps"][0]["run"] = "true"
    elif change == "caller_steps":
        caller["steps"] = [{"run": "true"}]
    elif change == "caller_path":
        caller["uses"] = "./.github/workflows/absent.yml"
    elif change == "caller_dependency":
        caller["needs"] = []
    elif change == "caller_permission":
        caller["permissions"] = {"contents": "write"}
    elif change == "caller_condition":
        caller["if"] = "false"
    elif change == "extra_trigger":
        child["on"]["push"] = {}
    elif change == "child_permission":
        child["permissions"] = {"contents": "write"}
    elif change == "child_environment":
        child["env"]["GIT_CONFIG_VALUE_0"] = "other"
    elif change == "missing_job":
        del child["jobs"]["lean-safety-proofs"]
    elif change == "child_dependency":
        job["needs"] = ["foreign-job"]
    elif change == "job_permission":
        job["permissions"] = {"contents": "write"}
    else:
        assert change == "mutable_action"
        job["steps"][0]["uses"] = "actions/checkout@main"
    coordinator_path.write_text(yaml.safe_dump(coordinator))
    child_path.write_text(yaml.safe_dump(child))
    result = subprocess.run(
        [sys.executable, str(TOOL), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode != 0
    assert diagnostic in result.stderr


@pytest.mark.parametrize(
    ("change", "diagnostic"),
    [
        ("schema", "unsupported ownership schema"),
        ("policy_mapping", "policy: expected a mapping"),
        ("groups_mapping", "groups: expected a mapping"),
        ("limits_mapping", "limits: expected a mapping"),
        ("empty_groups", "reusable workflow count limit"),
        ("count_limit", "reusable workflow count limit"),
        ("byte_limit", "workflow byte limit"),
        ("line_limit", "workflow line limit"),
        ("category_boolean", "category: lean-safety-proofs"),
        ("category_zero", "category: lean-safety-proofs"),
        ("category_overflow", "category: lean-safety-proofs"),
        ("outside_path", "workflow outside repository"),
    ],
)
def test_invalid_ownership_policy_is_rejected(tmp_path: Path, change: str, diagnostic: str) -> None:
    """The CLI rejects malformed policy, invalid categories and exceeded file limits."""
    shutil.copytree(ROOT / ".github", tmp_path / ".github")
    (tmp_path / "tools").mkdir()
    path = tmp_path / "tools/ci_workflow_ownership.json"
    policy = json.loads((ROOT / "tools/ci_workflow_ownership.json").read_text())
    if change == "schema":
        policy["schema"] = "unknown"
    elif change == "policy_mapping":
        policy = []
    elif change == "groups_mapping":
        policy["groups"] = []
    elif change == "limits_mapping":
        policy["limits"] = []
    elif change == "empty_groups":
        policy["groups"] = {}
    elif change == "count_limit":
        policy["limits"]["reusable_workflows"] = 1
    elif change == "byte_limit":
        policy["limits"]["bytes_per_file"] = 1
    elif change == "line_limit":
        policy["limits"]["lines_per_file"] = 1
    elif change.startswith("category_"):
        policy["groups"]["lean-safety-proofs"]["category"] = {
            "category_boolean": True,
            "category_zero": 0,
            "category_overflow": 12,
        }[change]
    else:
        assert change == "outside_path"
        policy["coordinator"] = "../outside.yml"
    path.write_text(json.dumps(policy))
    result = subprocess.run(
        [sys.executable, str(TOOL), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode != 0
    assert diagnostic in result.stderr


@pytest.mark.parametrize(
    ("change", "diagnostic"),
    [
        ("unknown", "unknown dependency"),
        ("self_cycle", "workflow dependency cycle"),
        ("two_node_cycle", "workflow dependency cycle"),
        ("duplicate", "duplicate group dependency"),
    ],
)
def test_agreed_invalid_dependency_graph_is_rejected(
    tmp_path: Path, change: str, diagnostic: str
) -> None:
    """Matching policy and workflow edits cannot admit invalid graph dependencies."""
    shutil.copytree(ROOT / ".github", tmp_path / ".github")
    (tmp_path / "tools").mkdir()
    policy_path = tmp_path / "tools/ci_workflow_ownership.json"
    policy = json.loads((ROOT / "tools/ci_workflow_ownership.json").read_text())
    coordinator_path = tmp_path / ".github/workflows/ci.yml"
    coordinator = yaml.load(coordinator_path.read_text(), Loader=yaml.BaseLoader)
    name = "lean-safety-proofs"
    if change == "unknown":
        dependencies = ["undeclared-group"]
    elif change == "self_cycle":
        dependencies = [name]
    elif change == "two_node_cycle":
        dependencies = ["ci-chain-guard"]
        policy["groups"]["ci-chain-guard"]["needs"] = [name]
        coordinator["jobs"]["ci-chain-guard"]["needs"] = [name]
    else:
        assert change == "duplicate"
        dependencies = ["ci-chain-guard", "ci-chain-guard"]
    policy["groups"][name]["needs"] = dependencies
    coordinator["jobs"][name]["needs"] = dependencies
    policy_path.write_text(json.dumps(policy))
    coordinator_path.write_text(yaml.safe_dump(coordinator))
    result = subprocess.run(
        [sys.executable, str(TOOL), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode != 0
    assert diagnostic in result.stderr


@pytest.mark.parametrize("status", ["failure", "cancelled", "skipped"])
@pytest.mark.parametrize(
    "category",
    sorted(json.loads((ROOT / "tools/ci_workflow_ownership.json").read_text())["groups"]),
)
def test_required_gate_rejects_unsuccessful_push_category(category: str, status: str) -> None:
    """Execute the actual aggregate shell: no required push category may fail or skip."""
    workflow = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    step = workflow["jobs"]["required-gate"]["steps"][0]
    expected = json.loads(step["env"]["EXPECTED_CATEGORIES"])
    results = {name: {"result": "success"} for name in expected}
    results[category]["result"] = status
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        env=os.environ
        | {
            "CATEGORY_RESULTS": json.dumps(results),
            "EXPECTED_CATEGORIES": json.dumps(expected),
            "EVENT_NAME": "push",
            "REF_NAME": "refs/heads/main",
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode != 0
    assert f"{category}: {status}" in result.stderr


@pytest.mark.parametrize(
    ("event", "stress", "change", "accepted"),
    [
        ("push", "success", "none", True),
        ("pull_request", "skipped", "none", True),
        ("pull_request", "failure", "none", False),
        ("push", "success", "missing", False),
        ("push", "success", "extra", False),
    ],
)
def test_required_gate_admission_and_inventory(
    event: str, stress: str, change: str, accepted: bool
) -> None:
    """The actual aggregate accepts complete success or the documented PR stress skip."""
    workflow = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    step = workflow["jobs"]["required-gate"]["steps"][0]
    expected = json.loads(step["env"]["EXPECTED_CATEGORIES"])
    results = {name: {"result": "success"} for name in expected}
    results["stress-test"]["result"] = stress
    if change == "missing":
        del results["ci-chain-guard"]
    elif change == "extra":
        results["unowned"] = {"result": "success"}
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        env=os.environ
        | {
            "CATEGORY_RESULTS": json.dumps(results),
            "EXPECTED_CATEGORIES": json.dumps(expected),
            "EVENT_NAME": event,
            "REF_NAME": "refs/heads/main",
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert (result.returncode == 0) is accepted


@pytest.mark.parametrize("filename", ["ci.yml", "ci-lean-safety-proofs.yml"])
def test_duplicate_workflow_keys_are_rejected(tmp_path: Path, filename: str) -> None:
    """Duplicate keys in actual coordinator or reusable files cannot shadow a contract."""
    shutil.copytree(ROOT / ".github", tmp_path / ".github")
    (tmp_path / "tools").mkdir()
    shutil.copyfile(
        ROOT / "tools/ci_workflow_ownership.json",
        tmp_path / "tools/ci_workflow_ownership.json",
    )
    path = tmp_path / ".github/workflows" / filename
    path.write_text(path.read_text() + "\npermissions: {}\n")
    result = subprocess.run(
        [sys.executable, str(TOOL), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode != 0
    assert "duplicate workflow mapping key" in result.stderr


@pytest.mark.parametrize(
    "group", sorted(json.loads((ROOT / "tools/ci_workflow_ownership.json").read_text())["groups"])
)
@pytest.mark.parametrize("surface", ["caller", "reusable"])
@pytest.mark.parametrize("replacement", ["inherit", {"UNDECLARED_TOKEN": "${{ secrets.OTHER }}"}])
def test_secret_routing_drift_is_rejected(
    tmp_path: Path, group: str, surface: str, replacement: object
) -> None:
    """Every group rejects inherited or undeclared secrets through the actual CLI."""
    shutil.copytree(ROOT / ".github", tmp_path / ".github")
    (tmp_path / "tools").mkdir()
    policy_path = ROOT / "tools/ci_workflow_ownership.json"
    shutil.copyfile(policy_path, tmp_path / "tools/ci_workflow_ownership.json")
    policy = json.loads(policy_path.read_text())
    relative = policy["coordinator"] if surface == "caller" else policy["groups"][group]["path"]
    path = tmp_path / relative
    workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
    if surface == "caller":
        workflow["jobs"][group]["secrets"] = replacement
        diagnostic = "caller secrets"
    else:
        workflow["on"]["workflow_call"]["secrets"] = replacement
        diagnostic = "reusable secret interface"
    path.write_text(yaml.safe_dump(workflow))
    result = subprocess.run(
        [sys.executable, str(TOOL), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode != 0
    assert f"{diagnostic}: {group}" in result.stderr


@pytest.mark.parametrize("group", ["python-tests", "rust-coverage"])
@pytest.mark.parametrize(
    ("surface", "replacement", "diagnostic"),
    [
        ("caller", {}, "caller secrets"),
        ("caller", {"CODECOV_TOKEN": "${{ secrets.OTHER }}"}, "caller secrets"),
        ("reusable", {}, "reusable secret interface"),
        (
            "reusable",
            {"secrets": {"CODECOV_TOKEN": {"required": "true"}}},
            "reusable secret interface",
        ),
    ],
)
def test_token_forwarding_and_declaration_are_exact(
    tmp_path: Path, group: str, surface: str, replacement: dict[str, object], diagnostic: str
) -> None:
    """Both token consumers reject removed forwarding, substituted values or required tokens."""
    shutil.copytree(ROOT / ".github", tmp_path / ".github")
    (tmp_path / "tools").mkdir()
    policy_path = ROOT / "tools/ci_workflow_ownership.json"
    shutil.copyfile(policy_path, tmp_path / "tools/ci_workflow_ownership.json")
    policy = json.loads(policy_path.read_text())
    relative = policy["coordinator"] if surface == "caller" else policy["groups"][group]["path"]
    path = tmp_path / relative
    workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
    if surface == "caller":
        workflow["jobs"][group]["secrets"] = replacement
    else:
        workflow["on"]["workflow_call"] = replacement
    path.write_text(yaml.safe_dump(workflow))
    result = subprocess.run(
        [sys.executable, str(TOOL), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode != 0
    assert f"{diagnostic}: {group}" in result.stderr
