# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Runtime security policy guards for deserialization and subprocess usage."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tools.sanitize_scorecard_sarif import (
    PLACEHOLDER_URI,
    REPOSITORY_ANCHORS,
    ScorecardSarifError,
    sanitize_file,
    sanitize_scorecard_sarif,
)


ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("src", "validation", "tools")


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for rel in SCAN_DIRS:
        base = ROOT / rel
        if not base.exists():
            continue
        files.extend(sorted(base.rglob("*.py")))
    return files


def test_numpy_load_always_disables_pickle() -> None:
    violations: list[str] = []
    for path in _iter_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "load":
                continue
            receiver = node.func.value
            is_numpy_call = (
                isinstance(receiver, ast.Name)
                and receiver.id == "np"
                or isinstance(receiver, ast.Name)
                and receiver.id == "numpy"
            )
            if not is_numpy_call:
                continue
            allow_kw = next((kw for kw in node.keywords if kw.arg == "allow_pickle"), None)
            if allow_kw is None:
                violations.append(f"{path}:{node.lineno} missing allow_pickle=False")
                continue
            if not isinstance(allow_kw.value, ast.Constant) or allow_kw.value.value is not False:
                violations.append(f"{path}:{node.lineno} allow_pickle must be False")
    assert not violations, "\n".join(violations)


def test_subprocess_calls_do_not_enable_shell_mode() -> None:
    violations: list[str] = []
    for path in _iter_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {"run", "Popen"}:
                continue
            if not isinstance(node.func.value, ast.Name) or node.func.value.id != "subprocess":
                continue
            shell_kw = next((kw for kw in node.keywords if kw.arg == "shell"), None)
            if shell_kw is None:
                continue
            if isinstance(shell_kw.value, ast.Constant) and shell_kw.value.value is True:
                violations.append(f"{path}:{node.lineno} subprocess.{node.func.attr}(shell=True)")
    assert not violations, "\n".join(violations)


def test_security_and_sbom_workflows_are_warning_clean() -> None:
    """Scheduled security workflows configure Git and validate every SBOM."""
    security = (ROOT / ".github/workflows/security-audit.yml").read_text(encoding="utf-8")
    sbom = (ROOT / ".github/workflows/sbom.yml").read_text(encoding="utf-8")

    for workflow in (security, sbom):
        assert 'GIT_CONFIG_COUNT: "1"' in workflow
        assert "GIT_CONFIG_KEY_0: init.defaultBranch" in workflow
        assert "GIT_CONFIG_VALUE_0: main" in workflow
    assert "--no-validate" not in sbom
    assert "python tools/validate_cyclonedx_sbom.py" in sbom
    assert "python -m pytest -q tests/test_cyclonedx_sbom_validator.py" in sbom
    assert "artifacts/sbom-python.cdx.json" in sbom
    assert "artifacts/sbom-rust/*.cdx.json" in sbom


def test_every_checkout_workflow_configures_default_branch() -> None:
    """Every checkout is protected from Git's runner-local branch-name hint."""
    workflow_dir = ROOT / ".github/workflows"
    violations: list[str] = []
    for path in sorted(workflow_dir.glob("*.yml")):
        workflow = path.read_text(encoding="utf-8")
        if "uses: actions/checkout@" not in workflow:
            continue
        for setting in (
            'GIT_CONFIG_COUNT: "1"',
            "GIT_CONFIG_KEY_0: init.defaultBranch",
            "GIT_CONFIG_VALUE_0: main",
        ):
            if setting not in workflow:
                violations.append(f"{path.name}: missing {setting}")
    assert not violations, "\n".join(violations)


def _scorecard_sarif(rule_id: str, *locations: object) -> dict[str, object]:
    return {
        "version": "2.1.0",
        "runs": [{"results": [{"ruleId": rule_id, "locations": list(locations)}]}],
    }


def test_scorecard_sarif_maps_placeholder_and_preserves_real_location() -> None:
    """Repository findings gain a real anchor without changing real locations."""
    real_location = {"physicalLocation": {"artifactLocation": {"uri": "src/scpn_fusion/core.py"}}}
    placeholder = {"physicalLocation": {"artifactLocation": {"uri": PLACEHOLDER_URI}}}
    payload = _scorecard_sarif("VulnerabilitiesID", placeholder, real_location)

    assert sanitize_scorecard_sarif(payload) == 1
    result = payload["runs"][0]["results"][0]  # type: ignore[index]
    locations = result["locations"]
    assert locations[0]["physicalLocation"]["artifactLocation"]["uri"] == "SECURITY.md"
    assert locations[1] == real_location
    assert result["properties"] == {
        "scpn.repositoryLevelAnchor": "SECURITY.md",
        "scpn.originalArtifactLocationUri": PLACEHOLDER_URI,
    }


def test_scorecard_sarif_maps_code_review_to_contributing_policy() -> None:
    """The code-review finding uses its semantically relevant policy anchor."""
    placeholder = {"physicalLocation": {"artifactLocation": {"uri": PLACEHOLDER_URI}}}
    payload = _scorecard_sarif("CodeReviewID", placeholder)

    assert sanitize_scorecard_sarif(payload) == 1
    result = payload["runs"][0]["results"][0]  # type: ignore[index]
    assert (
        result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == "CONTRIBUTING.md"
    )


def test_scorecard_sarif_rejects_unknown_repository_finding() -> None:
    """Future placeholder-bearing rules require an explicit truthful anchor."""
    placeholder = {"physicalLocation": {"artifactLocation": {"uri": PLACEHOLDER_URI}}}
    payload = _scorecard_sarif("UnknownRuleID", placeholder)

    with pytest.raises(ScorecardSarifError, match="unmapped repository-level"):
        sanitize_scorecard_sarif(payload)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"version": "2.0.0", "runs": []},
        {"version": "2.1.0", "runs": {}},
        {"version": "2.1.0", "runs": [{"results": {}}]},
    ],
)
def test_scorecard_sarif_rejects_invalid_structure(payload: object) -> None:
    """Malformed or unexpected SARIF fails closed before mutation."""
    with pytest.raises(ScorecardSarifError):
        sanitize_scorecard_sarif(payload)


def test_scorecard_sarif_file_replacement_is_deterministic(tmp_path: Path) -> None:
    """The CLI-facing file operation emits stable UTF-8 JSON."""
    placeholder = {"physicalLocation": {"artifactLocation": {"uri": PLACEHOLDER_URI}}}
    for anchor in REPOSITORY_ANCHORS.values():
        (tmp_path / anchor).write_text("policy\n", encoding="utf-8")
    path = tmp_path / "results.sarif"
    path.write_text(
        json.dumps(_scorecard_sarif("VulnerabilitiesID", placeholder)), encoding="utf-8"
    )

    assert sanitize_file(path) == 1
    first = path.read_text(encoding="utf-8")
    assert sanitize_file(path) == 0
    assert path.read_text(encoding="utf-8") == first


def test_scorecard_sarif_file_requires_tracked_policy_anchors(tmp_path: Path) -> None:
    """The CLI fails before writing when a configured anchor is absent."""
    placeholder = {"physicalLocation": {"artifactLocation": {"uri": PLACEHOLDER_URI}}}
    path = tmp_path / "results.sarif"
    original = json.dumps(_scorecard_sarif("VulnerabilitiesID", placeholder))
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ScorecardSarifError, match="repository anchor is missing"):
        sanitize_file(path)
    assert path.read_text(encoding="utf-8") == original


def test_scorecard_workflow_is_publishable_and_sanitizes_code_scanning_upload() -> None:
    """Keep the attested producer uses-only and sanitize its downstream SARIF."""
    workflow = (ROOT / ".github/workflows/scorecard.yml").read_text(encoding="utf-8")
    analysis_at = workflow.index("  analysis:")
    scorecard_at = workflow.index("uses: ossf/scorecard-action@")
    raw_artifact_at = workflow.index("name: scorecard-raw-results")
    upload_job_at = workflow.index("  upload-sarif:")
    sanitize_at = workflow.index("python tools/sanitize_scorecard_sarif.py results.sarif")
    sanitized_artifact_at = workflow.index("name: scorecard-results\n", sanitize_at)
    sarif_at = workflow.index("uses: github/codeql-action/upload-sarif@")

    analysis_job = workflow[analysis_at:upload_job_at]
    assert "run:" not in analysis_job
    assert scorecard_at < raw_artifact_at < upload_job_at
    assert upload_job_at < sanitize_at < sanitized_artifact_at < sarif_at
