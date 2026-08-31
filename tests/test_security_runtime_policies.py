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

from tools.validate_cyclonedx_sbom import SbomValidationError, validate_sbom
from tools.sanitize_scorecard_sarif import (
    PLACEHOLDER_URI,
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


def _write_sbom(tmp_path: Path, payload: object) -> Path:
    """Write one deterministic SBOM fixture for schema-validation tests."""
    path = tmp_path / "sbom.cdx.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cyclonedx_validator_accepts_declared_valid_schema(tmp_path: Path) -> None:
    """A complete minimal CycloneDX document is admitted."""
    path = _write_sbom(
        tmp_path,
        {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "serialNumber": "urn:uuid:12345678-1234-4234-8234-123456789abc",
            "version": 1,
            "components": [],
        },
    )

    assert validate_sbom(path).to_version() == "1.6"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "root must be a JSON object"),
        ({"bomFormat": "SPDX", "specVersion": "1.6"}, "bomFormat"),
        ({"bomFormat": "CycloneDX", "specVersion": 1.6}, "specVersion must be"),
        ({"bomFormat": "CycloneDX", "specVersion": "99.0"}, "unsupported"),
        (
            {"bomFormat": "CycloneDX", "specVersion": "1.6", "version": 0},
            "validation failed",
        ),
    ],
)
def test_cyclonedx_validator_rejects_invalid_documents(
    tmp_path: Path, payload: object, message: str
) -> None:
    """Format, version, and schema violations all fail closed."""
    with pytest.raises(SbomValidationError, match=message):
        validate_sbom(_write_sbom(tmp_path, payload))


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


def _scorecard_sarif(*locations: object) -> dict[str, object]:
    return {
        "version": "2.1.0",
        "runs": [{"results": [{"locations": list(locations)}]}],
    }


def test_scorecard_sarif_removes_only_placeholder_locations() -> None:
    """Repository-only findings lose the invalid upstream pseudo-URI."""
    real_location = {"physicalLocation": {"artifactLocation": {"uri": "src/scpn_fusion/core.py"}}}
    placeholder = {"physicalLocation": {"artifactLocation": {"uri": PLACEHOLDER_URI}}}
    payload = _scorecard_sarif(placeholder, real_location)

    assert sanitize_scorecard_sarif(payload) == 1
    result = payload["runs"][0]["results"][0]  # type: ignore[index]
    assert result["locations"] == [real_location]


def test_scorecard_sarif_removes_empty_locations_member() -> None:
    """A result with no file association remains valid as a global result."""
    placeholder = {"physicalLocation": {"artifactLocation": {"uri": PLACEHOLDER_URI}}}
    payload = _scorecard_sarif(placeholder)

    assert sanitize_scorecard_sarif(payload) == 1
    result = payload["runs"][0]["results"][0]  # type: ignore[index]
    assert "locations" not in result


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
    path = tmp_path / "results.sarif"
    path.write_text(json.dumps(_scorecard_sarif(placeholder)), encoding="utf-8")

    assert sanitize_file(path) == 1
    first = path.read_text(encoding="utf-8")
    assert sanitize_file(path) == 0
    assert path.read_text(encoding="utf-8") == first


def test_scorecard_workflow_sanitizes_before_upload() -> None:
    """Both the artifact and code-scanning upload consume sanitized SARIF."""
    workflow = (ROOT / ".github/workflows/scorecard.yml").read_text(encoding="utf-8")
    sanitize_at = workflow.index("python tools/sanitize_scorecard_sarif.py results.sarif")
    artifact_at = workflow.index("uses: actions/upload-artifact@")
    sarif_at = workflow.index("uses: github/codeql-action/upload-sarif@")
    assert sanitize_at < artifact_at < sarif_at
