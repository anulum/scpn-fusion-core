# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ----------------------------------------------------------------------
# SCPN Fusion Core -- Claims Audit Tests
# ----------------------------------------------------------------------
"""Tests for tools/claims_audit.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "claims_audit.py"
SPEC = importlib.util.spec_from_file_location("tools.claims_audit", MODULE_PATH)
assert SPEC and SPEC.loader
claims_audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = claims_audit
SPEC.loader.exec_module(claims_audit)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _manifest_path(tmp_path: Path, claims: list[dict[str, object]]) -> Path:
    path = tmp_path / "claims_manifest.json"
    _write_json(path, {"claims": claims})
    return path


def test_claims_manifest_passes_against_repo() -> None:
    manifest = ROOT / "validation" / "claims_manifest.json"
    claims = claims_audit.load_manifest(manifest)
    errors = claims_audit.run_audit(claims, ROOT)
    assert errors == []


def test_claims_audit_reports_missing_evidence_file(tmp_path: Path) -> None:
    path = _manifest_path(
        tmp_path,
        [
            {
                "id": "missing-evidence",
                "source_file": "README.md",
                "source_pattern": "SCPN Fusion Core",
                "evidence_files": ["does/not/exist.json"],
                "evidence_patterns": [],
            }
        ],
    )
    claims = claims_audit.load_manifest(path)
    errors = claims_audit.run_audit(claims, ROOT)
    assert any("evidence file missing" in err for err in errors)


def test_claims_audit_reports_missing_source_pattern(tmp_path: Path) -> None:
    path = _manifest_path(
        tmp_path,
        [
            {
                "id": "missing-source-pattern",
                "source_file": "README.md",
                "source_pattern": "THIS_PATTERN_SHOULD_NOT_EXIST_123456",
                "evidence_files": [],
                "evidence_patterns": [],
            }
        ],
    )
    claims = claims_audit.load_manifest(path)
    errors = claims_audit.run_audit(claims, ROOT)
    assert any("source pattern not found" in err for err in errors)


def test_claims_audit_reports_untracked_evidence_file() -> None:
    manifest = ROOT / "validation" / "claims_manifest.json"
    claims = claims_audit.load_manifest(manifest)
    errors = claims_audit.run_audit(claims, ROOT, tracked_files=set())
    assert any("evidence file not tracked by git" in err for err in errors)


def test_claims_audit_reports_source_and_pattern_errors(tmp_path: Path) -> None:
    source = tmp_path / "source.md"
    source.write_text("claim source\n", encoding="utf-8")
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("metric=0.9\n", encoding="utf-8")
    path = _manifest_path(
        tmp_path,
        [
            {
                "id": "missing-source",
                "source_file": "missing.md",
                "source_pattern": "anything",
                "evidence_files": [],
                "evidence_patterns": [],
            },
            {
                "id": "missing-pattern-file",
                "source_file": source.name,
                "source_pattern": "claim",
                "evidence_files": [],
                "evidence_patterns": [{"file": "missing-pattern.txt", "pattern": "metric"}],
            },
            {
                "id": "untracked-pattern-file",
                "source_file": source.name,
                "source_pattern": "claim",
                "evidence_files": [],
                "evidence_patterns": [{"file": evidence.name, "pattern": "metric"}],
            },
            {
                "id": "pattern-not-found",
                "source_file": source.name,
                "source_pattern": "claim",
                "evidence_files": [],
                "evidence_patterns": [{"file": evidence.name, "pattern": "absent"}],
            },
        ],
    )

    claims = claims_audit.load_manifest(path)
    errors = claims_audit.run_audit(
        claims,
        tmp_path,
        tracked_files={source.name, evidence.name} - {evidence.name},
    )
    pattern_errors = claims_audit.run_audit(
        claims[-1:],
        tmp_path,
        tracked_files={source.name, evidence.name},
    )

    joined = "\n".join(errors + pattern_errors)
    assert "source file missing" in joined
    assert "evidence pattern file missing" in joined
    assert "evidence pattern file not tracked by git" in joined
    assert "evidence pattern not found" in joined


def test_claims_audit_allows_git_tracking_fallback_none(tmp_path: Path) -> None:
    source = tmp_path / "source.md"
    source.write_text("claim source\n", encoding="utf-8")
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("metric=0.9\n", encoding="utf-8")
    path = _manifest_path(
        tmp_path,
        [
            {
                "id": "fallback",
                "source_file": source.name,
                "source_pattern": "claim",
                "evidence_files": [evidence.name],
                "evidence_patterns": [{"file": evidence.name, "pattern": "metric"}],
            }
        ],
    )

    claims = claims_audit.load_manifest(path)
    assert claims_audit.run_audit(claims, tmp_path, tracked_files=None) == []


def test_claims_manifest_defaults_optional_evidence_fields(tmp_path: Path) -> None:
    path = _manifest_path(
        tmp_path,
        [
            {
                "id": "minimal",
                "source_file": "source.md",
                "source_pattern": "claim",
                "evidence_patterns": None,
            }
        ],
    )

    claims = claims_audit.load_manifest(path)

    assert claims[0].evidence_files == ()
    assert claims[0].evidence_patterns == ()


def test_claims_manifest_schema_validation(tmp_path: Path) -> None:
    manifest = tmp_path / "claims_manifest.json"
    invalid_payloads: list[tuple[object, str]] = [
        ([], "JSON object"),
        ({}, "non-empty 'claims'"),
        ({"claims": []}, "non-empty 'claims'"),
        ({"claims": ["bad"]}, "must be an object"),
        ({"claims": [{"id": "", "source_file": "a", "source_pattern": "b"}]}, "non-empty"),
        ({"claims": [{"id": "x", "source_file": "", "source_pattern": "b"}]}, "non-empty"),
        ({"claims": [{"id": "x", "source_file": "a", "source_pattern": ""}]}, "non-empty"),
        (
            {
                "claims": [
                    {"id": "dup", "source_file": "a", "source_pattern": "b"},
                    {"id": "dup", "source_file": "c", "source_pattern": "d"},
                ]
            },
            "Duplicate claim id",
        ),
        (
            {
                "claims": [
                    {
                        "id": "x",
                        "source_file": "a",
                        "source_pattern": "b",
                        "evidence_files": "bad",
                    }
                ]
            },
            "list of strings",
        ),
        (
            {
                "claims": [
                    {
                        "id": "x",
                        "source_file": "a",
                        "source_pattern": "b",
                        "evidence_files": [""],
                    }
                ]
            },
            "non-empty",
        ),
        (
            {
                "claims": [
                    {
                        "id": "x",
                        "source_file": "a",
                        "source_pattern": "b",
                        "evidence_patterns": {},
                    }
                ]
            },
            "evidence_patterns",
        ),
        (
            {
                "claims": [
                    {
                        "id": "x",
                        "source_file": "a",
                        "source_pattern": "b",
                        "evidence_patterns": ["bad"],
                    }
                ]
            },
            "must be an object",
        ),
        (
            {
                "claims": [
                    {
                        "id": "x",
                        "source_file": "a",
                        "source_pattern": "b",
                        "evidence_patterns": [{"file": "", "pattern": "x"}],
                    }
                ]
            },
            "non-empty",
        ),
    ]

    for payload, message in invalid_payloads:
        _write_json(manifest, payload)
        with pytest.raises(ValueError, match=message):
            claims_audit.load_manifest(manifest)


def test_claims_audit_main_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.md"
    source.write_text("claim source\n", encoding="utf-8")
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("metric=0.9\n", encoding="utf-8")
    manifest = _manifest_path(
        tmp_path,
        [
            {
                "id": "claim",
                "source_file": source.name,
                "source_pattern": "claim",
                "evidence_files": [evidence.name],
                "evidence_patterns": [{"file": evidence.name, "pattern": "metric"}],
            }
        ],
    )
    monkeypatch.setattr(claims_audit, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(claims_audit, "_git_tracked_files", lambda repo_root: {evidence.name})

    assert claims_audit.main(["--manifest", "claims_manifest.json"]) == 0

    source.write_text("missing\n", encoding="utf-8")
    assert claims_audit.main(["--manifest", str(manifest)]) == 1

    with pytest.raises(FileNotFoundError, match="Claims manifest not found"):
        claims_audit.main(["--manifest", str(tmp_path / "missing.json")])


def test_git_tracked_files_uses_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs)
        return subprocess.CompletedProcess(cmd, 0, stdout="README.md\n", stderr="")

    monkeypatch.setattr(claims_audit.subprocess, "run", _fake_run)
    tracked = claims_audit._git_tracked_files(ROOT)
    assert tracked == {"README.md"}
    assert len(calls) == 1
    assert calls[0]["timeout"] == claims_audit._GIT_LS_FILES_TIMEOUT_SECONDS


def test_git_tracked_files_returns_none_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run(cmd: list[str], **kwargs: object) -> None:
        timeout_value = kwargs.get("timeout", 0.0)
        timeout = float(timeout_value) if isinstance(timeout_value, (int, float)) else 0.0
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr(claims_audit.subprocess, "run", _fake_run)
    assert claims_audit._git_tracked_files(ROOT) is None


def test_git_tracked_files_returns_none_on_os_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run(cmd: list[str], **kwargs: object) -> None:
        raise OSError("git unavailable")

    monkeypatch.setattr(claims_audit.subprocess, "run", _fake_run)
    assert claims_audit._git_tracked_files(ROOT) is None
