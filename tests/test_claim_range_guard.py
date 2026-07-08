# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ----------------------------------------------------------------------
# SCPN Fusion Core -- Claim Range Guard Tests
# ----------------------------------------------------------------------
"""Tests for tools/claim_range_guard.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "claim_range_guard.py"
SPEC = importlib.util.spec_from_file_location("tools.claim_range_guard", MODULE_PATH)
assert SPEC and SPEC.loader
claim_range_guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = claim_range_guard
SPEC.loader.exec_module(claim_range_guard)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_config(path: Path, checks: list[dict[str, object]]) -> None:
    _write_json(path, {"checks": checks})


def test_claim_range_guard_passes_with_repo_config() -> None:
    config = ROOT / "validation" / "claim_range_thresholds.json"
    checks = claim_range_guard.load_checks(config)
    errors, summary = claim_range_guard.run_checks(checks, repo_root=ROOT)
    assert errors == []
    assert summary["failed_checks"] == 0


def test_claim_range_guard_reports_threshold_violation(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    _write_json(artifact, {"metric": 5.0})
    config = tmp_path / "config.json"
    _write_config(
        config,
        [
            {
                "id": "max-check",
                "file": artifact.name,
                "path": ["metric"],
                "max": 1.0,
            }
        ],
    )

    checks = claim_range_guard.load_checks(config)
    errors, summary = claim_range_guard.run_checks(checks, repo_root=tmp_path)
    assert len(errors) == 1
    assert "expected <= 1.0" in errors[0]
    assert summary["failed_checks"] == 1


def test_claim_range_guard_reports_missing_path(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    _write_json(artifact, {"present": 1.0})
    config = tmp_path / "config.json"
    _write_config(
        config,
        [
            {
                "id": "missing-path",
                "file": artifact.name,
                "path": ["missing", "field"],
                "min": 0.0,
            }
        ],
    )

    checks = claim_range_guard.load_checks(config)
    errors, summary = claim_range_guard.run_checks(checks, repo_root=tmp_path)
    assert len(errors) == 1
    assert "missing key" in errors[0]
    assert summary["failed_checks"] == 1


def test_claim_range_guard_checks_ratio_equality_and_cache(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    _write_json(
        artifact,
        {
            "counts": {"passed": 9, "total": 10},
            "flags": [{"name": "claim_ready", "enabled": True}],
            "label": "accepted",
        },
    )
    config = tmp_path / "config.json"
    _write_config(
        config,
        [
            {
                "id": "ratio-check",
                "file": artifact.name,
                "ratio": {
                    "numerator": ["counts", "passed"],
                    "denominator": ["counts", "total"],
                },
                "min": 0.9,
                "max": 0.9,
                "description": "ratio branch",
            },
            {
                "id": "bool-check",
                "file": artifact.name,
                "path": ["flags", 0, "enabled"],
                "equals": True,
            },
            {
                "id": "string-check",
                "file": artifact.name,
                "path": ["label"],
                "equals": "accepted",
            },
        ],
    )

    checks = claim_range_guard.load_checks(config)
    errors, summary = claim_range_guard.run_checks(checks, repo_root=tmp_path)

    assert errors == []
    assert summary["total_checks"] == 3
    assert summary["checks"][0]["observed"] == 0.9
    assert summary["checks"][0]["ratio"] == {
        "numerator": ["counts", "passed"],
        "denominator": ["counts", "total"],
    }
    assert summary["checks"][1]["path"] == ["flags", 0, "enabled"]


def test_claim_range_guard_reports_runtime_artifact_errors(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    wrong_shape = tmp_path / "wrong_shape.json"
    _write_json(wrong_shape, {"items": {"not": "a list"}, "metric": "not numeric", "short": [1]})
    config = tmp_path / "config.json"
    _write_config(
        config,
        [
            {"id": "missing-file", "file": "missing.json", "path": ["metric"], "min": 0},
            {"id": "bad-json", "file": bad_json.name, "path": ["metric"], "min": 0},
            {"id": "expected-object", "file": wrong_shape.name, "path": ["metric", "x"], "min": 0},
            {"id": "expected-list", "file": wrong_shape.name, "path": ["items", 0], "min": 0},
            {"id": "index-range", "file": wrong_shape.name, "path": ["short", 2], "min": 0},
            {"id": "nonnumeric-min", "file": wrong_shape.name, "path": ["metric"], "min": 0},
            {"id": "numeric-equals", "file": wrong_shape.name, "path": ["metric"], "equals": 1},
        ],
    )

    checks = claim_range_guard.load_checks(config)
    errors, summary = claim_range_guard.run_checks(checks, repo_root=tmp_path)

    joined = "\n".join(errors)
    assert "missing.json" in joined
    assert "Expecting property name" in joined
    assert "expected object before key" in joined
    assert "expected list before index" in joined
    assert "index out of range" in joined
    assert "observed value is not numeric" in joined
    assert summary["failed_checks"] == 7


def test_claim_range_guard_reports_ratio_errors(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    _write_json(
        artifact,
        {
            "ratio": {"num": 5, "zero": 0, "text": "bad", "nan": float("nan")},
        },
    )
    config = tmp_path / "config.json"
    _write_config(
        config,
        [
            {
                "id": "zero-denominator",
                "file": artifact.name,
                "ratio": {"numerator": ["ratio", "num"], "denominator": ["ratio", "zero"]},
                "min": 0,
            },
            {
                "id": "nonnumeric-numerator",
                "file": artifact.name,
                "ratio": {"numerator": ["ratio", "text"], "denominator": ["ratio", "num"]},
                "min": 0,
            },
            {
                "id": "nonfinite-denominator",
                "file": artifact.name,
                "ratio": {"numerator": ["ratio", "num"], "denominator": ["ratio", "nan"]},
                "min": 0,
            },
        ],
    )

    checks = claim_range_guard.load_checks(config)
    errors, summary = claim_range_guard.run_checks(checks, repo_root=tmp_path)

    joined = "\n".join(errors)
    assert "denominator is zero" in joined
    assert "observed value is not numeric" in joined
    assert "observed value is not finite" in joined
    assert summary["failed_checks"] == 3


def test_claim_range_guard_reports_equality_and_minimum_failures(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    _write_json(artifact, {"metric": 2.0, "label": "wrong"})
    config = tmp_path / "config.json"
    _write_config(
        config,
        [
            {"id": "numeric-equals", "file": artifact.name, "path": ["metric"], "equals": 2.1},
            {"id": "string-equals", "file": artifact.name, "path": ["label"], "equals": "right"},
            {"id": "minimum", "file": artifact.name, "path": ["metric"], "min": 3.0},
        ],
    )

    checks = claim_range_guard.load_checks(config)
    errors, summary = claim_range_guard.run_checks(checks, repo_root=tmp_path)

    joined = "\n".join(errors)
    assert "expected == 2.1" in joined
    assert "expected == 'right'" in joined
    assert "expected >= 3.0" in joined
    assert summary["failed_checks"] == 3


def test_claim_range_guard_validates_config_schema(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    invalid_payloads: list[tuple[object, str]] = [
        ([], "JSON object"),
        ({}, "non-empty 'checks'"),
        ({"checks": []}, "non-empty 'checks'"),
        ({"checks": ["bad"]}, "must be an object"),
        ({"checks": [{"id": "", "file": "a", "path": ["x"], "min": 0}]}, "non-empty string"),
        ({"checks": [{"id": "x", "file": "", "path": ["x"], "min": 0}]}, "non-empty string"),
        ({"checks": [{"id": "x", "file": "a", "min": 0}]}, "either 'path' or 'ratio'"),
        (
            {"checks": [{"id": "x", "file": "a", "path": ["x"], "ratio": {}, "min": 0}]},
            "cannot define both",
        ),
        ({"checks": [{"id": "x", "file": "a", "path": [], "min": 0}]}, "non-empty list"),
        ({"checks": [{"id": "x", "file": "a", "path": [True], "min": 0}]}, "token"),
        ({"checks": [{"id": "x", "file": "a", "path": [""], "min": 0}]}, "token"),
        ({"checks": [{"id": "x", "file": "a", "path": ["m"]}]}, "at least one"),
        ({"checks": [{"id": "x", "file": "a", "path": ["m"], "min": "bad"}]}, "finite number"),
        ({"checks": [{"id": "x", "file": "a", "path": ["m"], "min": float("nan")}]}, "finite"),
        ({"checks": [{"id": "x", "file": "a", "path": ["m"], "min": 2, "max": 1}]}, "min"),
        (
            {"checks": [{"id": "x", "file": "a", "path": ["m"], "equals": {"bad": "shape"}}]},
            "equals",
        ),
        (
            {
                "checks": [
                    {"id": "dup", "file": "a", "path": ["m"], "min": 0},
                    {"id": "dup", "file": "a", "path": ["n"], "min": 0},
                ]
            },
            "Duplicate check id",
        ),
        (
            {"checks": [{"id": "x", "file": "a", "ratio": [], "min": 0}]},
            "ratio",
        ),
    ]

    for payload, message in invalid_payloads:
        _write_json(config, payload)
        with pytest.raises(ValueError, match=message):
            claim_range_guard.load_checks(config)


def test_claim_range_guard_main_writes_summary_and_reports_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact.json"
    _write_json(artifact, {"metric": 5.0})
    config = tmp_path / "config.json"
    summary_path = tmp_path / "summary" / "claim_range.json"
    _write_config(
        config,
        [{"id": "max-check", "file": artifact.name, "path": ["metric"], "max": 1.0}],
    )
    monkeypatch.setattr(claim_range_guard, "REPO_ROOT", tmp_path)

    rc = claim_range_guard.main(
        [
            "--config",
            str(config),
            "--summary-json",
            "summary/claim_range.json",
        ]
    )

    assert rc == 1
    assert json.loads(summary_path.read_text(encoding="utf-8"))["failed_checks"] == 1


def test_claim_range_guard_main_resolves_relative_config_and_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact.json"
    _write_json(artifact, {"metric": 1.0})
    config = tmp_path / "config.json"
    _write_config(
        config,
        [{"id": "min-check", "file": artifact.name, "path": ["metric"], "min": 1.0}],
    )
    monkeypatch.setattr(claim_range_guard, "REPO_ROOT", tmp_path)

    assert claim_range_guard.main(["--config", "config.json"]) == 0


def test_claim_range_guard_main_rejects_missing_config(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Claim range config not found"):
        claim_range_guard.main(["--config", str(tmp_path / "missing.json")])
