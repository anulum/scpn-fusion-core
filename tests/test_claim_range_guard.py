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
from dataclasses import replace
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


def test_claim_range_guard_reports_invalid_public_check_contract() -> None:
    """Return a failure for a malformed public RangeCheck even with assertions disabled."""
    checks = claim_range_guard.load_checks(ROOT / "validation/claim_range_thresholds.json")
    malformed = replace(checks[0], path=None, ratio=None)
    errors, summary = claim_range_guard.run_checks((malformed,), repo_root=ROOT)
    assert errors == [f"[{malformed.check_id}] check has neither a path nor a ratio."]
    assert summary["failed_checks"] == 1


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("schema",), "scpn-fusion-core.stress-campaign-results.v2"),
        (("campaign_complete",), True),
        (("promotion_eligible",), True),
        (("promotion_eligible",), 0),
        (("campaign_complete",), 0),
        (("promotion_eligible",), "false"),
        (
            ("campaign_identity", "payload", "evaluation_contract", "payload", "evidence_scope"),
            "promotion",
        ),
        (("controllers", "Rust-PID", "policy_implementation"), "different.plant.controller"),
    ],
)
def test_current_campaign_claim_contract_rejects_relabelled_evidence(
    tmp_path: Path, path: tuple[str, ...], replacement: object
) -> None:
    """Reject promotion or schema/controller substitution in the frozen real report."""
    relative = "validation/reports/stress_test_campaign.json"
    payload = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    target = payload
    for token in path[:-1]:
        target = target[token]
    target[path[-1]] = replacement
    artifact = tmp_path / relative
    artifact.parent.mkdir(parents=True)
    _write_json(artifact, payload)
    checks = tuple(
        check
        for check in claim_range_guard.load_checks(ROOT / "validation/claim_range_thresholds.json")
        if check.file == relative
    )
    errors, _ = claim_range_guard.run_checks(checks, repo_root=tmp_path)
    assert any("expected ==" in error for error in errors), errors


def test_historical_latency_contract_retains_observations_not_release_bounds() -> None:
    """Keep legacy latency outside live release guarantees and preserve all other bounds."""
    checks = claim_range_guard.load_checks(ROOT / "validation/claim_range_thresholds.json")
    retired = {
        "rust_pid_p50_latency_us",
        "rust_pid_speedup_vs_python_pid",
        "rust_pid_disruption_rate",
    }
    assert retired.isdisjoint(check.check_id for check in checks)
    historical = [check for check in checks if "historical_controller_latency.json" in check.file]
    assert historical
    assert all(
        check.minimum is None and check.maximum is None and check.ratio is None
        for check in historical
    )
    by_id = {check.check_id: check for check in checks}
    assert by_id["qlknn_test_relative_l2"].maximum == 0.25
    assert by_id["real_shot_disruption_recall"].minimum == 0.6
    assert by_id["real_shot_disruption_fpr"].maximum == 0.4
    assert by_id["freegs_overall_psi_nrmse"].maximum == 0.01


@pytest.mark.parametrize("substitution", ["missing", "historical", "different_method"])
def test_controller_claim_contract_rejects_missing_or_incompatible_artifacts(
    tmp_path: Path, substitution: str
) -> None:
    """Reject missing evidence, legacy-for-current substitution and altered historical timers."""
    historical_path = "papers/submissions/002_neuromorphic_vertical_stability_control/evidence/historical_controller_latency.json"
    current_path = "validation/reports/stress_test_campaign.json"
    checks = tuple(
        check
        for check in claim_range_guard.load_checks(ROOT / "validation/claim_range_thresholds.json")
        if check.file in {historical_path, current_path}
    )
    for relative in (historical_path, current_path):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    errors, _ = claim_range_guard.run_checks(checks, repo_root=tmp_path)
    assert errors == []
    if substitution == "missing":
        (tmp_path / current_path).unlink()
    elif substitution == "historical":
        (tmp_path / current_path).write_bytes((ROOT / historical_path).read_bytes())
    else:
        historical = json.loads((ROOT / historical_path).read_text(encoding="utf-8"))
        historical["methodology"]["latency_metric"] = "policy_only_kernel_timer"
        _write_json(tmp_path / historical_path, historical)
    errors, summary = claim_range_guard.run_checks(checks, repo_root=tmp_path)
    assert errors
    assert summary["failed_checks"] > 0


def test_controller_claim_registries_share_exact_evidence_scope() -> None:
    """Require public claim and metric registries to bind the same historical/current files."""
    manifest = json.loads((ROOT / "validation/claims_manifest.json").read_text(encoding="utf-8"))
    claims = {claim["id"]: claim for claim in manifest["claims"]}
    checks = claim_range_guard.load_checks(ROOT / "validation/claim_range_thresholds.json")
    evidence = set(claims["readme_controller_promotion_withheld"]["evidence_files"])
    assert evidence == {
        check.file
        for check in checks
        if check.check_id.startswith(("historical_", "current_controller_", "current_rust_pid_"))
    }
    assert set(claims["readme_historical_controller_latency_boundary"]["evidence_files"]) < evidence


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
