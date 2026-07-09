# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Fallback Budget Guard Tests
"""Tests for fallback-budget benchmark artifact gating."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "fallback_budget_guard.py"
SPEC = importlib.util.spec_from_file_location("fallback_budget_guard", MODULE_PATH)
assert SPEC and SPEC.loader
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _real_torax_payload(*, passes: bool = True) -> dict[str, Any]:
    """Return a minimal real-TORAX parity payload for fallback-budget tests."""
    return {
        "divergence_metrics": {
            "core_te_ratio_fine_over_torax": 0.07,
            "normalised_te_shape_rel_l2_fine": 0.02,
        },
        "passes_thresholds": passes,
        "physics_equivalence_claimed": False,
        "reference": {
            "profile_checksum_sha256": "a" * 64,
            "provenance": {
                "code": "TORAX",
                "config_name": "basic_config",
                "config_sha256": "b" * 64,
                "licence": "Apache-2.0",
                "torax_version": "1.4.3",
            },
        },
        "schema": guard.TORAX_REAL_PARITY_SCHEMA,
    }


def _passing_thresholds() -> dict[str, Any]:
    """Return minimal thresholds for successful legacy benchmark payloads."""
    return {
        "freegs": {"allowed_modes": ["freegs"]},
        "sparc": {
            "allowed_backends": ["neural_equilibrium"],
            "max_fallback_rate": 0.0,
            "preferred_backend": "neural_equilibrium",
        },
        "torax": {
            "allowed_backends": ["neural_transport"],
            "max_fallback_rate": 0.0,
            "preferred_backend": "neural_transport",
        },
    }


def test_evaluate_passes_when_all_budgets_satisfied() -> None:
    """Fallback budgets pass when every domain satisfies its configured limits."""
    summary = guard.evaluate(
        torax={
            "cases": [
                {"transport_backend": "neural_transport", "passes": True},
                {"transport_backend": "neural_transport", "passes": True},
            ]
        },
        sparc={
            "cases": [
                {"surrogate_backend": "reduced_order_proxy", "passes": True},
                {"surrogate_backend": "reduced_order_proxy", "passes": True},
            ]
        },
        freegs={
            "cases": [
                {"mode": "solovev_manufactured_source", "passes": True},
            ]
        },
        thresholds={
            "torax": {
                "preferred_backend": "neural_transport",
                "allowed_backends": ["neural_transport"],
                "max_fallback_rate": 0.0,
                "require_all_cases_pass": True,
            },
            "sparc": {
                "preferred_backend": "neural_equilibrium",
                "allowed_backends": ["neural_equilibrium", "reduced_order_proxy"],
                "max_fallback_rate": 1.0,
                "require_all_cases_pass": True,
            },
            "freegs": {
                "allowed_modes": ["solovev_manufactured_source", "freegs"],
                "require_all_cases_pass": True,
            },
        },
    )
    assert summary["overall_pass"] is True
    assert summary["torax"]["passes"] is True
    assert summary["sparc"]["passes"] is True
    assert summary["freegs"]["passes"] is True


def test_evaluate_ignores_diagnostic_non_gated_case_failures() -> None:
    """Diagnostic non-gated case failures do not fail domain acceptance."""
    summary = guard.evaluate(
        torax={"cases": [{"transport_backend": "neural_transport", "passes": True}]},
        sparc={
            "cases": [
                {
                    "surrogate_backend": "neural_equilibrium",
                    "gated": True,
                    "passes": True,
                },
                {
                    "surrogate_backend": "reduced_order_proxy",
                    "gated": False,
                    "passes": False,
                },
            ]
        },
        freegs={"cases": [{"mode": "solovev_manufactured_source", "passes": True}]},
        thresholds={
            "torax": {
                "preferred_backend": "neural_transport",
                "allowed_backends": ["neural_transport"],
                "max_fallback_rate": 0.0,
                "require_all_cases_pass": True,
            },
            "sparc": {
                "preferred_backend": "neural_equilibrium",
                "allowed_backends": ["neural_equilibrium", "reduced_order_proxy"],
                "max_fallback_rate": 1.0,
                "require_all_cases_pass": True,
            },
            "freegs": {
                "allowed_modes": ["solovev_manufactured_source", "freegs"],
                "require_all_cases_pass": True,
            },
        },
    )

    assert summary["sparc"]["passes"] is True
    assert summary["overall_pass"] is True


def test_evaluate_fails_when_torax_fallback_exceeds_budget() -> None:
    """TORAX fallback use fails when it exceeds the allowed budget."""
    summary = guard.evaluate(
        torax={"cases": [{"transport_backend": "analytic_fallback", "passes": True}]},
        sparc={"cases": [{"surrogate_backend": "reduced_order_proxy", "passes": True}]},
        freegs={"cases": [{"mode": "solovev_manufactured_source", "passes": True}]},
        thresholds={
            "torax": {
                "preferred_backend": "neural_transport",
                "allowed_backends": ["neural_transport", "analytic_fallback"],
                "max_fallback_rate": 0.0,
            },
            "sparc": {
                "preferred_backend": "neural_equilibrium",
                "allowed_backends": ["neural_equilibrium", "reduced_order_proxy"],
                "max_fallback_rate": 1.0,
            },
            "freegs": {"allowed_modes": ["solovev_manufactured_source", "freegs"]},
        },
    )
    assert summary["torax"]["passes"] is False
    assert summary["overall_pass"] is False


def test_evaluate_accepts_real_torax_parity_payload() -> None:
    """The current real-TORAX parity schema is accepted as the TORAX budget row."""
    summary = guard.evaluate(
        torax=_real_torax_payload(),
        sparc={"cases": [{"surrogate_backend": "neural_equilibrium", "passes": True}]},
        freegs={"cases": [{"mode": "solovev_manufactured_source", "passes": True}]},
        thresholds={
            "torax": {
                "allowed_backends": ["real_torax_reference"],
                "max_fallback_rate": 0.0,
                "min_case_count": 1,
                "min_preferred_backend_rate": 1.0,
                "preferred_backend": "real_torax_reference",
                "require_backend_requirement_satisfied": True,
                "require_all_cases_pass": True,
            },
            "sparc": {
                "allowed_backends": ["neural_equilibrium"],
                "max_fallback_rate": 0.0,
                "preferred_backend": "neural_equilibrium",
            },
            "freegs": {"allowed_modes": ["solovev_manufactured_source"]},
        },
    )

    assert summary["overall_pass"] is True
    assert summary["torax"]["artifact_schema"] == guard.TORAX_REAL_PARITY_SCHEMA
    assert summary["torax"]["case_count"] == 1
    assert summary["torax"]["observed_backends"] == ["real_torax_reference"]


def test_evaluate_rejects_invalid_real_torax_parity_payload() -> None:
    """Invalid real-TORAX parity payloads fail the TORAX budget row."""
    torax = _real_torax_payload(passes=False)

    summary = guard.evaluate(
        torax=torax,
        sparc={"cases": [{"surrogate_backend": "neural_equilibrium", "passes": True}]},
        freegs={"cases": [{"mode": "solovev_manufactured_source", "passes": True}]},
        thresholds={
            "torax": {
                "allowed_backends": ["real_torax_reference"],
                "max_fallback_rate": 0.0,
                "preferred_backend": "real_torax_reference",
                "require_backend_requirement_satisfied": True,
                "require_all_cases_pass": True,
            },
            "sparc": {
                "allowed_backends": ["neural_equilibrium"],
                "max_fallback_rate": 0.0,
                "preferred_backend": "neural_equilibrium",
            },
            "freegs": {"allowed_modes": ["solovev_manufactured_source"]},
        },
    )

    assert summary["overall_pass"] is False
    assert summary["torax"]["backend_requirement_satisfied_rate"] == 0.0
    assert summary["torax"]["passes"] is False


def test_main_writes_summary_and_returns_nonzero_on_failure(tmp_path: Path) -> None:
    """CLI execution writes a summary and returns non-zero on failed budgets."""
    torax = tmp_path / "torax.json"
    sparc = tmp_path / "sparc.json"
    freegs = tmp_path / "freegs.json"
    thresholds = tmp_path / "thresholds.json"
    summary_path = tmp_path / "summary.json"

    _write_json(torax, {"cases": [{"transport_backend": "analytic_fallback", "passes": True}]})
    _write_json(sparc, {"cases": [{"surrogate_backend": "reduced_order_proxy", "passes": True}]})
    _write_json(freegs, {"cases": [{"mode": "solovev_manufactured_source", "passes": True}]})
    _write_json(
        thresholds,
        {
            "torax": {
                "preferred_backend": "neural_transport",
                "allowed_backends": ["neural_transport", "analytic_fallback"],
                "max_fallback_rate": 0.0,
            },
            "sparc": {
                "preferred_backend": "neural_equilibrium",
                "allowed_backends": ["neural_equilibrium", "reduced_order_proxy"],
                "max_fallback_rate": 1.0,
            },
            "freegs": {"allowed_modes": ["solovev_manufactured_source", "freegs"]},
        },
    )
    rc = guard.main(
        [
            "--torax",
            str(torax),
            "--sparc",
            str(sparc),
            "--freegs",
            str(freegs),
            "--thresholds",
            str(thresholds),
            "--summary-json",
            str(summary_path),
        ]
    )
    assert rc == 1
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["overall_pass"] is False


def test_evaluate_requires_freegs_mode_when_available() -> None:
    """FreeGS mode is required when configured and the backend is available."""
    summary = guard.evaluate(
        torax={"cases": [{"transport_backend": "neural_transport", "passes": True}]},
        sparc={"cases": [{"surrogate_backend": "neural_equilibrium", "passes": True}]},
        freegs={
            "freegs_available": True,
            "force_solovev": False,
            "cases": [{"mode": "solovev_manufactured_source", "passes": True}],
        },
        thresholds={
            "torax": {
                "preferred_backend": "neural_transport",
                "allowed_backends": ["neural_transport"],
                "max_fallback_rate": 0.0,
            },
            "sparc": {
                "preferred_backend": "neural_equilibrium",
                "allowed_backends": ["neural_equilibrium"],
                "max_fallback_rate": 0.0,
            },
            "freegs": {
                "allowed_modes": ["solovev_manufactured_source", "freegs"],
                "require_freegs_mode_when_available": True,
                "min_freegs_cases_when_available": 1,
            },
        },
    )
    assert summary["freegs"]["passes"] is False
    assert summary["overall_pass"] is False


def test_evaluate_allows_force_solovev_even_when_freegs_is_available() -> None:
    """Forced Solov'ev mode remains allowed even when FreeGS is installed."""
    summary = guard.evaluate(
        torax={"cases": [{"transport_backend": "neural_transport", "passes": True}]},
        sparc={"cases": [{"surrogate_backend": "neural_equilibrium", "passes": True}]},
        freegs={
            "freegs_available": True,
            "force_solovev": True,
            "cases": [{"mode": "solovev_manufactured_source", "passes": True}],
        },
        thresholds={
            "torax": {
                "preferred_backend": "neural_transport",
                "allowed_backends": ["neural_transport"],
                "max_fallback_rate": 0.0,
            },
            "sparc": {
                "preferred_backend": "neural_equilibrium",
                "allowed_backends": ["neural_equilibrium"],
                "max_fallback_rate": 0.0,
            },
            "freegs": {
                "allowed_modes": ["solovev_manufactured_source", "freegs"],
                "require_freegs_mode_when_available": True,
                "min_freegs_cases_when_available": 1,
            },
        },
    )
    assert summary["freegs"]["passes"] is True
    assert summary["overall_pass"] is True


def test_evaluate_enforces_preferred_backend_rate_when_configured() -> None:
    """Preferred backend rate thresholds fail when the observed rate is too low."""
    summary = guard.evaluate(
        torax={
            "cases": [
                {"transport_backend": "analytic_fallback", "passes": True},
                {"transport_backend": "neural_transport", "passes": True},
            ]
        },
        sparc={"cases": [{"surrogate_backend": "neural_equilibrium", "passes": True}]},
        freegs={"cases": [{"mode": "freegs", "reference_backend": "freegs", "passes": True}]},
        thresholds={
            "torax": {
                "preferred_backend": "neural_transport",
                "allowed_backends": ["neural_transport", "analytic_fallback"],
                "max_fallback_rate": 1.0,
                "min_preferred_backend_rate": 1.0,
            },
            "sparc": {
                "preferred_backend": "neural_equilibrium",
                "allowed_backends": ["neural_equilibrium"],
                "max_fallback_rate": 0.0,
            },
            "freegs": {
                "allowed_modes": ["freegs"],
            },
        },
    )
    assert summary["torax"]["preferred_backend_rate"] < 1.0
    assert summary["torax"]["passes"] is False
    assert summary["overall_pass"] is False


def test_helper_validation_branches_are_explicit(tmp_path: Path) -> None:
    """Helper branches reject invalid payloads and handle empty case lists."""
    assert guard._resolve("artifacts/example.json") == ROOT / "artifacts" / "example.json"
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="expected JSON object"):
        guard._load_json(invalid_json)

    assert (
        guard._fallback_rate_for_cases(
            [],
            backend_key="transport_backend",
            preferred_backend="neural_transport",
        )
        == 1.0
    )
    assert (
        guard._rate_for_backend(
            [],
            backend_key="transport_backend",
            preferred_backend="neural_transport",
        )
        == 0.0
    )


def test_evaluate_rejects_missing_domain_cases() -> None:
    """Each benchmark domain must provide at least one evaluable row."""
    sparc = {"cases": [{"surrogate_backend": "neural_equilibrium", "passes": True}]}
    freegs = {"cases": [{"mode": "freegs", "passes": True}]}
    torax = {"cases": [{"transport_backend": "neural_transport", "passes": True}]}

    with pytest.raises(ValueError, match="torax benchmark artifact has zero cases"):
        guard.evaluate(torax={}, sparc=sparc, freegs=freegs, thresholds=_passing_thresholds())
    with pytest.raises(ValueError, match="sparc benchmark artifact has zero cases"):
        guard.evaluate(torax=torax, sparc={}, freegs=freegs, thresholds=_passing_thresholds())
    with pytest.raises(ValueError, match="freegs benchmark artifact has zero cases"):
        guard.evaluate(torax=torax, sparc=sparc, freegs={}, thresholds=_passing_thresholds())


def test_evaluate_applies_strict_flags_and_runtime_normalisation() -> None:
    """Strict backend flags and malformed runtime telemetry are normalised."""
    summary = guard.evaluate(
        torax={
            "cases": [{"transport_backend": "neural_transport", "passes": True}],
            "require_neural_transport": True,
        },
        sparc={
            "cases": [{"surrogate_backend": "neural_equilibrium", "passes": True}],
            "require_neural_backend": True,
        },
        freegs={
            "cases": [{"mode": "freegs", "reference_backend": "freegs", "passes": True}],
            "freegs_available": True,
            "require_freegs_backend": True,
        },
        thresholds={
            **_passing_thresholds(),
            "freegs": {
                "allowed_modes": ["freegs"],
                "require_freegs_mode_when_available": True,
            },
            "runtime": {
                "max_domain_events": ["bad"],
                "max_total_events": 0,
            },
        },
        runtime_telemetry={"domain_counts": ["bad"], "total_count": 0},
    )

    assert summary["overall_pass"] is True
    assert summary["torax"]["strict_requested"] is True
    assert summary["sparc"]["strict_requested"] is True
    assert summary["freegs"]["strict_requested"] is True
    assert summary["runtime"]["domain_counts"] == {}
    assert summary["runtime"]["max_domain_events"] == {}


def test_main_writes_successful_summary(tmp_path: Path) -> None:
    """CLI execution returns zero and writes a passing summary when all gates pass."""
    torax = tmp_path / "torax.json"
    sparc = tmp_path / "sparc.json"
    freegs = tmp_path / "freegs.json"
    thresholds = tmp_path / "thresholds.json"
    summary_path = tmp_path / "summary.json"

    _write_json(torax, _real_torax_payload())
    _write_json(sparc, {"cases": [{"surrogate_backend": "neural_equilibrium", "passes": True}]})
    _write_json(freegs, {"cases": [{"mode": "freegs", "passes": True}]})
    _write_json(
        thresholds,
        {
            "freegs": {"allowed_modes": ["freegs"]},
            "sparc": {
                "allowed_backends": ["neural_equilibrium"],
                "max_fallback_rate": 0.0,
                "preferred_backend": "neural_equilibrium",
            },
            "torax": {
                "allowed_backends": ["real_torax_reference"],
                "max_fallback_rate": 0.0,
                "preferred_backend": "real_torax_reference",
            },
        },
    )

    rc = guard.main(
        [
            "--freegs",
            str(freegs),
            "--sparc",
            str(sparc),
            "--summary-json",
            str(summary_path),
            "--thresholds",
            str(thresholds),
            "--torax",
            str(torax),
        ]
    )

    assert rc == 0
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["overall_pass"] is True
