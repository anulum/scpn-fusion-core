# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Fallback Budget Guard Tests

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "fallback_budget_guard.py"
SPEC = importlib.util.spec_from_file_location("fallback_budget_guard", MODULE_PATH)
assert SPEC and SPEC.loader
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_evaluate_passes_when_all_budgets_satisfied() -> None:
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


def test_evaluate_fails_when_torax_fallback_exceeds_budget() -> None:
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


def test_main_writes_summary_and_returns_nonzero_on_failure(tmp_path: Path) -> None:
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
