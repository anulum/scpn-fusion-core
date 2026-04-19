# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "deprecated_default_lane_guard.py"
SPEC = importlib.util.spec_from_file_location("deprecated_default_lane_guard", MODULE_PATH)
assert SPEC and SPEC.loader
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


def test_evaluate_passes_when_fno_is_non_default_and_non_public() -> None:
    summary = guard.evaluate(
        mode_specs={
            "kernel": {"module": "scpn_fusion.core.fusion_kernel", "maturity": "public"},
            "fno-training": {
                "module": "scpn_fusion.core.fno_jax_training",
                "maturity": "surrogate",
            },
        },
        default_modes=["kernel"],
        release_commands=["scpn-fusion flight --mode=neuro-control"],
    )
    assert summary["overall_pass"] is True
    assert summary["default_contains_deprecated_fno"] is False
    assert summary["fno_public_modes"] == []


def test_evaluate_fails_when_default_includes_deprecated_fno_module() -> None:
    summary = guard.evaluate(
        mode_specs={
            "kernel": {
                "module": "scpn_fusion.core.fno_turbulence_suppressor",
                "maturity": "public",
            },
        },
        default_modes=["kernel"],
        release_commands=[],
    )
    assert summary["default_contains_deprecated_fno"] is True
    assert summary["overall_pass"] is False


def test_evaluate_fails_when_fno_mode_is_public() -> None:
    summary = guard.evaluate(
        mode_specs={
            "fno-training": {"module": "scpn_fusion.core.fno_jax_training", "maturity": "public"},
        },
        default_modes=[],
        release_commands=[],
    )
    assert summary["fno_public_modes"] == ["fno-training"]
    assert summary["overall_pass"] is False


def test_evaluate_fails_when_release_docs_expose_fno_without_surrogate_unlock() -> None:
    summary = guard.evaluate(
        mode_specs={
            "fno-training": {
                "module": "scpn_fusion.core.fno_jax_training",
                "maturity": "surrogate",
            },
        },
        default_modes=[],
        release_commands=["scpn-fusion fno-training"],
    )
    assert summary["docs_violations"] == ["scpn-fusion fno-training"]
    assert summary["overall_pass"] is False


def test_main_writes_summary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        guard,
        "_load_runtime_state",
        lambda: (
            {
                "kernel": {"module": "scpn_fusion.core.fusion_kernel", "maturity": "public"},
                "fno-training": {
                    "module": "scpn_fusion.core.fno_jax_training",
                    "maturity": "surrogate",
                },
            },
            ["kernel"],
        ),
    )
    monkeypatch.setattr(guard, "_load_release_commands", lambda: ["scpn-fusion flight"])

    summary_path = tmp_path / "summary.json"
    rc = guard.main(["--summary-json", str(summary_path)])
    assert rc == 0
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["overall_pass"] is True
