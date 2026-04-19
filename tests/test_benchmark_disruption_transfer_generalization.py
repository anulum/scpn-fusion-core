# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for validation/benchmark_disruption_transfer_generalization.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "benchmark_disruption_transfer_generalization.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_disruption_transfer_generalization", MODULE_PATH
)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def _fake_disruption_payload() -> dict:
    return {
        "shots": [
            {"file": "shot_1_hmode.npz", "is_disruption": True, "detected": True},
            {"file": "shot_2_hmode_safe.npz", "is_disruption": False, "detected": False},
            {"file": "shot_3_hybrid.npz", "is_disruption": True, "detected": True},
            {"file": "shot_4_beta_limit.npz", "is_disruption": True, "detected": True},
            {"file": "shot_5_beta_limit.npz", "is_disruption": False, "detected": False},
            {"file": "shot_6_vde.npz", "is_disruption": True, "detected": False},
            {"file": "shot_7_vde.npz", "is_disruption": False, "detected": False},
        ]
    }


def test_run_benchmark_splits_source_vs_target(monkeypatch) -> None:
    monkeypatch.setattr(
        mod, "validate_disruption", lambda *_args, **_kwargs: _fake_disruption_payload()
    )
    report = mod.run_benchmark(
        disruption_dir=ROOT / "validation" / "reference_data" / "diiid" / "disruption_shots",
        source_scenarios=("hmode", "hmode_safe", "hybrid"),
    )
    assert report["source_group"]["n_shots"] == 3
    assert report["target_group"]["n_shots"] == 4
    assert report["source_group"]["recall"] == 1.0
    assert report["target_group"]["recall"] == 0.5
    assert report["transfer_efficiency"] == 0.5
    assert report["passes"] is False


def test_main_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        mod, "validate_disruption", lambda *_args, **_kwargs: _fake_disruption_payload()
    )
    out_json = tmp_path / "transfer.json"
    out_md = tmp_path / "transfer.md"
    rc = mod.main(
        [
            "--disruption-dir",
            str(ROOT / "validation" / "reference_data" / "diiid" / "disruption_shots"),
            "--source-scenarios",
            "hmode,hmode_safe,hybrid",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    )
    assert rc == 0
    assert out_json.exists()
    assert out_md.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "checks" in payload
    assert isinstance(payload.get("passes"), bool)
