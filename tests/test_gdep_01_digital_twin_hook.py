# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GDEP-01 Digital Twin Hook Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for GDEP-01 digital twin ingestion and scenario planning."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from scpn_fusion.control.digital_twin_ingest import RealtimeTwinHook, generate_emulated_stream


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "gdep_01_digital_twin_hook.py"
SPEC = importlib.util.spec_from_file_location("gdep_01_digital_twin_hook", MODULE_PATH)
assert SPEC and SPEC.loader
gdep_01_digital_twin_hook = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gdep_01_digital_twin_hook)


def test_stream_generation_basic_shape() -> None:
    packets = generate_emulated_stream("NSTX-U", seed=7, samples=80)
    assert len(packets) == 80
    assert packets[0].machine == "NSTX-U"
    assert packets[-1].t_ms > packets[0].t_ms


def test_realtime_hook_scenario_plan_smoke() -> None:
    packets = generate_emulated_stream("SPARC", seed=9, samples=96)
    hook = RealtimeTwinHook("SPARC", seed=9)
    for packet in packets:
        hook.ingest(packet)
    plan = hook.scenario_plan(horizon=24)
    assert plan["safe_horizon_rate"] >= 0.90
    assert plan["mean_risk"] <= 0.75
    assert plan["latency_ms"] <= 6.0


def test_gdep_01_campaign_passes_thresholds() -> None:
    out = gdep_01_digital_twin_hook.run_campaign(seed=42, samples_per_machine=220)
    assert out["passes_thresholds"] is True
