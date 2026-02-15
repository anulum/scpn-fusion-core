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

import pytest

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


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"samples": 16}, "samples"),
        ({"dt_ms": 0}, "dt_ms"),
    ],
)
def test_generate_emulated_stream_rejects_invalid_runtime_inputs(
    kwargs: dict[str, int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        generate_emulated_stream("NSTX-U", seed=1, **kwargs)


def test_realtime_hook_rejects_invalid_max_buffer() -> None:
    with pytest.raises(ValueError, match="max_buffer"):
        RealtimeTwinHook("SPARC", max_buffer=8, seed=1)


def test_realtime_hook_rejects_invalid_horizon() -> None:
    packets = generate_emulated_stream("SPARC", seed=9, samples=96)
    hook = RealtimeTwinHook("SPARC", seed=9)
    for packet in packets[:8]:
        hook.ingest(packet)
    with pytest.raises(ValueError, match="horizon"):
        hook.scenario_plan(horizon=3)


def test_gdep_01_campaign_passes_thresholds() -> None:
    out = gdep_01_digital_twin_hook.run_campaign(seed=42, samples_per_machine=220)
    assert out["passes_thresholds"] is True
    for key in (
        "chaos_channels_total",
        "chaos_dropouts_total",
        "chaos_dropout_rate",
        "chaos_noise_injections_total",
        "chaos_noise_injection_rate",
    ):
        assert key in out
    for machine in out["machines"]:
        assert "chaos_channels_total" in machine
        assert "chaos_dropouts_total" in machine
        assert "chaos_dropout_rate" in machine
        assert "chaos_noise_injections_total" in machine
        assert "chaos_noise_injection_rate" in machine


def test_gdep_01_chaos_campaign_is_deterministic_for_seed() -> None:
    a = gdep_01_digital_twin_hook.run_campaign(
        seed=11,
        samples_per_machine=160,
        chaos_dropout_prob=0.02,
        chaos_noise_std=0.005,
    )
    b = gdep_01_digital_twin_hook.run_campaign(
        seed=11,
        samples_per_machine=160,
        chaos_dropout_prob=0.02,
        chaos_noise_std=0.005,
    )
    assert a["passes_thresholds"] == b["passes_thresholds"]
    assert a["machines"][0]["planning_success_rate"] == b["machines"][0]["planning_success_rate"]
    assert a["machines"][1]["mean_risk"] == b["machines"][1]["mean_risk"]
    assert a["machines"][0]["chaos_dropouts_total"] == b["machines"][0]["chaos_dropouts_total"]
    assert a["machines"][1]["chaos_noise_injections_total"] == b["machines"][1]["chaos_noise_injections_total"]
    assert a["chaos_dropouts_total"] == b["chaos_dropouts_total"]
    assert a["chaos_noise_injections_total"] == b["chaos_noise_injections_total"]


def test_gdep_01_campaign_full_dropout_counts_all_channels() -> None:
    out = gdep_01_digital_twin_hook.run_campaign(
        seed=4,
        samples_per_machine=64,
        chaos_dropout_prob=1.0,
        chaos_noise_std=0.0,
    )
    assert out["chaos_channels_total"] == 2 * 64 * 4
    assert out["chaos_dropouts_total"] == out["chaos_channels_total"]
    assert out["chaos_dropout_rate"] == 1.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"samples_per_machine": 16}, "samples_per_machine"),
        ({"chaos_dropout_prob": 1.2}, "chaos_dropout_prob"),
        ({"chaos_dropout_prob": float("nan")}, "chaos_dropout_prob"),
        ({"chaos_noise_std": -0.1}, "chaos_noise_std"),
        ({"chaos_noise_std": float("inf")}, "chaos_noise_std"),
    ],
)
def test_gdep_01_campaign_rejects_invalid_inputs(
    kwargs: dict[str, float | int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        gdep_01_digital_twin_hook.run_campaign(seed=1, **kwargs)
