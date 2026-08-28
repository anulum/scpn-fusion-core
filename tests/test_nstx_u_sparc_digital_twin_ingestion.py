# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — NSTX-U/SPARC Digital-Twin Ingestion Tests
"""Real-surface tests for NSTX-U/SPARC digital-twin ingestion and planning."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
import runpy
import sys
from typing import Any

import pytest

from scpn_fusion.control.digital_twin_ingest import (
    RealtimeTwinHook,
    TelemetryPacket,
    generate_emulated_stream,
    run_realtime_twin_session,
)


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "nstx_u_sparc_digital_twin_ingestion.py"
SPEC = importlib.util.spec_from_file_location("nstx_u_sparc_digital_twin_ingestion", MODULE_PATH)
assert SPEC and SPEC.loader
validation_cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validation_cli
SPEC.loader.exec_module(validation_cli)


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


def test_stream_generation_rejects_unsupported_machine() -> None:
    with pytest.raises(ValueError, match="machine must be"):
        generate_emulated_stream("JET", seed=1, samples=64)


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


def test_realtime_hook_ring_buffer_trims_to_max() -> None:
    packets = generate_emulated_stream("SPARC", seed=3, samples=70)
    hook = RealtimeTwinHook("SPARC", max_buffer=64, seed=3)
    for packet in packets:
        hook.ingest(packet)
    # The ring buffer retains only the most recent ``max_buffer`` packets.
    assert len(hook.buffer) == 64
    assert hook.buffer[-1] == packets[-1]
    assert hook.buffer[0] == packets[-64]


def test_scenario_plan_empty_buffer_raises() -> None:
    hook = RealtimeTwinHook("SPARC", seed=1)
    with pytest.raises(RuntimeError, match="No telemetry packets ingested"):
        hook.scenario_plan(horizon=24)


def test_scenario_plan_fails_closed_for_extreme_risk_packet() -> None:
    hook = RealtimeTwinHook("SPARC", seed=1)
    packet = TelemetryPacket(
        t_ms=0,
        machine="SPARC",
        ip_ma=8.7,
        beta_n=100.0,
        q95=-100.0,
        density_1e19=1000.0,
    )
    for _ in range(64):
        hook.ingest(packet)

    plan = hook.scenario_plan(horizon=4)
    assert plan["safe_horizon_rate"] == 0.0
    assert plan["mean_risk"] == 1.0
    assert plan["passes"] is False


def test_campaign_passes_thresholds() -> None:
    out = validation_cli.run_campaign(seed=42, samples_per_machine=220)
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


def test_chaos_campaign_is_deterministic_for_seed() -> None:
    a = validation_cli.run_campaign(
        seed=11,
        samples_per_machine=160,
        chaos_dropout_prob=0.02,
        chaos_noise_std=0.005,
    )
    b = validation_cli.run_campaign(
        seed=11,
        samples_per_machine=160,
        chaos_dropout_prob=0.02,
        chaos_noise_std=0.005,
    )
    assert a["passes_thresholds"] == b["passes_thresholds"]
    assert a["machines"][0]["planning_success_rate"] == b["machines"][0]["planning_success_rate"]
    assert a["machines"][1]["mean_risk"] == b["machines"][1]["mean_risk"]
    assert a["machines"][0]["chaos_dropouts_total"] == b["machines"][0]["chaos_dropouts_total"]
    assert (
        a["machines"][1]["chaos_noise_injections_total"]
        == b["machines"][1]["chaos_noise_injections_total"]
    )
    assert a["chaos_dropouts_total"] == b["chaos_dropouts_total"]
    assert a["chaos_noise_injections_total"] == b["chaos_noise_injections_total"]


def test_campaign_full_dropout_counts_all_channels() -> None:
    out = validation_cli.run_campaign(
        seed=4,
        samples_per_machine=64,
        chaos_dropout_prob=1.0,
        chaos_noise_std=0.0,
    )
    assert out["chaos_channels_total"] == 2 * 64 * 4
    assert out["chaos_dropouts_total"] == out["chaos_channels_total"]
    assert out["chaos_dropout_rate"] == 1.0


def test_report_contract_and_markdown_are_descriptive() -> None:
    report = validation_cli.generate_report(
        seed=8,
        samples_per_machine=64,
        chaos_dropout_prob=0.02,
        chaos_noise_std=0.005,
    )
    payload = validation_cli.validate_report(report)
    text = validation_cli.render_markdown(report)
    assert report["schema_version"] == 2
    assert report["report_kind"] == "nstx_u_sparc_digital_twin_ingestion"
    assert payload["passes_thresholds"] is True
    assert text.startswith("# NSTX-U/SPARC Digital-Twin Ingestion Validation")
    assert "## Chaos Campaign" in text
    assert "Config dropout probability" in text
    assert "Observed dropout rate" in text
    assert "Observed noise injection rate" in text


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"samples_per_machine": 16}, "samples_per_machine"),
        ({"chaos_dropout_prob": 1.2}, "chaos_dropout_prob"),
        ({"chaos_dropout_prob": float("nan")}, "chaos_dropout_prob"),
        ({"chaos_noise_std": -0.1}, "chaos_noise_std"),
        ({"chaos_noise_std": float("inf")}, "chaos_noise_std"),
        ({"min_planning_success_rate": -0.1}, "min_planning_success_rate"),
        ({"min_planning_success_rate": float("nan")}, "min_planning_success_rate"),
        ({"max_mean_risk": 1.1}, "max_mean_risk"),
        ({"max_mean_risk": float("inf")}, "max_mean_risk"),
        ({"max_p95_latency_ms": 0.0}, "max_p95_latency_ms"),
        ({"max_p95_latency_ms": float("nan")}, "max_p95_latency_ms"),
    ],
)
def test_campaign_rejects_invalid_inputs(kwargs: dict[str, float | int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validation_cli.run_campaign(seed=1, **kwargs)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 1}, "unsupported report schema_version"),
        ({"report_kind": "obsolete"}, "unsupported report_kind"),
        ({"generated_at_utc": ""}, "generated_at_utc must be"),
        ({"generated_at_utc": None}, "generated_at_utc must be"),
        ({"nstx_u_sparc_digital_twin_ingestion": []}, "must be an object"),
    ],
)
def test_report_contract_rejects_invalid_current_payloads(
    change: dict[str, Any], message: str
) -> None:
    report = validation_cli.generate_report(seed=5, samples_per_machine=64)
    report.update(change)
    with pytest.raises(ValueError, match=message):
        validation_cli.validate_report(report)


def test_report_contract_rejects_obsolete_coded_payload_exactly() -> None:
    stale_report = {
        "generated_at_utc": "2026-08-26T00:00:00+00:00",
        "gdep_01": {"passes_thresholds": True},
    }
    with pytest.raises(ValueError, match="current descriptive contract"):
        validation_cli.validate_report(stale_report)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: run_realtime_twin_session("SPARC", seed=1, samples=16), "samples"),
        (lambda: run_realtime_twin_session("SPARC", seed=1, dt_ms=0), "dt_ms"),
        (lambda: run_realtime_twin_session("SPARC", seed=1, horizon=3), "horizon"),
        (lambda: run_realtime_twin_session("SPARC", seed=1, plan_every=0), "plan_every"),
        (
            lambda: run_realtime_twin_session("SPARC", seed=1, chaos_dropout_prob=-0.1),
            "chaos_dropout_prob",
        ),
        (
            lambda: run_realtime_twin_session("SPARC", seed=1, chaos_dropout_prob=float("nan")),
            "chaos_dropout_prob",
        ),
        (
            lambda: run_realtime_twin_session("SPARC", seed=1, chaos_noise_std=-0.1),
            "chaos_noise_std",
        ),
        (
            lambda: run_realtime_twin_session("SPARC", seed=1, chaos_noise_std=float("inf")),
            "chaos_noise_std",
        ),
    ],
)
def test_realtime_session_rejects_invalid_inputs(
    call: Callable[[], dict[str, Any]], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        call()


def test_realtime_session_reports_no_plans_when_cadence_exceeds_stream() -> None:
    out = run_realtime_twin_session("NSTX-U", samples=32, plan_every=64, seed=3)
    assert out["plan_count"] == 0
    assert out["planning_success_rate"] == 0.0
    assert out["mean_risk"] == 1.0
    assert out["p95_latency_ms"] == 999.0
    assert out["passes_thresholds"] is False


def test_cli_writes_current_contract_and_enforces_strict_thresholds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_json = tmp_path / "digital_twin_ingestion.json"
    output_md = tmp_path / "digital_twin_ingestion.md"
    common_args = [
        "--seed",
        "13",
        "--samples-per-machine",
        "64",
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]

    assert validation_cli.main(common_args) == 0
    assert validation_cli.main([*common_args, "--strict"]) == 0
    written = json.loads(output_json.read_text(encoding="utf-8"))
    validation_cli.validate_report(written)
    assert output_md.read_text(encoding="utf-8").startswith(
        "# NSTX-U/SPARC Digital-Twin Ingestion Validation"
    )
    assert "ingestion validation complete" in capsys.readouterr().out

    assert (
        validation_cli.main(
            [
                *common_args,
                "--strict",
                "--max-mean-risk",
                "0.0",
            ]
        )
        == 2
    )


def test_script_entry_point_runs_the_real_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_json = tmp_path / "script-report.json"
    output_md = tmp_path / "script-report.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--samples-per-machine",
            "64",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--strict",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(MODULE_PATH), run_name="__main__")

    assert exc_info.value.code == 0
    assert output_json.is_file()
    assert output_md.is_file()
