# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — SCPN vs PID/MPC Benchmark Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for deterministic SCPN-vs-PID-vs-MPC benchmark lane."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "scpn_pid_mpc_benchmark.py"
SPEC = importlib.util.spec_from_file_location("scpn_pid_mpc_benchmark", MODULE_PATH)
assert SPEC and SPEC.loader
scpn_pid_mpc_benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scpn_pid_mpc_benchmark
SPEC.loader.exec_module(scpn_pid_mpc_benchmark)


def test_campaign_is_deterministic_for_seed() -> None:
    a = scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=180)
    b = scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=180)
    assert a["pid"]["rmse"] == b["pid"]["rmse"]
    assert a["mpc"]["rmse"] == b["mpc"]["rmse"]
    assert a["scpn"]["rmse"] == b["scpn"]["rmse"]
    assert a["ratios"]["scpn_vs_pid_rmse_ratio"] == b["ratios"]["scpn_vs_pid_rmse_ratio"]
    assert a["ratios"]["scpn_vs_mpc_rmse_ratio"] == b["ratios"]["scpn_vs_mpc_rmse_ratio"]


def test_campaign_meets_thresholds_smoke() -> None:
    out = scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=240)
    assert out["passes_thresholds"] is True
    assert out["mpc"]["rmse"] <= out["pid"]["rmse"]
    assert out["ratios"]["scpn_vs_pid_rmse_ratio"] <= out["thresholds"]["max_scpn_vs_pid_rmse_ratio"]
    assert out["ratios"]["scpn_vs_mpc_rmse_ratio"] <= out["thresholds"]["max_scpn_vs_mpc_rmse_ratio"]


def test_campaign_controller_uses_nonzero_binary_margin() -> None:
    controller = scpn_pid_mpc_benchmark._build_scpn_controller()
    assert getattr(controller, "_sc_binary_margin", 0.0) > 0.0


def test_traceable_runtime_lane_is_exposed_and_deterministic() -> None:
    a = scpn_pid_mpc_benchmark.run_campaign(
        seed=42, steps=160, scpn_runtime_profile="traceable"
    )
    b = scpn_pid_mpc_benchmark.run_campaign(
        seed=42, steps=160, scpn_runtime_profile="traceable"
    )
    assert a["runtime_lane"]["runtime_profile"] == "traceable"
    assert a["runtime_lane"]["uses_traceable_step"] is True
    assert a["scpn"]["rmse"] == b["scpn"]["rmse"]


def test_render_markdown_contains_sections() -> None:
    report = scpn_pid_mpc_benchmark.generate_report(seed=11, steps=120)
    text = scpn_pid_mpc_benchmark.render_markdown(report)
    assert "# SCPN vs PID/MPC Benchmark" in text
    assert "SCPN runtime profile" in text
    assert "RMSE" in text
    assert "Ratios" in text
    assert "Threshold Pass" in text
