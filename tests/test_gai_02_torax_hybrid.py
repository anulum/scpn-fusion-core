# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GAI-02 TORAX Hybrid Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for GAI-02 TORAX-hybrid realtime validation lane."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from scpn_fusion.control.torax_hybrid_loop import run_nstxu_torax_hybrid_campaign

pytestmark = pytest.mark.experimental


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PATH = ROOT / "validation" / "gai_02_torax_hybrid.py"
VALIDATION_SPEC = importlib.util.spec_from_file_location("gai_02_torax_hybrid", VALIDATION_PATH)
assert VALIDATION_SPEC and VALIDATION_SPEC.loader
gai_02_torax_hybrid = importlib.util.module_from_spec(VALIDATION_SPEC)
sys.modules[VALIDATION_SPEC.name] = gai_02_torax_hybrid
VALIDATION_SPEC.loader.exec_module(gai_02_torax_hybrid)

RUNTIME_PATH = ROOT / "run_realtime_simulation.py"
RUNTIME_SPEC = importlib.util.spec_from_file_location("run_realtime_simulation", RUNTIME_PATH)
assert RUNTIME_SPEC and RUNTIME_SPEC.loader
run_realtime_simulation = importlib.util.module_from_spec(RUNTIME_SPEC)
sys.modules[RUNTIME_SPEC.name] = run_realtime_simulation
RUNTIME_SPEC.loader.exec_module(run_realtime_simulation)


def test_campaign_meets_thresholds_smoke() -> None:
    out = run_nstxu_torax_hybrid_campaign(seed=42, episodes=8, steps_per_episode=160)
    assert out.disruption_avoidance_rate >= 0.90
    assert out.torax_parity_pct >= 95.0
    assert out.p95_loop_latency_ms <= 1.0
    assert out.passes_thresholds is True


def test_validation_report_contains_threshold_pass() -> None:
    report = gai_02_torax_hybrid.generate_report(seed=7, episodes=6, steps_per_episode=140)
    g = report["gai_02"]
    text = gai_02_torax_hybrid.render_markdown(report)
    assert g["passes_thresholds"] is True
    assert "GAI-02 TORAX Hybrid Validation" in text
    assert "TORAX parity" in text


def test_run_realtime_hybrid_smoke_function() -> None:
    summary = run_realtime_simulation.run_torax_hybrid_smoke(
        seed=10, episodes=5, steps_per_episode=120
    )
    assert summary["passes_thresholds"] is True
    assert summary["torax_parity_pct"] >= 95.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"episodes": 0}, "episodes"),
        ({"steps_per_episode": 16}, "steps_per_episode"),
    ],
)
def test_campaign_rejects_invalid_runtime_inputs(
    kwargs: dict[str, int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        run_nstxu_torax_hybrid_campaign(seed=42, **kwargs)
