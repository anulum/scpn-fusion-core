# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Control Resilience Campaign Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for validation/control_resilience_campaign.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "control_resilience_campaign.py"
SPEC = importlib.util.spec_from_file_location("control_resilience_campaign", MODULE_PATH)
assert SPEC and SPEC.loader
control_resilience_campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control_resilience_campaign)


def test_generate_campaign_report_has_expected_keys() -> None:
    report = control_resilience_campaign.generate_campaign_report(
        seed=7, episodes=8, window=48, noise_std=0.02, bit_flip_interval=7
    )
    assert "generated_at_utc" in report
    assert "runtime_seconds" in report
    assert "campaign" in report
    campaign = report["campaign"]
    for key in (
        "mean_abs_risk_error",
        "p95_abs_risk_error",
        "recovery_steps_p95",
        "recovery_success_rate",
        "passes_thresholds",
    ):
        assert key in campaign


def test_campaign_report_is_deterministic_for_same_seed() -> None:
    r1 = control_resilience_campaign.generate_campaign_report(
        seed=99, episodes=6, window=32, noise_std=0.01, bit_flip_interval=5
    )["campaign"]
    r2 = control_resilience_campaign.generate_campaign_report(
        seed=99, episodes=6, window=32, noise_std=0.01, bit_flip_interval=5
    )["campaign"]
    assert r1["mean_abs_risk_error"] == r2["mean_abs_risk_error"]
    assert r1["p95_abs_risk_error"] == r2["p95_abs_risk_error"]
    assert r1["recovery_steps_p95"] == r2["recovery_steps_p95"]
    assert r1["recovery_success_rate"] == r2["recovery_success_rate"]


def test_render_markdown_contains_metrics_section() -> None:
    report = control_resilience_campaign.generate_campaign_report(
        seed=11, episodes=4, window=24
    )
    text = control_resilience_campaign.render_markdown(report)
    assert "# Control Resilience Campaign" in text
    assert "## Metrics" in text
    assert "Threshold pass" in text


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"episodes": 0}, "episodes"),
        ({"window": 8}, "window"),
        ({"noise_std": -0.1}, "noise_std"),
        ({"noise_std": float("nan")}, "noise_std"),
        ({"bit_flip_interval": 0}, "bit_flip_interval"),
        ({"recovery_window": 0}, "recovery_window"),
        ({"recovery_epsilon": 0.0}, "recovery_epsilon"),
    ],
)
def test_generate_campaign_report_validates_inputs(
    kwargs: dict[str, float | int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        control_resilience_campaign.generate_campaign_report(**kwargs)


def test_generate_campaign_report_does_not_mutate_global_numpy_rng_state() -> None:
    np.random.seed(777)
    state = np.random.get_state()

    control_resilience_campaign.generate_campaign_report(
        seed=13,
        episodes=6,
        window=32,
        noise_std=0.01,
        bit_flip_interval=5,
    )

    observed = float(np.random.random())
    np.random.set_state(state)
    expected = float(np.random.random())
    assert observed == expected
