# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Digital Twin Ingest Runtime Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Runtime-level deterministic tests for digital twin ingest session helper."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.digital_twin_ingest import run_realtime_twin_session


def test_run_realtime_twin_session_returns_finite_summary() -> None:
    summary = run_realtime_twin_session(
        "SPARC",
        seed=9,
        samples=96,
        horizon=24,
        plan_every=8,
    )
    for key in (
        "machine",
        "seed",
        "samples",
        "horizon",
        "plan_every",
        "plan_count",
        "planning_success_rate",
        "mean_risk",
        "p95_latency_ms",
        "passes_thresholds",
    ):
        assert key in summary
    assert summary["machine"] == "SPARC"
    assert summary["plan_count"] > 0
    assert 0.0 <= summary["planning_success_rate"] <= 1.0
    assert np.isfinite(summary["mean_risk"])
    assert np.isfinite(summary["p95_latency_ms"])


def test_run_realtime_twin_session_is_deterministic_for_same_seed() -> None:
    kwargs = dict(
        machine="NSTX-U",
        seed=42,
        samples=112,
        horizon=24,
        plan_every=8,
        chaos_dropout_prob=0.02,
        chaos_noise_std=0.005,
    )
    a = run_realtime_twin_session(**kwargs)
    b = run_realtime_twin_session(**kwargs)
    for key in (
        "plan_count",
        "planning_success_rate",
        "mean_risk",
        "p95_latency_ms",
    ):
        assert a[key] == pytest.approx(b[key], rel=0.0, abs=0.0)
    assert a["passes_thresholds"] == b["passes_thresholds"]


def test_run_realtime_twin_session_rejects_invalid_machine() -> None:
    with pytest.raises(ValueError, match="machine must be"):
        run_realtime_twin_session("ITER", seed=1, samples=64)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"samples": 16}, "samples"),
        ({"dt_ms": 0}, "dt_ms"),
        ({"horizon": 3}, "horizon"),
        ({"plan_every": 0}, "plan_every"),
        ({"chaos_dropout_prob": 1.5}, "chaos_dropout_prob"),
        ({"chaos_dropout_prob": float("nan")}, "chaos_dropout_prob"),
        ({"chaos_noise_std": -0.1}, "chaos_noise_std"),
    ],
)
def test_run_realtime_twin_session_rejects_invalid_runtime_inputs(
    kwargs: dict[str, float | int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        run_realtime_twin_session("SPARC", seed=1, **kwargs)
