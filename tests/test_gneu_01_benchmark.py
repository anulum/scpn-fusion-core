# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GNEU-01 Benchmark Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for validation/gneu_01_benchmark.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "gneu_01_benchmark.py"
SPEC = importlib.util.spec_from_file_location("gneu_01_benchmark", MODULE_PATH)
assert SPEC and SPEC.loader
gneu_01_benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gneu_01_benchmark)


def test_run_benchmark_is_deterministic_for_seed() -> None:
    a = gneu_01_benchmark.run_benchmark(seed=17, episodes=6, window=48)
    b = gneu_01_benchmark.run_benchmark(seed=17, episodes=6, window=48)
    assert a["agreement"] == b["agreement"]
    assert a["mean_abs_delta"] == b["mean_abs_delta"]
    assert a["oracle_sc_mean_abs_delta"] == b["oracle_sc_mean_abs_delta"]
    assert a["recovery_ms_p95"] == b["recovery_ms_p95"]


def test_run_benchmark_meets_thresholds_on_smoke_config() -> None:
    out = gneu_01_benchmark.run_benchmark(seed=42, episodes=8, window=64)
    assert out["agreement"] >= 0.95
    assert out["mean_abs_delta"] <= 0.08
    assert out["oracle_sc_mean_abs_delta"] <= 0.05
    assert out["recovery_ms_p95"] <= 1.0
    assert out["passes_thresholds"] is True


def test_gneu_controller_uses_nonzero_binary_margin() -> None:
    controller = gneu_01_benchmark._build_controller()
    assert getattr(controller, "_sc_binary_margin", 0.0) > 0.0


def test_render_markdown_contains_key_sections() -> None:
    report = gneu_01_benchmark.generate_report(seed=5, episodes=4, window=40)
    text = gneu_01_benchmark.render_markdown(report)
    assert "# GNEU-01 Benchmark" in text
    assert "## Metrics" in text
    assert "TORAX parity estimate" in text
    assert "oracle-vs-SC marking delta" in text


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"episodes": 0}, "episodes"),
        ({"window": 8}, "window"),
        ({"recovery_window_steps": 0}, "recovery_window_steps"),
        ({"recovery_epsilon": 0.0}, "recovery_epsilon"),
        ({"recovery_epsilon": float("nan")}, "recovery_epsilon"),
        ({"dt_ms": 0.0}, "dt_ms"),
    ],
)
def test_run_benchmark_rejects_invalid_inputs(
    kwargs: dict[str, float | int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        gneu_01_benchmark.run_benchmark(**kwargs)
