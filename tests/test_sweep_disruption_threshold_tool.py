# ----------------------------------------------------------------------
# SCPN Fusion Core -- Sweep Disruption Threshold Tool Tests
# ----------------------------------------------------------------------
"""Tests for tools/sweep_disruption_threshold.py hardening paths."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "sweep_disruption_threshold.py"
SPEC = importlib.util.spec_from_file_location("sweep_disruption_threshold", MODULE_PATH)
assert SPEC and SPEC.loader
sweep_tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sweep_tool
SPEC.loader.exec_module(sweep_tool)


def test_load_shots_extracts_arrays_without_retaining_npz_handle(tmp_path: Path) -> None:
    shot = tmp_path / "shot_valid.npz"
    n = 200
    np.savez(
        shot,
        is_disruption=np.bool_(True),
        disruption_time_idx=np.int64(120),
        dBdt_gauss_per_s=np.linspace(0.0, 1.0, n, dtype=np.float64),
        n1_amp=np.linspace(0.1, 0.2, n, dtype=np.float64),
        n2_amp=np.linspace(0.05, 0.1, n, dtype=np.float64),
    )
    rows = sweep_tool.load_shots(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["file"] == "shot_valid.npz"
    assert row["is_disruption"] is True
    assert int(row["disruption_time_idx"]) == 120
    assert row["signal"].shape == (n,)
    assert row["n1_amp"].shape == (n,)
    assert row["n2_amp"].shape == (n,)
    assert "data" not in row


def test_load_shots_rejects_object_payload_under_secure_defaults(tmp_path: Path) -> None:
    shot = tmp_path / "shot_bad.npz"
    np.savez(
        shot,
        is_disruption=np.array([True], dtype=object),
        disruption_time_idx=np.array([100], dtype=object),
        dBdt_gauss_per_s=np.linspace(0.0, 1.0, 200, dtype=np.float64),
    )
    with pytest.raises(ValueError):
        sweep_tool.load_shots(tmp_path)


def test_precompute_unbiased_logits_uses_safe_toroidal_defaults() -> None:
    signal = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    pre = sweep_tool.precompute_unbiased_logits(
        [
            {
                "file": "shot",
                "is_disruption": False,
                "disruption_time_idx": -1,
                "signal": signal,
                "n1_amp": None,
                "n2_amp": None,
            }
        ]
    )
    assert len(pre) == 1
    logits = pre[0]["unbiased_logits"]
    assert logits.ndim == 1
    assert logits.size > 0
    assert bool(np.all(np.isfinite(logits)))
