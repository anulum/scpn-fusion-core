# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GMVR-01 Compact Constraint Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for GMVR-01 compact-constraint scan validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core.global_design_scanner import GlobalDesignExplorer


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "gmvr_01_compact_constraints.py"
SPEC = importlib.util.spec_from_file_location("gmvr_01_compact_constraints", MODULE_PATH)
assert SPEC and SPEC.loader
gmvr_01_compact_constraints = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gmvr_01_compact_constraints)


def test_compact_scan_is_deterministic_for_seed() -> None:
    explorer = GlobalDesignExplorer("dummy")
    a = explorer.run_compact_scan(n_samples=600, seed=7)
    b = explorer.run_compact_scan(n_samples=600, seed=7)
    assert len(a) == len(b)
    assert list(a["R"].head(8)) == list(b["R"].head(8))


def test_campaign_finds_feasible_design_in_target_window() -> None:
    out = gmvr_01_compact_constraints.run_campaign(seed=42, scan_samples=1800)
    assert out["passes_thresholds"] is True
    assert out["feasible_count"] > 0
    best = out["best_design"]
    assert best is not None
    assert 1.2 <= best["R_m"] <= 1.5
    assert best["Q"] > 5.0
    assert best["Div_Load_Optimized_MW_m2"] <= 45.0
    assert best["Zeff_Est"] <= 0.4
    assert best["B_peak_HTS_T"] <= 21.0


def test_compact_scan_does_not_mutate_global_numpy_rng_state() -> None:
    np.random.seed(2222)
    state = np.random.get_state()

    explorer = GlobalDesignExplorer("dummy")
    _ = explorer.run_compact_scan(n_samples=200, seed=5)

    observed = float(np.random.random())
    np.random.set_state(state)
    expected = float(np.random.random())
    assert observed == expected


def test_run_scan_rejects_invalid_bounds() -> None:
    explorer = GlobalDesignExplorer("dummy")
    with pytest.raises(ValueError, match="r_bounds"):
        explorer.run_scan(n_samples=10, r_bounds=(2.0, 2.0))
