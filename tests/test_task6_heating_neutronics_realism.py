# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Task 6 Heating + Neutronics Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for validation/task6_heating_neutronics_realism.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "task6_heating_neutronics_realism.py"
SPEC = importlib.util.spec_from_file_location(
    "task6_heating_neutronics_realism",
    MODULE_PATH,
)
assert SPEC and SPEC.loader
task6_heating_neutronics_realism = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task6_heating_neutronics_realism)


def test_task6_campaign_passes_thresholds_smoke() -> None:
    report = task6_heating_neutronics_realism.run_campaign(
        seed=42,
        scan_candidates=56,
        target_optimized_configs=10,
        shortlist_size=16,
    )
    g = report["task6_heating_neutronics_realism"]
    assert g["passes_thresholds"] is True
    assert g["metrics"]["optimized_config_count"] >= 10
    assert g["metrics"]["min_q"] >= 10.0
    assert g["metrics"]["min_tbr"] >= 1.05
    assert g["metrics"]["mean_tbr_mc"] > 0.0
    assert g["metrics"]["mean_rf_reflection_rate"] <= 0.55
    assert g["metrics"]["mean_neutron_leakage_rate"] <= 0.50


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"scan_candidates": 0}, "scan_candidates"),
        ({"target_optimized_configs": 1}, "target_optimized_configs"),
        ({"scan_candidates": 8, "target_optimized_configs": 10}, "scan_candidates"),
        ({"shortlist_size": 0}, "shortlist_size"),
    ],
)
def test_task6_campaign_rejects_invalid_inputs(
    kwargs: dict[str, int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        task6_heating_neutronics_realism.run_campaign(**kwargs)


def test_task6_markdown_contains_required_sections() -> None:
    report = task6_heating_neutronics_realism.generate_report(
        seed=7,
        scan_candidates=48,
        target_optimized_configs=10,
        shortlist_size=14,
    )
    text = task6_heating_neutronics_realism.render_markdown(report)
    assert "# Task 6 Heating + Neutronics Realism" in text
    assert "GENRAY-Like Heating Proxies" in text
    assert "MCNP-Lite Neutronics Optimization (MVR-0.96 Lane)" in text
    assert "ARIES-AT Scaling Parity" in text
