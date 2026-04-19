# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — GDEP-03 Blind Validation Tests
"""Tests for GDEP-03 blind validation dashboard lane."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "gdep_03_blind_validation.py"
SPEC = importlib.util.spec_from_file_location("gdep_03_blind_validation", MODULE_PATH)
assert SPEC and SPEC.loader
gdep_03_blind_validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gdep_03_blind_validation)


def test_blind_reference_loader_contains_expected_machines() -> None:
    rows = gdep_03_blind_validation.load_blind_references(
        ROOT / "validation" / "reference_data" / "blind"
    )
    machines = {row["machine"] for row in rows}
    assert machines == {"EU-DEMO", "K-DEMO"}
    assert len(rows) >= 10


def test_gdep_03_campaign_passes_thresholds() -> None:
    out = gdep_03_blind_validation.run_campaign()
    assert out["passes_thresholds"] is True
    assert out["aggregate"]["parity_pct"] >= out["thresholds"]["min_parity_pct"]
    assert all(machine["passes_thresholds"] for machine in out["machines"])


def test_render_markdown_contains_sections() -> None:
    report = gdep_03_blind_validation.generate_report()
    text = gdep_03_blind_validation.render_markdown(report)
    assert "# GDEP-03 Blind Validation Dashboard" in text
    assert "Aggregate Metrics" in text
    assert "Overall pass" in text
