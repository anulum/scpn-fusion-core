# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GMVR-03 Stellarator Extension Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for GMVR-03 stellarator extension campaign."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "gmvr_03_stellarator_extension.py"
SPEC = importlib.util.spec_from_file_location("gmvr_03_stellarator_extension", MODULE_PATH)
assert SPEC and SPEC.loader
gmvr_03_stellarator_extension = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gmvr_03_stellarator_extension)


def test_gmvr_03_campaign_passes_thresholds_smoke() -> None:
    out = gmvr_03_stellarator_extension.run_campaign(iterations=6)
    assert out["passes_thresholds"] is True
    assert out["final_instability_metric"] <= 0.09
    assert out["improvement_pct"] >= 10.0
    assert out["vmec_parity_pct"] >= 95.0


def test_gmvr_03_markdown_contains_key_sections() -> None:
    report = gmvr_03_stellarator_extension.generate_report(iterations=6)
    text = gmvr_03_stellarator_extension.render_markdown(report)
    assert "GMVR-03 Stellarator Extension Validation" in text
    assert "VMEC++ Proxy Parity" in text
