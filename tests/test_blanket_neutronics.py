# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Blanket Neutronics Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Unit tests for reduced 3D volumetric blanket surrogate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "scpn_fusion" / "nuclear" / "blanket_neutronics.py"
SPEC = importlib.util.spec_from_file_location("blanket_neutronics", MODULE_PATH)
assert SPEC and SPEC.loader
blanket_neutronics = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = blanket_neutronics
SPEC.loader.exec_module(blanket_neutronics)

BreedingBlanket = blanket_neutronics.BreedingBlanket
VolumetricBlanketReport = blanket_neutronics.VolumetricBlanketReport


def test_volumetric_surrogate_returns_finite_positive_report() -> None:
    blanket = BreedingBlanket(thickness_cm=80.0, li6_enrichment=0.9)
    report = blanket.calculate_volumetric_tbr(
        major_radius_m=6.2,
        minor_radius_m=2.0,
        elongation=1.7,
        radial_cells=10,
        poloidal_cells=20,
        toroidal_cells=16,
    )
    assert isinstance(report, VolumetricBlanketReport)
    assert report.tbr > 0.0
    assert report.total_production_per_s > 0.0
    assert report.incident_neutrons_per_s > 0.0
    assert report.blanket_volume_m3 > 0.0


def test_thicker_blanket_increases_volumetric_tbr() -> None:
    thin = BreedingBlanket(thickness_cm=40.0, li6_enrichment=0.9)
    thick = BreedingBlanket(thickness_cm=100.0, li6_enrichment=0.9)

    thin_report = thin.calculate_volumetric_tbr(
        radial_cells=8, poloidal_cells=16, toroidal_cells=12
    )
    thick_report = thick.calculate_volumetric_tbr(
        radial_cells=8, poloidal_cells=16, toroidal_cells=12
    )

    assert thick_report.tbr > thin_report.tbr


def test_higher_li6_enrichment_increases_volumetric_tbr() -> None:
    low = BreedingBlanket(thickness_cm=80.0, li6_enrichment=0.5)
    high = BreedingBlanket(thickness_cm=80.0, li6_enrichment=0.95)

    low_report = low.calculate_volumetric_tbr(
        radial_cells=8, poloidal_cells=16, toroidal_cells=12
    )
    high_report = high.calculate_volumetric_tbr(
        radial_cells=8, poloidal_cells=16, toroidal_cells=12
    )

    assert high_report.tbr > low_report.tbr
