# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Nuclear Package Export Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Smoke tests for scpn_fusion.nuclear package exports."""

from __future__ import annotations

from scpn_fusion.nuclear import (
    BreedingBlanket,
    NuclearEngineeringLab,
    SputteringPhysics,
    TEMHD_Stabilizer,
    VolumetricBlanketReport,
)


def test_nuclear_package_exports_are_importable() -> None:
    assert NuclearEngineeringLab is not None
    assert BreedingBlanket is not None
    assert VolumetricBlanketReport is not None
    assert SputteringPhysics is not None
    assert TEMHD_Stabilizer is not None


def test_exported_types_construct_smoke() -> None:
    blanket = BreedingBlanket(thickness_cm=80.0, li6_enrichment=0.9)
    assert blanket.thickness == 80.0
    pwi = SputteringPhysics("Tungsten")
    assert pwi.material == "Tungsten"
    temhd = TEMHD_Stabilizer(layer_thickness_mm=5.0, B_field=10.0)
    assert temhd.N > 0
