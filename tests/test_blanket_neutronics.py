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

import numpy as np
import pytest

from scpn_fusion.engineering.cad_raytrace import estimate_surface_loading


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "scpn_fusion" / "nuclear" / "blanket_neutronics.py"
SPEC = importlib.util.spec_from_file_location("blanket_neutronics", MODULE_PATH)
assert SPEC and SPEC.loader
blanket_neutronics = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = blanket_neutronics
SPEC.loader.exec_module(blanket_neutronics)

BreedingBlanket = blanket_neutronics.BreedingBlanket
VolumetricBlanketReport = blanket_neutronics.VolumetricBlanketReport
run_breeding_sim = blanket_neutronics.run_breeding_sim


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"thickness_cm": 0.0}, "thickness_cm"),
        ({"thickness_cm": float("nan")}, "thickness_cm"),
        ({"li6_enrichment": -0.1}, "li6_enrichment"),
        ({"li6_enrichment": 1.1}, "li6_enrichment"),
    ],
)
def test_breeding_blanket_constructor_rejects_invalid_inputs(
    kwargs: dict[str, float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        BreedingBlanket(**kwargs)


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


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"major_radius_m": 0.05}, "major_radius_m"),
        ({"minor_radius_m": 0.01}, "minor_radius_m"),
        ({"elongation": 0.0}, "elongation"),
        ({"radial_cells": 1}, "radial_cells"),
        ({"poloidal_cells": 4}, "poloidal_cells"),
        ({"toroidal_cells": 4}, "toroidal_cells"),
        ({"incident_flux": 0.0}, "incident_flux"),
        ({"incident_flux": float("nan")}, "incident_flux"),
    ],
)
def test_volumetric_surrogate_rejects_invalid_inputs(
    kwargs: dict[str, float | int], match: str
) -> None:
    blanket = BreedingBlanket(thickness_cm=80.0, li6_enrichment=0.9)
    with pytest.raises(ValueError, match=match):
        blanket.calculate_volumetric_tbr(**kwargs)


def test_rear_albedo_increases_1d_tbr() -> None:
    blanket = BreedingBlanket(thickness_cm=80.0, li6_enrichment=0.9)
    phi_sink = blanket.solve_transport(incident_flux=1.0e14, rear_albedo=0.0)
    phi_reflect = blanket.solve_transport(incident_flux=1.0e14, rear_albedo=0.7)
    tbr_sink, _ = blanket.calculate_tbr(phi_sink)
    tbr_reflect, _ = blanket.calculate_tbr(phi_reflect)
    assert tbr_reflect > tbr_sink


def test_rear_albedo_requires_valid_range() -> None:
    blanket = BreedingBlanket(thickness_cm=80.0, li6_enrichment=0.9)
    with pytest.raises(ValueError, match="rear_albedo"):
        blanket.solve_transport(rear_albedo=-0.1)
    with pytest.raises(ValueError, match="rear_albedo"):
        blanket.solve_transport(rear_albedo=1.0)


def test_incident_flux_requires_positive_finite_value() -> None:
    blanket = BreedingBlanket(thickness_cm=80.0, li6_enrichment=0.9)
    with pytest.raises(ValueError, match="incident_flux"):
        blanket.solve_transport(incident_flux=0.0)
    with pytest.raises(ValueError, match="incident_flux"):
        blanket.solve_transport(incident_flux=float("nan"))


def test_run_breeding_sim_returns_finite_summary_without_plot() -> None:
    summary = run_breeding_sim(
        thickness_cm=80.0,
        li6_enrichment=0.9,
        incident_flux=1.0e14,
        rear_albedo=0.5,
        save_plot=False,
        verbose=False,
    )
    for key in (
        "thickness_cm",
        "li6_enrichment",
        "incident_flux",
        "rear_albedo",
        "tbr",
        "status",
        "flux_peak",
        "flux_mean",
        "production_peak",
        "production_mean",
        "plot_saved",
    ):
        assert key in summary
    assert summary["plot_saved"] is False
    assert np.isfinite(summary["tbr"])
    assert np.isfinite(summary["flux_peak"])
    assert np.isfinite(summary["production_peak"])


def test_run_breeding_sim_is_deterministic_for_fixed_inputs() -> None:
    kwargs = dict(
        thickness_cm=85.0,
        li6_enrichment=0.85,
        incident_flux=9.0e13,
        rear_albedo=0.4,
        save_plot=False,
        verbose=False,
    )
    a = run_breeding_sim(**kwargs)
    b = run_breeding_sim(**kwargs)
    assert a["tbr"] == b["tbr"]
    assert a["flux_peak"] == b["flux_peak"]
    assert a["production_peak"] == b["production_peak"]


def test_cad_surface_loading_smoke() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=np.int64,
    )
    source_points = np.asarray([[2.0, 2.0, 2.0], [1.5, 2.0, 1.2]], dtype=np.float64)
    source_strength = np.asarray([4.0e6, 2.0e6], dtype=np.float64)
    report = estimate_surface_loading(vertices, faces, source_points, source_strength)
    assert report.face_loading_w_m2.shape == (4,)
    assert np.all(np.isfinite(report.face_loading_w_m2))
    assert report.peak_loading_w_m2 >= report.mean_loading_w_m2 >= 0.0
