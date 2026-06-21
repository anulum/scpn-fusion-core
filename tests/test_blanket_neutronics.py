# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Blanket Neutronics Tests
"""Unit tests for the reduced 1D and 3-group volumetric breeding-blanket surrogates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_fusion.engineering.cad_raytrace import estimate_surface_loading
from scpn_fusion.nuclear.blanket_neutronics import (
    BreedingBlanket,
    MultiGroupBlanket,
    VolumetricBlanketReport,
    run_breeding_sim,
)


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
    """The 1D blanket constructor rejects non-physical geometry/enrichment."""
    with pytest.raises(ValueError, match=match):
        BreedingBlanket(**kwargs)


def test_volumetric_surrogate_returns_finite_positive_report() -> None:
    """The volumetric TBR surrogate returns a finite, positive report."""
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
    """A thicker blanket captures more neutrons, raising the volumetric TBR."""
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
    """Higher Li-6 enrichment increases thermal capture and the volumetric TBR."""
    low = BreedingBlanket(thickness_cm=80.0, li6_enrichment=0.5)
    high = BreedingBlanket(thickness_cm=80.0, li6_enrichment=0.95)

    low_report = low.calculate_volumetric_tbr(radial_cells=8, poloidal_cells=16, toroidal_cells=12)
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
def test_volumetric_surrogate_rejects_invalid_inputs(kwargs: dict[str, Any], match: str) -> None:
    """The volumetric TBR surrogate rejects non-physical geometry/mesh/flux inputs."""
    blanket = BreedingBlanket(thickness_cm=80.0, li6_enrichment=0.9)
    with pytest.raises(ValueError, match=match):
        blanket.calculate_volumetric_tbr(**kwargs)


def test_rear_albedo_increases_1d_tbr() -> None:
    """A reflecting rear boundary returns neutrons and raises the 1D TBR."""
    blanket = BreedingBlanket(thickness_cm=80.0, li6_enrichment=0.9)
    phi_sink = blanket.solve_transport(incident_flux=1.0e14, rear_albedo=0.0)
    phi_reflect = blanket.solve_transport(incident_flux=1.0e14, rear_albedo=0.7)
    tbr_sink, _ = blanket.calculate_tbr(phi_sink)
    tbr_reflect, _ = blanket.calculate_tbr(phi_reflect)
    assert tbr_reflect > tbr_sink


def test_rear_albedo_requires_valid_range() -> None:
    """The 1D solver rejects a rear albedo outside [0, 1)."""
    blanket = BreedingBlanket(thickness_cm=80.0, li6_enrichment=0.9)
    with pytest.raises(ValueError, match="rear_albedo"):
        blanket.solve_transport(rear_albedo=-0.1)
    with pytest.raises(ValueError, match="rear_albedo"):
        blanket.solve_transport(rear_albedo=1.0)


def test_incident_flux_requires_positive_finite_value() -> None:
    """The 1D solver rejects a non-positive or non-finite incident flux."""
    blanket = BreedingBlanket(thickness_cm=80.0, li6_enrichment=0.9)
    with pytest.raises(ValueError, match="incident_flux"):
        blanket.solve_transport(incident_flux=0.0)
    with pytest.raises(ValueError, match="incident_flux"):
        blanket.solve_transport(incident_flux=float("nan"))


def test_run_breeding_sim_returns_finite_summary_without_plot() -> None:
    """run_breeding_sim returns a complete finite summary when plotting is off."""
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
    for key in ("tbr", "flux_peak", "production_peak"):
        value = summary[key]
        assert isinstance(value, float)
        assert np.isfinite(value)


def test_run_breeding_sim_is_deterministic_for_fixed_inputs() -> None:
    """run_breeding_sim is bit-for-bit reproducible for identical inputs."""
    a = run_breeding_sim(
        thickness_cm=85.0,
        li6_enrichment=0.85,
        incident_flux=9.0e13,
        rear_albedo=0.4,
        save_plot=False,
        verbose=False,
    )
    b = run_breeding_sim(
        thickness_cm=85.0,
        li6_enrichment=0.85,
        incident_flux=9.0e13,
        rear_albedo=0.4,
        save_plot=False,
        verbose=False,
    )
    assert a["tbr"] == b["tbr"]
    assert a["flux_peak"] == b["flux_peak"]
    assert a["production_peak"] == b["production_peak"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"thickness_cm": 0.0}, "thickness_cm"),
        ({"thickness_cm": float("nan")}, "thickness_cm"),
        ({"li6_enrichment": -0.1}, "li6_enrichment"),
        ({"li6_enrichment": 1.1}, "li6_enrichment"),
    ],
)
def test_run_breeding_sim_rejects_invalid_inputs(kwargs: dict[str, Any], match: str) -> None:
    """run_breeding_sim validates thickness and enrichment before solving."""
    with pytest.raises(ValueError, match=match):
        run_breeding_sim(save_plot=False, verbose=False, **kwargs)


def test_run_breeding_sim_verbose_path_runs(capsys: pytest.CaptureFixture[str]) -> None:
    """The verbose branch emits progress output and still returns a summary."""
    summary = run_breeding_sim(thickness_cm=80.0, li6_enrichment=0.9, save_plot=False, verbose=True)
    captured = capsys.readouterr()
    assert "TBR" in captured.out
    assert isinstance(summary["tbr"], float)


def test_cad_surface_loading_smoke() -> None:
    """The CAD ray-trace surface-loading estimate is finite and ordered."""
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


def test_solve_transport_rejects_supercritical_low_enrichment() -> None:
    """The 1D solver rejects a supercritical (negative net-removal) blanket."""
    # Net removal = absorption - net (n,2n) source; it goes negative
    # (supercritical) below ~0.29 Li-6 enrichment with the default Be multiplier,
    # which would otherwise yield an unphysical negative breeding ratio.
    blanket = BreedingBlanket(thickness_cm=100.0, li6_enrichment=0.07)
    with pytest.raises(ValueError, match="Supercritical blanket"):
        blanket.solve_transport()


def test_solve_transport_breeding_ratio_positive_for_enriched_blanket() -> None:
    """A well-enriched 1D blanket yields a positive breeding ratio."""
    blanket = BreedingBlanket(thickness_cm=100.0, li6_enrichment=0.9)
    tbr, _ = blanket.calculate_tbr(blanket.solve_transport())
    assert tbr > 0.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"thickness_cm": 0.0}, "thickness_cm"),
        ({"r_inner_cm": 1.0}, "r_inner_cm"),
        ({"li6_enrichment": 1.5}, "li6_enrichment"),
    ],
)
def test_multigroup_constructor_rejects_invalid_inputs(kwargs: dict[str, Any], match: str) -> None:
    """The 3-group blanket constructor rejects non-physical geometry/enrichment."""
    with pytest.raises(ValueError, match=match):
        MultiGroupBlanket(**kwargs)


def test_multigroup_constructor_floors_cell_count_to_integer() -> None:
    """n_cells is coerced to at least the thickness-derived integer floor."""
    blanket = MultiGroupBlanket(thickness_cm=80.0, n_cells=10)
    assert isinstance(blanket.n_cells, int)
    assert blanket.n_cells >= int(80.0 * 2.5)
    assert blanket.r.shape == (blanket.n_cells,)


def test_multigroup_constructor_rejects_non_integer_cells() -> None:
    """A non-integer n_cells is rejected by the integer guard."""
    with pytest.raises(ValueError, match="n_cells"):
        MultiGroupBlanket(n_cells=10.5)  # type: ignore[arg-type]


def test_multigroup_solve_transport_returns_positive_breeding() -> None:
    """The 3-group solve returns finite group fluxes and a positive TBR breakdown."""
    blanket = MultiGroupBlanket(thickness_cm=80.0, li6_enrichment=0.9, n_cells=60)
    result = blanket.solve_transport(incident_flux=1.0e14)
    for key in ("phi_g1", "phi_g2", "phi_g3", "total_production"):
        flux = result[key]
        assert isinstance(flux, np.ndarray)
        assert flux.shape == (blanket.n_cells,)
        assert np.all(np.isfinite(flux))
        assert np.all(flux >= 0.0)
    tbr = result["tbr"]
    assert isinstance(tbr, float)
    assert tbr > 0.0
    by_group = result["tbr_by_group"]
    assert isinstance(by_group, dict)
    assert set(by_group) == {"fast", "epithermal", "thermal"}
    assert by_group["thermal"] >= by_group["fast"]
    assert isinstance(result["flux_clamp_total"], int)


def test_multigroup_thermal_dominates_breeding() -> None:
    """Thermal Li-6 capture dominates the breeding contribution at high enrichment."""
    blanket = MultiGroupBlanket(thickness_cm=90.0, li6_enrichment=0.95, n_cells=60)
    result = blanket.solve_transport(incident_flux=1.0e14)
    by_group = result["tbr_by_group"]
    assert isinstance(by_group, dict)
    assert by_group["thermal"] > by_group["epithermal"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"incident_flux": 0.0}, "incident_flux"),
        ({"port_coverage_factor": 0.0}, "port_coverage_factor"),
        ({"port_coverage_factor": 1.5}, "port_coverage_factor"),
        ({"streaming_factor": 0.0}, "streaming_factor"),
        ({"streaming_factor": 1.5}, "streaming_factor"),
    ],
)
def test_multigroup_solve_transport_rejects_invalid_inputs(
    kwargs: dict[str, float], match: str
) -> None:
    """The 3-group solve rejects non-physical flux and correction factors."""
    blanket = MultiGroupBlanket(thickness_cm=80.0, li6_enrichment=0.9, n_cells=40)
    with pytest.raises(ValueError, match=match):
        blanket.solve_transport(**kwargs)


def test_run_breeding_sim_writes_plot_artifact(tmp_path: object) -> None:
    """The save_plot branch renders and persists a PNG artifact."""
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    out = tmp_path / "tbr.png"
    summary = run_breeding_sim(
        thickness_cm=80.0,
        li6_enrichment=0.9,
        save_plot=True,
        output_path=str(out),
        verbose=True,
    )
    assert summary["plot_saved"] is True
    assert out.exists()


def test_multigroup_rejects_cell_count_below_minimum() -> None:
    """n_cells below the integer minimum is rejected by the guard."""
    with pytest.raises(ValueError, match="n_cells"):
        MultiGroupBlanket(thickness_cm=80.0, n_cells=2)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"port_coverage_factor": 0.0}, "port_coverage_factor"),
        ({"port_coverage_factor": 1.5}, "port_coverage_factor"),
        ({"streaming_factor": 0.0}, "streaming_factor"),
        ({"streaming_factor": 1.5}, "streaming_factor"),
        ({"blanket_fill_factor": 0.0}, "blanket_fill_factor"),
        ({"blanket_fill_factor": 1.5}, "blanket_fill_factor"),
    ],
)
def test_volumetric_surrogate_rejects_invalid_correction_factors(
    kwargs: dict[str, Any], match: str
) -> None:
    """The volumetric surrogate rejects correction factors outside (0, 1]."""
    blanket = BreedingBlanket(thickness_cm=80.0, li6_enrichment=0.9)
    with pytest.raises(ValueError, match=match):
        blanket.calculate_volumetric_tbr(
            radial_cells=8, poloidal_cells=16, toroidal_cells=12, **kwargs
        )


def test_run_breeding_sim_records_plot_error_on_render_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plotting failure is caught and reported without aborting the run."""
    import matplotlib.pyplot as plt

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("render backend unavailable")

    monkeypatch.setattr(plt, "savefig", _boom)
    summary = run_breeding_sim(thickness_cm=80.0, li6_enrichment=0.9, save_plot=True, verbose=True)
    assert summary["plot_saved"] is False
    assert isinstance(summary["plot_error"], str)
    assert "render backend unavailable" in summary["plot_error"]


def test_multigroup_cylindrical_group_supports_neumann_right_boundary() -> None:
    """The single-group cylindrical solve honours a Neumann right boundary."""
    blanket = MultiGroupBlanket(thickness_cm=80.0, li6_enrichment=0.9, n_cells=40)
    source = np.ones(blanket.n_cells, dtype=np.float64)
    phi = blanket._solve_cylindrical_group(
        D=0.5,
        sigma_rem=0.1,
        source=source,
        bc_left=("dirichlet", 1.0e14),
        bc_right=("neumann", 0.0),
    )
    assert phi.shape == (blanket.n_cells,)
    assert np.all(np.isfinite(phi))
