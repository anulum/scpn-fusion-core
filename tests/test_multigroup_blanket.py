# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Multi-Group Blanket Neutronics Tests
"""Tests for the 3-group cylindrical blanket neutronics model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_fusion.nuclear.multigroup_blanket import MultiGroupBlanket


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


def test_multigroup_rejects_cell_count_below_minimum() -> None:
    """n_cells below the integer minimum is rejected by the guard."""
    with pytest.raises(ValueError, match="n_cells"):
        MultiGroupBlanket(thickness_cm=80.0, n_cells=2)


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
