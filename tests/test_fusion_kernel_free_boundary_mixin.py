# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FusionKernel Free-Boundary Mixin Delegation Tests
"""Real-surface delegation tests for the free-boundary FusionKernel methods.

Exercise the public/private free-boundary methods that route through
``FusionKernelFreeBoundaryMixin`` on a real :class:`FusionKernel` grid and check
that each forwards faithfully to its validated adapter in
:mod:`scpn_fusion.core.fusion_kernel_free_boundary`. Complements
``tests/test_coil_optimization.py`` (which covers the coilset-config,
external-flux, mutual-inductance, boundary-flux, optimisation, interpolation, and
outer-loop routes) to complete the mixin's coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core import fusion_kernel_free_boundary as runtime
from scpn_fusion.core.fusion_kernel import CoilSet, FusionKernel

_MOCK_CONFIG = {
    "reactor_name": "Free-Boundary-Mixin-Test",
    "grid_resolution": [20, 20],
    "dimensions": {"R_min": 4.0, "R_max": 8.0, "Z_min": -4.0, "Z_max": 4.0},
    "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
    "coils": [{"name": "CS", "r": 1.7, "z": 0.0, "current": 0.15}],
    "solver": {
        "max_iterations": 10,
        "convergence_threshold": 1e-4,
        "relaxation_factor": 0.1,
    },
}

_FLUX_POINTS = np.array([[5.0, -0.5], [6.5, 0.5]], dtype=np.float64)
_B_PROBE_POINTS = np.array([[5.5, 0.0], [6.0, 0.0]], dtype=np.float64)
_B_PROBE_DIRECTIONS = ["R", "Z"]
_TRUE_CURRENTS = np.array([0.6, -0.4], dtype=np.float64)


@pytest.fixture
def kernel(tmp_path: Path) -> FusionKernel:
    """Build a small real FusionKernel with the free-boundary mixin attached."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(_MOCK_CONFIG), encoding="utf-8")
    return FusionKernel(str(cfg))


def _coils() -> CoilSet:
    """Return a two-coil external set bracketing the plasma grid."""
    return CoilSet(
        positions=[(3.0, -2.0), (9.0, 2.0)],
        currents=_TRUE_CURRENTS.copy(),
        turns=[8, 8],
    )


def test_green_function_forwards_to_runtime(kernel: FusionKernel) -> None:
    """The ``_green_function`` static wrapper matches the adapter value."""
    value = kernel._green_function(5.0, 0.0, 6.0, 0.5)
    assert value == pytest.approx(runtime.green_function(5.0, 0.0, 6.0, 0.5))


def test_probe_response_matrix_forwards_to_runtime(kernel: FusionKernel) -> None:
    """The probe-response wrapper reproduces the adapter response matrix."""
    coils = _coils()
    via_mixin = kernel._build_magnetic_probe_response_matrix(
        coils,
        flux_points=_FLUX_POINTS,
        b_probe_points=_B_PROBE_POINTS,
        b_probe_directions=_B_PROBE_DIRECTIONS,
    )
    via_runtime = runtime.build_magnetic_probe_response_matrix(
        kernel,
        coils,
        flux_points=_FLUX_POINTS,
        b_probe_points=_B_PROBE_POINTS,
        b_probe_directions=_B_PROBE_DIRECTIONS,
    )
    assert via_mixin.shape == (4, 2)
    np.testing.assert_allclose(via_mixin, via_runtime)


def test_reconstruct_coil_currents_recovers_true_currents(kernel: FusionKernel) -> None:
    """The reconstruction wrapper recovers the synthesised coil currents."""
    coils = _coils()
    response = runtime.build_magnetic_probe_response_matrix(
        kernel,
        coils,
        flux_points=_FLUX_POINTS,
        b_probe_points=_B_PROBE_POINTS,
        b_probe_directions=_B_PROBE_DIRECTIONS,
    )
    measurements = response @ _TRUE_CURRENTS
    n_flux = _FLUX_POINTS.shape[0]

    result = kernel.reconstruct_coil_currents_from_magnetic_probes(
        coils,
        flux_points=_FLUX_POINTS,
        flux_measurements=measurements[:n_flux],
        b_probe_points=_B_PROBE_POINTS,
        b_probe_directions=_B_PROBE_DIRECTIONS,
        b_probe_measurements=measurements[n_flux:],
        tikhonov_alpha=0.0,
    )
    np.testing.assert_allclose(result["coil_currents"], _TRUE_CURRENTS, atol=1e-9)


def test_resolve_shape_target_flux_returns_explicit_values(kernel: FusionKernel) -> None:
    """The shape-target wrapper returns the coil set's explicit target values."""
    target_values = np.array([0.1, 0.2], dtype=np.float64)
    coils = CoilSet(
        positions=[(3.0, -2.0), (9.0, 2.0)],
        currents=_TRUE_CURRENTS.copy(),
        turns=[8, 8],
        target_flux_points=_FLUX_POINTS,
        target_flux_values=target_values,
    )
    resolved = kernel._resolve_shape_target_flux(coils)
    np.testing.assert_allclose(resolved, target_values)
