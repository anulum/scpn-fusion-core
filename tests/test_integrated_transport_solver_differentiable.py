# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Differentiable Coupled Transport Tests
"""Public/runtime tests for four-state transport gradients and optimisation."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core import CoupledTransportControls, CoupledTransportTarget
from validation import benchmark_torax_differentiable_transport as benchmark


def test_public_differentiable_rollout_matches_production_and_fd() -> None:
    reference = benchmark.load_reference()
    case = reference["primary"]
    target = benchmark._target(case)
    controls = CoupledTransportControls()
    result = benchmark._solver(case).differentiate_coupled_transport(
        benchmark._input_history(), controls, target
    )
    production = benchmark.production_rollout(case, controls)
    differentiated = (
        result.ion_temperature_kev,
        result.electron_temperature_kev,
        result.electron_density_1e19_m3,
        result.poloidal_flux_wb_per_rad,
    )
    for observed, expected in zip(differentiated, production, strict=True):
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1.0e-12)
    finite_difference = benchmark._finite_difference(case, target)
    relative_error = np.abs(result.gradient - finite_difference) / np.maximum(
        np.maximum(np.abs(result.gradient), np.abs(finite_difference)), 1.0e-14
    )
    assert np.all(np.isfinite(result.gradient))
    assert np.all(np.abs(result.gradient) >= 1.0e-12)
    assert float(np.max(relative_error)) <= 0.01


def test_bounded_optimisation_is_deterministic_and_improves_objective() -> None:
    reference = benchmark.load_reference()
    case = reference["primary"]
    target = benchmark._target(case)
    solver = benchmark._solver(case)
    first = solver.optimise_coupled_transport(
        benchmark._input_history(),
        CoupledTransportControls(),
        target,
        iterations=4,
        learning_rate=0.08,
    )
    second = solver.optimise_coupled_transport(
        benchmark._input_history(),
        CoupledTransportControls(),
        target,
        iterations=4,
        learning_rate=0.08,
    )
    assert first.final_objective < first.initial_objective
    np.testing.assert_array_equal(first.objective_history, second.objective_history)
    np.testing.assert_array_equal(first.control_history, second.control_history)
    assert np.all((first.control_history >= 0.5) & (first.control_history <= 1.5))


@pytest.mark.parametrize(
    "controls",
    [
        (0.0, 1.0, 1.0),
        (1.0, float("nan"), 1.0),
        (1.0, 1.0, -1.0),
    ],
)
def test_controls_reject_invalid_values(controls: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        CoupledTransportControls(*controls)


def test_target_and_optimiser_reject_invalid_contracts() -> None:
    reference = benchmark.load_reference()
    case = reference["primary"]
    target = benchmark._target(case)
    solver = benchmark._solver(case)
    invalid = CoupledTransportTarget(
        target.ion_temperature_kev[:-1],
        target.electron_temperature_kev,
        target.electron_density_1e19_m3,
        target.poloidal_flux_wb_per_rad,
    )
    with pytest.raises(ValueError, match="match rho"):
        solver.differentiate_coupled_transport(
            benchmark._input_history(), CoupledTransportControls(), invalid
        )
    with pytest.raises(ValueError, match="iterations"):
        solver.optimise_coupled_transport(
            benchmark._input_history(), CoupledTransportControls(), target, iterations=0
        )
