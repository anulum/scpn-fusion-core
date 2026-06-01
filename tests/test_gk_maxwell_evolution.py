#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for local electromagnetic Maxwell field evolution evidence."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.gk_maxwell_evolution import (
    MaxwellEvolutionConfig,
    run_local_maxwell_evolution,
)


def test_local_maxwell_evolution_conserves_source_free_field_energy() -> None:
    result = run_local_maxwell_evolution(
        MaxwellEvolutionConfig(n_kx=5, n_ky=5, n_steps=12, dt=2.5e-12, seed=17)
    )

    assert result.schema == "gk-maxwell-evolution.v1"
    assert result.faraday_induction_supported is True
    assert result.ampere_maxwell_displacement_current_supported is True
    assert result.inductive_parallel_electric_field_supported is True
    assert result.self_consistent_kinetic_current_supported is False
    assert np.all(np.isfinite(result.total_field_energy_t))
    assert result.relative_total_field_energy_drift <= result.relative_energy_tolerance
    assert result.max_faraday_linf_residual <= result.residual_tolerance
    assert result.max_ampere_maxwell_linf_residual <= result.residual_tolerance
    assert result.max_inductive_e_parallel_linf_residual <= result.residual_tolerance
    assert result.max_magnetic_divergence_linf_residual <= result.residual_tolerance
    assert result.A_parallel_energy_t.shape == result.time_s.shape
    assert result.B_parallel_energy_t.shape == result.time_s.shape


def test_local_maxwell_evolution_rejects_unstable_courant_step() -> None:
    config = MaxwellEvolutionConfig(n_kx=4, n_ky=4, n_steps=2, dt=1.0e-6)

    with pytest.raises(ValueError, match="Courant"):
        run_local_maxwell_evolution(config)
