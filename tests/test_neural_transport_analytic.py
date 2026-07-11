# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the analytic reduced-gyrokinetic transport closure.

Covers the gyro-Bohm estimate, dominant-channel selection, the single-point
critical-gradient model, and the radial profile driver together with its full
input-validation contract — the fallback closure the neural transport backend
uses when no trained surrogate is available.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_fusion.core._neural_transport_analytic import (
    _TRANSPORT_FLOOR,
    _dominant_channel,
    _gyro_bohm_diffusivity,
    critical_gradient_model,
    reduced_gyrokinetic_profile_model,
)
from scpn_fusion.core._neural_transport_types import TransportInputs


def _inputs(**overrides: float) -> TransportInputs:
    """Build a driven H-mode-like transport input with optional overrides."""
    base: dict[str, float] = {"grad_te": 20.0, "grad_ti": 15.0, "grad_ne": 4.0}
    base.update(overrides)
    return TransportInputs(**base)


class TestGyroBohmDiffusivity:
    """Gyro-Bohm diffusivity estimate."""

    def test_nominal_diffusivity_is_positive_finite(self) -> None:
        """A nominal input yields a positive, finite diffusivity above the floor."""
        chi_gb = _gyro_bohm_diffusivity(_inputs())
        assert np.isfinite(chi_gb)
        assert chi_gb >= _TRANSPORT_FLOOR

    def test_non_finite_input_falls_to_floor(self) -> None:
        """A non-finite temperature collapses the estimate to the transport floor."""
        chi_gb = _gyro_bohm_diffusivity(_inputs(te_kev=float("inf")))
        assert chi_gb == _TRANSPORT_FLOOR


class TestDominantChannel:
    """Dominant-channel selection."""

    def test_itg_dominates_when_ion_channel_strongest(self) -> None:
        """The ITG channel wins when its combined strength is largest."""
        channel = _dominant_channel(chi_i_itg=3.0, chi_e_itg=1.0, chi_e_tem=1.0, chi_e_etg=0.5)
        assert channel == "ITG"

    def test_all_zero_strengths_report_stable(self) -> None:
        """A fully quenched state reports the stable channel."""
        channel = _dominant_channel(chi_i_itg=0.0, chi_e_itg=0.0, chi_e_tem=0.0, chi_e_etg=0.0)
        assert channel == "stable"


class TestCriticalGradientModel:
    """Single-point critical-gradient closure."""

    def test_driven_input_produces_positive_fluxes(self) -> None:
        """Strong gradients above threshold drive positive multichannel fluxes."""
        fluxes = critical_gradient_model(_inputs())
        assert fluxes.chi_i >= 0.0
        assert fluxes.chi_e >= 0.0
        assert fluxes.channel in {"ITG", "TEM", "ETG", "stable"}

    def test_quiescent_input_is_stable(self) -> None:
        """Below-threshold gradients leave the plasma in the stable channel."""
        fluxes = critical_gradient_model(_inputs(grad_te=0.0, grad_ti=0.0))
        assert fluxes.channel == "stable"
        assert fluxes.chi_i == 0.0

    def test_out_of_range_stiffness_raises(self) -> None:
        """A stiffness exponent outside the physical band is rejected."""
        with pytest.raises(ValueError, match="stiffness"):
            critical_gradient_model(_inputs(), stiffness=0.5)


def _profile(n: int = 8, rho_max: float = 0.95) -> dict[str, Any]:
    """Build a monotonic, physically ordered radial profile of length ``n``."""
    rho = np.linspace(0.1, rho_max, n)
    te = np.linspace(8.0, 1.0, n)
    ti = np.linspace(7.0, 1.0, n)
    ne = np.linspace(6.0, 2.0, n)
    q = np.linspace(1.0, 4.0, n)
    s_hat = np.linspace(0.2, 2.0, n)
    return {"rho": rho, "te": te, "ti": ti, "ne": ne, "q_profile": q, "s_hat_profile": s_hat}


class TestReducedProfileModel:
    """Radial profile driver and its validation contract."""

    def test_edge_profile_returns_full_metadata(self) -> None:
        """An edge-reaching profile returns coefficient arrays and metadata."""
        chi_e, chi_i, d_e, meta = reduced_gyrokinetic_profile_model(**_profile())
        assert chi_e.shape == chi_i.shape == d_e.shape == (8,)
        assert meta["model"] == "reduced_multichannel_analytic"
        assert set(meta["channel_counts"]) == {"ITG", "TEM", "ETG", "stable"}
        assert 0.0 <= meta["edge_etg_fraction"] <= 1.0

    def test_core_only_profile_zeroes_edge_fraction(self) -> None:
        """A profile confined to the core reports a zero edge-ETG fraction."""
        _, _, _, meta = reduced_gyrokinetic_profile_model(**_profile(rho_max=0.6))
        assert meta["edge_etg_fraction"] == 0.0

    def test_flat_profile_reports_stable_dominant(self) -> None:
        """A gradient-free profile drives no transport, so the dominant channel is stable."""
        prof = _profile()
        prof["te"] = np.full(8, 5.0)
        prof["ti"] = np.full(8, 5.0)
        prof["ne"] = np.full(8, 5.0)
        chi_e, chi_i, _, meta = reduced_gyrokinetic_profile_model(**prof)
        assert meta["dominant_channel"] == "stable"
        assert np.allclose(chi_i, 0.0)
        assert np.allclose(chi_e, 0.0)

    def test_non_1d_array_raises(self) -> None:
        """A non-1D input array is rejected."""
        prof = _profile()
        prof["te"] = prof["te"].reshape(2, 4)
        with pytest.raises(ValueError, match="1D arrays"):
            reduced_gyrokinetic_profile_model(**prof)

    def test_too_few_points_raises(self) -> None:
        """A profile shorter than three points is rejected."""
        with pytest.raises(ValueError, match="at least 3 points"):
            reduced_gyrokinetic_profile_model(**_profile(n=2))

    def test_mismatched_length_raises(self) -> None:
        """Profile arrays of differing lengths are rejected."""
        prof = _profile()
        prof["te"] = prof["te"][:-1]
        with pytest.raises(ValueError, match="identical length"):
            reduced_gyrokinetic_profile_model(**prof)

    @pytest.mark.parametrize("field", ["rho", "te", "ti", "ne", "q_profile", "s_hat_profile"])
    def test_non_finite_field_raises(self, field: str) -> None:
        """A non-finite entry in any profile array is rejected."""
        prof = _profile()
        prof[field] = prof[field].copy()
        prof[field][1] = np.nan
        with pytest.raises(ValueError, match="finite values"):
            reduced_gyrokinetic_profile_model(**prof)

    def test_non_increasing_rho_raises(self) -> None:
        """A non-strictly-increasing rho grid is rejected."""
        prof = _profile()
        prof["rho"] = prof["rho"].copy()
        prof["rho"][2] = prof["rho"][1]
        with pytest.raises(ValueError, match="strictly increasing"):
            reduced_gyrokinetic_profile_model(**prof)

    def test_out_of_domain_rho_raises(self) -> None:
        """A rho grid outside ``[0, 1.2]`` is rejected."""
        prof = _profile(rho_max=1.5)
        with pytest.raises(ValueError, match="0 <= rho <= 1.2"):
            reduced_gyrokinetic_profile_model(**prof)

    @pytest.mark.parametrize("geom", ["r_major", "a_minor", "b_toroidal"])
    def test_non_positive_geometry_raises(self, geom: str) -> None:
        """A non-positive geometry scalar is rejected."""
        geom_override: dict[str, Any] = {geom: 0.0}
        with pytest.raises(ValueError, match=geom):
            reduced_gyrokinetic_profile_model(**_profile(), **geom_override)
