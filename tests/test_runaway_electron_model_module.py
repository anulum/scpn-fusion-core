# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Direct module-link tests for the extracted runaway electron model.

Exercises the Rosenbluth-Putvinski avalanche model end-to-end plus every
input-validation raise and every physics/numerical guard branch: sub-Dreicer
fields, strong-screening ratios, disabled relativistic losses, cold-electron
hot-tail suppression, and the overflow re-guards that keep the fixed-step
integrator degrading gracefully rather than emitting non-finite output.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from scpn_fusion.control.runaway_electron_model import (
    RunawayElectronModel,
    RunawayElectronResult,
)


class TestSimulate:
    """End-to-end integration behaviour of ``simulate``."""

    def test_simulate_smoke(self) -> None:
        """A nominal quench returns a well-formed, finite result."""
        model = RunawayElectronModel(n_e=1e20, T_e_keV=20.0, z_eff=2.0)
        out = model.simulate(duration_s=0.005, dt_s=5e-5)
        assert isinstance(out, RunawayElectronResult)
        assert out.peak_re_current_ma >= 0.0
        assert out.final_re_current_ma >= 0.0
        assert out.avalanche_gain >= 1.0

    def test_simulate_series_lengths_and_finiteness(self) -> None:
        """Every returned time series has one sample per integration step."""
        model = RunawayElectronModel()
        out = model.simulate(duration_s=0.01, dt_s=1e-4)
        n = len(out.time_ms)
        assert n == 100
        for series in (
            out.runaway_current_ma,
            out.dreicer_rate_per_s,
            out.avalanche_rate_per_s,
            out.electric_field_v_m,
        ):
            assert len(series) == n
            assert all(np.isfinite(v) for v in series)

    def test_minimum_step_floor(self) -> None:
        """A very short duration is floored to at least ten integration steps."""
        model = RunawayElectronModel()
        out = model.simulate(duration_s=1e-4, dt_s=1e-4)
        assert len(out.time_ms) == 10

    def test_explicit_neon_mol_overrides_default(self) -> None:
        """Passing ``neon_mol`` mutates the impurity state used by the run."""
        model = RunawayElectronModel(neon_mol=0.0)
        model.simulate(neon_mol=0.5, duration_s=0.005, dt_s=5e-5)
        assert model.neon_mol == 0.5
        assert model.n_e_tot > model.n_e_free

    def test_heavy_neon_deconfinement_suppresses_avalanche(self) -> None:
        """Above the 0.3 mol neon threshold the avalanche term is deconfined."""
        model = RunawayElectronModel()
        model._update_impurity_state(neon_mol=0.5, z_eff=3.0)
        suppressed = model._avalanche_rate(1.0, 1e15)
        model._update_impurity_state(neon_mol=0.0, z_eff=1.0)
        nominal = model._avalanche_rate(1.0, 1e15)
        assert 0.0 <= suppressed < nominal


class TestInputValidation:
    """Constructor and ``simulate`` domain guards raise ``ValueError``."""

    @pytest.mark.parametrize(
        "build",
        [
            lambda: RunawayElectronModel(n_e=float("nan")),
            lambda: RunawayElectronModel(n_e=float("inf")),
            lambda: RunawayElectronModel(T_e_keV=float("nan")),
        ],
    )
    def test_non_finite_constructor_input_raises(
        self, build: Callable[[], RunawayElectronModel]
    ) -> None:
        """Non-finite plasma parameters are rejected at construction."""
        with pytest.raises(ValueError, match="finite"):
            build()

    @pytest.mark.parametrize("value", [-1.0, 0.0])
    def test_non_positive_density_raises(self, value: float) -> None:
        """A density that is not strictly positive is rejected."""
        with pytest.raises(ValueError, match="> 0"):
            RunawayElectronModel(n_e=value)

    def test_negative_neon_mol_raises(self) -> None:
        """A negative neon molar concentration is rejected."""
        with pytest.raises(ValueError, match=">= 0"):
            RunawayElectronModel(neon_mol=-1.0)

    def test_z_eff_below_one_raises(self) -> None:
        """An effective charge below unity is unphysical and rejected."""
        with pytest.raises(ValueError, match="z_eff must be >= 1.0"):
            RunawayElectronModel(z_eff=0.5)

    def test_step_larger_than_duration_raises(self) -> None:
        """A step size exceeding the total duration is rejected."""
        model = RunawayElectronModel()
        with pytest.raises(ValueError, match="must be <= duration_s"):
            model.simulate(duration_s=0.001, dt_s=0.01)

    @pytest.mark.parametrize("seed", [0.0, 2.0, -0.1, float("nan")])
    def test_seed_fraction_out_of_domain_raises(self, seed: float) -> None:
        """A seed runaway fraction outside ``(0, 1]`` is rejected."""
        model = RunawayElectronModel()
        with pytest.raises(ValueError):
            model.simulate(seed_re_fraction=seed)


class TestDreicerRate:
    """Dreicer generation guards across field and temperature regimes."""

    def test_non_finite_field_returns_zero(self) -> None:
        """A non-finite field yields no Dreicer generation."""
        model = RunawayElectronModel()
        assert model._dreicer_rate(float("inf"), 20.0) == 0.0

    @pytest.mark.parametrize("e_field,t_e", [(0.0, 20.0), (1.0, 0.005)])
    def test_sub_threshold_field_or_cold_electrons_return_zero(
        self, e_field: float, t_e: float
    ) -> None:
        """A non-positive field or near-zero temperature yields no generation."""
        model = RunawayElectronModel()
        assert model._dreicer_rate(e_field, t_e) == 0.0

    def test_strong_screening_ratio_returns_zero(self) -> None:
        """A very small field drives ``E_D/E`` past the screening cut-off."""
        model = RunawayElectronModel()
        assert model._dreicer_rate(1e-30, 20.0) == 0.0

    def test_nominal_field_gives_positive_rate(self) -> None:
        """A field near the Dreicer field produces a finite positive rate."""
        model = RunawayElectronModel()
        rate = model._dreicer_rate(model.E_D * 0.2, 0.5)
        assert rate > 0.0 and np.isfinite(rate)


class TestAvalancheRate:
    """Avalanche generation guards."""

    def test_non_finite_input_returns_zero(self) -> None:
        """A non-finite field yields no avalanche growth."""
        model = RunawayElectronModel()
        assert model._avalanche_rate(float("inf"), 1e15) == 0.0

    def test_sub_critical_field_returns_zero(self) -> None:
        """Below the critical field the avalanche does not grow."""
        model = RunawayElectronModel()
        assert model._avalanche_rate(model.E_c * 0.5, 1e15) == 0.0

    def test_growth_overflow_is_guarded(self) -> None:
        """An extreme runaway population overflows growth and is guarded."""
        model = RunawayElectronModel()
        assert model._avalanche_rate(1e6, 1e308) == 0.0


class TestFokkerPlanckGeneration:
    """Hot-tail Fokker-Planck seed generation guards."""

    def test_non_finite_input_returns_zero(self) -> None:
        """A non-finite field yields no hot-tail seed."""
        model = RunawayElectronModel()
        assert model._fokker_planck_generation(float("inf"), 1e15) == 0.0

    @pytest.mark.parametrize("e_field,t_e", [(0.0, 0.5), (1.0, 0.005)])
    def test_sub_critical_or_cold_returns_zero(self, e_field: float, t_e: float) -> None:
        """A sub-critical field or near-zero temperature suppresses the seed."""
        model = RunawayElectronModel()
        assert model._fokker_planck_generation(e_field, 1e15, T_e_keV=t_e) == 0.0

    def test_cold_electron_hot_tail_underflow_returns_zero(self) -> None:
        """A cold-electron regime underflows the exponential to zero."""
        model = RunawayElectronModel()
        assert model._fokker_planck_generation(0.078, 1e15, T_e_keV=0.02) == 0.0

    def test_nominal_regime_gives_positive_seed(self) -> None:
        """A super-critical field with warm electrons produces a positive seed."""
        model = RunawayElectronModel()
        rate = model._fokker_planck_generation(model.E_c * 5.0, 1e15, T_e_keV=0.5)
        assert rate > 0.0 and np.isfinite(rate)


class TestRelativisticLossRate:
    """Synchrotron/bremsstrahlung loss guards."""

    def test_disabled_returns_zero(self) -> None:
        """With relativistic losses disabled the rate is identically zero."""
        model = RunawayElectronModel(enable_relativistic_losses=False)
        assert model._relativistic_loss_rate(E=1.0, n_re=1e15) == 0.0

    @pytest.mark.parametrize("e_field,n_re", [(float("inf"), 1e15), (1.0, 0.0)])
    def test_non_finite_or_empty_population_returns_zero(self, e_field: float, n_re: float) -> None:
        """A non-finite field or empty population yields no loss."""
        model = RunawayElectronModel()
        assert model._relativistic_loss_rate(E=e_field, n_re=n_re) == 0.0

    def test_loss_overflow_is_guarded(self) -> None:
        """An extreme population overflows the loss and is guarded to zero."""
        model = RunawayElectronModel()
        assert model._relativistic_loss_rate(E=1.0, n_re=1e308) == 0.0

    def test_nominal_population_gives_positive_loss(self) -> None:
        """A finite super-critical population produces a positive loss."""
        model = RunawayElectronModel()
        loss = model._relativistic_loss_rate(E=model.E_c * 5.0, n_re=1e15)
        assert loss > 0.0 and np.isfinite(loss)


class TestIntegratorRobustness:
    """The fixed-step integrator degrades gracefully on non-finite sub-terms."""

    def test_non_finite_generation_term_is_absorbed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A sub-term returning non-finite leaves the trajectory finite.

        Monkeypatching the Dreicer term to a non-finite value drives the
        source rate non-finite; the increment guard must zero the step so the
        integrator continues and every emitted sample stays finite.
        """
        model = RunawayElectronModel()
        monkeypatch.setattr(model, "_dreicer_rate", lambda E, T_e_keV: float("inf"))
        out = model.simulate(duration_s=0.005, dt_s=5e-5)
        assert np.isfinite(out.peak_re_current_ma)
        assert np.isfinite(out.final_re_current_ma)
        assert all(np.isfinite(v) for v in out.runaway_current_ma)
