# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the neuro-symbolic controller backend runtime mixin.

The mixin dispatches the controller's numeric kernels across a NumPy path and
an optional Rust path. These tests drive it through a controlled state double
over both backends, covering dense activations, marking update, the float
oracle step, the deterministic stochastic step (fractional/binary firing, the
antithetic and binomial samplers, and the Rust firing sampler), and the
transition-timing delay ring.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_fusion.scpn.controller as controller_mod
from scpn_fusion.scpn.controller_backend_mixin import (
    NeuroSymbolicControllerBackendMixin,
    _controller_module,
)
from scpn_fusion.scpn.controller_features_mixin import NeuroSymbolicControllerFeaturesMixin

_NT = 3
_NP = 4


class _Controller(NeuroSymbolicControllerBackendMixin, NeuroSymbolicControllerFeaturesMixin):
    """State double supplying the buffers and configuration the mixin dispatches on."""

    def __init__(self, *, backend: str = "numpy") -> None:
        rng = np.random.default_rng(7)
        self._runtime_backend = backend
        self._W_in = rng.random((_NT, _NP))
        self._W_out = rng.random((_NP, _NT))
        self._W_in_t = np.ascontiguousarray(self._W_in.T)
        self._tmp_activations = np.zeros(_NT, dtype=np.float64)
        self._tmp_consumption = np.zeros(_NP, dtype=np.float64)
        self._tmp_production = np.zeros(_NP, dtype=np.float64)
        self._tmp_marking_oracle = np.zeros(_NP, dtype=np.float64)
        self._tmp_marking_sc = np.zeros(_NP, dtype=np.float64)

        self._firing_mode = "fractional"
        self._margins = np.full(_NT, 0.5, dtype=np.float64)
        self._thresholds = np.full(_NT, 0.3, dtype=np.float64)

        self._oracle_pending = np.zeros((1, _NT), dtype=np.float64)
        self._oracle_cursor = 0
        self._sc_pending = np.zeros((1, _NT), dtype=np.float64)
        self._sc_cursor = 0

        self._sc_binary_margin = 0.2
        self._sc_n_passes = 1
        self._sc_antithetic = False
        self._sc_bitflip_rate = 0.0
        self.seed_base = 12345
        self._tmp_sc_counts = np.zeros(_NT, dtype=np.int64)
        self._nT = _NT
        self._sc_antithetic_chunk_size = 16

        # No transition delay by default.
        self._max_delay_ticks = 0
        self._delay_immediate_idx = np.zeros(0, dtype=np.int64)
        self._delay_delayed_idx = np.zeros(0, dtype=np.int64)
        self._delay_delayed_offsets = np.zeros(0, dtype=np.int64)
        self._tmp_delay_slots = np.zeros(0, dtype=np.int64)

    def _enable_delays(self) -> None:
        """Switch on a 2-tick delay ring with one immediate and two delayed transitions."""
        self._max_delay_ticks = 2
        ring = self._max_delay_ticks + 1
        self._oracle_pending = np.zeros((ring, _NT), dtype=np.float64)
        self._sc_pending = np.zeros((ring, _NT), dtype=np.float64)
        self._delay_immediate_idx = np.array([0], dtype=np.int64)
        self._delay_delayed_idx = np.array([1, 2], dtype=np.int64)
        self._delay_delayed_offsets = np.array([1, 2], dtype=np.int64)
        self._tmp_delay_slots = np.zeros(2, dtype=np.int64)


def _marking() -> NDArray[np.float64]:
    """A bounded marking vector in the unit interval."""
    return np.linspace(0.2, 0.9, _NP)


def test_controller_module_resolves_lazily() -> None:
    """The lazy helper resolves the host controller module."""
    assert _controller_module() is controller_mod


class TestDenseActivations:
    """Dense activation kernel over both backends."""

    @pytest.mark.parametrize("backend", ["numpy", "rust"])
    def test_activations_match_reference(self, backend: str) -> None:
        """Both backends reproduce the W_in @ marking activations."""
        ctrl = _Controller(backend=backend)
        marking = _marking()
        out = ctrl._dense_activations(marking)
        assert np.allclose(out, ctrl._W_in @ marking)


class TestMarkingUpdate:
    """Marking-update kernel over both backends."""

    @pytest.mark.parametrize("backend", ["numpy", "rust"])
    def test_update_stays_in_unit_interval(self, backend: str) -> None:
        """The updated marking is clamped to the unit interval on both backends."""
        ctrl = _Controller(backend=backend)
        marking = _marking()
        firing = np.full(_NT, 0.5, dtype=np.float64)
        out = np.zeros(_NP, dtype=np.float64)
        result = ctrl._marking_update(marking, firing, out)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


class TestOracleStep:
    """Float-path Petri oracle step."""

    def test_fractional_firing(self) -> None:
        """Fractional mode produces graded firing and a bounded next marking."""
        ctrl = _Controller()
        f, m2 = ctrl._oracle_step(_marking())
        assert f.shape == (_NT,)
        assert np.all((f >= 0.0) & (f <= 1.0))
        assert np.all((m2 >= 0.0) & (m2 <= 1.0))

    def test_binary_firing(self) -> None:
        """Binary mode produces a 0/1 firing vector."""
        ctrl = _Controller()
        ctrl._firing_mode = "binary"
        f, _ = ctrl._oracle_step(_marking())
        assert set(np.unique(f)).issubset({0.0, 1.0})


class TestStochasticStep:
    """Deterministic stochastic step across firing modes and samplers."""

    def test_single_pass_fractional(self) -> None:
        """A single-pass fractional step returns the firing probability directly."""
        ctrl = _Controller()
        f, m2 = ctrl._sc_step(_marking(), k=0)
        assert np.all((f >= 0.0) & (f <= 1.0))
        assert np.all((m2 >= 0.0) & (m2 <= 1.0))

    def test_single_pass_binary_with_margin(self) -> None:
        """Binary mode with a soft margin produces a graded firing probability."""
        ctrl = _Controller()
        ctrl._firing_mode = "binary"
        f, _ = ctrl._sc_step(_marking(), k=1)
        assert np.all((f >= 0.0) & (f <= 1.0))

    def test_single_pass_binary_hard_threshold(self) -> None:
        """Binary mode without a margin produces a hard 0/1 firing vector."""
        ctrl = _Controller()
        ctrl._firing_mode = "binary"
        ctrl._sc_binary_margin = 0.0
        f, _ = ctrl._sc_step(_marking(), k=2)
        assert set(np.unique(f)).issubset({0.0, 1.0})

    def test_multi_pass_binomial_numpy(self) -> None:
        """A multi-pass non-antithetic step uses the binomial sampler."""
        ctrl = _Controller()
        ctrl._sc_n_passes = 4
        f, _ = ctrl._sc_step(_marking(), k=3)
        assert np.all((f >= 0.0) & (f <= 1.0))

    def test_multi_pass_antithetic_even_vectorised(self) -> None:
        """An even-pass antithetic step over the vectorised path stays bounded."""
        ctrl = _Controller()
        ctrl._sc_n_passes = 4
        ctrl._sc_antithetic = True
        f, _ = ctrl._sc_step(_marking(), k=4)
        assert np.all((f >= 0.0) & (f <= 1.0))

    def test_multi_pass_antithetic_odd_vectorised(self) -> None:
        """An odd-pass antithetic step over the vectorised path stays bounded."""
        ctrl = _Controller()
        ctrl._sc_n_passes = 5
        ctrl._sc_antithetic = True
        f, _ = ctrl._sc_step(_marking(), k=5)
        assert np.all((f >= 0.0) & (f <= 1.0))

    def test_multi_pass_antithetic_even_chunked(self) -> None:
        """An even-pass antithetic step over the chunked path stays bounded."""
        ctrl = _Controller()
        ctrl._sc_n_passes = 4
        ctrl._sc_antithetic = True
        ctrl._sc_antithetic_chunk_size = 2
        f, _ = ctrl._sc_step(_marking(), k=6)
        assert np.all((f >= 0.0) & (f <= 1.0))

    def test_multi_pass_antithetic_odd_chunked(self) -> None:
        """An odd-pass antithetic step over the chunked path stays bounded."""
        ctrl = _Controller()
        ctrl._sc_n_passes = 5
        ctrl._sc_antithetic = True
        ctrl._sc_antithetic_chunk_size = 2
        f, _ = ctrl._sc_step(_marking(), k=7)
        assert np.all((f >= 0.0) & (f <= 1.0))

    def test_rust_sampler_multi_pass(self) -> None:
        """The Rust firing sampler drives the multi-pass step when the backend is Rust."""
        ctrl = _Controller(backend="rust")
        ctrl._sc_n_passes = 4
        f, _ = ctrl._sc_step(_marking(), k=8)
        assert np.all((f >= 0.0) & (f <= 1.0))

    def test_bitflip_faults_applied_single_pass(self) -> None:
        """A positive bit-flip rate injects faults into firing and marking outputs."""
        ctrl = _Controller()
        ctrl._sc_bitflip_rate = 0.5
        f, m2 = ctrl._sc_step(_marking(), k=9)
        assert np.all((f >= 0.0) & (f <= 1.0))
        assert np.all((m2 >= 0.0) & (m2 <= 1.0))

    def test_rust_sampler_with_bitflip(self) -> None:
        """The Rust sampler path also honours bit-flip fault injection."""
        ctrl = _Controller(backend="rust")
        ctrl._sc_n_passes = 4
        ctrl._sc_bitflip_rate = 0.5
        f, m2 = ctrl._sc_step(_marking(), k=10)
        assert np.all((f >= 0.0) & (f <= 1.0))
        assert np.all((m2 >= 0.0) & (m2 <= 1.0))


class TestRustKernelConsistencyGuards:
    """Guards for an inconsistent Rust runtime (advertised available, kernel missing)."""

    def test_dense_missing_kernel_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing Rust dense kernel raises, then falls back to the NumPy path."""
        monkeypatch.setattr(controller_mod, "_rust_dense_activations", None, raising=False)
        ctrl = _Controller(backend="rust")
        marking = _marking()
        out = ctrl._dense_activations(marking)
        assert np.allclose(out, ctrl._W_in @ marking)
        assert ctrl._runtime_backend == "numpy"

    def test_marking_missing_kernel_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing Rust marking kernel raises, then falls back to the NumPy path."""
        monkeypatch.setattr(controller_mod, "_rust_marking_update", None, raising=False)
        ctrl = _Controller(backend="rust")
        out = np.zeros(_NP, dtype=np.float64)
        result = ctrl._marking_update(_marking(), np.full(_NT, 0.5), out)
        assert np.all((result >= 0.0) & (result <= 1.0))
        assert ctrl._runtime_backend == "numpy"


class TestTransitionTiming:
    """Transition-timing delay ring."""

    def test_no_delay_passes_through(self) -> None:
        """With no delay configured the desired firing passes straight through."""
        ctrl = _Controller()
        desired = np.array([0.2, 1.5, -0.3], dtype=np.float64)
        fired, cursor = ctrl._apply_transition_timing(desired, ctrl._sc_pending, 0)
        assert np.allclose(fired, np.clip(desired, 0.0, 1.0))
        assert cursor == 0

    def test_delayed_firing_routes_through_ring(self) -> None:
        """Immediate transitions fire now; delayed transitions are buffered ahead."""
        ctrl = _Controller()
        ctrl._enable_delays()
        desired = np.full(_NT, 1.0, dtype=np.float64)
        fired, next_cursor = ctrl._apply_transition_timing(desired, ctrl._sc_pending, 0)
        # The immediate transition fires this tick; delayed ones are still pending.
        assert fired[0] == 1.0
        assert next_cursor == 1
        assert np.any(ctrl._sc_pending > 0.0)
