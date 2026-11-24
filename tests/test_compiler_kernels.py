# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests: compiler forward-pass kernels and compile guards

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from scpn_fusion.neurocore_compat import (
    RNG,
    SC_NEUROCORE_AVAILABLE,
    generate_bernoulli_bitstream,
    pack_bitstream,
)
from scpn_fusion.scpn import compiler as compiler_mod
from scpn_fusion.scpn.compiler import CompiledNet, FusionCompiler
from scpn_fusion.scpn.structure import StochasticPetriNet


def _identity_net(n_transitions: int = 1, *, bitstream_length: int = 128) -> CompiledNet:
    """A minimal CompiledNet with no LIF neurons (pure numpy comparator path)."""
    return CompiledNet(
        n_places=1,
        n_transitions=n_transitions,
        place_names=["p0"],
        transition_names=[f"t{i}" for i in range(n_transitions)],
        W_in=np.zeros((n_transitions, 1), dtype=np.float64),
        W_out=np.zeros((1, n_transitions), dtype=np.float64),
        neurons=[],
        thresholds=np.full(n_transitions, 0.5, dtype=np.float64),
        firing_mode="binary",
        bitstream_length=bitstream_length,
    )


class TestLifFireNumpyPath:
    """Binary firing without sc_neurocore neurons uses the numpy threshold."""

    def test_threshold_comparator_without_neurons(self) -> None:
        net = _identity_net(n_transitions=2)
        fired = net.lif_fire(np.array([1.0, 0.0]))
        assert fired.tolist() == [1.0, 0.0]


class TestDenseForwardGuards:
    """``dense_forward`` requires sc_neurocore and matches its own fallback."""

    def test_raises_without_sc_neurocore(self, monkeypatch: pytest.MonkeyPatch) -> None:
        net = _identity_net(bitstream_length=64)
        monkeypatch.setattr(compiler_mod, "_HAS_SC_NEUROCORE", False)
        with pytest.raises(RuntimeError, match="dense_forward requires sc_neurocore"):
            net.dense_forward(np.zeros((1, 1, 1), dtype=np.uint64), np.array([0.5]))

    @pytest.mark.parametrize(("n_out", "n_in", "length"), [(1, 1, 128), (3, 4, 256), (2, 5, 64)])
    def test_vectorised_popcount_matches_a_per_stream_oracle(
        self, n_out: int, n_in: int, length: int
    ) -> None:
        """The vectorised AND + bitwise_count equals an independent per-stream count."""
        assert SC_NEUROCORE_AVAILABLE
        net = _identity_net(bitstream_length=length)
        rng = np.random.default_rng(20261115)
        n_words = length // 64
        w_packed = rng.integers(0, 2**64, size=(n_out, n_in, n_words), dtype=np.uint64)
        w_packed[0, 0, 0] = np.uint64(0xFFFFFFFFFFFFFFFF)  # every bit set
        w_packed[-1, -1, -1] = np.uint64(0)  # no bit set
        input_probs = rng.uniform(0.0, 1.0, size=n_in)
        vectorised = net.dense_forward(w_packed, input_probs)

        # Oracle: rebuild the same input streams, then count each AND word by
        # Python's int.bit_count, one stream pair at a time.
        expected = np.zeros(n_out, dtype=np.float64)
        for j in range(n_in):
            bits = generate_bernoulli_bitstream(
                float(np.clip(input_probs[j], 0.0, 1.0)),
                length,
                rng=RNG(net.seed + 1_000_000 + j),
            )
            stream = pack_bitstream(bits)
            for i in range(n_out):
                words = np.bitwise_and(w_packed[i, j, :], stream)
                expected[i] += sum(int(word).bit_count() for word in words)
        expected /= length

        assert vectorised.shape == (n_out,)
        np.testing.assert_array_equal(vectorised, expected)


class TestCompileWeightGuards:
    """``compile`` fails closed when the Petri net yields no weight matrix."""

    def test_raises_when_w_in_missing(self) -> None:
        compiler = FusionCompiler(bitstream_length=64)
        net = cast(
            "StochasticPetriNet",
            SimpleNamespace(is_compiled=True, W_in=None, W_out=None),
        )
        with pytest.raises(RuntimeError, match="no W_in matrix"):
            compiler.compile(net)

    def test_raises_when_w_out_missing(self) -> None:
        compiler = FusionCompiler(bitstream_length=64)
        net = cast(
            "StochasticPetriNet",
            SimpleNamespace(is_compiled=True, W_in=object(), W_out=None),
        )
        with pytest.raises(RuntimeError, match="no W_out matrix"):
            compiler.compile(net)


def _relay_net() -> StochasticPetriNet:
    """Two places joined through one transition: p_in -> t_relay -> p_out."""
    net = StochasticPetriNet()
    net.add_place("p_in", initial_tokens=0.75)
    net.add_place("p_out", initial_tokens=0.0)
    net.add_transition("t_relay", threshold=0.25)
    net.add_arc("p_in", "t_relay", weight=1.0)
    net.add_arc("t_relay", "p_out", weight=0.5)
    return net


class TestCompileWithoutScNeurocore:
    """Without sc_neurocore, ``compile`` builds the dense net with no stochastic path."""

    def test_dense_contract_without_neurons_or_packed_weights(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The compiled net keeps its dense weights and fires through the numpy comparator."""
        monkeypatch.setattr(compiler_mod, "_HAS_SC_NEUROCORE", False)
        compiled = FusionCompiler(bitstream_length=64, seed=7).compile(_relay_net())

        assert compiled.neurons == []
        assert compiled.W_in_packed is None
        assert compiled.W_out_packed is None
        assert compiled.has_stochastic_path is False
        np.testing.assert_array_equal(compiled.W_in, np.array([[1.0, 0.0]]))
        np.testing.assert_array_equal(compiled.W_out, np.array([[0.0], [0.5]]))
        np.testing.assert_array_equal(compiled.thresholds, np.array([0.25]))
        np.testing.assert_array_equal(compiled.initial_marking, np.array([0.75, 0.0]))
        # The numpy threshold comparator fires the relay once its input crosses 0.25.
        assert compiled.lif_fire(np.array([0.75])).tolist() == [1.0]
        assert compiled.lif_fire(np.array([0.10])).tolist() == [0.0]

    @pytest.mark.skipif(not SC_NEUROCORE_AVAILABLE, reason="sc_neurocore not installed")
    def test_dense_matrices_match_the_sc_neurocore_compile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dropping the backend changes only the stochastic path, not the dense contract."""
        with_backend = FusionCompiler(bitstream_length=64, seed=7).compile(_relay_net())
        monkeypatch.setattr(compiler_mod, "_HAS_SC_NEUROCORE", False)
        without_backend = FusionCompiler(bitstream_length=64, seed=7).compile(_relay_net())

        assert with_backend.has_stochastic_path is True
        np.testing.assert_array_equal(without_backend.W_in, with_backend.W_in)
        np.testing.assert_array_equal(without_backend.W_out, with_backend.W_out)
        np.testing.assert_array_equal(without_backend.thresholds, with_backend.thresholds)
