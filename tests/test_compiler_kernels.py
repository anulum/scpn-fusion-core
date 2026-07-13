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
from numpy.typing import NDArray

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

    def test_vectorized_bit_count_matches_popcount_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if not getattr(compiler_mod, "_HAS_SC_NEUROCORE", False):
            pytest.skip("dense_forward numeric path requires sc_neurocore")
        net = _identity_net(bitstream_length=128)  # 128 bits -> 2 uint64 words
        w_packed = np.array([[[0xF0F0F0F0F0F0F0F0, 0x0F0F0F0F0F0F0F0F]]], dtype=np.uint64)
        input_probs = np.array([0.7])

        # Fallback path (no numpy bit_count): explicit sc_neurocore popcount.
        fallback = net.dense_forward(w_packed, input_probs)

        def _bit_count(values: NDArray[np.uint64]) -> NDArray[np.uint64]:
            popcount = np.vectorize(lambda x: int(x).bit_count(), otypes=[np.uint64])
            return cast("NDArray[np.uint64]", popcount(values))

        # Vectorized path taken once numpy exposes bit_count; must agree exactly.
        monkeypatch.setattr(np, "bit_count", _bit_count, raising=False)
        vectorized = net.dense_forward(w_packed, input_probs)

        assert np.allclose(vectorized, fallback)


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
