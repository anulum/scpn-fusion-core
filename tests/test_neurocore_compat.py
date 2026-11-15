# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Embedded NeuroCore Compatibility Tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.neurocore_compat import (
    RNG,
    SC_NEUROCORE_AVAILABLE,
    StochasticLIFNeuron,
    generate_bernoulli_bitstream,
    pack_bitstream,
    vec_and,
    vec_popcount,
)


def test_embedded_backend_is_available() -> None:
    assert SC_NEUROCORE_AVAILABLE is True


def test_bitstream_mean_tracks_probability() -> None:
    L = 4096
    rng = RNG(123)
    bits = generate_bernoulli_bitstream(0.3, L, rng=rng)
    est = float(np.mean(bits))
    assert abs(est - 0.3) < 3.0 / np.sqrt(L)


def test_packed_and_popcount_product_estimate() -> None:
    L = 4096
    bw = generate_bernoulli_bitstream(0.7, L, rng=RNG(1))
    bx = generate_bernoulli_bitstream(0.4, L, rng=RNG(2))
    est = vec_popcount(vec_and(pack_bitstream(bw), pack_bitstream(bx))) / L
    assert abs(est - 0.28) < 3.0 / np.sqrt(L)


@pytest.mark.parametrize(
    "words",
    [
        [0],
        [0xFFFFFFFFFFFFFFFF],
        [1 << 63, 1],
        [0xF0F0F0F0F0F0F0F0, 0x0F0F0F0F0F0F0F0F, 0x8000000000000001],
    ],
)
def test_vec_popcount_matches_python_bit_count(words: list[int]) -> None:
    """Vectorised popcount equals int.bit_count on boundary words (empty, full, top bit)."""
    packed = np.asarray(words, dtype=np.uint64)
    assert vec_popcount(packed) == sum(word.bit_count() for word in words)


def test_vec_popcount_flattens_any_shape_and_handles_empty_input() -> None:
    rng = np.random.default_rng(7)
    packed = rng.integers(0, 2**64, size=(3, 4, 5), dtype=np.uint64)
    assert vec_popcount(packed) == sum(int(word).bit_count() for word in packed.reshape(-1))
    assert vec_popcount(np.zeros(0, dtype=np.uint64)) == 0


def test_lif_neuron_emits_spikes_for_sustained_drive() -> None:
    neuron = StochasticLIFNeuron(
        v_threshold=0.5,
        tau_mem=5.0,
        dt=1.0,
        noise_std=0.0,
        resistance=1.0,
        seed=7,
    )
    spikes = [neuron.step(1.0) for _ in range(20)]
    assert any(spikes)


def test_stochastic_lif_replay_is_deterministic_for_fixed_seed() -> None:
    neuron_inputs = np.linspace(0.0, 0.8, 32)

    def run_once(seed: int) -> list[bool]:
        neuron = StochasticLIFNeuron(
            v_threshold=0.5,
            tau_mem=5.0,
            dt=1.0,
            noise_std=0.15,
            resistance=1.0,
            seed=seed,
        )
        return [neuron.step(float(v)) for v in neuron_inputs]

    out_a = run_once(42)
    out_b = run_once(42)
    np.testing.assert_array_equal(out_a, out_b)
