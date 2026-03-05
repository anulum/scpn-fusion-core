# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Embedded NeuroCore Compatibility Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np

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
