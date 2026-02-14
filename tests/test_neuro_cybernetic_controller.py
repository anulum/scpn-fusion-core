# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Neuro Cybernetic Controller Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Deterministic tests for reduced spiking-controller pool behavior."""

from __future__ import annotations

import numpy as np

from scpn_fusion.control.neuro_cybernetic_controller import SpikingControllerPool


def test_spiking_pool_is_deterministic_for_same_seed() -> None:
    kwargs = dict(
        n_neurons=24,
        gain=2.0,
        tau_window=8,
        seed=19,
        use_quantum=False,
    )
    p1 = SpikingControllerPool(**kwargs)
    p2 = SpikingControllerPool(**kwargs)
    o1 = np.asarray([p1.step(0.15) for _ in range(24)], dtype=np.float64)
    o2 = np.asarray([p2.step(0.15) for _ in range(24)], dtype=np.float64)
    np.testing.assert_allclose(o1, o2, atol=0.0, rtol=0.0)


def test_spiking_pool_push_pull_sign_response() -> None:
    pos_pool = SpikingControllerPool(
        n_neurons=20,
        gain=3.0,
        tau_window=6,
        seed=31,
        use_quantum=False,
    )
    neg_pool = SpikingControllerPool(
        n_neurons=20,
        gain=3.0,
        tau_window=6,
        seed=31,
        use_quantum=False,
    )

    pos = [pos_pool.step(0.2) for _ in range(32)]
    neg = [neg_pool.step(-0.2) for _ in range(32)]
    assert float(np.mean(pos[-8:])) > 0.0
    assert float(np.mean(neg[-8:])) < 0.0


def test_spiking_pool_exposes_backend_name() -> None:
    pool = SpikingControllerPool(n_neurons=8, gain=1.0, tau_window=4, seed=11)
    assert pool.backend in {"sc_neurocore", "numpy_lif"}
