# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Split Module Smoke Tests

from __future__ import annotations

import numpy as np

import scpn_fusion.core.fno_training_multi_regime as mr


def test_split_module_exports_regime_constants() -> None:
    assert {"itg", "tem", "etg"}.issubset(set(mr.SPARC_REGIMES))


def test_split_module_sampling_and_generation_are_callable() -> None:
    params = mr._sample_regime_params(np.random.default_rng(1), "itg")
    assert "alpha" in params and np.isfinite(params["alpha"])

    x, y, meta = mr._generate_multi_regime_pairs(n_samples=3, grid_size=16, seed=2)
    assert x.shape == (3, 16, 16)
    assert y.shape == (3, 16, 16)
    assert len(meta) == 3
