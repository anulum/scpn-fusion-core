# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — MAST Validation Tool Tests

from __future__ import annotations

from pathlib import Path

import numpy as np

from scpn_fusion.io.mast_ingestor import default_mast_cache_dir
from tools.train_mast_snn import HardwareSNN, initialise_base_weights, resolve_mast_cache_dir


def test_default_mast_cache_dir_stays_under_repo_data() -> None:
    cache_dir = default_mast_cache_dir()

    assert cache_dir == Path.cwd() / "data" / "mast_cache"


def test_environment_override_controls_mast_cache_dir(monkeypatch, tmp_path) -> None:
    cache_dir = tmp_path / "mast-cache"
    monkeypatch.setenv("SCPN_MAST_CACHE_DIR", str(cache_dir))

    assert resolve_mast_cache_dir() == cache_dir


def test_hardware_snn_base_weights_are_deterministic() -> None:
    first = initialise_base_weights(n_neurons=8, seed=123)
    second = initialise_base_weights(n_neurons=8, seed=123)

    np.testing.assert_allclose(first, second)


def test_hardware_snn_requires_valid_neuron_count() -> None:
    try:
        HardwareSNN(n_neurons=0)
    except ValueError as exc:
        assert "n_neurons" in str(exc)
    else:
        raise AssertionError("HardwareSNN accepted a non-positive neuron count")
