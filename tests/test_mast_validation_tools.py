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
from tools.train_mast_snn import (
    HardwareSNN,
    classify_full_fidelity_status,
    initialise_base_weights,
    load_local_npz_shot,
    resolve_mast_cache_dir,
)


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


def test_local_npz_shot_loader_reads_materialised_mast_artifact(tmp_path) -> None:
    time = np.linspace(0.0, 1.0, 6)
    ip = np.array([0.0, 0.9, 1.0, 0.95, 0.1, 0.0])
    magnetic = np.arange(12, dtype=np.float64).reshape(2, 6)
    np.savez(
        tmp_path / "mast_shot_12345.npz",
        shot_id=np.array([12345]),
        time=time,
        ip=ip,
        mag_b_field_pol_probe_cc_field=magnetic,
    )

    trace = load_local_npz_shot(tmp_path, 12345)

    assert trace is not None
    assert trace.source == "local_npz:mast_shot_12345.npz"
    np.testing.assert_allclose(trace.time_s, time)
    np.testing.assert_allclose(trace.plasma_current_a, ip)
    assert trace.magnetic_trace_t.shape == time.shape


def test_full_fidelity_status_blocks_until_enough_real_shots_are_detected() -> None:
    assert (
        classify_full_fidelity_status(
            train_available_count=1,
            validation_report=[{"shot_id": 1, "status": "detected"}],
            min_train_shots=3,
            min_validation_shots=3,
        )
        == "blocked_insufficient_training_shots"
    )
    assert (
        classify_full_fidelity_status(
            train_available_count=3,
            validation_report=[{"shot_id": 1, "status": "detected"}],
            min_train_shots=3,
            min_validation_shots=3,
        )
        == "blocked_insufficient_detected_validation_shots"
    )
    assert (
        classify_full_fidelity_status(
            train_available_count=3,
            validation_report=[
                {"shot_id": 1, "status": "detected"},
                {"shot_id": 2, "status": "detected"},
                {"shot_id": 3, "status": "detected"},
            ],
            min_train_shots=3,
            min_validation_shots=3,
        )
        == "full_fidelity_local_evidence_ready"
    )
