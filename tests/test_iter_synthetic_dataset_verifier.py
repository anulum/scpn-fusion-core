# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — tests for the ITER-like dataset provenance verifier
"""Tests for deterministic ITER-like dataset provenance verification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scpn_fusion.core.neural_equilibrium import NeuralEquilibriumAccelerator
from tools.verify_iter_synthetic_dataset import (
    expected_plasma_current_feature,
    sha256_file,
    verify_dataset,
)

REPO = Path(__file__).resolve().parents[1]


def test_expected_plasma_current_feature_preserves_worker_partition() -> None:
    values = expected_plasma_current_feature(
        samples=5,
        workers=2,
        base_seed=42,
        coil_count=2,
        base_current=15.0,
        coil_multiplier_range=(0.85, 1.15),
        current_multiplier_range=(0.8, 1.2),
        divisor=1_000_000.0,
    )
    expected: list[float] = []
    for seed, count in ((42, 3), (43, 2)):
        rng = np.random.default_rng(seed)
        for _ in range(count):
            rng.uniform(0.85, 1.15, size=2)
            expected.append(15.0 * rng.uniform(0.8, 1.2) / 1_000_000.0)
    np.testing.assert_array_equal(values, expected)


def test_verify_dataset_checks_hash_shape_rng_and_finiteness(tmp_path: Path) -> None:
    x = np.zeros((5, 2), dtype=np.float64)
    x[:, 0] = expected_plasma_current_feature(
        samples=5,
        workers=2,
        base_seed=42,
        coil_count=2,
        base_current=15.0,
        coil_multiplier_range=(0.85, 1.15),
        current_multiplier_range=(0.8, 1.2),
        divisor=1_000_000.0,
    )
    y = np.arange(20, dtype=np.float64).reshape(5, 4)
    x_path = tmp_path / "iter_X.npy"
    y_path = tmp_path / "iter_Y.npy"
    np.save(x_path, x)
    np.save(y_path, y)
    manifest = {
        "dataset_id": "test",
        "generation": {
            "requested_samples": 5,
            "workers": 2,
            "worker_seeds": [42, 43],
            "coil_count": 2,
            "coil_current_multiplier_range": [0.85, 1.15],
            "plasma_current_base_config_value": 15.0,
            "plasma_current_multiplier_range": [0.8, 1.2],
            "plasma_current_feature_divisor": 1_000_000.0,
        },
        "arrays": {
            "X": {
                "release_asset": x_path.name,
                "shape": [5, 2],
                "dtype": "float64",
                "size_bytes": x_path.stat().st_size,
                "sha256": sha256_file(x_path),
            },
            "Y": {
                "release_asset": y_path.name,
                "shape": [5, 4],
                "dtype": "float64",
                "size_bytes": y_path.stat().st_size,
                "sha256": sha256_file(y_path),
            },
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = verify_dataset(
        data_dir=tmp_path, manifest_path=manifest_path, full_field_scan=True
    )

    assert result["status"] == "passed"
    assert result["rng_rows_verified"] == 5
    assert result["failures"] == []


def test_published_v2_weights_match_selection_report_and_runtime() -> None:
    report = json.loads(
        (REPO / "validation/reports/iter_surrogate_v2_selection.json").read_text(
            encoding="utf-8"
        )
    )
    weights_path = REPO / report["artifact"]

    assert sha256_file(weights_path) == report["artifact_sha256"]
    accelerator = NeuralEquilibriumAccelerator()
    accelerator.load_weights(weights_path)
    with np.load(weights_path, allow_pickle=False) as weights:
        input_mean = np.asarray(weights["input_mean"], dtype=np.float64)
    prediction = accelerator.predict(input_mean)

    assert prediction.shape == tuple(report["runtime_contract"]["runtime_prediction_shape"])
    assert np.all(np.isfinite(prediction))
