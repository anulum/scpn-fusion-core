# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — ITER Surrogate Tool Tests

from __future__ import annotations

import numpy as np

from tools.train_iter_surrogate import default_iter_dataset_paths, inspect_iter_dataset, load_iter_dataset


def test_default_iter_dataset_paths_are_directory_relative(tmp_path) -> None:
    x_path, y_path = default_iter_dataset_paths(tmp_path)

    assert x_path == tmp_path / "iter_X.npy"
    assert y_path == tmp_path / "iter_Y.npy"


def test_load_iter_dataset_from_directory(tmp_path) -> None:
    x = np.arange(6, dtype=np.float64).reshape(2, 3)
    y = np.arange(8, dtype=np.float64).reshape(2, 4)
    np.save(tmp_path / "iter_X.npy", x)
    np.save(tmp_path / "iter_Y.npy", y)

    loaded_x, loaded_y = load_iter_dataset(tmp_path)

    np.testing.assert_allclose(loaded_x, x)
    np.testing.assert_allclose(loaded_y, y)


def test_load_iter_dataset_from_npz(tmp_path) -> None:
    x = np.arange(4, dtype=np.float64).reshape(2, 2)
    y = np.arange(6, dtype=np.float64).reshape(2, 3)
    dataset_path = tmp_path / "iter_dataset.npz"
    np.savez(dataset_path, X=x, Y=y)

    loaded_x, loaded_y = load_iter_dataset(dataset_path)

    np.testing.assert_allclose(loaded_x, x)
    np.testing.assert_allclose(loaded_y, y)


def test_iter_dataset_report_distinguishes_development_from_full_fidelity() -> None:
    x = np.ones((10, 12), dtype=np.float64)
    y = np.ones((10, 16), dtype=np.float64)

    dev_report = inspect_iter_dataset(x, y, min_full_fidelity_samples=50)
    ready_report = inspect_iter_dataset(x, y, min_full_fidelity_samples=10)

    assert dev_report["status"] == "development_dataset_below_full_fidelity_sample_count"
    assert ready_report["status"] == "full_fidelity_iter_dataset_ready"


def test_iter_dataset_report_blocks_invalid_shapes_and_nonfinite_values() -> None:
    x = np.ones((10, 11), dtype=np.float64)
    y = np.ones((10, 16), dtype=np.float64)
    assert inspect_iter_dataset(x, y)["status"] == "blocked_invalid_feature_shape"

    x = np.ones((10, 12), dtype=np.float64)
    y = np.ones((9, 16), dtype=np.float64)
    assert inspect_iter_dataset(x, y)["status"] == "blocked_invalid_field_shape"

    y = np.ones((10, 16), dtype=np.float64)
    y[0, 0] = np.nan
    assert inspect_iter_dataset(x, y)["status"] == "blocked_nonfinite_values"
