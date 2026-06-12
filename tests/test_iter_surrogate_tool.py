# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — ITER Surrogate Tool Tests

from __future__ import annotations

import numpy as np

from tools.train_iter_surrogate import default_iter_dataset_paths, load_iter_dataset


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
