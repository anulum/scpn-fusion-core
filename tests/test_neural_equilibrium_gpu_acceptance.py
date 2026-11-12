# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Acceptance-gate regressions for GPU neural-equilibrium training."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "train_neural_equilibrium_gpu.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("train_neural_equilibrium_gpu", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_failed_acceptance_writes_only_rejected_weights(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    requested_path = tmp_path / "neural_equilibrium_augmented.npz"
    written_paths: list[Path] = []

    def fake_save_weights(path: Path, *args) -> None:
        written_paths.append(Path(path))

    monkeypatch.setattr(module, "save_weights", fake_save_weights)

    persisted = module.persist_training_artifacts(
        requested_path,
        mlp=object(),
        pca=object(),
        input_mean=object(),
        input_std=object(),
        result=object(),
        criteria_met=False,
    )

    assert persisted == requested_path.with_stem(requested_path.stem + "_rejected")
    assert written_paths == [requested_path.with_stem(requested_path.stem + "_rejected")]
    assert requested_path not in written_paths


def test_passed_acceptance_writes_requested_weights(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    requested_path = tmp_path / "neural_equilibrium_augmented.npz"
    written_paths: list[Path] = []

    monkeypatch.setattr(
        module, "save_weights", lambda path, *args: written_paths.append(Path(path))
    )

    persisted = module.persist_training_artifacts(
        requested_path,
        mlp=object(),
        pca=object(),
        input_mean=object(),
        input_std=object(),
        result=object(),
        criteria_met=True,
    )

    assert persisted == requested_path
    assert written_paths == [requested_path]


def test_load_and_normalise_uses_canonical_geqdsk_boundary_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    from scpn_fusion.core import eqdsk

    fake_eq = SimpleNamespace(
        r=np.linspace(1.0, 2.0, 4, dtype=np.float64),
        z=np.linspace(-1.0, 1.0, 4, dtype=np.float64),
        psirz=np.arange(16, dtype=np.float64).reshape(4, 4),
        simag=0.0,
        sibry=15.0,
        rbdry=np.array([1.0, 1.5, 2.0, 1.5, 1.0], dtype=np.float64),
        zbdry=np.array([0.0, 2.0, 0.0, -2.0, 0.0], dtype=np.float64),
        qpsi=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        current=1.5e6,
        bcentr=5.3,
        rmaxis=1.5,
        zmaxis=0.0,
    )
    monkeypatch.setattr(eqdsk, "read_geqdsk", lambda _path: fake_eq)

    features, fields, labels = module.load_and_normalise(
        {"test-machine": [tmp_path / "case.geqdsk"]},
        n_perturbations=0,
        rng=np.random.default_rng(7),
    )

    assert features.shape == (1, 12)
    assert fields.shape == (1, module.TARGET_GRID**2)
    assert labels == ["test-machine"]
    assert features[0, 8] == pytest.approx(4.0)
    assert features[0, 9] == pytest.approx(0.0)
    assert features[0, 10] == pytest.approx(0.0)
    assert features[0, 11] == pytest.approx(4.0)


def test_save_weights_writes_a_pickle_free_archive_and_metrics(tmp_path: Path) -> None:
    """Persist PCA, network and scalar metadata as plain arrays plus metrics JSON."""
    from scpn_fusion.io.safe_loaders import checked_np_load

    module = _load_module()
    weights = [np.ones((12, 3)), np.ones((3, 2))]
    biases = [np.zeros(3), np.zeros(2)]
    mlp = SimpleNamespace(weights=weights, biases=biases)
    pca = SimpleNamespace(
        n_components=2,
        mean_=np.arange(4.0),
        components_=np.eye(2, 4),
        explained_variance_ratio_=np.array([0.75, 0.25]),
    )
    result = SimpleNamespace(
        n_samples=10,
        n_components=2,
        explained_variance=1.0,
        final_train_loss=0.1,
        best_val_loss=0.2,
        test_mse=0.3,
        test_max_error=0.4,
        train_time_s=1.5,
        n_epochs_run=3,
        lr_initial=1e-3,
        lr_final=1e-4,
        per_machine={},
    )
    path = tmp_path / "weights" / "neural_equilibrium.npz"

    module.save_weights(path, mlp, pca, np.zeros(12), np.ones(12), result)

    with checked_np_load(path, allow_pickle=False) as archive:
        assert "allow_pickle" not in archive.files
        assert int(archive["n_layers"][0]) == 2
        assert int(archive["grid_nh"][0]) == module.TARGET_GRID
        np.testing.assert_array_equal(archive["pca_mean"], pca.mean_)
        np.testing.assert_array_equal(archive["w1"], weights[1])
    metrics = json.loads(path.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_samples"] == 10
    assert metrics["psi_normalized"] is True
