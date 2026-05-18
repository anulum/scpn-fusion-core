# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Acceptance-gate regressions for GPU neural-equilibrium training."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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
