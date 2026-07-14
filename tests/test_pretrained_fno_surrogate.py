# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Pretrained Eurofusion/JET FNO Surrogate Tests
"""Tests for the pretrained Eurofusion/JET FNO surrogate."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pytest

from scpn_fusion.core._pretrained_surrogate_config import DEFAULT_JET_DIR
from scpn_fusion.core.pretrained_fno_surrogate import (
    _build_jet_fno_dataset,
    _resize_2d,
    evaluate_pretrained_fno,
)
from scpn_fusion.core.pretrained_surrogates import bundle_pretrained_surrogates


def test_pretrained_fno_eval_returns_finite_metrics(tmp_path: Path) -> None:
    fno_path = tmp_path / "fno_eval.npz"
    _ = bundle_pretrained_surrogates(
        force_retrain=True,
        seed=17,
        weights_dir=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        mlp_path=tmp_path / "mlp.npz",
        fno_path=fno_path,
        mlp_hidden=12,
        mlp_epochs=120,
        fno_modes=4,
        fno_width=8,
        fno_epochs=2,
        fno_batch_size=4,
        fno_augment_per_file=2,
    )
    out = evaluate_pretrained_fno(
        fno_path=fno_path,
        augment_per_file=1,
        max_samples=4,
    )
    assert out["eval_samples"] == 4.0
    assert np.isfinite(out["eval_relative_l2_mean"])
    assert np.isfinite(out["eval_relative_l2_p95"])


def test_resize_2d_reshapes_and_rejects_non_2d() -> None:
    # Direct-surface test for the FNO dataset's bilinear resize helper.
    src = np.arange(6, dtype=np.float64).reshape(2, 3)
    out = _resize_2d(src, out_h=4, out_w=5)
    assert out.shape == (4, 5)
    assert np.all(np.isfinite(out))
    with pytest.raises(ValueError, match="Expected 2D array"):
        _resize_2d(np.zeros(3, dtype=np.float64))


def test_build_jet_fno_dataset_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No JET GEQDSK files"):
        _build_jet_fno_dataset(jet_dir=tmp_path)


def test_evaluate_pretrained_fno_rejects_nonpositive_max_samples(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_samples must be > 0"):
        evaluate_pretrained_fno(fno_path=tmp_path / "unused.npz", max_samples=0)


def test_evaluate_pretrained_fno_rejects_empty_dataset(tmp_path: Path) -> None:
    # Zero augmentations over the real JET references yields no samples, so the
    # evaluation guard fires before any weights are loaded.
    with pytest.raises(ValueError, match="FNO evaluation dataset is empty"):
        evaluate_pretrained_fno(
            fno_path=tmp_path / "unused.npz",
            jet_dir=DEFAULT_JET_DIR,
            augment_per_file=0,
        )
