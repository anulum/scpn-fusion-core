# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Disruption Predictor Checkpoint Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Checkpoint load/train tests for disruption predictor."""

from __future__ import annotations

from pathlib import Path

import pytest

import scpn_fusion.control.disruption_predictor as dp


pytestmark = pytest.mark.skipif(dp.torch is None, reason="torch is required for checkpoint tests")


def test_load_or_train_creates_checkpoint_when_missing(tmp_path: Path) -> None:
    model_path = tmp_path / "disruption_model.pth"
    model, meta = dp.load_or_train_predictor(
        model_path=model_path,
        seq_len=32,
        train_kwargs={
            "seq_len": 32,
            "n_shots": 8,
            "epochs": 1,
            "seed": 7,
            "save_plot": False,
        },
    )
    assert model is not None
    assert meta["trained"] is True
    assert meta["seq_len"] == 32
    assert model_path.exists()


def test_load_or_train_reuses_existing_checkpoint(tmp_path: Path) -> None:
    model_path = tmp_path / "disruption_model.pth"

    dp.train_predictor(
        seq_len=24,
        n_shots=8,
        epochs=1,
        model_path=model_path,
        seed=11,
        save_plot=False,
    )
    model, meta = dp.load_or_train_predictor(
        model_path=model_path,
        seq_len=64,
        force_retrain=False,
        train_kwargs={
            "seq_len": 64,
            "n_shots": 8,
            "epochs": 1,
            "seed": 12,
            "save_plot": False,
        },
    )
    assert model is not None
    assert meta["trained"] is False
    assert meta["seq_len"] == 24
    assert Path(meta["model_path"]) == model_path
