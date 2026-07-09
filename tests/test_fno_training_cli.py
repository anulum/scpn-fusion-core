# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FNO Training CLI Tests
"""Tests for the lightweight FNO training command-line entrypoint."""

from __future__ import annotations

from pathlib import Path

import pytest

from scpn_fusion.core import fno_training


def test_fno_training_cli_logs_legacy_summary(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The legacy CLI mode logs the summary returned by the training path."""

    def _fake_train_fno(**kwargs: object) -> dict[str, object]:
        return {"saved_path": Path("legacy.npz"), "best_val_loss": 0.25, "kwargs": kwargs}

    monkeypatch.setattr(fno_training, "train_fno", _fake_train_fno)

    with caplog.at_level("INFO", logger=fno_training.__name__):
        summary = fno_training._run_training_smoke_cli(["legacy"])

    assert summary["saved_path"] == Path("legacy.npz")
    assert "FNO legacy smoke training complete" in caplog.text
    assert "Best val loss" in caplog.text


def test_fno_training_cli_logs_gs_transport_summary(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The GS-transport CLI mode logs validation and machine-class summaries."""

    def _fake_train_gs(**kwargs: object) -> dict[str, object]:
        return {
            "saved_path": Path("gs.npz"),
            "best_val_loss": 0.12,
            "test_rel_l2": 0.03,
            "machine_class_counts": {"MAST": 2},
            "kwargs": kwargs,
        }

    monkeypatch.setattr(fno_training, "train_gs_transport_surrogate", _fake_train_gs)

    with caplog.at_level("INFO", logger=fno_training.__name__):
        summary = fno_training._run_training_smoke_cli(["gs_transport"])

    assert summary["test_rel_l2"] == 0.03
    assert "GS-transport surrogate training complete" in caplog.text
    assert "Machine-class distribution" in caplog.text


def test_fno_training_cli_logs_multi_regime_summary(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The default CLI mode logs multi-regime aggregate and per-regime metrics."""

    def _fake_train_multi(**kwargs: object) -> dict[str, object]:
        return {
            "saved_path": Path("multi.npz"),
            "best_val_loss": 0.08,
            "regime_counts": {"core": 3},
            "regime_val_losses": {"core": {"mean": 0.08, "n": 3.0}},
            "kwargs": kwargs,
        }

    monkeypatch.setattr(fno_training, "train_fno_multi_regime", _fake_train_multi)

    with caplog.at_level("INFO", logger=fno_training.__name__):
        summary = fno_training._run_training_smoke_cli([])

    assert summary["regime_counts"] == {"core": 3}
    assert "FNO multi-regime SPARC training complete" in caplog.text
    assert "Regime validation: regime=core" in caplog.text
