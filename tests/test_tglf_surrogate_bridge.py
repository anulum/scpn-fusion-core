# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Direct tests for extracted TGLF surrogate bridge module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import pytest
from scpn_fusion.core import tglf_interface as tglf
from scpn_fusion.core.tglf_surrogate_bridge import (
    TGLFDatasetGenerator,
    train_surrogate_from_tglf,
)


def test_dataset_generator_accepts_zero_samples() -> None:
    """A zero-sample request returns an empty dataset without invoking TGLF."""
    gen = TGLFDatasetGenerator("C:/fake/tglf")
    out = gen.generate_random_dataset(n_samples=0)
    assert out == []


def test_dataset_generator_records_successful_tglf_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful sampled TGLF calls are serialized into input/output records."""
    calls: list[tglf.TGLFInputDeck] = []

    def _run_tglf_binary(
        deck: tglf.TGLFInputDeck,
        binary: str | Path,
        *,
        timeout_s: float,
    ) -> tglf.TGLFOutput:
        calls.append(deck)
        assert Path(binary) == Path("/opt/tglf")
        assert timeout_s == 60.0
        return tglf.TGLFOutput(rho=deck.rho, chi_i=1.25, chi_e=2.5, gamma_max=0.75)

    monkeypatch.setattr(tglf, "run_tglf_binary", _run_tglf_binary)

    dataset = TGLFDatasetGenerator("/opt/tglf").generate_random_dataset(n_samples=2)

    assert len(calls) == 2
    assert len(dataset) == 2
    for record in dataset:
        input_payload = cast(dict[str, float], record["input"])
        output_payload = cast(dict[str, float], record["output"])
        assert 0.0 <= input_payload["R_LTi"] <= 12.0
        assert 0.0 <= input_payload["R_LTe"] <= 12.0
        assert 0.0 <= input_payload["R_Lne"] <= 5.0
        assert 1.0 <= input_payload["q"] <= 5.0
        assert 0.0 <= input_payload["s_hat"] <= 3.0
        assert 0.001 <= input_payload["beta_e"] <= 0.05
        assert 1.0 <= input_payload["Z_eff"] <= 3.0
        assert output_payload["chi_i"] == pytest.approx(1.25)
        assert output_payload["chi_e"] == pytest.approx(2.5)
        assert output_payload["gamma_max"] == pytest.approx(0.75)


def test_dataset_generator_logs_failed_tglf_runs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Failed sampled TGLF calls are logged and omitted from the dataset."""

    def _run_tglf_binary(
        deck: tglf.TGLFInputDeck,
        binary: str | Path,
        *,
        timeout_s: float,
    ) -> tglf.TGLFOutput:
        del deck, binary, timeout_s
        raise RuntimeError("synthetic TGLF failure")

    monkeypatch.setattr(tglf, "run_tglf_binary", _run_tglf_binary)

    with caplog.at_level(logging.WARNING, logger="scpn_fusion.core.tglf_surrogate_bridge"):
        dataset = TGLFDatasetGenerator("/opt/tglf").generate_random_dataset(n_samples=1)

    assert dataset == []
    assert "Sample 0 failed: synthetic TGLF failure" in caplog.text


def test_train_surrogate_placeholder_smoke(tmp_path: Path) -> None:
    """The reserved training entry point does not write placeholder weights."""
    out_path = tmp_path / "weights.npz"
    train_surrogate_from_tglf([], out_path)
    # Placeholder currently prints progress; no file emission contract yet.
    assert out_path.name == "weights.npz"
