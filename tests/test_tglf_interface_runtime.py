# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests for TGLF Interface Runtime
"""Focused tests for the extracted TGLF subprocess runtime."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import cast

import pytest

import scpn_fusion.core.tglf_interface as public_tglf
from scpn_fusion.core._tglf_interface_runtime import (
    _normalize_tglf_max_retries,
    _normalize_tglf_timeout_seconds,
    _parse_tglf_run_output,
    run_tglf_binary,
    write_tglf_input_file,
)
from scpn_fusion.core._tglf_interface_types import TGLFInputDeck, TGLFOutput


def _completed(returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    """Build a typed subprocess result for mocked TGLF runs."""
    return subprocess.CompletedProcess(
        args=["tglf"],
        returncode=returncode,
        stdout="",
        stderr=stderr,
    )


def test_normalizers_accept_and_reject_boundary_values() -> None:
    """Runtime argument normalizers reject unsafe timeout and retry settings."""
    assert _normalize_tglf_timeout_seconds(2) == 2.0
    assert _normalize_tglf_max_retries(0) == 0

    for timeout_s in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="timeout_s must be finite and > 0"):
            _normalize_tglf_timeout_seconds(timeout_s)

    for max_retries in (-1, 11, True, cast(int, 1.5)):
        with pytest.raises(ValueError, match="max_retries must be an integer"):
            _normalize_tglf_max_retries(max_retries)


def test_write_tglf_input_file_creates_nested_deck(tmp_path: Path) -> None:
    """The deck writer creates parent directories and writes mapped TGLF keys."""
    deck = TGLFInputDeck(
        rho=0.4,
        q=2.1,
        q_prime_loc=1.25,
        p_prime_loc=-4200.0,
        alpha_mhd=0.18,
        xnue=0.07,
        kappa=1.85,
        T_e_keV=8.0,
        T_i_keV=4.0,
    )

    path = write_tglf_input_file(deck, tmp_path / "nested" / "run")

    text = path.read_text(encoding="utf-8")
    assert path.name == "input.tglf"
    assert "Q_LOC = 2.100000" in text
    assert "Q_PRIME_LOC = 1.250000" in text
    assert "P_PRIME_LOC = -4200.000000" in text
    assert "ALPHA_LOC = 0.180000" in text
    assert "XNUE = 0.070000" in text
    assert "TAUS_2 = 2.000000" in text


def test_parse_tglf_run_output_handles_aliases_and_bad_values(tmp_path: Path) -> None:
    """The text parser ignores malformed values while accepting TGLF aliases."""
    output_path = tmp_path / "out.tglf.run"
    output_path.write_text(
        "\n".join(
            [
                "noise without assignment",
                "CHI_I = nan",
                "CHIEFF_I = 1.25",
                "CHI_E = not-a-number",
                "CHIEFF_E = 2.50",
                "GAMMA_MAX = inf",
                "GAMMA_MAX = 0.75",
            ]
        ),
        encoding="utf-8",
    )

    output = _parse_tglf_run_output(output_path, rho=0.35)

    assert output == TGLFOutput(rho=0.35, chi_i=1.25, chi_e=2.50, gamma_max=0.75)


def test_run_tglf_binary_parses_run_file_and_clips_nonfinite_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A successful binary run writes an input deck, clips bad deck values, and parses output."""
    binary = tmp_path / "tglf"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    work_dir = tmp_path / "work"

    def _run(
        args: list[str],
        *,
        cwd: str,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        Path(cwd, "out.tglf.run").write_text(
            "CHI_I = 3.0\nCHI_E = 4.0\nGAMMA_MAX = 0.2\n",
            encoding="utf-8",
        )
        return _completed()

    deck = TGLFInputDeck(R_LTi=float("nan"))
    monkeypatch.setattr("scpn_fusion.core._tglf_interface_runtime.subprocess.run", _run)

    output = run_tglf_binary(deck, binary, work_dir=work_dir, max_retries=0)

    assert output == TGLFOutput(rho=deck.rho, chi_i=3.0, chi_e=4.0, gamma_max=0.2)
    assert deck.R_LTi == 0.0
    assert "RLTS_1 = 0.000000" in (work_dir / "input.tglf").read_text(encoding="utf-8")


def test_run_tglf_binary_retries_and_cleans_auto_work_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A retry success cleans the auto-created work directory instead of leaking it."""
    binary = tmp_path / "tglf"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    auto_dir = tmp_path / "auto-run"
    calls = 0
    sleeps: list[float] = []

    def _mkdtemp(prefix: str) -> str:
        assert prefix == "tglf_"
        auto_dir.mkdir()
        return str(auto_dir)

    def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    def _run(
        args: list[str],
        *,
        cwd: str,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _completed(returncode=2, stderr="transient failure")
        Path(cwd, "out.tglf.run").write_text("CHI_I = 5.0\n", encoding="utf-8")
        return _completed()

    monkeypatch.setattr("scpn_fusion.core._tglf_interface_runtime.tempfile.mkdtemp", _mkdtemp)
    monkeypatch.setattr("scpn_fusion.core._tglf_interface_runtime.subprocess.run", _run)
    monkeypatch.setattr(time, "sleep", _sleep)

    output = run_tglf_binary(TGLFInputDeck(), binary, max_retries=2)

    assert output.chi_i == 5.0
    assert calls == 2
    assert sleeps == [1.0]
    assert not auto_dir.exists()


def test_run_tglf_binary_uses_json_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When text output is absent, the runtime falls back to public JSON parsing."""
    binary = tmp_path / "tglf"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")

    def _run(
        args: list[str],
        *,
        cwd: str,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        Path(cwd, "output.json").write_text("{}", encoding="utf-8")
        return _completed()

    def _parse(output_dir: str | Path) -> list[TGLFOutput]:
        assert Path(output_dir).name == "work"
        return [TGLFOutput(rho=0.6, chi_i=7.0, chi_e=8.0, gamma_max=0.4)]

    monkeypatch.setattr("scpn_fusion.core._tglf_interface_runtime.subprocess.run", _run)
    monkeypatch.setattr(public_tglf, "parse_tglf_output", _parse)

    output = run_tglf_binary(TGLFInputDeck(rho=0.6), binary, work_dir=tmp_path / "work")

    assert output == TGLFOutput(rho=0.6, chi_i=7.0, chi_e=8.0, gamma_max=0.4)


def test_run_tglf_binary_returns_default_after_unparseable_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A successful subprocess with no recognized output fails closed to zero transport."""
    binary = tmp_path / "tglf"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")

    def _run(
        args: list[str],
        *,
        cwd: str,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        return _completed()

    monkeypatch.setattr("scpn_fusion.core._tglf_interface_runtime.subprocess.run", _run)

    output = run_tglf_binary(TGLFInputDeck(rho=0.42), binary, work_dir=tmp_path, max_retries=0)

    assert output == TGLFOutput(rho=0.42)


def test_run_tglf_binary_returns_default_after_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A subprocess timeout is caught and converted to fail-closed zero transport."""
    binary = tmp_path / "tglf"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")

    def _run(
        args: list[str],
        *,
        cwd: str,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)

    monkeypatch.setattr("scpn_fusion.core._tglf_interface_runtime.subprocess.run", _run)

    output = run_tglf_binary(TGLFInputDeck(rho=0.77), binary, work_dir=tmp_path, max_retries=0)

    assert output == TGLFOutput(rho=0.77)


def test_run_tglf_binary_requires_existing_binary(tmp_path: Path) -> None:
    """A missing external binary raises before any work directory is created."""
    with pytest.raises(FileNotFoundError, match="TGLF binary not found"):
        run_tglf_binary(TGLFInputDeck(), tmp_path / "missing-tglf")
