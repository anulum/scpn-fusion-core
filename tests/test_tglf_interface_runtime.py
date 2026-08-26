# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests for TGLF Interface Runtime
"""Focused tests for the current-GACODE TGLF subprocess runtime."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from scpn_fusion.core._tglf_interface_runtime import (
    _normalize_tglf_max_retries,
    _normalize_tglf_timeout_seconds,
    _parse_tglf_run_output,
    _resolve_tglf_command,
    _validate_tglf_command_name,
    run_tglf_binary,
    write_tglf_input_file,
)
from scpn_fusion.core._tglf_interface_types import TGLFInputDeck


def _completed(returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["/resolved/tglf-test", "-e", "."],
        returncode=returncode,
        stdout="",
        stderr=stderr,
    )


def _write_current_outputs(run_dir: Path) -> None:
    """Write a minimal authentic current-GACODE output pair."""
    (run_dir / "out.tglf.gbflux").write_text(
        "-0.2 -0.2 4.0 8.0 0.0 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    (run_dir / "out.tglf.eigenvalue_spectrum").write_text(
        "# gamma/frequency pairs\n# mode 1 mode 2\n0.1 -0.2 0.05 0.1\n0.3 -0.8 0.2 0.4\n",
        encoding="utf-8",
    )


def _mock_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scpn_fusion.core._tglf_interface_runtime.shutil.which",
        lambda command: "/resolved/tglf-test" if command == "tglf-test" else None,
    )


def test_normalizers_accept_and_reject_boundary_values() -> None:
    assert _normalize_tglf_timeout_seconds(2) == 2.0
    assert _normalize_tglf_max_retries(0) == 0
    for timeout_s in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="timeout_s must be finite and > 0"):
            _normalize_tglf_timeout_seconds(timeout_s)
    for max_retries in (-1, 11, True, cast(int, 1.5)):
        with pytest.raises(ValueError, match="max_retries must be an integer"):
            _normalize_tglf_max_retries(max_retries)


def test_path_resolution_rejects_filesystem_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_resolution(monkeypatch)
    assert _resolve_tglf_command("tglf-test") == "/resolved/tglf-test"
    assert _validate_tglf_command_name("  tglf-test  ") == "tglf-test"
    with pytest.raises(ValueError, match="resolved through PATH"):
        _resolve_tglf_command("/opt/gacode/tglf")
    with pytest.raises(FileNotFoundError, match="not found on PATH"):
        _resolve_tglf_command("missing-tglf")


def test_write_tglf_input_file_uses_current_gacode_schema(tmp_path: Path) -> None:
    deck = TGLFInputDeck(
        rho=0.4,
        q=2.1,
        s_hat=1.25,
        alpha_mhd=0.18,
        xnue=0.07,
        kappa=1.85,
        T_e_keV=8.0,
        T_i_keV=4.0,
        R_major=6.0,
        a_minor=2.0,
    )
    path = write_tglf_input_file(deck, tmp_path / "nested" / "run")
    text = path.read_text(encoding="utf-8")
    assert path.name == "input.tglf"
    assert "&tglf_namelist" not in text
    assert "Q_LOC = 2.1" in text
    assert "RMAJ_LOC = 3" in text
    assert "RMIN_LOC = 0.4" in text
    assert "RLTS_1 = 2" in text
    assert "RLTS_2 = 2" in text
    assert "MASS_1 = 2.723000e-4" in text
    assert "ZS_1 = -1.0" in text
    assert "USE_BPER = .false." in text


def test_parse_tglf_run_output_reads_species_summary(tmp_path: Path) -> None:
    output_path = tmp_path / "out.tglf.run"
    output_path.write_text(
        "header\nelec -0.25 6.5 0.0\nion1 -0.25 18.2 0.0\n",
        encoding="utf-8",
    )
    output = _parse_tglf_run_output(output_path, rho=0.35)
    assert output.particle_e == pytest.approx(-0.25)
    assert output.particle_i == pytest.approx(-0.25)
    assert output.q_e == pytest.approx(6.5)
    assert output.q_i == pytest.approx(18.2)


def test_run_tglf_binary_executes_current_cli_and_parses_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mock_resolution(monkeypatch)
    work_dir = tmp_path / "work"

    def _run(args, *, cwd, capture_output, text, timeout):
        assert args == ["/resolved/tglf-test", "-e", "."]
        assert capture_output and text and timeout == 30.0
        _write_current_outputs(Path(cwd))
        return _completed()

    monkeypatch.setattr("scpn_fusion.core._tglf_interface_runtime.subprocess.run", _run)
    deck = TGLFInputDeck(rho=0.5, R_LTi=6.0, R_LTe=6.0, R_Lne=2.0, R_Lni=2.0)
    output = run_tglf_binary(
        deck,
        tglf_command="tglf-test",
        work_dir=work_dir,
        timeout_s=30.0,
        max_retries=0,
    )
    assert output.q_i == pytest.approx(8.0)
    assert output.q_e == pytest.approx(4.0)
    assert output.particle_e == pytest.approx(-0.2)
    assert output.particle_i == pytest.approx(-0.2)
    assert output.gamma_max == pytest.approx(0.3)
    assert np.isfinite([output.chi_i, output.chi_e, output.d_i, output.d_e]).all()
    assert (work_dir / "input.tglf").is_file()


def test_run_tglf_binary_rejects_nonfinite_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mock_resolution(monkeypatch)
    with pytest.raises(ValueError, match="R_LTi.*finite"):
        run_tglf_binary(
            TGLFInputDeck(R_LTi=float("nan")),
            tglf_command="tglf-test",
            work_dir=tmp_path,
        )


def test_run_tglf_binary_retries_and_cleans_auto_work_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mock_resolution(monkeypatch)
    auto_dir = tmp_path / "auto-run"
    calls = 0
    sleeps: list[float] = []

    def _mkdtemp(prefix: str) -> str:
        assert prefix == "tglf_"
        auto_dir.mkdir()
        return str(auto_dir)

    def _run(args, *, cwd, capture_output, text, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _completed(returncode=2, stderr="transient failure")
        _write_current_outputs(Path(cwd))
        return _completed()

    monkeypatch.setattr("scpn_fusion.core._tglf_interface_runtime.tempfile.mkdtemp", _mkdtemp)
    monkeypatch.setattr("scpn_fusion.core._tglf_interface_runtime.subprocess.run", _run)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    output = run_tglf_binary(TGLFInputDeck(), tglf_command="tglf-test", max_retries=2)
    assert output.q_i == pytest.approx(8.0)
    assert calls == 2
    assert sleeps == [1.0]
    assert not auto_dir.exists()


@pytest.mark.parametrize("failure", ["unparseable", "timeout"])
def test_run_tglf_binary_fails_closed_after_final_attempt(
    failure: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mock_resolution(monkeypatch)

    def _run(args, *, cwd, capture_output, text, timeout):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)
        return _completed()

    monkeypatch.setattr("scpn_fusion.core._tglf_interface_runtime.subprocess.run", _run)
    with pytest.raises(RuntimeError, match="failed after 1 attempts"):
        run_tglf_binary(
            TGLFInputDeck(),
            tglf_command="tglf-test",
            work_dir=tmp_path,
            max_retries=0,
        )
