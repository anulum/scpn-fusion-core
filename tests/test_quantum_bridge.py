# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Quantum Bridge Hardening Tests
# ──────────────────────────────────────────────────────────────────────
"""Tests for hardened quantum bridge process orchestration."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from scpn_fusion.core import quantum_bridge as qb


def _write_dummy_scripts(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in qb._QUANTUM_SCRIPT_NAMES:
        (root / name).write_text("print('ok')\n", encoding="utf-8")


def test_run_quantum_suite_rejects_missing_lab_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Quantum Lab not found"):
        qb.run_quantum_suite(base_path=tmp_path / "missing_lab")


def test_run_quantum_suite_rejects_missing_required_scripts(tmp_path: Path) -> None:
    lab = tmp_path / "QUANTUM_LAB"
    lab.mkdir(parents=True, exist_ok=True)
    (lab / qb._QUANTUM_SCRIPT_NAMES[0]).write_text("print('ok')\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing required scripts"):
        qb.run_quantum_suite(base_path=lab)


def test_run_quantum_suite_executes_scripts_with_check_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lab = tmp_path / "QUANTUM_LAB"
    _write_dummy_scripts(lab)
    calls: list[tuple[list[str], bool, float]] = []

    def _fake_run(cmd: list[str], check: bool, timeout: float) -> None:
        calls.append((cmd, check, timeout))

    monkeypatch.setattr(qb.subprocess, "run", _fake_run)
    result = qb.run_quantum_suite(base_path=lab)

    assert result["ok"] is True
    assert result["scripts"] == list(qb._QUANTUM_SCRIPT_NAMES)
    assert len(calls) == len(qb._QUANTUM_SCRIPT_NAMES)
    for idx, (cmd, check, timeout) in enumerate(calls):
        assert check is True
        assert timeout == qb._QUANTUM_SCRIPT_TIMEOUT_SECONDS
        assert cmd[0] == sys.executable
        assert Path(cmd[1]).name == qb._QUANTUM_SCRIPT_NAMES[idx]


def test_run_quantum_suite_raises_on_subprocess_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lab = tmp_path / "QUANTUM_LAB"
    _write_dummy_scripts(lab)
    target = qb._QUANTUM_SCRIPT_NAMES[1]

    def _fake_run(cmd: list[str], check: bool, timeout: float) -> None:
        _ = (check, timeout)
        if Path(cmd[1]).name == target:
            raise subprocess.CalledProcessError(returncode=2, cmd=cmd)

    monkeypatch.setattr(qb.subprocess, "run", _fake_run)
    with pytest.raises(RuntimeError, match=target):
        qb.run_quantum_suite(base_path=lab)


def test_run_quantum_suite_raises_on_subprocess_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lab = tmp_path / "QUANTUM_LAB"
    _write_dummy_scripts(lab)
    target = qb._QUANTUM_SCRIPT_NAMES[2]

    def _fake_run(cmd: list[str], check: bool, timeout: float) -> None:
        _ = check
        if Path(cmd[1]).name == target:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr(qb.subprocess, "run", _fake_run)
    with pytest.raises(RuntimeError, match=f"timed out: {target}"):
        qb.run_quantum_suite(base_path=lab, script_timeout_seconds=12.5)


def test_run_quantum_suite_rejects_non_positive_timeout(tmp_path: Path) -> None:
    lab = tmp_path / "QUANTUM_LAB"
    _write_dummy_scripts(lab)
    with pytest.raises(ValueError, match="script_timeout_seconds must be finite and > 0."):
        qb.run_quantum_suite(base_path=lab, script_timeout_seconds=0.0)
