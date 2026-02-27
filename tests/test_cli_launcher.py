# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Unified CLI Launcher Tests
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

import subprocess
import sys

from click.testing import CliRunner

import scpn_fusion.cli as cli_mod


def test_single_mode_invokes_subprocess(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_mod,
        "MODE_SPECS",
        {"kernel": cli_mod.ModeSpec("scpn_fusion.core.fusion_kernel", "public", "kernel")},
    )

    calls: list[tuple[list[str], str, bool, float]] = []

    def fake_run(cmd, cwd, check, timeout):  # type: ignore[no-untyped-def]
        calls.append((cmd, cwd, check, timeout))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["kernel"])

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0][0][:3] == [sys.executable, "-m", "scpn_fusion.core.fusion_kernel"]
    assert calls[0][2] is False
    assert calls[0][3] == cli_mod.DEFAULT_MODE_TIMEOUT_SECONDS


def test_all_mode_fail_fast_stops_after_first_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_mod,
        "MODE_SPECS",
        {
            "m1": cli_mod.ModeSpec("mod.one", "public", "m1"),
            "m2": cli_mod.ModeSpec("mod.two", "public", "m2"),
        },
    )

    calls: list[list[str]] = []

    def fake_run(cmd, cwd, check, timeout):  # type: ignore[no-untyped-def]
        _ = (cwd, check, timeout)
        calls.append(cmd)
        code = 1 if len(calls) == 1 else 0
        return subprocess.CompletedProcess(cmd, code)

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["all"])

    assert result.exit_code != 0
    assert len(calls) == 1
    assert "One or more modes failed" in result.output


def test_all_mode_continue_on_error_runs_full_plan(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_mod,
        "MODE_SPECS",
        {
            "m1": cli_mod.ModeSpec("mod.one", "public", "m1"),
            "m2": cli_mod.ModeSpec("mod.two", "public", "m2"),
        },
    )

    calls: list[list[str]] = []

    def fake_run(cmd, cwd, check, timeout):  # type: ignore[no-untyped-def]
        _ = (cwd, check, timeout)
        calls.append(cmd)
        code = 1 if len(calls) == 1 else 0
        return subprocess.CompletedProcess(cmd, code)

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["all", "--continue-on-error"])

    assert result.exit_code != 0
    assert len(calls) == 2


def test_surrogate_mode_locked_without_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_mod,
        "MODE_SPECS",
        {"neural": cli_mod.ModeSpec("mod.neural", "surrogate", "neural")},
    )
    monkeypatch.delenv("SCPN_SURROGATE", raising=False)

    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["neural"])

    assert result.exit_code != 0
    assert "surrogate mode locked" in result.output


def test_run_mode_returns_timeout_code_on_timeout(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_mod,
        "MODE_SPECS",
        {"kernel": cli_mod.ModeSpec("scpn_fusion.core.fusion_kernel", "public", "kernel")},
    )

    def fake_run(cmd, cwd, check, timeout):  # type: ignore[no-untyped-def]
        _ = (cmd, cwd, check, timeout)
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=0.01)

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    code = cli_mod._run_mode(
        "kernel",
        python_bin=sys.executable,
        script_args=(),
        mode_timeout_seconds=0.01,
        dry_run=False,
    )
    assert code == 124


def test_cli_rejects_non_positive_timeout() -> None:
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["kernel", "--mode-timeout-seconds", "0"])
    assert result.exit_code != 0
    assert "--mode-timeout-seconds must be finite and > 0." in result.output
