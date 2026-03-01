# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Unified CLI Launcher Tests
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import subprocess
import sys
from types import SimpleNamespace

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


def test_experimental_mode_requires_ack(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_mod,
        "MODE_SPECS",
        {"quantum": cli_mod.ModeSpec("mod.quantum", "experimental", "quantum")},
    )
    monkeypatch.delenv("SCPN_EXPERIMENTAL", raising=False)
    monkeypatch.delenv("SCPN_EXPERIMENTAL_ACK", raising=False)

    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["quantum", "--experimental"])
    assert result.exit_code != 0
    assert "experimental acknowledgement missing" in result.output


def test_experimental_mode_runs_with_ack(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_mod,
        "MODE_SPECS",
        {"quantum": cli_mod.ModeSpec("mod.quantum", "experimental", "quantum")},
    )
    monkeypatch.delenv("SCPN_EXPERIMENTAL", raising=False)

    calls: list[tuple[list[str], str, bool, float]] = []

    def fake_run(cmd, cwd, check, timeout):  # type: ignore[no-untyped-def]
        calls.append((cmd, cwd, check, timeout))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "quantum",
            "--experimental",
            "--experimental-ack",
            cli_mod.EXPERIMENTAL_ACK_TOKEN,
        ],
    )
    assert result.exit_code == 0
    assert len(calls) == 1


def test_cli_rejects_non_positive_timeout() -> None:
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["kernel", "--mode-timeout-seconds", "0"])
    assert result.exit_code != 0
    assert "--mode-timeout-seconds must be finite and > 0." in result.output


def test_cli_rejects_non_finite_timeout_nan() -> None:
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["kernel", "--mode-timeout-seconds", "nan"])
    assert result.exit_code != 0
    assert "--mode-timeout-seconds must be finite and > 0." in result.output


def test_all_mode_rejects_positional_args(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_mod,
        "MODE_SPECS",
        {"kernel": cli_mod.ModeSpec("mod.kernel", "public", "kernel")},
    )
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["all", "unexpected-arg"])
    assert result.exit_code != 0
    assert "Positional script arguments are not supported with mode 'all'" in result.output


def test_list_modes_renders_lock_state(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_mod,
        "MODE_SPECS",
        {
            "kernel": cli_mod.ModeSpec("mod.kernel", "public", "kernel"),
            "neural": cli_mod.ModeSpec("mod.neural", "surrogate", "neural"),
            "quantum": cli_mod.ModeSpec("mod.quantum", "experimental", "quantum"),
        },
    )
    monkeypatch.delenv("SCPN_SURROGATE", raising=False)
    monkeypatch.delenv("SCPN_EXPERIMENTAL", raising=False)

    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["--list-modes"])
    assert result.exit_code == 0
    assert "mode | maturity | unlocked | description" in result.output
    assert "kernel | public | yes | kernel" in result.output
    assert "neural | surrogate | no | neural" in result.output
    assert "quantum | experimental | no | quantum" in result.output


def test_execution_plan_unknown_mode_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_mod,
        "MODE_SPECS",
        {"kernel": cli_mod.ModeSpec("mod.kernel", "public", "kernel")},
    )
    try:
        cli_mod._execution_plan(
            "missing",
            include_surrogate=False,
            include_experimental=False,
            experimental_ack=None,
        )
    except Exception as exc:  # click.ClickException without importing click in test
        assert "Unknown mode 'missing'" in str(exc)
    else:
        raise AssertionError("Expected unknown mode to raise ClickException")


def test_execution_plan_all_experimental_requires_ack(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_mod,
        "MODE_SPECS",
        {
            "kernel": cli_mod.ModeSpec("mod.kernel", "public", "kernel"),
            "quantum": cli_mod.ModeSpec("mod.quantum", "experimental", "quantum"),
        },
    )
    try:
        cli_mod._execution_plan(
            "all",
            include_surrogate=False,
            include_experimental=True,
            experimental_ack="",
        )
    except Exception as exc:
        assert "experimental acknowledgement missing" in str(exc)
    else:
        raise AssertionError("Expected experimental all-plan to require acknowledgement")


def test_run_mode_dry_run_skips_subprocess(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_mod,
        "MODE_SPECS",
        {"kernel": cli_mod.ModeSpec("mod.kernel", "public", "kernel")},
    )

    called = False

    def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        _ = (args, kwargs)
        nonlocal called
        called = True
        return subprocess.CompletedProcess(["python"], 0)

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    code = cli_mod._run_mode(
        "kernel",
        python_bin=sys.executable,
        script_args=("a",),
        mode_timeout_seconds=1.0,
        dry_run=True,
    )
    assert code == 0
    assert called is False


def test_system_health_check_logs_low_resources(monkeypatch, caplog) -> None:
    monkeypatch.setattr(cli_mod.os, "cpu_count", lambda: 1)
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(
            virtual_memory=lambda: SimpleNamespace(
                total=int(4 * (1024**3)),
                available=int(1 * (1024**3)),
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(__version__="0.9.0"))
    monkeypatch.setitem(sys.modules, "scipy", SimpleNamespace(__version__="1.0.0"))
    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(devices=lambda: [SimpleNamespace(platform="cpu")]),
    )

    caplog.set_level(logging.WARNING, logger="scpn_fusion.cli")
    cli_mod._system_health_check()
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "Low CPU core count" in messages
    assert "System RAM < 8GB" in messages
    assert "Low available RAM" in messages


def test_main_returns_click_exception_exit_code(monkeypatch) -> None:
    class _Boom:
        @staticmethod
        def main(*args, **kwargs):  # type: ignore[no-untyped-def]
            _ = (args, kwargs)
            raise cli_mod.click.ClickException("boom")

    monkeypatch.setattr(cli_mod, "cli", _Boom())
    assert cli_mod.main() == 1


def test_main_returns_non_integer_system_exit_as_failure(monkeypatch) -> None:
    class _Boom:
        @staticmethod
        def main(*args, **kwargs):  # type: ignore[no-untyped-def]
            _ = (args, kwargs)
            raise SystemExit("non-int")

    monkeypatch.setattr(cli_mod, "cli", _Boom())
    assert cli_mod.main() == 1
