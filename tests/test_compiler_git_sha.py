# ----------------------------------------------------------------------
# SCPN Fusion Core -- Compiler Git SHA Hardening Tests
# ----------------------------------------------------------------------
"""Tests for _resolve_git_sha timeout/fallback behavior."""

from __future__ import annotations

import subprocess

from scpn_fusion.scpn import compiler


def test_resolve_git_sha_prefers_environment(monkeypatch) -> None:
    monkeypatch.setenv("SCPN_GIT_SHA", "abcdef123456")
    assert compiler._resolve_git_sha() == "abcdef1"


def test_resolve_git_sha_uses_timeout_for_git_probe(monkeypatch) -> None:
    monkeypatch.delenv("SCPN_GIT_SHA", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("CI_COMMIT_SHA", raising=False)

    calls: list[dict[str, object]] = []

    def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        _ = args
        calls.append(dict(kwargs))
        return subprocess.CompletedProcess(
            args=["git", "rev-parse", "--short", "HEAD"],
            returncode=0,
            stdout="1234567\n",
            stderr="",
        )

    monkeypatch.setattr(compiler.subprocess, "run", fake_run)
    sha = compiler._resolve_git_sha()
    assert sha == "1234567"
    assert calls
    assert calls[0]["timeout"] == compiler._GIT_SHA_TIMEOUT_SECONDS


def test_resolve_git_sha_falls_back_on_timeout(monkeypatch) -> None:
    monkeypatch.delenv("SCPN_GIT_SHA", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("CI_COMMIT_SHA", raising=False)

    def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1.0)

    monkeypatch.setattr(compiler.subprocess, "run", fake_run)
    assert compiler._resolve_git_sha() == "0000000"
