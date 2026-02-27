from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "run_mypy_strict.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_mypy_strict", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load tools/run_mypy_strict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_sets_pythonpath_when_missing(monkeypatch):
    module = _load_module()
    monkeypatch.delenv("PYTHONPATH", raising=False)

    recorded: dict[str, object] = {}

    def fake_run(cmd, cwd, env, check, timeout):
        recorded["cmd"] = cmd
        recorded["cwd"] = cwd
        recorded["env"] = env
        recorded["check"] = check
        recorded["timeout"] = timeout
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.sys, "argv", ["run_mypy_strict.py"])
    monkeypatch.setattr(module.sys, "executable", "python-test")

    rc = module.main()
    assert rc == 0

    expected_root = SCRIPT_PATH.resolve().parents[1]
    expected_src = str(expected_root / "src")

    assert recorded["cwd"] == expected_root
    assert recorded["cmd"] == [
        "python-test",
        "-m",
        "mypy",
        "--no-incremental",
        "--no-warn-unused-configs",
    ]
    assert recorded["check"] is False
    assert recorded["timeout"] == module.DEFAULT_MYPY_TIMEOUT_SECONDS
    assert recorded["env"]["PYTHONPATH"] == expected_src


def test_main_preserves_existing_pythonpath_and_args(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("PYTHONPATH", "existing_path")

    recorded: dict[str, object] = {}

    def fake_run(cmd, cwd, env, check, timeout):
        recorded["cmd"] = cmd
        recorded["cwd"] = cwd
        recorded["env"] = env
        recorded["check"] = check
        recorded["timeout"] = timeout
        return subprocess.CompletedProcess(cmd, 7)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["run_mypy_strict.py", "src/scpn_fusion/control/__init__.py"],
    )
    monkeypatch.setattr(module.sys, "executable", "python-test")

    rc = module.main()
    assert rc == 7

    expected_root = SCRIPT_PATH.resolve().parents[1]
    expected_src = str(expected_root / "src")

    assert recorded["cwd"] == expected_root
    assert recorded["cmd"] == [
        "python-test",
        "-m",
        "mypy",
        "--no-incremental",
        "--no-warn-unused-configs",
        "src/scpn_fusion/control/__init__.py",
    ]
    assert recorded["check"] is False
    assert recorded["timeout"] == module.DEFAULT_MYPY_TIMEOUT_SECONDS
    assert recorded["env"]["PYTHONPATH"] == f"{expected_src}{os.pathsep}existing_path"


def test_main_returns_timeout_exit_code(monkeypatch):
    module = _load_module()

    def fake_run(cmd, cwd, env, check, timeout):  # type: ignore[no-untyped-def]
        _ = (cmd, cwd, env, check, timeout)
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.sys, "argv", ["run_mypy_strict.py", "--timeout-seconds", "0.01"])
    monkeypatch.setattr(module.sys, "executable", "python-test")

    rc = module.main()
    assert rc == 124


def test_main_rejects_invalid_timeout(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module.sys, "argv", ["run_mypy_strict.py", "--timeout-seconds", "0"])
    monkeypatch.setattr(module.sys, "executable", "python-test")

    rc = module.main()
    assert rc == 2
