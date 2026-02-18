from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "run_python_preflight.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_python_preflight", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load tools/run_python_preflight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_runs_default_checks_in_order(monkeypatch):
    module = _load_module()
    calls: list[tuple[list[str], Path]] = []

    def fake_call(cmd, cwd):
        calls.append((cmd, cwd))
        return 0

    monkeypatch.setattr(module.subprocess, "call", fake_call)
    monkeypatch.setattr(module.sys, "argv", ["run_python_preflight.py"])
    monkeypatch.setattr(module.sys, "executable", "python-test")

    rc = module.main()
    assert rc == 0

    assert calls == [
        (
            [
                "python-test",
                "-m",
                "pytest",
                "tests/test_version_metadata.py",
                "-q",
            ],
            SCRIPT_PATH.resolve().parents[1],
        ),
        (
            [
                "python-test",
                "-m",
                "pytest",
                "tests/test_task5_disruption_mitigation_integration.py::test_task5_campaign_passes_thresholds_smoke",
                "tests/test_task6_heating_neutronics_realism.py::test_task6_campaign_passes_thresholds_smoke",
                "-q",
            ],
            SCRIPT_PATH.resolve().parents[1],
        ),
        (
            ["python-test", "tools/run_mypy_strict.py"],
            SCRIPT_PATH.resolve().parents[1],
        ),
    ]


def test_main_honors_skip_flags(monkeypatch):
    module = _load_module()
    calls: list[tuple[list[str], Path]] = []

    def fake_call(cmd, cwd):
        calls.append((cmd, cwd))
        return 0

    monkeypatch.setattr(module.subprocess, "call", fake_call)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "run_python_preflight.py",
            "--skip-version-metadata",
            "--skip-threshold-smoke",
        ],
    )
    monkeypatch.setattr(module.sys, "executable", "python-test")

    rc = module.main()
    assert rc == 0
    assert calls == [
        (
            ["python-test", "tools/run_mypy_strict.py"],
            SCRIPT_PATH.resolve().parents[1],
        )
    ]


def test_main_stops_at_first_failure(monkeypatch):
    module = _load_module()
    calls: list[tuple[list[str], Path]] = []
    results = iter([17, 0, 0])

    def fake_call(cmd, cwd):
        calls.append((cmd, cwd))
        return next(results)

    monkeypatch.setattr(module.subprocess, "call", fake_call)
    monkeypatch.setattr(module.sys, "argv", ["run_python_preflight.py"])
    monkeypatch.setattr(module.sys, "executable", "python-test")

    rc = module.main()
    assert rc == 17
    assert calls == [
        (
            [
                "python-test",
                "-m",
                "pytest",
                "tests/test_version_metadata.py",
                "-q",
            ],
            SCRIPT_PATH.resolve().parents[1],
        )
    ]
