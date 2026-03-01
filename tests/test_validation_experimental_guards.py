"""Regression tests for validation experimental entrypoint guards."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATHS = [
    ROOT / "validation" / "full_validation_pipeline.py",
    ROOT / "validation" / "run_experimental_validation.py",
    ROOT / "validation" / "validate_against_sparc.py",
]


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("module_path", MODULE_PATHS)
def test_guard_rejects_missing_experimental_unlock(
    module_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(module_path)
    monkeypatch.delenv("SCPN_EXPERIMENTAL", raising=False)
    monkeypatch.delenv("SCPN_EXPERIMENTAL_ACK", raising=False)
    with pytest.raises(SystemExit, match="locked"):
        module.require_experimental_opt_in(
            allow_experimental=False,
            experimental_ack="",
        )


@pytest.mark.parametrize("module_path", MODULE_PATHS)
def test_guard_rejects_missing_ack(
    module_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(module_path)
    monkeypatch.delenv("SCPN_EXPERIMENTAL_ACK", raising=False)
    with pytest.raises(SystemExit, match="acknowledgement missing"):
        module.require_experimental_opt_in(
            allow_experimental=True,
            experimental_ack="wrong",
        )


@pytest.mark.parametrize("module_path", MODULE_PATHS)
def test_guard_accepts_valid_ack(
    module_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(module_path)
    monkeypatch.delenv("SCPN_EXPERIMENTAL", raising=False)
    monkeypatch.delenv("SCPN_EXPERIMENTAL_ACK", raising=False)
    module.require_experimental_opt_in(
        allow_experimental=True,
        experimental_ack=module.EXPERIMENTAL_ACK_TOKEN,
    )
