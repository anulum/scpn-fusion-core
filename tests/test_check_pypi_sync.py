# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for tools/check_pypi_sync.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "check_pypi_sync.py"
SPEC = importlib.util.spec_from_file_location("check_pypi_sync", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load tools/check_pypi_sync.py")
check_pypi_sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_pypi_sync)


def test_read_local_version(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "demo"\nversion = "1.2.3"\n',
        encoding="utf-8",
    )
    assert check_pypi_sync.read_local_version(pyproject) == "1.2.3"


def test_read_local_version_without_tomllib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "demo"\nversion = "9.8.7"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(check_pypi_sync, "tomllib", None)
    assert check_pypi_sync.read_local_version(pyproject) == "9.8.7"


@pytest.mark.parametrize(
    ("local", "remote", "mode", "expected"),
    [
        ("3.9.3", "3.9.3", "equal", True),
        ("3.9.3", "3.9.2", "not-behind", True),
        ("3.9.2", "3.9.3", "not-behind", False),
        ("3.9.3", "3.9.2", "equal", False),
    ],
)
def test_compare_versions(local: str, remote: str, mode: str, expected: bool) -> None:
    ok, _ = check_pypi_sync.compare_versions(local, remote, mode=mode)
    assert ok is expected


def test_main_allows_network_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        check_pypi_sync,
        "fetch_pypi_version",
        lambda package, timeout: (_ for _ in ()).throw(RuntimeError("network down")),
    )
    rc = check_pypi_sync.main(
        [
            "--local-version",
            "3.9.3",
            "--allow-network-failure",
            "--retries",
            "1",
        ]
    )
    assert rc == 0


def test_main_detects_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check_pypi_sync, "fetch_pypi_version", lambda package, timeout: "3.9.2")
    rc = check_pypi_sync.main(
        [
            "--local-version",
            "3.9.3",
            "--mode",
            "equal",
            "--retries",
            "1",
        ]
    )
    assert rc == 1
