# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for tools/check_pypi_sync.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import TracebackType
from typing import Literal, Self
from urllib.request import Request

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "check_pypi_sync.py"
SPEC = importlib.util.spec_from_file_location("tools.check_pypi_sync", MODULE_PATH)
assert SPEC and SPEC.loader
check_pypi_sync = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_pypi_sync
SPEC.loader.exec_module(check_pypi_sync)


class _JsonResponse:
    """Minimal response object for ``json.load`` and context-manager use."""

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        return False

    def read(self) -> str:
        """Return the JSON payload consumed by ``json.load``."""
        return self._payload


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
    monkeypatch.setattr(check_pypi_sync, "TOML_LOADS", None)
    assert check_pypi_sync.read_local_version(pyproject) == "9.8.7"


def test_read_local_version_rejects_missing_project_version(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '# comment\n\n[tool.demo]\nname = "demo"\n[project]\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unable to determine project.version"):
        check_pypi_sync.read_local_version(pyproject)


def test_read_local_version_falls_back_when_project_table_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "7.8.9"\n', encoding="utf-8")
    monkeypatch.setattr(check_pypi_sync, "TOML_LOADS", lambda text: {"project": "bad"})

    assert check_pypi_sync.read_local_version(pyproject) == "7.8.9"


def test_fetch_pypi_version_parses_and_validates_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, float]] = []

    def fake_urlopen(
        request: Request,
        *,
        timeout: float,
    ) -> _JsonResponse:
        calls.append((request.full_url, timeout))
        return _JsonResponse('{"info": {"version": " 3.9.4 "}}')

    monkeypatch.setattr(check_pypi_sync.urllib.request, "urlopen", fake_urlopen)

    assert check_pypi_sync.fetch_pypi_version("scpn-fusion", timeout=1.5) == "3.9.4"
    assert calls == [
        (
            "https://pypi.org/pypi/scpn-fusion/json",
            1.5,
        )
    ]


def test_fetch_pypi_version_rejects_missing_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: object, *, timeout: float) -> _JsonResponse:
        return _JsonResponse('{"info": {}}')

    monkeypatch.setattr(check_pypi_sync.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(ValueError, match="missing info.version"):
        check_pypi_sync.fetch_pypi_version("scpn-fusion", timeout=1.5)


@pytest.mark.parametrize(
    ("local", "remote", "mode", "expected"),
    [
        ("3.9.3", "3.9.3", "equal", True),
        ("3.9.3", "3.9.2", "not-behind", True),
        ("3.9.2", "3.9.3", "not-behind", False),
        ("3.9.3", "3.9.2", "equal", False),
        ("beta", "alpha", "not-behind", True),
    ],
)
def test_compare_versions(local: str, remote: str, mode: str, expected: bool) -> None:
    ok, _ = check_pypi_sync.compare_versions(local, remote, mode=mode)
    assert ok is expected


def test_compare_versions_rejects_unsupported_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported mode"):
        check_pypi_sync.compare_versions("1", "1", mode="newest")


def test_normalize_version_strips_optional_prefix() -> None:
    assert check_pypi_sync.normalize_version(" v1.2.3 ", strip_v_prefix=True) == "1.2.3"
    assert check_pypi_sync.normalize_version(" v1.2.3 ", strip_v_prefix=False) == "v1.2.3"


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


def test_main_reports_network_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_fetch(package: str, *, timeout: float) -> str:
        raise RuntimeError(f"{package}:{timeout}")

    monkeypatch.setattr(check_pypi_sync, "fetch_pypi_version", fail_fetch)

    rc = check_pypi_sync.main(
        [
            "--package",
            "scpn-fusion",
            "--local-version",
            "3.9.3",
            "--timeout",
            "0.5",
            "--retries",
            "1",
        ]
    )

    assert rc == 2


def test_main_reads_relative_pyproject_and_strips_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "v3.9.3"\n', encoding="utf-8")

    monkeypatch.setattr(check_pypi_sync, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        check_pypi_sync,
        "fetch_pypi_version",
        lambda package, timeout: "3.9.3",
    )

    rc = check_pypi_sync.main(
        [
            "--pyproject",
            "pyproject.toml",
            "--strip-v-prefix",
            "--mode",
            "equal",
            "--retries",
            "1",
        ]
    )

    assert rc == 0


def test_main_retries_transient_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = iter(["3.9.2", "3.9.5"])
    sleeps: list[float] = []

    monkeypatch.setattr(
        check_pypi_sync,
        "fetch_pypi_version",
        lambda package, timeout: next(versions),
    )
    monkeypatch.setattr(check_pypi_sync.time, "sleep", sleeps.append)

    rc = check_pypi_sync.main(
        [
            "--local-version",
            "3.9.5",
            "--mode",
            "equal",
            "--retries",
            "2",
            "--retry-delay",
            "0.25",
        ]
    )

    assert rc == 0
    assert sleeps == [0.25]
