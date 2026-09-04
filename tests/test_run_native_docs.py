# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Behavioural contract tests for the polyglot native-doc runner."""

from __future__ import annotations

from pathlib import Path
import subprocess
import zlib
from urllib.error import URLError
from urllib.request import Request

import pytest

from tools import run_native_docs as native_docs


class _InventoryResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> _InventoryResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        return self._payload


def _inventory_payload(project: str) -> bytes:
    records = b"example py:function 1 example.html Example\n"
    return (
        b"# Sphinx inventory version 2\n"
        + f"# Project: {project}\n".encode()
        + b"# Version: 1.0\n"
        + b"# The remainder of this file is compressed using zlib.\n"
        + zlib.compress(records)
    )


def test_requested_languages_expands_all_once_in_canonical_order() -> None:
    assert native_docs._requested_languages(("go", "all", "go")) == (
        "go",
        "python",
        "rust",
        "julia",
        "cpp",
        "lean",
    )


def test_main_fails_closed_when_generator_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("tools.run_native_docs.shutil.which", lambda _name: None)

    assert native_docs.main(["--language", "cpp"]) == 127
    assert "missing executable doxygen" in capsys.readouterr().err


def test_cpp_docs_require_graphviz_dot(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "tools.run_native_docs.shutil.which",
        lambda name: None if name == "dot" else f"/usr/bin/{name}",
    )

    assert native_docs.main(["--language", "cpp"]) == 127
    assert "missing executable dot" in capsys.readouterr().err


def test_main_can_skip_only_unavailable_generators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "tools.run_native_docs.shutil.which",
        lambda name: None if name == "doxygen" else f"/usr/bin/{name}",
    )

    def fake_go_builder() -> int:
        calls.append("go")
        return 0

    monkeypatch.setitem(native_docs.BUILDERS, "go", fake_go_builder)

    assert (
        native_docs.main(
            [
                "--language",
                "cpp",
                "--language",
                "go",
                "--skip-unavailable",
            ]
        )
        == 0
    )
    assert calls == ["go"]


def test_inventory_download_uses_validated_official_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    payload = _inventory_payload("SciPy")

    def fake_urlopen(request: Request, *, timeout: float) -> _InventoryResponse:
        assert timeout == native_docs.INTERSPHINX_TIMEOUT_SECONDS
        url = request.full_url
        calls.append(url)
        if url.endswith("primary/objects.inv"):
            raise URLError("primary unavailable")
        return _InventoryResponse(payload)

    monkeypatch.setattr(native_docs, "urlopen", fake_urlopen)
    monkeypatch.setattr(native_docs, "INTERSPHINX_ATTEMPTS", 1)
    target = tmp_path / "scipy.objects.inv"

    assert native_docs._download_intersphinx_inventory(
        "scipy",
        (
            "https://example.invalid/primary/objects.inv",
            "https://example.invalid/fallback/objects.inv",
        ),
        target,
    )
    assert calls == [
        "https://example.invalid/primary/objects.inv",
        "https://example.invalid/fallback/objects.inv",
    ]
    assert target.read_bytes() == payload


def test_inventory_download_rejects_wrong_project_and_corrupt_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    responses = iter(
        (
            _inventory_payload("NumPy"),
            (
                b"# Sphinx inventory version 2\n"
                b"# Project: SciPy\n"
                b"# Version: 1.0\n"
                b"# The remainder of this file is compressed using zlib.\n"
                b"not-zlib"
            ),
        )
    )

    def fake_urlopen(_request: Request, *, timeout: float) -> _InventoryResponse:
        assert timeout == native_docs.INTERSPHINX_TIMEOUT_SECONDS
        return _InventoryResponse(next(responses))

    monkeypatch.setattr(native_docs, "urlopen", fake_urlopen)
    monkeypatch.setattr(native_docs, "INTERSPHINX_ATTEMPTS", 1)
    target = tmp_path / "scipy.objects.inv"

    assert not native_docs._download_intersphinx_inventory(
        "scipy",
        (
            "https://example.invalid/wrong-project.objects.inv",
            "https://example.invalid/corrupt.objects.inv",
        ),
        target,
    )
    assert not target.exists()


def test_rust_docs_deny_warnings_and_missing_docs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], Path, dict[str, str] | None]] = []

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None,
        check: bool,
        text: bool,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[str]:
        del check, text, capture_output
        calls.append((command, cwd, env))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("tools.run_native_docs.subprocess.run", fake_run)
    monkeypatch.setenv("RUSTDOCFLAGS", "--cfg docsrs")

    assert native_docs._build_rust_docs() == 0
    command, cwd, env = calls[0]
    assert command[:2] == ["cargo", "doc"]
    assert cwd == native_docs.REPO_ROOT / "scpn-fusion-rs"
    assert env is not None
    assert env["RUSTDOCFLAGS"] == "--cfg docsrs -D warnings -D missing_docs"


def test_go_docs_require_coverage_and_render_every_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None,
        check: bool,
        text: bool,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env, check, text, capture_output
        calls.append(command)
        if command[:2] == ["go", "list"]:
            return subprocess.CompletedProcess(command, 0, "example/a\nexample/b\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("tools.run_native_docs.subprocess.run", fake_run)

    assert native_docs._build_go_docs() == 0
    assert calls == [
        ["go", "run", "./cmd/doccheck", "./..."],
        ["go", "list", "./..."],
        ["go", "doc", "-all", "example/a"],
        ["go", "doc", "-all", "example/b"],
    ]


def test_cpp_docs_create_output_parent_in_clean_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(native_docs, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(native_docs, "_run", fake_run)

    assert native_docs._build_cpp_docs() == 0
    assert (tmp_path / "docs" / "_build").is_dir()
    assert calls == [("doxygen", "docs/Doxyfile")]


def test_lean_docs_build_targets_sequentially_without_equation_derivation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], Path, dict[str, str] | None]] = []

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None,
        check: bool,
        text: bool,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[str]:
        del check, text, capture_output
        calls.append((command, cwd, env))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("tools.run_native_docs.subprocess.run", fake_run)

    assert native_docs._build_lean_docs() == 0
    assert [command for command, _cwd, _env in calls] == [
        ["lake", "build", target] for target in native_docs.LEAN_DOC_TARGETS
    ]
    for _command, cwd, env in calls:
        assert cwd == native_docs.REPO_ROOT / "scpn-fusion-lean" / "docbuild"
        assert env is not None
        assert env["DISABLE_EQUATIONS"] == "1"


def test_lean_docs_stop_after_the_first_failed_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None,
        check: bool,
        text: bool,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env, check, text, capture_output
        calls.append(command)
        return subprocess.CompletedProcess(command, 23 if len(calls) == 2 else 0, "", "")

    monkeypatch.setattr("tools.run_native_docs.subprocess.run", fake_run)

    assert native_docs._build_lean_docs() == 23
    assert calls == [
        ["lake", "build", native_docs.LEAN_DOC_TARGETS[0]],
        ["lake", "build", native_docs.LEAN_DOC_TARGETS[1]],
    ]
