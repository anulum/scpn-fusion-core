# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DIII-D EFIT GEQDSK downloader tests
"""Behavioral tests for the fail-closed EFIT GEQDSK downloader."""

from __future__ import annotations

import importlib
import logging
import runpy
import sys
import tempfile
from pathlib import Path

import pytest

from tools import download_efit_geqdsk as tool


class _FakeValue:
    """MDSplus-like wrapper around one provider payload."""

    def __init__(self, payload: object) -> None:
        self.payload = payload

    def data(self) -> object:
        """Return the wrapped payload."""
        return self.payload


class _FakeConnection:
    """Record MDSplus calls and return a configured payload."""

    def __init__(self, payload: object, *, failure: Exception | None = None) -> None:
        self.payload = payload
        self.failure = failure
        self.opened: tuple[str, int] | None = None
        self.node: str | None = None

    def openTree(self, tree: str, shot: int) -> None:
        """Record the tree/shot request or raise the configured failure."""
        if self.failure is not None:
            raise self.failure
        self.opened = (tree, shot)

    def get(self, node: str) -> object:
        """Record the node request and return the configured payload."""
        self.node = node
        return self.payload


class _FakeMDSplus:
    """Provide one fake connection through the legacy module API."""

    def __init__(self, connection: _FakeConnection) -> None:
        self.connection = connection
        self.host: str | None = None

    def Connection(self, host: str) -> _FakeConnection:
        """Record the host and return the fake connection."""
        self.host = host
        return self.connection


def test_filename_uses_canonical_and_unknown_scenarios() -> None:
    """Render stable cache names for canonical and caller-supplied shots."""
    assert tool._geqdsk_filename(163303) == "diiid_hmode_163303.geqdsk"
    assert tool._geqdsk_filename(999999) == "diiid_unknown_999999.geqdsk"


def test_cache_requires_exact_shot_identity_and_bounded_regular_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject fuzzy synthetic, empty, oversized, missing, and symlink candidates."""
    fuzzy = tmp_path / "diiid_hmode_1p5MA.geqdsk"
    fuzzy.write_text("synthetic proxy", encoding="ascii")
    numeric_collision = tmp_path / "shot_1633030.geqdsk"
    numeric_collision.write_text("different shot", encoding="ascii")
    directory = tmp_path / "shot_163303.geqdsk"
    directory.mkdir()
    assert tool._check_cache(163303, tmp_path) is None
    directory.rmdir()

    canonical = tmp_path / tool._geqdsk_filename(163303)
    canonical.touch()
    assert tool._check_cache(163303, tmp_path) is None
    canonical.write_text("exact shot", encoding="ascii")
    assert tool._check_cache(163303, tmp_path) == canonical

    canonical.unlink()
    alternate = tmp_path / "g163303.04200.geqdsk"
    alternate.write_text("exact shot alias", encoding="ascii")
    assert tool._check_cache(163303, tmp_path) == alternate

    alternate.unlink()
    monkeypatch.setattr(tool, "MAX_GEQDSK_DOWNLOAD_BYTES", 4)
    oversized = tmp_path / "shot_163303.geqdsk"
    oversized.write_text("large", encoding="ascii")
    assert tool._check_cache(163303, tmp_path) is None
    assert tool._is_usable_cache_file(tmp_path / "missing.geqdsk") is False

    target = tmp_path / "target.geqdsk"
    target.write_text("x", encoding="ascii")
    symlink = tmp_path / tool._geqdsk_filename(163303)
    symlink.symlink_to(target)
    assert tool._is_usable_cache_file(symlink) is False


def test_optional_mdsplus_loader_handles_absent_and_present_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return none on ImportError and the imported legacy module otherwise."""

    def missing(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", missing)
    assert tool._load_mdsplus() is None

    provider = _FakeMDSplus(_FakeConnection(b"payload"))
    monkeypatch.setattr(importlib, "import_module", lambda name: provider)
    assert tool._load_mdsplus() is provider


@pytest.mark.parametrize(
    ("raw", "error_fragment"),
    [
        (object(), "non-string data"),
        ("non-ascii-π", "non-ASCII"),
        (b"  \n", "empty"),
    ],
)
def test_payload_validation_rejects_unsafe_provider_values(
    raw: object,
    error_fragment: str,
) -> None:
    """Reject structured, non-ASCII, and empty provider payloads."""
    payload, error = tool._payload_bytes(raw, shot=163303)
    assert payload is None
    assert error is not None and error_fragment in error


def test_payload_validation_accepts_wrapped_text_and_bounds_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unwrap ASCII text while rejecting payloads beyond the fixed byte limit."""
    payload, error = tool._payload_bytes(_FakeValue("GEQDSK\n"), shot=163303)
    assert payload == b"GEQDSK\n"
    assert error is None

    monkeypatch.setattr(tool, "MAX_GEQDSK_DOWNLOAD_BYTES", 4)
    payload, error = tool._payload_bytes(b"12345", shot=163303)
    assert payload is None
    assert error is not None and "too large" in error


def test_atomic_writer_replaces_output_and_cleans_failed_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish complete bytes atomically and remove failed partial files."""
    output = tmp_path / "nested" / "shot.geqdsk"
    tool._write_payload_atomically(output, b"complete")
    assert output.read_bytes() == b"complete"
    assert list(output.parent.glob("*.part")) == []

    def reject_replace(self: Path, target: Path) -> Path:
        raise OSError(f"cannot replace {target}")

    monkeypatch.setattr(Path, "replace", reject_replace)
    with pytest.raises(OSError, match="cannot replace"):
        tool._write_payload_atomically(output, b"replacement")
    assert output.read_bytes() == b"complete"
    assert list(output.parent.glob("*.part")) == []


def test_atomic_writer_leaves_no_file_when_temporary_creation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave the destination absent if a temporary file cannot be created."""

    def reject_temporary(*args: object, **kwargs: object) -> object:
        raise OSError("temporary unavailable")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", reject_temporary)
    output = tmp_path / "nested" / "shot.geqdsk"
    with pytest.raises(OSError, match="temporary unavailable"):
        tool._write_payload_atomically(output, b"payload")
    assert not output.exists()


def test_mdsplus_download_success_uses_exact_provider_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write a validated fake payload after the expected host/tree/node calls."""
    connection = _FakeConnection(_FakeValue("GEQDSK payload\n"))
    provider = _FakeMDSplus(connection)
    monkeypatch.setattr(tool, "_load_mdsplus", lambda: provider)

    path, error = tool._try_mdsplus_download(
        163303,
        tmp_path,
        host="example.invalid",
        tree="efit-test",
        node="\\node",
    )

    assert error is None
    assert path == tmp_path / "diiid_hmode_163303.geqdsk"
    assert path.read_text(encoding="ascii") == "GEQDSK payload\n"
    assert provider.host == "example.invalid"
    assert connection.opened == ("efit-test", 163303)
    assert connection.node == "\\node"


def test_mdsplus_download_reports_absence_invalid_data_and_provider_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Convert optional-client, payload, and provider failures into status errors."""
    monkeypatch.setattr(tool, "_load_mdsplus", lambda: None)
    assert tool._try_mdsplus_download(163303, tmp_path)[1] == (
        "MDSplus Python module not installed"
    )

    monkeypatch.setattr(
        tool,
        "_load_mdsplus",
        lambda: _FakeMDSplus(_FakeConnection(object())),
    )
    assert "non-string data" in (tool._try_mdsplus_download(163303, tmp_path)[1] or "")

    monkeypatch.setattr(
        tool,
        "_load_mdsplus",
        lambda: _FakeMDSplus(_FakeConnection(b"unused", failure=RuntimeError("offline"))),
    )
    assert "offline" in (tool._try_mdsplus_download(163303, tmp_path)[1] or "")


@pytest.mark.parametrize(
    ("shots", "message"),
    [
        ([], "At least one"),
        ([0], "positive integers"),
        ([True], "positive integers"),
        ([163303, 163303], "unique"),
    ],
)
def test_download_rejects_invalid_shot_collections(
    tmp_path: Path,
    shots: list[int],
    message: str,
) -> None:
    """Reject empty, non-positive, boolean, and duplicate shot requests."""
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match=message):
        tool.download_geqdsks(cache_dir=cache, shots=shots, try_mdsplus=False)
    assert not cache.exists()


def test_download_prefers_exact_cache_and_reports_disabled_missing(tmp_path: Path) -> None:
    """Use only exact-shot cache entries and fail closed when downloads are disabled."""
    cache = tmp_path / "cache"
    exact = cache / tool._geqdsk_filename(163303)
    cache.mkdir()
    exact.write_text("exact shot", encoding="ascii")

    results = tool.download_geqdsks(
        cache_dir=cache,
        shots=[163303, 999999],
        try_mdsplus=False,
    )

    assert results == [
        tool.ShotStatus(163303, "H-mode", True, "cache", exact),
        tool.ShotStatus(
            999999,
            "unknown",
            False,
            "missing",
            None,
            "MDSplus download disabled",
        ),
    ]


def test_download_records_provider_success_and_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Continue across shots while retaining provider success and error custody."""
    downloaded = tmp_path / tool._geqdsk_filename(163303)

    def download(shot: int, cache_dir: Path, **kwargs: object) -> tuple[Path | None, str | None]:
        if shot == 163303:
            downloaded.parent.mkdir(parents=True, exist_ok=True)
            downloaded.write_text("provider", encoding="ascii")
            return downloaded, None
        return None, "provider unavailable"

    monkeypatch.setattr(tool, "_try_mdsplus_download", download)
    with caplog.at_level(logging.WARNING):
        results = tool.download_geqdsks(cache_dir=tmp_path, shots=[163303, 154406])

    assert [result.source for result in results] == ["mdsplus", "missing"]
    assert results[1].error == "provider unavailable"
    assert "provider unavailable" in caplog.text


def test_status_printer_covers_complete_missing_and_empty_results(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Render complete and missing summaries while rejecting an empty collection."""
    path = tmp_path / "shot.geqdsk"
    path.write_text("x", encoding="ascii")
    tool._print_status([tool.ShotStatus(163303, "H-mode", True, "cache", path)])
    assert "All target shots are available." in capsys.readouterr().out

    tool._print_status([tool.ShotStatus(154406, "hybrid", False, "missing", None, "disabled")])
    output = capsys.readouterr().out
    assert "1 file(s) missing" in output
    assert "disabled" in output

    with pytest.raises(ValueError, match="empty"):
        tool._print_status([])


def test_cli_and_direct_script_entry_are_network_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise successful and missing CLI exits plus the actual script guard."""
    cache = tmp_path / "cache"
    cache.mkdir()
    exact = cache / tool._geqdsk_filename(163303)
    exact.write_text("exact shot", encoding="ascii")

    assert tool.main(["--cache-dir", str(cache), "--no-mdsplus", "--shots", "163303", "-v"]) == 0
    assert "1/1 available" in capsys.readouterr().out

    exact.unlink()
    assert tool.main(["--cache-dir", str(cache), "--no-mdsplus", "--shots", "163303"]) == 1
    assert "0/1 available" in capsys.readouterr().out

    module_path = Path(tool.__file__)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(module_path), "--cache-dir", str(cache), "--no-mdsplus", "--shots", "163303"],
    )
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(module_path), run_name="__main__")
    assert exit_info.value.code == 1
    assert "0/1 available" in capsys.readouterr().out
