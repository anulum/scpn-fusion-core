# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Release Checksum Verifier Tests
"""Contract tests for the release SHA256 manifest verifier tool."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.verify_release_checksums import (
    _parse_manifest,
    _resolve,
    main,
    verify_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _artifact(dist: Path, name: str, content: bytes) -> str:
    """Write an artifact into ``dist`` and return its SHA256 digest."""
    dist.mkdir(parents=True, exist_ok=True)
    (dist / name).write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _write_manifest(dist: Path, lines: list[str]) -> Path:
    """Write a SHA256SUMS manifest with the given lines into ``dist``."""
    manifest = dist / "SHA256SUMS"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


class TestResolveAndParse:
    """Path resolution and manifest parsing helpers."""

    def test_resolve_relative_is_repo_anchored(self) -> None:
        """A relative path resolves under the repository root."""
        assert _resolve("dist/SHA256SUMS") == REPO_ROOT / "dist" / "SHA256SUMS"

    def test_resolve_absolute_is_unchanged(self, tmp_path: Path) -> None:
        """An already-absolute path resolves to itself."""
        assert _resolve(str(tmp_path)) == tmp_path

    def test_parse_manifest_skips_blank_lines(self, tmp_path: Path) -> None:
        """Blank manifest lines are skipped during parsing."""
        manifest = _write_manifest(tmp_path, ["", "abc  pkg.whl", "   "])
        assert _parse_manifest(manifest) == [("abc", "pkg.whl")]

    def test_parse_manifest_rejects_malformed_line(self, tmp_path: Path) -> None:
        """A manifest line without exactly two fields is rejected."""
        manifest = _write_manifest(tmp_path, ["only-one-field"])
        with pytest.raises(ValueError, match="Malformed checksum manifest line"):
            _parse_manifest(manifest)

    def test_parse_manifest_rejects_empty(self, tmp_path: Path) -> None:
        """A manifest with no data lines is rejected as empty."""
        manifest = _write_manifest(tmp_path, ["", "   "])
        with pytest.raises(ValueError, match="manifest is empty"):
            _parse_manifest(manifest)


class TestVerifyManifest:
    """The manifest verifier over a release directory."""

    def test_all_checksums_match(self, tmp_path: Path) -> None:
        """A consistent manifest and directory verify as passing."""
        dist = tmp_path / "dist"
        digest = _artifact(dist, "pkg.whl", b"payload")
        _write_manifest(dist, [f"{digest}  pkg.whl"])
        summary = verify_manifest(dist_dir=dist, manifest=dist / "SHA256SUMS")
        assert summary["overall_pass"] is True
        assert summary["verified_entries"] == 1

    def test_missing_artifact_fails(self, tmp_path: Path) -> None:
        """A declared file absent from the directory fails verification."""
        dist = tmp_path / "dist"
        dist.mkdir()
        _write_manifest(dist, ["deadbeef  ghost.whl"])
        summary = verify_manifest(dist_dir=dist, manifest=dist / "SHA256SUMS")
        assert summary["overall_pass"] is False
        assert any("missing artifact: ghost.whl" in row for row in summary["failures"])

    def test_digest_mismatch_fails(self, tmp_path: Path) -> None:
        """A file whose digest differs from the manifest fails verification."""
        dist = tmp_path / "dist"
        _artifact(dist, "pkg.whl", b"payload")
        _write_manifest(dist, ["0000  pkg.whl"])
        summary = verify_manifest(dist_dir=dist, manifest=dist / "SHA256SUMS")
        assert summary["overall_pass"] is False
        assert any("digest mismatch: pkg.whl" in row for row in summary["failures"])

    def test_untracked_artifact_fails(self, tmp_path: Path) -> None:
        """A directory file absent from the manifest is reported as untracked."""
        dist = tmp_path / "dist"
        digest = _artifact(dist, "pkg.whl", b"payload")
        _artifact(dist, "extra.whl", b"other")
        _write_manifest(dist, [f"{digest}  pkg.whl"])
        summary = verify_manifest(dist_dir=dist, manifest=dist / "SHA256SUMS")
        assert summary["overall_pass"] is False
        assert any("untracked artifact: extra.whl" in row for row in summary["failures"])


class TestMain:
    """The command-line entry point."""

    def test_main_returns_zero_on_success(self, tmp_path: Path) -> None:
        """A passing verification writes a summary and returns zero."""
        dist = tmp_path / "dist"
        digest = _artifact(dist, "pkg.whl", b"payload")
        manifest = _write_manifest(dist, [f"{digest}  pkg.whl"])
        summary_json = tmp_path / "summary.json"
        rc = main(
            [
                "--dist-dir",
                str(dist),
                "--manifest",
                str(manifest),
                "--summary-json",
                str(summary_json),
            ]
        )
        assert rc == 0
        assert json.loads(summary_json.read_text(encoding="utf-8"))["overall_pass"] is True

    def test_main_returns_one_on_failure(self, tmp_path: Path) -> None:
        """A failing verification prints failures and returns one."""
        dist = tmp_path / "dist"
        dist.mkdir()
        manifest = _write_manifest(dist, ["deadbeef  ghost.whl"])
        summary_json = tmp_path / "summary.json"
        rc = main(
            [
                "--dist-dir",
                str(dist),
                "--manifest",
                str(manifest),
                "--summary-json",
                str(summary_json),
            ]
        )
        assert rc == 1
        assert json.loads(summary_json.read_text(encoding="utf-8"))["overall_pass"] is False
