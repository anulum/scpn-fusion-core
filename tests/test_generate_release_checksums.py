# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Release Checksum Generator Tests
"""Contract tests for the release SHA256 manifest generator tool."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.generate_release_checksums import (
    _display_path,
    _iter_release_files,
    _resolve,
    _sha256,
    generate_manifest,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _artifact(dist: Path, name: str, content: bytes) -> Path:
    """Write a release artifact into ``dist`` and return its path."""
    dist.mkdir(parents=True, exist_ok=True)
    path = dist / name
    path.write_bytes(content)
    return path


class TestHelpers:
    """Path, digest, and listing helpers used by the manifest generator."""

    def test_resolve_relative_is_repo_anchored(self) -> None:
        """A relative path resolves under the repository root."""
        assert _resolve("dist/pkg.whl") == REPO_ROOT / "dist" / "pkg.whl"

    def test_resolve_absolute_is_unchanged(self, tmp_path: Path) -> None:
        """An already-absolute path resolves to itself."""
        assert _resolve(str(tmp_path)) == tmp_path

    def test_sha256_matches_hashlib(self, tmp_path: Path) -> None:
        """The streamed digest matches a direct hashlib digest."""
        artifact = tmp_path / "a.bin"
        artifact.write_bytes(b"payload")
        assert _sha256(artifact) == hashlib.sha256(b"payload").hexdigest()

    def test_iter_release_files_excludes_manifest(self, tmp_path: Path) -> None:
        """The manifest file itself is excluded from the artifact list."""
        _artifact(tmp_path, "pkg.whl", b"x")
        _artifact(tmp_path, "SHA256SUMS", b"stale")
        files = _iter_release_files(tmp_path, "SHA256SUMS")
        assert [f.name for f in files] == ["pkg.whl"]

    def test_display_path_outside_repo_is_absolute(self, tmp_path: Path) -> None:
        """A path outside the repository root is displayed unchanged."""
        assert _display_path(tmp_path) == str(tmp_path)


class TestGenerateManifest:
    """The manifest generator over a release directory."""

    def test_roundtrip_writes_manifest_and_summary(self, tmp_path: Path) -> None:
        """The manifest lists each artifact with its digest and byte size."""
        dist = tmp_path / "dist"
        _artifact(dist, "pkg_a.whl", b"artifact-a")
        manifest = dist / "SHA256SUMS"

        summary = generate_manifest(dist, manifest)

        assert summary["count"] == 1
        assert manifest.exists()
        expected_line = f"{hashlib.sha256(b'artifact-a').hexdigest()}  pkg_a.whl"
        assert expected_line in manifest.read_text(encoding="utf-8")
        assert summary["entries"][0]["size_bytes"] == len(b"artifact-a")

    def test_empty_directory_is_rejected(self, tmp_path: Path) -> None:
        """An empty release directory raises a located error."""
        dist = tmp_path / "empty"
        dist.mkdir()
        with pytest.raises(ValueError, match="No release artifacts"):
            generate_manifest(dist, dist / "SHA256SUMS")


class TestMain:
    """The command-line entry point."""

    def test_main_generates_manifest_and_summary(self, tmp_path: Path) -> None:
        """The CLI writes the manifest and JSON summary and returns zero."""
        dist = tmp_path / "dist"
        _artifact(dist, "pkg.whl", b"z")
        manifest = dist / "SHA256SUMS"
        summary_json = tmp_path / "summary.json"

        return_code = main(
            [
                "--dist-dir",
                str(dist),
                "--output-file",
                str(manifest),
                "--summary-json",
                str(summary_json),
            ]
        )

        assert return_code == 0
        assert manifest.exists()
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        assert payload["count"] == 1
