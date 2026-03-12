"""Tests for release checksum tooling."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

GEN_PATH = ROOT / "tools" / "generate_release_checksums.py"
GEN_SPEC = importlib.util.spec_from_file_location("generate_release_checksums", GEN_PATH)
assert GEN_SPEC and GEN_SPEC.loader
generate_release_checksums = importlib.util.module_from_spec(GEN_SPEC)
GEN_SPEC.loader.exec_module(generate_release_checksums)

VERIFY_PATH = ROOT / "tools" / "verify_release_checksums.py"
VERIFY_SPEC = importlib.util.spec_from_file_location("verify_release_checksums", VERIFY_PATH)
assert VERIFY_SPEC and VERIFY_SPEC.loader
verify_release_checksums = importlib.util.module_from_spec(VERIFY_SPEC)
VERIFY_SPEC.loader.exec_module(verify_release_checksums)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_generate_and_verify_manifest_roundtrip(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    _write(dist / "pkg_a.whl", "artifact-a")
    _write(dist / "pkg_b.tar.gz", "artifact-b")
    manifest = dist / "SHA256SUMS"

    summary = generate_release_checksums.generate_manifest(dist, manifest)
    assert summary["count"] == 2
    assert manifest.exists()

    verify_summary = verify_release_checksums.verify_manifest(dist_dir=dist, manifest=manifest)
    assert verify_summary["overall_pass"] is True
    assert verify_summary["failures"] == []
    assert verify_summary["verified_entries"] == 2


def test_verify_detects_tamper(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    _write(dist / "pkg_a.whl", "artifact-a")
    _write(dist / "pkg_b.tar.gz", "artifact-b")
    manifest = dist / "SHA256SUMS"
    generate_release_checksums.generate_manifest(dist, manifest)

    # Tamper artifact after manifest generation.
    _write(dist / "pkg_a.whl", "artifact-a-modified")

    verify_summary = verify_release_checksums.verify_manifest(dist_dir=dist, manifest=manifest)
    assert verify_summary["overall_pass"] is False
    assert any("digest mismatch: pkg_a.whl" in item for item in verify_summary["failures"])


def test_cli_writes_summary_files(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    _write(dist / "pkg_a.whl", "artifact-a")
    _write(dist / "pkg_b.tar.gz", "artifact-b")
    manifest = dist / "SHA256SUMS"
    gen_summary = tmp_path / "gen_summary.json"
    verify_summary = tmp_path / "verify_summary.json"

    rc_gen = generate_release_checksums.main(
        [
            "--dist-dir",
            str(dist),
            "--output-file",
            str(manifest),
            "--summary-json",
            str(gen_summary),
        ]
    )
    assert rc_gen == 0
    gen_payload = json.loads(gen_summary.read_text(encoding="utf-8"))
    assert int(gen_payload["count"]) == 2

    rc_verify = verify_release_checksums.main(
        [
            "--dist-dir",
            str(dist),
            "--manifest",
            str(manifest),
            "--summary-json",
            str(verify_summary),
        ]
    )
    assert rc_verify == 0
    verify_payload = json.loads(verify_summary.read_text(encoding="utf-8"))
    assert verify_payload["overall_pass"] is True
