# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Reference Data Provenance Manifest Tests
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "generate_reference_data_provenance_manifest.py"
SPEC = importlib.util.spec_from_file_location("generate_reference_data_provenance_manifest", MODULE_PATH)
assert SPEC and SPEC.loader
prov = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prov
SPEC.loader.exec_module(prov)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_build_manifest_assigns_license_for_all_files(tmp_path: Path) -> None:
    root = tmp_path / "reference_data"
    root.mkdir(parents=True)
    (root / "README.md").write_text("reference data\n", encoding="utf-8")
    (root / "dataset.npz").write_bytes(b"abc123")
    policy = root / "provenance_policy.json"
    manifest = root / "provenance_manifest.json"
    _write_json(
        policy,
        {
            "rules": [
                {
                    "id": "metadata",
                    "glob": "README.md",
                    "source_type": "documentation",
                    "source": "docs",
                    "license": "AGPL-3.0-or-later",
                },
                {
                    "id": "synthetic_data",
                    "glob": "*.npz",
                    "source_type": "synthetic_generated",
                    "source": "tooling",
                    "license": "synthetic-v1",
                },
                {
                    "id": "policy",
                    "glob": "provenance_policy.json",
                    "source_type": "documentation",
                    "source": "policy",
                    "license": "AGPL-3.0-or-later",
                },
            ]
        },
    )

    payload = prov.build_manifest(root=root, policy_path=policy, manifest_path=manifest)
    assert payload["file_count"] == 3
    licenses = {row["license"] for row in payload["files"]}
    assert "AGPL-3.0-or-later" in licenses
    assert "synthetic-v1" in licenses


def test_main_check_mode_detects_stale_manifest(tmp_path: Path) -> None:
    root = tmp_path / "reference_data"
    root.mkdir(parents=True)
    (root / "README.md").write_text("reference data\n", encoding="utf-8")
    (root / "sample.npz").write_bytes(b"abc123")
    policy = root / "provenance_policy.json"
    manifest = root / "provenance_manifest.json"
    _write_json(
        policy,
        {
            "rules": [
                {
                    "id": "metadata",
                    "glob": "README.md",
                    "source_type": "documentation",
                    "source": "docs",
                    "license": "AGPL-3.0-or-later",
                },
                {
                    "id": "synthetic_data",
                    "glob": "*.npz",
                    "source_type": "synthetic_generated",
                    "source": "tooling",
                    "license": "synthetic-v1",
                },
                {
                    "id": "policy",
                    "glob": "provenance_policy.json",
                    "source_type": "documentation",
                    "source": "policy",
                    "license": "AGPL-3.0-or-later",
                },
            ]
        },
    )

    assert (
        prov.main(
            [
                "--root",
                str(root),
                "--policy",
                str(policy),
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )
    assert prov.main(
        [
            "--root",
            str(root),
            "--policy",
            str(policy),
            "--manifest",
            str(manifest),
            "--check",
        ]
    ) == 0

    (root / "sample_extra.npz").write_bytes(b"changed")
    assert prov.main(
        [
            "--root",
            str(root),
            "--policy",
            str(policy),
            "--manifest",
            str(manifest),
            "--check",
        ]
    ) == 1


def test_build_manifest_raises_on_unmatched_file(tmp_path: Path) -> None:
    root = tmp_path / "reference_data"
    root.mkdir(parents=True)
    (root / "unmatched.dat").write_bytes(b"x")
    policy = root / "provenance_policy.json"
    manifest = root / "provenance_manifest.json"
    _write_json(
        policy,
        {
            "rules": [
                {
                    "id": "policy",
                    "glob": "provenance_policy.json",
                    "source_type": "documentation",
                    "source": "policy",
                    "license": "AGPL-3.0-or-later",
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="No provenance policy rule matched file"):
        prov.build_manifest(root=root, policy_path=policy, manifest_path=manifest)


def test_build_manifest_normalizes_text_line_endings_for_hash_and_size(tmp_path: Path) -> None:
    root = tmp_path / "reference_data"
    root.mkdir(parents=True)
    text_path = root / "README.md"
    text_path.write_bytes(b"line1\r\nline2\r\n")
    policy = root / "provenance_policy.json"
    manifest = root / "provenance_manifest.json"
    _write_json(
        policy,
        {
            "rules": [
                {
                    "id": "metadata",
                    "glob": "README.md",
                    "source_type": "documentation",
                    "source": "docs",
                    "license": "AGPL-3.0-or-later",
                },
                {
                    "id": "policy",
                    "glob": "provenance_policy.json",
                    "source_type": "documentation",
                    "source": "policy",
                    "license": "AGPL-3.0-or-later",
                },
            ]
        },
    )

    crlf_manifest = prov.build_manifest(root=root, policy_path=policy, manifest_path=manifest)
    crlf_row = next(row for row in crlf_manifest["files"] if row["path"] == "README.md")

    text_path.write_bytes(b"line1\nline2\n")
    lf_manifest = prov.build_manifest(root=root, policy_path=policy, manifest_path=manifest)
    lf_row = next(row for row in lf_manifest["files"] if row["path"] == "README.md")

    assert crlf_row["sha256"] == lf_row["sha256"]
    assert crlf_row["size_bytes"] == lf_row["size_bytes"]


def test_build_manifest_sorts_files_by_relative_posix_path(tmp_path: Path) -> None:
    root = tmp_path / "reference_data"
    (root / "b").mkdir(parents=True)
    (root / "a").mkdir(parents=True)
    (root / "b" / "y.txt").write_text("y\n", encoding="utf-8")
    (root / "a" / "x.txt").write_text("x\n", encoding="utf-8")
    policy = root / "provenance_policy.json"
    manifest = root / "provenance_manifest.json"
    _write_json(
        policy,
        {
            "rules": [
                {
                    "id": "txt",
                    "glob": "**/*.txt",
                    "source_type": "documentation",
                    "source": "docs",
                    "license": "AGPL-3.0-or-later",
                },
                {
                    "id": "policy",
                    "glob": "provenance_policy.json",
                    "source_type": "documentation",
                    "source": "policy",
                    "license": "AGPL-3.0-or-later",
                },
            ]
        },
    )

    payload = prov.build_manifest(root=root, policy_path=policy, manifest_path=manifest)
    paths = [row["path"] for row in payload["files"]]
    assert paths == sorted(paths)


def test_normalize_for_check_sorts_datasets_and_files() -> None:
    payload = {
        "generated_at_utc": "2026-03-03T00:00:00+00:00",
        "datasets": [
            {"id": "zeta", "file_count": 1},
            {"id": "alpha", "file_count": 2},
        ],
        "files": [
            {"path": "z/file.json", "sha256": "a"},
            {"path": "a/file.json", "sha256": "b"},
        ],
    }

    normalized = prov._normalize_for_check(payload)
    assert normalized["generated_at_utc"] == "<normalized>"
    assert [row["id"] for row in normalized["datasets"]] == ["alpha", "zeta"]
    assert [row["path"] for row in normalized["files"]] == [
        "a/file.json",
        "z/file.json",
    ]


def test_normalize_for_check_ignores_byte_fingerprint_fields() -> None:
    payload_a = {
        "generated_at_utc": "2026-03-03T00:00:00+00:00",
        "datasets": [
            {"id": "sparc", "file_count": 2, "total_bytes": 12345},
        ],
        "files": [
            {
                "path": "sparc/example.npz",
                "dataset_id": "sparc",
                "source_type": "public_reference",
                "source": "SPARC",
                "license": "CC-BY-4.0",
                "size_bytes": 111,
                "sha256": "aaa",
            }
        ],
    }
    payload_b = {
        "generated_at_utc": "2026-03-04T00:00:00+00:00",
        "datasets": [
            {"id": "sparc", "file_count": 2, "total_bytes": 99999},
        ],
        "files": [
            {
                "path": "sparc/example.npz",
                "dataset_id": "sparc",
                "source_type": "public_reference",
                "source": "SPARC",
                "license": "CC-BY-4.0",
                "size_bytes": 222,
                "sha256": "bbb",
            }
        ],
    }

    assert prov._normalize_for_check(payload_a) == prov._normalize_for_check(payload_b)


def test_build_manifest_ignores_untracked_files_when_git_index_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "reference_data"
    root.mkdir(parents=True)
    (root / "README.md").write_text("reference data\n", encoding="utf-8")
    (root / "untracked_extra.npz").write_bytes(b"extra")
    policy = root / "provenance_policy.json"
    manifest = root / "provenance_manifest.json"
    _write_json(
        policy,
        {
            "rules": [
                {
                    "id": "metadata",
                    "glob": "README.md",
                    "source_type": "documentation",
                    "source": "docs",
                    "license": "AGPL-3.0-or-later",
                },
                {
                    "id": "policy",
                    "glob": "provenance_policy.json",
                    "source_type": "documentation",
                    "source": "policy",
                    "license": "AGPL-3.0-or-later",
                },
            ]
        },
    )

    monkeypatch.setattr(
        prov,
        "_git_tracked_paths",
        lambda _root: {"README.md", "provenance_policy.json"},
    )
    payload = prov.build_manifest(root=root, policy_path=policy, manifest_path=manifest)
    assert payload["file_count"] == 2
    assert sorted(row["path"] for row in payload["files"]) == [
        "README.md",
        "provenance_policy.json",
    ]
