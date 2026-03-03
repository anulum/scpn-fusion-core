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

    (root / "sample.npz").write_bytes(b"changed")
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

