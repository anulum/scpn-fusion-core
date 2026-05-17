# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for Git LFS hygiene guard enforcement."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "tools" / "check_lfs_hygiene.py"
REQUIRED_NPZ = "validation/reference_data/diiid/disruption_shots/shot_154406_hybrid.npz"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_lfs_hygiene", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load tools/check_lfs_hygiene.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_required_file(tmp_path: Path) -> Path:
    artifact = tmp_path / REQUIRED_NPZ
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"PK\x03\x04synthetic-npz")
    (tmp_path / ".gitattributes").write_text(
        "validation/reference_data/**/*.npz filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    return artifact


def test_lfs_hygiene_rejects_lfs_attr_full_blob(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _load_module()
    _prepare_required_file(tmp_path)

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "_tracked_files", lambda: [REQUIRED_NPZ])
    monkeypatch.setattr(module, "_check_attr_filter", lambda _paths: {REQUIRED_NPZ: "lfs"})
    monkeypatch.setattr(module, "_index_blob_head", lambda _path: b"PK\x03\x04synthetic-npz")

    assert module.main(["--max-nonlfs-bytes", "1048576"]) == 1
    assert "committed as full blobs" in capsys.readouterr().err


def test_lfs_hygiene_accepts_lfs_pointer_blob(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    _prepare_required_file(tmp_path)

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "_tracked_files", lambda: [REQUIRED_NPZ])
    monkeypatch.setattr(module, "_check_attr_filter", lambda _paths: {REQUIRED_NPZ: "lfs"})
    monkeypatch.setattr(
        module,
        "_index_blob_head",
        lambda _path: b"version https://git-lfs.github.com/spec/v1\n",
    )

    assert module.main(["--max-nonlfs-bytes", "1048576"]) == 0
