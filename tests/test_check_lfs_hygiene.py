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
from types import ModuleType
from types import SimpleNamespace
from typing import Any, cast

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "tools" / "check_lfs_hygiene.py"
REQUIRED_NPZ = "weights/fno_turbulence_jax.npz"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tools.check_lfs_hygiene", SCRIPT_PATH)
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
        "weights/fno_turbulence_jax.npz filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    return artifact


def test_tracked_files_splits_git_nul_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracked-file collection drops empty NUL-separated records."""
    module = _load_module()
    typed_module = cast(Any, module)

    monkeypatch.setattr(typed_module, "_git", lambda *_args: "a.py\0\0b.npz\0")

    assert typed_module._tracked_files() == ["a.py", "b.npz"]


def test_git_adapter_returns_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The git subprocess adapter returns command stdout."""
    module = _load_module()
    typed_module = cast(Any, module)

    def fake_run(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(stdout="ok")

    monkeypatch.setattr(typed_module.subprocess, "run", fake_run)

    assert typed_module._git("status", "--short") == "ok"


def test_required_glob_matches_training_artifacts() -> None:
    """Required LFS globs cover declared model and training artifact families."""
    module = _load_module()
    typed_module = cast(Any, module)

    assert typed_module._matches_required_glob(REQUIRED_NPZ)
    assert typed_module._matches_required_glob("training_logs/run/model.npz")
    assert not typed_module._matches_required_glob("docs/example.npz")


def test_check_attr_filter_parses_git_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Git attribute output is parsed into path-to-filter mappings."""
    module = _load_module()
    typed_module = cast(Any, module)

    def fake_run(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(stdout=f"{REQUIRED_NPZ}: filter: lfs\nbad line\nx.npy: filter: unset\n")

    monkeypatch.setattr(typed_module.subprocess, "run", fake_run)

    assert typed_module._check_attr_filter([]) == {}
    assert typed_module._check_attr_filter([REQUIRED_NPZ]) == {REQUIRED_NPZ: "lfs", "x.npy": "unset"}


def test_index_blob_head_truncates_git_blob(monkeypatch: pytest.MonkeyPatch) -> None:
    """Index blob reads return only the requested byte prefix."""
    module = _load_module()
    typed_module = cast(Any, module)

    def fake_run(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(stdout=b"abcdef")

    monkeypatch.setattr(typed_module.subprocess, "run", fake_run)

    assert typed_module._index_blob_head("weights/x.npz", byte_count=3) == b"abc"
    assert typed_module._is_lfs_pointer_blob(b"version https://git-lfs.github.com/spec/v1\nabc")


def test_lfs_hygiene_rejects_lfs_attr_full_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    _prepare_required_file(tmp_path)

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "_tracked_files", lambda: [REQUIRED_NPZ])
    monkeypatch.setattr(module, "_check_attr_filter", lambda _paths: {REQUIRED_NPZ: "lfs"})
    monkeypatch.setattr(module, "_index_blob_head", lambda _path: b"PK\x03\x04synthetic-npz")

    typed_module = cast(Any, module)
    assert typed_module.main(["--max-nonlfs-bytes", "1048576"]) == 1
    assert "committed as full blobs" in capsys.readouterr().err


def test_lfs_hygiene_rejects_missing_policy_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A repository without ``.gitattributes`` fails closed."""
    module = _load_module()
    typed_module = cast(Any, module)

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    assert typed_module.main([]) == 1
    assert ".gitattributes missing" in capsys.readouterr().err


def test_lfs_hygiene_reports_missing_required_and_oversize_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing LFS attributes and oversize binary blobs are reported together."""
    module = _load_module()
    typed_module = cast(Any, module)
    (tmp_path / ".gitattributes").write_text("", encoding="utf-8")
    big_binary = tmp_path / "data" / "big.npy"
    big_binary.parent.mkdir()
    big_binary.write_bytes(b"123456")

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "_tracked_files",
        lambda: [REQUIRED_NPZ, "data/missing.npy", "data/big.npy"],
    )
    monkeypatch.setattr(module, "_check_attr_filter", lambda _paths: {"data/big.npy": "unspecified"})

    assert typed_module.main(["--max-nonlfs-bytes", "4"]) == 1
    stderr = capsys.readouterr().err
    assert "not tracked via Git LFS" in stderr
    assert "Oversize tracked binary files" in stderr
    assert "data/big.npy" in stderr


def test_lfs_hygiene_accepts_lfs_pointer_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    typed_module = cast(Any, module)
    assert typed_module.main(["--max-nonlfs-bytes", "1048576"]) == 0
