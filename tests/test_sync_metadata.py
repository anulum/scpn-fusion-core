# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "sync_metadata.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sync_metadata", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load tools/sync_metadata.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sync_metadata_check_detects_drift(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    repo = tmp_path / "repo"
    version_file = repo / "src" / "scpn_fusion" / "VERSION"
    pyproject = repo / "pyproject.toml"
    version_file.parent.mkdir(parents=True, exist_ok=True)
    version_file.write_text("9.9.9\n", encoding="utf-8")
    pyproject.write_text(
        '[project]\nname = "scpn-fusion"\nversion = "0.0.1"\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "REPO_ROOT", repo)
    monkeypatch.setattr(module, "VERSION_FILE", version_file)
    assert module.main(["--check"]) == 1


def test_sync_metadata_apply_updates_file(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    repo = tmp_path / "repo"
    version_file = repo / "src" / "scpn_fusion" / "VERSION"
    pyproject = repo / "pyproject.toml"
    version_file.parent.mkdir(parents=True, exist_ok=True)
    version_file.write_text("3.9.9\n", encoding="utf-8")
    pyproject.write_text(
        '[project]\nname = "scpn-fusion"\nversion = "0.0.1"\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "REPO_ROOT", repo)
    monkeypatch.setattr(module, "VERSION_FILE", version_file)
    assert module.main([]) == 0
    assert 'version = "3.9.9"' in pyproject.read_text(encoding="utf-8")
    assert module.main(["--check"]) == 0
