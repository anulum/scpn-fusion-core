"""Tests for tools/check_test_module_linkage.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "check_test_module_linkage.py"
SPEC = importlib.util.spec_from_file_location("check_test_module_linkage", MODULE_PATH)
assert SPEC and SPEC.loader
linkage = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = linkage
SPEC.loader.exec_module(linkage)


def test_collect_unlinked_modules_returns_known_paths() -> None:
    unlinked = linkage.collect_unlinked_modules(
        source_root=ROOT / "src" / "scpn_fusion",
        test_root=ROOT / "tests",
    )
    assert len(unlinked) > 0
    assert "src/scpn_fusion/core/force_balance.py" in unlinked


def test_main_passes_with_repo_allowlist() -> None:
    rc = linkage.main([])
    assert rc == 0


def test_main_fails_with_empty_allowlist(tmp_path: Path) -> None:
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                "allowlisted_modules": [],
            }
        ),
        encoding="utf-8",
    )
    rc = linkage.main(
        [
            "--allowlist",
            str(allowlist),
        ]
    )
    assert rc == 1


def test_collect_unlinked_modules_detects_ast_import_linkage(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "scpn_fusion"
    module_dir = source_root / "core"
    module_dir.mkdir(parents=True)
    (module_dir / "sample_module.py").write_text("VALUE = 1\n", encoding="utf-8")

    test_root = tmp_path / "tests"
    test_root.mkdir(parents=True)
    (test_root / "test_sample_module.py").write_text(
        "from scpn_fusion.core import sample_module\n"
        "def test_linkage() -> None:\n"
        "    assert sample_module.VALUE == 1\n",
        encoding="utf-8",
    )

    unlinked = linkage.collect_unlinked_modules(
        source_root=source_root,
        test_root=test_root,
    )
    assert unlinked == []
