# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for tools/check_test_module_linkage.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "check_test_module_linkage.py"
SPEC = importlib.util.spec_from_file_location("tools.check_test_module_linkage", MODULE_PATH)
assert SPEC and SPEC.loader
linkage = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = linkage
SPEC.loader.exec_module(linkage)


def test_collect_unlinked_modules_returns_known_paths() -> None:
    unlinked = linkage.collect_unlinked_modules(
        source_root=ROOT / "src" / "scpn_fusion",
        test_root=ROOT / "tests",
    )
    assert isinstance(unlinked, list)
    assert "src/scpn_fusion/core/force_balance.py" not in unlinked


def test_main_passes_with_repo_allowlist() -> None:
    rc = linkage.main([])
    assert rc == 0


def test_main_writes_summary_json(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    rc = linkage.main(["--summary-json", str(summary_path)])
    assert rc == 0
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["overall_pass"] is True
    assert payload["unexpected_count"] == 0


def test_main_reports_unexpected_modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(linkage, "REPO_ROOT", tmp_path)
    source_root = tmp_path / "src" / "scpn_fusion"
    module_dir = source_root / "core"
    module_dir.mkdir(parents=True)
    (module_dir / "unlinked.py").write_text("VALUE = 1\n", encoding="utf-8")

    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_other.py").write_text(
        "def test_other() -> None:\n    assert True\n",
        encoding="utf-8",
    )

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
            "--source-root",
            str(source_root),
            "--test-root",
            str(test_root),
            "--allowlist",
            str(allowlist),
        ]
    )
    assert rc == 1


def test_main_reports_stale_allowlist_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(linkage, "REPO_ROOT", tmp_path)
    source_root = tmp_path / "src" / "scpn_fusion"
    source_root.mkdir(parents=True)
    test_root = tmp_path / "tests"
    test_root.mkdir()
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                "allowlisted_modules": [{"path": "src/scpn_fusion/core/stale.py"}],
            }
        ),
        encoding="utf-8",
    )

    rc = linkage.main(
        [
            "--source-root",
            str(source_root),
            "--test-root",
            str(test_root),
            "--allowlist",
            str(allowlist),
        ]
    )
    assert rc == 1


def test_main_allows_stale_allowlist_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(linkage, "REPO_ROOT", tmp_path)
    source_root = tmp_path / "src" / "scpn_fusion"
    source_root.mkdir(parents=True)
    test_root = tmp_path / "tests"
    test_root.mkdir()
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                "allowlisted_modules": [{"path": "src/scpn_fusion/core/stale.py"}],
            }
        ),
        encoding="utf-8",
    )

    rc = linkage.main(
        [
            "--source-root",
            str(source_root),
            "--test-root",
            str(test_root),
            "--allowlist",
            str(allowlist),
            "--allow-stale-allowlist",
        ]
    )
    assert rc == 0


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


def test_collect_unlinked_modules_detects_stem_and_text_linkage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(linkage, "REPO_ROOT", tmp_path)
    source_root = tmp_path / "src" / "scpn_fusion"
    module_dir = source_root / "core"
    module_dir.mkdir(parents=True)
    (module_dir / "__init__.py").write_text("", encoding="utf-8")
    (module_dir / "stem_linked.py").write_text("VALUE = 1\n", encoding="utf-8")
    (module_dir / "corpus_linked.py").write_text("VALUE = 2\n", encoding="utf-8")
    (module_dir / "name_linked.py").write_text("VALUE = 3\n", encoding="utf-8")
    (module_dir / "missing_link.py").write_text("VALUE = 4\n", encoding="utf-8")

    test_root = tmp_path / "tests"
    test_root.mkdir(parents=True)
    (test_root / "test_stem_linked.py").write_text(
        "def test_marker() -> None:\n    assert True\n",
        encoding="utf-8",
    )
    (test_root / "test_textual.py").write_text(
        "MODULE = 'scpn_fusion.core.corpus_linked'\nNAME = 'test_name_linked'\n",
        encoding="utf-8",
    )

    unlinked = linkage.collect_unlinked_modules(
        source_root=source_root,
        test_root=test_root,
    )

    assert unlinked == ["src/scpn_fusion/core/missing_link.py"]


def test_collect_import_targets_handles_syntax_and_import_forms(tmp_path: Path) -> None:
    syntax_error = tmp_path / "test_bad.py"
    syntax_error.write_text("def bad(:\n", encoding="utf-8")
    assert linkage._collect_import_targets(syntax_error) == set()

    imports_file = tmp_path / "test_imports.py"
    imports_file.write_text(
        "import scpn_fusion.core.force_balance as fb\n"
        "from scpn_fusion.core import current_diffusion\n"
        "from scpn_fusion.core import *\n"
        "from .local import helper\n",
        encoding="utf-8",
    )

    assert linkage._collect_import_targets(imports_file) == {
        "scpn_fusion.core.force_balance",
        "scpn_fusion.core",
        "scpn_fusion.core.current_diffusion",
    }


def test_load_allowlist_validates_schema(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Allowlist file not found"):
        linkage.load_allowlist(tmp_path / "missing.json")

    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        linkage.load_allowlist(allowlist)

    allowlist.write_text(json.dumps({"allowlisted_modules": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list"):
        linkage.load_allowlist(allowlist)

    allowlist.write_text(json.dumps({"allowlisted_modules": ["bad"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        linkage.load_allowlist(allowlist)

    allowlist.write_text(json.dumps({"allowlisted_modules": [{"path": ""}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty string"):
        linkage.load_allowlist(allowlist)


def test_resolve_handles_relative_and_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(linkage, "REPO_ROOT", tmp_path)

    assert linkage._resolve("relative/path") == tmp_path / "relative/path"
    assert linkage._resolve(str(tmp_path / "absolute")) == tmp_path / "absolute"
