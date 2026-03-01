#!/usr/bin/env python
"""Guard against source modules with no direct test linkage."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "src" / "scpn_fusion"
DEFAULT_TEST_ROOT = REPO_ROOT / "tests"
DEFAULT_ALLOWLIST = REPO_ROOT / "tools" / "untested_module_allowlist.json"


@dataclass(frozen=True)
class TestLinkageIndex:
    imports: set[str]
    test_module_stems: set[str]
    corpus: str


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def collect_source_modules(source_root: Path) -> list[Path]:
    modules: list[Path] = []
    for path in source_root.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        modules.append(path)
    return sorted(modules)


def _module_import_path(source_root: Path, module_path: Path) -> str:
    rel = module_path.relative_to(source_root).with_suffix("")
    return "scpn_fusion." + ".".join(rel.parts)


def _iter_test_files(test_root: Path) -> list[Path]:
    return sorted(test_root.rglob("test_*.py"))


def _build_test_corpus(test_root: Path) -> str:
    parts: list[str] = []
    for path in _iter_test_files(test_root):
        parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


def _collect_import_targets(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or node.module is None:
                continue
            imports.add(node.module)
            for alias in node.names:
                if alias.name == "*":
                    continue
                imports.add(f"{node.module}.{alias.name}")
    return imports


def _build_test_linkage_index(test_root: Path) -> TestLinkageIndex:
    imports: set[str] = set()
    stems: set[str] = set()
    corpus_parts: list[str] = []

    for path in _iter_test_files(test_root):
        corpus_parts.append(path.read_text(encoding="utf-8", errors="ignore"))
        imports.update(_collect_import_targets(path))
        stem = path.stem
        if stem.startswith("test_") and len(stem) > len("test_"):
            stems.add(stem[len("test_"):])

    return TestLinkageIndex(
        imports=imports,
        test_module_stems=stems,
        corpus="\n".join(corpus_parts),
    )


def _is_import_linked(import_path: str, imports: set[str]) -> bool:
    if import_path in imports:
        return True
    prefix = import_path + "."
    return any(candidate.startswith(prefix) for candidate in imports)


def collect_unlinked_modules(
    *,
    source_root: Path,
    test_root: Path,
) -> list[str]:
    linkage = _build_test_linkage_index(test_root)
    unlinked: list[str] = []
    for module_path in collect_source_modules(source_root):
        import_path = _module_import_path(source_root, module_path)
        stem = module_path.stem
        if _is_import_linked(import_path, linkage.imports):
            continue
        if stem in linkage.test_module_stems:
            continue

        # Backward-compatible textual heuristic for dynamic import patterns.
        if import_path in linkage.corpus:
            continue
        if f"test_{stem}" in linkage.corpus:
            continue
        unlinked.append(module_path.relative_to(REPO_ROOT).as_posix())
    return sorted(unlinked)


def load_allowlist(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Allowlist file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Allowlist must be a JSON object.")
    entries = payload.get("allowlisted_modules")
    if not isinstance(entries, list):
        raise ValueError("allowlisted_modules must be a list.")
    paths: set[str] = set()
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"allowlisted_modules[{idx}] must be an object.")
        path_value = entry.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"allowlisted_modules[{idx}].path must be a non-empty string.")
        paths.add(path_value)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help="Source root to scan for modules.",
    )
    parser.add_argument(
        "--test-root",
        default=str(DEFAULT_TEST_ROOT),
        help="Test root to scan for direct linkage.",
    )
    parser.add_argument(
        "--allowlist",
        default=str(DEFAULT_ALLOWLIST),
        help="Allowlist JSON for known-unlinked modules.",
    )
    parser.add_argument(
        "--allow-stale-allowlist",
        action="store_true",
        help="Allow allowlist entries that are no longer unlinked.",
    )
    args = parser.parse_args(argv)

    source_root = _resolve(args.source_root)
    test_root = _resolve(args.test_root)
    allowlist_path = _resolve(args.allowlist)

    unlinked = set(
        collect_unlinked_modules(
            source_root=source_root,
            test_root=test_root,
        )
    )
    allowlisted = load_allowlist(allowlist_path)

    unexpected = sorted(unlinked - allowlisted)
    stale = sorted(allowlisted - unlinked)

    print(f"Unlinked modules detected: {len(unlinked)}")
    print(f"Allowlisted modules: {len(allowlisted)}")
    print(f"Unexpected modules: {len(unexpected)}")
    print(f"Stale allowlist entries: {len(stale)}")

    if unexpected:
        print("Guard FAILED: new modules without direct test linkage:")
        for path in unexpected:
            print(f"- {path}")
        return 1

    if stale and not args.allow_stale_allowlist:
        print("Guard FAILED: stale allowlist entries should be removed:")
        for path in stale:
            print(f"- {path}")
        return 1

    print("Untested-module guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
