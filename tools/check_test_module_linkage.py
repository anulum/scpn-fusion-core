#!/usr/bin/env python
"""Guard against source modules with no direct test linkage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "src" / "scpn_fusion"
DEFAULT_TEST_ROOT = REPO_ROOT / "tests"
DEFAULT_ALLOWLIST = REPO_ROOT / "tools" / "untested_module_allowlist.json"


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


def _build_test_corpus(test_root: Path) -> str:
    parts: list[str] = []
    for path in sorted(test_root.rglob("test_*.py")):
        parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


def collect_unlinked_modules(
    *,
    source_root: Path,
    test_root: Path,
) -> list[str]:
    corpus = _build_test_corpus(test_root)
    unlinked: list[str] = []
    for module_path in collect_source_modules(source_root):
        import_path = _module_import_path(source_root, module_path)
        stem = module_path.stem
        if import_path in corpus:
            continue
        if f"test_{stem}" in corpus:
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
