#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Audit public Python docstring coverage against a non-regression baseline."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = REPO_ROOT / "tools" / "docstring_coverage_baseline.json"
DEFAULT_ROOTS = ("src", "validation", "tools")
EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "docs",
    "tests",
}
KIND_SORT_ORDER = {"module": 0, "class": 1, "function": 2, "method": 3}


@dataclass(frozen=True, order=True)
class DocstringIssue:
    """Single missing-docstring finding for a public Python object."""

    path: str
    line: int
    kind: str
    qualname: str


@dataclass(frozen=True)
class DocstringSummary:
    """Aggregate docstring audit result."""

    total_issues: int
    by_kind: dict[str, int]
    files_with_issues: int


def _run_git_ls_files(root: Path) -> list[Path] | None:
    """Return tracked files from git, or ``None`` when git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "*.py"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [root / line for line in result.stdout.splitlines() if line]


def iter_python_files(root: Path, included_roots: Sequence[str]) -> list[Path]:
    """List production Python files covered by the docstring audit."""
    tracked = _run_git_ls_files(root)
    if tracked is None:
        tracked = sorted(root.rglob("*.py"))

    included_prefixes = tuple(Path(item).parts for item in included_roots)
    files: list[Path] = []
    for path in tracked:
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        if not any(rel.parts[: len(prefix)] == prefix for prefix in included_prefixes):
            continue
        files.append(path)
    return sorted(files)


def _has_docstring(node: ast.AST) -> bool:
    """Return ``True`` when an AST node owns a non-empty docstring."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
        return bool(ast.get_docstring(node))
    return False


def _is_public_name(name: str) -> bool:
    """Return whether ``name`` should be treated as public API surface."""
    return not name.startswith("_")


def _issue(path: Path, root: Path, node: ast.AST, kind: str, qualname: str) -> DocstringIssue:
    """Build a normalised issue record for ``node``."""
    return DocstringIssue(
        path=path.relative_to(root).as_posix(),
        line=int(getattr(node, "lineno", 1)),
        kind=kind,
        qualname=qualname,
    )


def collect_docstring_issues(root: Path, included_roots: Sequence[str]) -> list[DocstringIssue]:
    """Collect missing docstrings for modules and public top-level objects."""
    issues: list[DocstringIssue] = []
    for path in iter_python_files(root, included_roots):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_name = ".".join(path.with_suffix("").relative_to(root).parts)
        if not _has_docstring(tree):
            issues.append(
                DocstringIssue(path.relative_to(root).as_posix(), 1, "module", module_name)
            )

        for node in tree.body:
            if isinstance(node, ast.ClassDef) and _is_public_name(node.name):
                if not _has_docstring(node):
                    issues.append(_issue(path, root, node, "class", node.name))
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if _is_public_name(child.name) and not _has_docstring(child):
                            issues.append(
                                _issue(path, root, child, "method", f"{node.name}.{child.name}")
                            )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if _is_public_name(node.name) and not _has_docstring(node):
                    issues.append(_issue(path, root, node, "function", node.name))
    return sorted(
        issues,
        key=lambda issue: (
            issue.path,
            issue.line,
            KIND_SORT_ORDER.get(issue.kind, 99),
            issue.qualname,
        ),
    )


def summarise(issues: Iterable[DocstringIssue]) -> DocstringSummary:
    """Summarise docstring findings by kind and file count."""
    materialised = list(issues)
    by_kind: dict[str, int] = {}
    for issue in materialised:
        by_kind[issue.kind] = by_kind.get(issue.kind, 0) + 1
    return DocstringSummary(
        total_issues=len(materialised),
        by_kind=dict(sorted(by_kind.items())),
        files_with_issues=len({issue.path for issue in materialised}),
    )


def _load_baseline(path: Path) -> dict[str, object]:
    """Load a JSON baseline file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing docstring baseline: {path}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("total_issues"), int):
        raise SystemExit(f"Invalid docstring baseline: {path}")
    return data


def _write_baseline(path: Path, summary: DocstringSummary) -> None:
    """Write a deterministic JSON baseline file."""
    payload = {
        "schema_version": "docstring-coverage-baseline.v1",
        "scope": list(DEFAULT_ROOTS),
        **asdict(summary),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable summary.")
    parser.add_argument("--max-list", type=int, default=25, help="Maximum findings to print.")
    parser.add_argument("--roots", nargs="+", default=list(DEFAULT_ROOTS))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the docstring non-regression audit."""
    args = build_parser().parse_args(argv)
    issues = collect_docstring_issues(REPO_ROOT, tuple(args.roots))
    summary = summarise(issues)

    if args.write_baseline:
        _write_baseline(args.baseline, summary)

    baseline = _load_baseline(args.baseline)
    baseline_total = int(cast(int, baseline["total_issues"]))
    status = {
        **asdict(summary),
        "baseline_total_issues": baseline_total,
        "passes": summary.total_issues <= baseline_total,
    }
    if args.json:
        print(json.dumps(status, sort_keys=True))
    else:
        print(
            "Docstring coverage audit: "
            f"{summary.total_issues} issues, baseline {baseline_total}, "
            f"{summary.files_with_issues} files."
        )
        if issues:
            print("First findings:")
            for issue in issues[: max(0, int(args.max_list))]:
                print(f"  {issue.path}:{issue.line}: {issue.kind}: {issue.qualname}")

    if not status["passes"]:
        print(
            "Docstring coverage regression: "
            f"{summary.total_issues} issues exceeds baseline {baseline_total}.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
