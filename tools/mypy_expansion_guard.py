#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — MyPy Expansion Guard
"""Fail when the strict MyPy surface shrinks or gains untyped escape hatches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 compatibility.
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = REPO_ROOT / "tools" / "mypy_expansion_baseline.json"
DEFAULT_PYPROJECT = REPO_ROOT / "pyproject.toml"
DEFAULT_SUMMARY = REPO_ROOT / "artifacts" / "mypy_expansion_guard_summary.json"


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _load_mypy_config(pyproject_path: Path) -> dict[str, Any]:
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    tool = payload.get("tool")
    if not isinstance(tool, dict):
        raise ValueError("pyproject.toml missing [tool] table")
    mypy = tool.get("mypy")
    if not isinstance(mypy, dict):
        raise ValueError("pyproject.toml missing [tool.mypy] table")
    return mypy


def _string_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    out: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{label}[{idx}] must be a non-empty string")
        out.append(item)
    return out


def _module_tuple(value: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(value, str) and value:
        return (value,)
    return tuple(_string_list(value, label=label))


def _allowed_missing_import_modules(baseline: dict[str, Any]) -> set[tuple[str, ...]]:
    rows = baseline.get("allowed_missing_import_modules")
    if not isinstance(rows, list):
        raise ValueError("baseline missing allowed_missing_import_modules list")
    allowed: set[tuple[str, ...]] = set()
    for idx, row in enumerate(rows):
        allowed.add(_module_tuple(row, label=f"allowed_missing_import_modules[{idx}]"))
    return allowed


def _typed_files(mypy: dict[str, Any]) -> list[str]:
    files = _string_list(mypy.get("files"), label="tool.mypy.files")
    duplicates = sorted({path for path in files if files.count(path) > 1})
    if duplicates:
        raise ValueError("Duplicate MyPy typed files: " + ", ".join(duplicates))
    return files


def _ignore_missing_import_modules(mypy: dict[str, Any]) -> set[tuple[str, ...]]:
    rows = mypy.get("overrides", [])
    if not isinstance(rows, list):
        raise ValueError("tool.mypy.overrides must be a list when present")
    ignored: set[tuple[str, ...]] = set()
    for idx, override in enumerate(rows):
        if not isinstance(override, dict):
            raise ValueError(f"tool.mypy.overrides[{idx}] must be an object")
        if override.get("ignore_missing_imports") is True:
            ignored.add(_module_tuple(override.get("module"), label=f"overrides[{idx}].module"))
    return ignored


def evaluate(*, pyproject_path: Path, baseline_path: Path) -> dict[str, Any]:
    """Evaluate current MyPy policy against the pinned expansion baseline."""
    baseline = _load_json_object(baseline_path)
    mypy = _load_mypy_config(pyproject_path)
    files = _typed_files(mypy)
    file_set = set(files)
    required_files = set(_string_list(baseline.get("required_typed_files"), label="required"))
    min_count = int(baseline.get("min_typed_file_count", len(required_files)))

    allowed_ignored = _allowed_missing_import_modules(baseline)
    current_ignored = _ignore_missing_import_modules(mypy)

    failures: list[str] = []
    if mypy.get("strict") is not True:
        failures.append("tool.mypy.strict must remain true")
    if mypy.get("ignore_missing_imports") is not False:
        failures.append("tool.mypy.ignore_missing_imports must remain false")
    if mypy.get("exclude") not in (None, "", []):
        failures.append("tool.mypy.exclude must not hide files from the strict cohort")
    if len(files) < min_count:
        failures.append(f"typed file count shrank: {len(files)} < {min_count}")

    missing_required = sorted(required_files - file_set)
    if missing_required:
        failures.append("required typed files removed: " + ", ".join(missing_required))

    missing_paths = sorted(path for path in files if not (REPO_ROOT / path).exists())
    if missing_paths:
        failures.append("configured typed files do not exist: " + ", ".join(missing_paths))

    new_ignored = sorted(current_ignored - allowed_ignored)
    if new_ignored:
        formatted = [",".join(row) for row in new_ignored]
        failures.append("new ignore_missing_imports overrides: " + "; ".join(formatted))

    return {
        "baseline": _display_path(baseline_path),
        "pyproject": _display_path(pyproject_path),
        "strict_enabled": mypy.get("strict") is True,
        "global_ignore_missing_imports_disabled": mypy.get("ignore_missing_imports") is False,
        "typed_file_count": len(files),
        "min_typed_file_count": min_count,
        "missing_required_typed_files": missing_required,
        "missing_configured_typed_files": missing_paths,
        "allowed_missing_import_overrides": [list(row) for row in sorted(allowed_ignored)],
        "current_missing_import_overrides": [list(row) for row in sorted(current_ignored)],
        "new_missing_import_overrides": [list(row) for row in new_ignored],
        "failures": failures,
        "overall_pass": not failures,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the MyPy expansion guard and write a machine-readable summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", default=str(DEFAULT_PYPROJECT))
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args(argv)

    summary = evaluate(
        pyproject_path=_resolve(args.pyproject),
        baseline_path=_resolve(args.baseline),
    )
    summary_path = _resolve(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "MyPy expansion guard: "
        f"typed_files={summary['typed_file_count']} "
        f"min={summary['min_typed_file_count']} "
        f"failures={len(summary['failures'])}"
    )
    if not bool(summary["overall_pass"]):
        for failure in summary["failures"]:
            print(f" - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
