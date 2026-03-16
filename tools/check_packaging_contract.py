#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Validate packaging contracts used by release/install workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - exercised on 3.9/3.10 CI lanes
    tomllib = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYPROJECT = REPO_ROOT / "pyproject.toml"
DEFAULT_SUMMARY = REPO_ROOT / "artifacts" / "packaging_contract_summary.json"

HEAVY_BASE_BLOCKLIST = {"streamlit", "jax", "jaxlib", "gymnasium"}
REQUIRED_OPTIONAL_EXTRAS = {"ui", "ml", "rl", "snn", "full-physics", "rust", "full"}


def _canonical_requirement_name(requirement: str) -> str:
    token = requirement.strip()
    for sep in (";", "[", "<", ">", "=", "!", "~", " "):
        if sep in token:
            token = token.split(sep, 1)[0]
    return token.strip().lower()


def _load_pyproject(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if tomllib is not None:
        data = tomllib.loads(text)
        if not isinstance(data, dict):
            raise ValueError("pyproject payload is not a table")
        return data

    # Minimal fallback parser for Python environments without TOML parser.
    dependencies: list[str] = []
    optional: dict[str, list[str]] = {}
    section = ""
    collecting: tuple[str, list[str]] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if collecting is not None:
            name, buffer = collecting
            buffer.append(stripped)
            if "]" in stripped:
                values = " ".join(buffer)
                items = [
                    part.strip().strip("\"'")
                    for part in values.split("[", 1)[1].split("]", 1)[0].split(",")
                    if part.strip()
                ]
                if section == "[project]":
                    dependencies.extend(items)
                elif section == "[project.optional-dependencies]":
                    optional[name] = items
                collecting = None
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped
            continue
        if section == "[project]" and stripped.startswith("dependencies"):
            collecting = ("dependencies", [stripped])
            continue
        if section == "[project.optional-dependencies]" and "=" in stripped:
            name = stripped.split("=", 1)[0].strip()
            collecting = (name, [stripped])
            continue

    return {
        "project": {
            "dependencies": dependencies,
            "optional-dependencies": optional,
        }
    }


def evaluate_contract(pyproject: dict[str, Any]) -> dict[str, Any]:
    project = pyproject.get("project", {})
    if not isinstance(project, dict):
        raise ValueError("[project] table missing from pyproject.toml")

    dependencies = project.get("dependencies", [])
    if not isinstance(dependencies, list):
        raise ValueError("[project].dependencies must be a list")
    dependency_names = sorted(
        {_canonical_requirement_name(str(item)) for item in dependencies if str(item).strip()}
    )
    blocked_in_base = sorted(set(dependency_names) & HEAVY_BASE_BLOCKLIST)

    optional = project.get("optional-dependencies", {})
    if not isinstance(optional, dict):
        raise ValueError("[project.optional-dependencies] must be a table")
    missing_required_extras = sorted(REQUIRED_OPTIONAL_EXTRAS - set(optional.keys()))

    full_extra = optional.get("full", [])
    if not isinstance(full_extra, list):
        raise ValueError("[project.optional-dependencies].full must be a list")
    full_names = {
        _canonical_requirement_name(str(item)) for item in full_extra if str(item).strip()
    }

    family_union: set[str] = set()
    for family in ("ui", "ml", "rl", "snn", "full-physics", "rust"):
        family_requirements = optional.get(family, [])
        if not isinstance(family_requirements, list):
            raise ValueError(f"[project.optional-dependencies].{family} must be a list")
        family_union.update(
            _canonical_requirement_name(str(item))
            for item in family_requirements
            if str(item).strip()
        )
    missing_from_full = sorted(family_union - full_names)

    return {
        "base_dependency_names": dependency_names,
        "blocked_in_base": blocked_in_base,
        "missing_required_extras": missing_required_extras,
        "full_extra_size": len(full_names),
        "expected_full_union_size": len(family_union),
        "missing_from_full_extra": missing_from_full,
        "overall_pass": not blocked_in_base
        and not missing_required_extras
        and not missing_from_full,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", default=str(DEFAULT_PYPROJECT))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args(argv)

    pyproject_path = Path(args.pyproject)
    if not pyproject_path.is_absolute():
        pyproject_path = (REPO_ROOT / pyproject_path).resolve()
    summary_path = Path(args.summary_json)
    if not summary_path.is_absolute():
        summary_path = (REPO_ROOT / summary_path).resolve()

    summary = evaluate_contract(_load_pyproject(pyproject_path))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Packaging contract: "
        f"blocked_in_base={len(summary['blocked_in_base'])} "
        f"missing_extras={len(summary['missing_required_extras'])} "
        f"missing_from_full={len(summary['missing_from_full_extra'])}"
    )
    if not bool(summary["overall_pass"]):
        print("Packaging contract check failed.")
        return 1
    print("Packaging contract check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
