#!/usr/bin/env python
"""Validate docs/RELEASE_ACCEPTANCE_CHECKLIST.md readiness."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKLIST = REPO_ROOT / "docs" / "RELEASE_ACCEPTANCE_CHECKLIST.md"

REQUIRED_ITEMS = (
    "Release preflight (`python tools/run_python_preflight.py --gate release`)",
    "Research preflight (`python tools/run_python_preflight.py --gate research`)",
    "Claims audit and claims evidence map are up to date",
    "Underdeveloped register regenerated in current branch",
    "Version metadata and release docs are consistent",
    "Changelog contains the release section and date",
    "CI workflow on `main` is green for the release commit",
    "Tag/release notes reviewed and approved",
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _parse_release_version(text: str) -> str | None:
    match = re.search(r"(?mi)^Release Version:\s*`?([^`\r\n]+)`?\s*$", text)
    return match.group(1).strip() if match else None


def _parse_checklist_state(text: str) -> str | None:
    match = re.search(r"(?mi)^Checklist State:\s*`?([^`\r\n]+)`?\s*$", text)
    return match.group(1).strip() if match else None


def _parse_check_items(text: str) -> dict[str, bool]:
    items: dict[str, bool] = {}
    for match in re.finditer(r"(?mi)^\s*-\s*\[([ xX])\]\s*(.+?)\s*$", text):
        checked = match.group(1).lower() == "x"
        label = match.group(2).strip()
        items[_normalize(label)] = checked
    return items


def check_release_acceptance(
    checklist_path: Path,
    *,
    expected_version: str | None,
    require_ready_state: bool,
) -> list[str]:
    errors: list[str] = []
    if not checklist_path.exists():
        return [f"Checklist file missing: {checklist_path}"]

    text = checklist_path.read_text(encoding="utf-8")
    release_version = _parse_release_version(text)
    if not release_version:
        errors.append("Missing 'Release Version:' line.")
    elif expected_version and release_version != expected_version:
        errors.append(
            f"Checklist release version mismatch: expected {expected_version}, "
            f"found {release_version}."
        )

    state = _parse_checklist_state(text)
    if require_ready_state:
        if not state:
            errors.append("Missing 'Checklist State:' line.")
        elif state.lower() != "ready":
            errors.append(f"Checklist state must be 'ready' (found: {state}).")

    parsed_items = _parse_check_items(text)
    for required in REQUIRED_ITEMS:
        key = _normalize(required)
        if key not in parsed_items:
            errors.append(f"Missing checklist item: {required}")
            continue
        if not parsed_items[key]:
            errors.append(f"Checklist item is not checked: {required}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checklist",
        default=str(DEFAULT_CHECKLIST),
        help="Path to release checklist markdown.",
    )
    parser.add_argument(
        "--expected-version",
        default=None,
        help="Expected release version (for tag workflows), e.g. v3.5.0.",
    )
    parser.add_argument(
        "--allow-not-ready",
        action="store_true",
        help="Allow checklist state other than 'ready'.",
    )
    args = parser.parse_args(argv)

    checklist_path = Path(args.checklist)
    if not checklist_path.is_absolute():
        checklist_path = REPO_ROOT / checklist_path

    errors = check_release_acceptance(
        checklist_path,
        expected_version=args.expected_version,
        require_ready_state=not args.allow_not_ready,
    )
    if errors:
        print(f"Release acceptance checklist FAILED ({len(errors)} issue(s))")
        for error in errors:
            print(f" - {error}")
        return 1

    print(
        "Release acceptance checklist passed: "
        f"{checklist_path.relative_to(REPO_ROOT).as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
