# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Dependabot Python Runtime Contract Tests
"""Regression contract for the Python used by the requirements updater."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TORNADO_SECURITY_FLOOR = (6, 5, 8)


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(component) for component in value.split("."))


def test_dependabot_runtime_matches_the_hash_lock_generation_lane() -> None:
    selected = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    components = selected.split(".")

    assert len(components) == 3
    assert all(component.isdigit() for component in components)
    assert components[:2] == ["3", "12"]
    assert (ROOT / "requirements/ci-py312.txt").is_file()
    dependabot = (ROOT / ".github/dependabot.yml").read_text(encoding="utf-8")
    assert 'directory: "/requirements"' in dependabot


def test_tornado_security_floor_is_locked_in_every_requirement_profile() -> None:
    requirements = ROOT / "requirements"
    source_matches: list[tuple[Path, str]] = []
    lock_matches: list[tuple[Path, str]] = []
    for path in requirements.glob("*.in"):
        match = re.search(r"(?m)^tornado>=([0-9.]+)$", path.read_text(encoding="utf-8"))
        if match is not None:
            source_matches.append((path, match.group(1)))
    for path in requirements.glob("*.txt"):
        match = re.search(r"(?m)^tornado==([0-9.]+) \\$", path.read_text(encoding="utf-8"))
        if match is not None:
            lock_matches.append((path, match.group(1)))

    assert {path.name for path, _version in source_matches} == {
        "build.in",
        "ci.in",
        "docs.in",
        "full.in",
    }
    assert len(lock_matches) == 9
    assert all(
        _version_tuple(version) >= TORNADO_SECURITY_FLOOR for _path, version in source_matches
    )
    assert all(_version_tuple(version) >= TORNADO_SECURITY_FLOOR for _path, version in lock_matches)
