#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Guard that deprecated FNO lanes never become default release/runtime paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = REPO_ROOT / "artifacts" / "deprecated_default_lane_guard_summary.json"
RELEASE_SURFACES = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "RESULTS.md",
    REPO_ROOT / "VALIDATION.md",
    REPO_ROOT / "docs" / "V3_9_3_RELEASE_CHECKLIST.md",
)
DEPRECATED_FNO_TOKENS = (
    "fno_turbulence_suppressor",
    "fno_training",
    "legacy_fno",
)


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _contains_deprecated_fno_token(value: str) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in DEPRECATED_FNO_TOKENS)


def _extract_release_commands(text: str) -> list[str]:
    commands: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "scpn-fusion" not in line:
            continue
        commands.append(line)
    return commands


def evaluate(
    *,
    mode_specs: dict[str, dict[str, str]],
    default_modes: list[str],
    release_commands: list[str],
) -> dict[str, Any]:
    default_modules = [str(mode_specs.get(mode, {}).get("module", "")) for mode in default_modes]
    default_has_deprecated = any(
        _contains_deprecated_fno_token(module) for module in default_modules
    )

    fno_modes: dict[str, dict[str, str]] = {
        mode: spec
        for mode, spec in mode_specs.items()
        if "fno" in mode.lower() or _contains_deprecated_fno_token(str(spec.get("module", "")))
    }
    fno_public_modes = sorted(
        mode
        for mode, spec in fno_modes.items()
        if str(spec.get("maturity", "")).lower() == "public"
    )

    docs_violations: list[str] = []
    for command in release_commands:
        if "fno-training" not in command:
            continue
        has_surrogate_unlock = "--surrogate" in command or "SCPN_SURROGATE=1" in command
        if not has_surrogate_unlock:
            docs_violations.append(command)

    return {
        "default_modes": default_modes,
        "default_modules": default_modules,
        "default_contains_deprecated_fno": bool(default_has_deprecated),
        "fno_modes": fno_modes,
        "fno_public_modes": fno_public_modes,
        "release_command_count": len(release_commands),
        "docs_violations": docs_violations,
        "overall_pass": bool(
            (not default_has_deprecated) and (not fno_public_modes) and (not docs_violations)
        ),
    }


def _load_runtime_state() -> tuple[dict[str, dict[str, str]], list[str]]:
    src_root = REPO_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from scpn_fusion import cli as scpn_cli

    mode_specs = {
        mode: {
            "module": str(spec.module),
            "maturity": str(spec.maturity),
            "description": str(spec.description),
        }
        for mode, spec in scpn_cli.MODE_SPECS.items()
    }
    default_modes = scpn_cli._execution_plan(
        "all",
        include_surrogate=False,
        include_experimental=False,
        experimental_ack=None,
    )
    return mode_specs, list(default_modes)


def _load_release_commands() -> list[str]:
    commands: list[str] = []
    for path in RELEASE_SURFACES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        commands.extend(_extract_release_commands(text))
    return commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args(argv)

    mode_specs, default_modes = _load_runtime_state()
    summary = evaluate(
        mode_specs=mode_specs,
        default_modes=default_modes,
        release_commands=_load_release_commands(),
    )

    summary_path = _resolve(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Deprecated-default-lane guard summary: "
        f"default_has_deprecated={summary['default_contains_deprecated_fno']} "
        f"fno_public_modes={len(summary['fno_public_modes'])} "
        f"docs_violations={len(summary['docs_violations'])}"
    )
    if not bool(summary["overall_pass"]):
        print("Deprecated default lane guard failed.")
        return 1
    print("Deprecated default lane guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
