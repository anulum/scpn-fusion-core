#!/usr/bin/env python3
# ----------------------------------------------------------------------
# SCPN Fusion Core -- Deprecated Mode Exclusion Benchmark
# ----------------------------------------------------------------------
"""Benchmark to ensure deprecated FNO lanes never leak into default runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from scpn_fusion import cli as scpn_cli


DEPRECATED_MODULE_TOKENS = (
    "fno_turbulence_suppressor",
    "fno_training",
)
FNO_TEST_FILES = (
    ROOT / "tests" / "test_fno_training.py",
    ROOT / "tests" / "test_fno_multi_regime.py",
)


def _contains_deprecated_module(module_name: str) -> bool:
    mod = module_name.lower()
    return any(token in mod for token in DEPRECATED_MODULE_TOKENS)


def _file_has_experimental_marker(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return "pytestmark = pytest.mark.experimental" in text


def run_benchmark() -> dict[str, Any]:
    default_modes = scpn_cli._execution_plan(
        "all",
        include_surrogate=False,
        include_experimental=False,
    )
    default_modules = [scpn_cli.MODE_SPECS[m].module for m in default_modes]
    default_exclusion_pass = bool(
        all(not _contains_deprecated_module(module) for module in default_modules)
    )

    public_modules = [
        spec.module
        for spec in scpn_cli.MODE_SPECS.values()
        if spec.maturity == "public"
    ]
    public_exclusion_pass = bool(
        all(not _contains_deprecated_module(module) for module in public_modules)
    )

    opt_in_modes = [
        mode
        for mode, spec in scpn_cli.MODE_SPECS.items()
        if _contains_deprecated_module(spec.module)
    ]
    opt_in_maturity_pass = bool(
        all(scpn_cli.MODE_SPECS[mode].maturity != "public" for mode in opt_in_modes)
    )

    research_marker_results = {
        path.as_posix().split("/", maxsplit=0)[-1] if False else str(path.relative_to(ROOT).as_posix()): _file_has_experimental_marker(path)
        for path in FNO_TEST_FILES
    }
    research_marker_pass = bool(all(research_marker_results.values()))

    passes = bool(
        default_exclusion_pass
        and public_exclusion_pass
        and opt_in_maturity_pass
        and research_marker_pass
    )

    return {
        "deprecated_mode_exclusion_benchmark": {
            "default_modes": default_modes,
            "default_modules": default_modules,
            "default_exclusion_pass": default_exclusion_pass,
            "public_exclusion_pass": public_exclusion_pass,
            "opt_in_maturity_pass": opt_in_maturity_pass,
            "research_marker_pass": research_marker_pass,
            "passes_thresholds": passes,
            "deprecated_module_tokens": list(DEPRECATED_MODULE_TOKENS),
            "opt_in_modes": opt_in_modes,
            "research_marker_files": research_marker_results,
        }
    }


def render_markdown(report: dict[str, Any]) -> str:
    g = report["deprecated_mode_exclusion_benchmark"]
    lines = [
        "# Deprecated Mode Exclusion Benchmark",
        "",
        f"- Default exclusion pass: `{'YES' if g['default_exclusion_pass'] else 'NO'}`",
        f"- Public exclusion pass: `{'YES' if g['public_exclusion_pass'] else 'NO'}`",
        f"- Opt-in maturity pass: `{'YES' if g['opt_in_maturity_pass'] else 'NO'}`",
        f"- Research marker pass: `{'YES' if g['research_marker_pass'] else 'NO'}`",
        f"- Overall pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
        "| Default mode | Module |",
        "|--------------|--------|",
    ]
    for mode, module in zip(g["default_modes"], g["default_modules"], strict=False):
        lines.append(f"| {mode} | {module} |")

    lines.extend(
        [
            "",
            "| Research file | Experimental marker |",
            "|---------------|---------------------|",
        ]
    )
    for rel_path, has_marker in g["research_marker_files"].items():
        lines.append(f"| {rel_path} | {'YES' if has_marker else 'NO'} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "deprecated_mode_exclusion_benchmark.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "deprecated_mode_exclusion_benchmark.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = run_benchmark()
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["deprecated_mode_exclusion_benchmark"]
    print("Deprecated mode exclusion benchmark complete.")
    print(
        "default_pass={d}, public_pass={p}, marker_pass={m}, pass={all_pass}".format(
            d=g["default_exclusion_pass"],
            p=g["public_exclusion_pass"],
            m=g["research_marker_pass"],
            all_pass=g["passes_thresholds"],
        )
    )
    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
