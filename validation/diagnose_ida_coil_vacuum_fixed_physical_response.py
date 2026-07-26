#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Fixed-Physical Coil-Vacuum Diagnostic
"""Execute the fixed-physical CVGC2 forcing and inverse-response diagnostic."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib.machinery
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
_CHILD_FLAG = "--cvgc2-sanitized-child"
_PASSTHROUGH_ENVIRONMENT = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "JAX_ENABLE_X64",
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
        "LANG",
        "LC_ALL",
        "MKL_NUM_THREADS",
        "NVIDIA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "TF_CPP_MIN_LOG_LEVEL",
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    }
)
_FIXED_CHILD_ENVIRONMENT = {
    "HOME": "/nonexistent",
    "PATH": "/usr/bin:/bin",
}
_ALLOWED_CHILD_ENVIRONMENT = _PASSTHROUGH_ENVIRONMENT | _FIXED_CHILD_ENVIRONMENT.keys()
_SOURCE_ONLY_BOOTSTRAP = False
_SOURCE_ONLY_CACHE_PREFIX: Path | None = None
_SANITIZED_INTERPRETER = False


def _sanitized_interpreter_flags_are_valid(
    flags: Any,
    *,
    version: tuple[int, int],
) -> bool:
    """Return whether isolated child flags meet the versioned safety contract."""
    safe_path = getattr(flags, "safe_path", None)
    safe_path_is_valid = safe_path is True or (version == (3, 10) and safe_path is None)
    return bool(
        flags.isolated == 1
        and flags.no_site == 1
        and flags.ignore_environment == 1
        and safe_path_is_valid
    )


if __name__ == "__main__":
    if _CHILD_FLAG not in sys.argv:
        allowed_environment = {
            name: value for name, value in os.environ.items() if name in _PASSTHROUGH_ENVIRONMENT
        }
        allowed_environment.update(_FIXED_CHILD_ENVIRONMENT)
        os.execve(
            sys.executable,
            [
                sys.executable,
                "-I",
                "-S",
                str(Path(__file__).resolve()),
                _CHILD_FLAG,
                *sys.argv[1:],
            ],
            allowed_environment,
        )
    unexpected_environment = set(os.environ) - _ALLOWED_CHILD_ENVIRONMENT
    if unexpected_environment or any(
        os.environ.get(name) != value for name, value in _FIXED_CHILD_ENVIRONMENT.items()
    ):
        raise RuntimeError("CVGC2 sanitized child environment is invalid")
    if not _sanitized_interpreter_flags_are_valid(
        sys.flags,
        version=sys.version_info[:2],
    ):
        raise RuntimeError("CVGC2 sanitized child interpreter flags are invalid")
    sys.argv.remove(_CHILD_FLAG)
    if any(name.startswith(("validation", "scpn_fusion")) for name in sys.modules):
        raise RuntimeError("CVGC2 project modules were preloaded before the sanitized boundary")
    expected_meta_path: tuple[object, ...] = (
        importlib.machinery.BuiltinImporter,
        importlib.machinery.FrozenImporter,
        importlib.machinery.PathFinder,
    )
    if len(sys.meta_path) != len(expected_meta_path) or any(
        actual is not expected
        for actual, expected in zip(sys.meta_path, expected_meta_path, strict=True)
    ):
        raise RuntimeError("CVGC2 sanitized child has an unexpected import finder")
    venv_site_packages = (
        Path(sys.executable).parent.parent
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if not venv_site_packages.is_dir():
        raise RuntimeError("CVGC2 exact virtual-environment site-packages path is missing")
    _SOURCE_ONLY_BOOTSTRAP = __spec__ is None
    _SANITIZED_INTERPRETER = True
    _SOURCE_ONLY_CACHE_PREFIX = Path(tempfile.mkdtemp(prefix="scpn-cvgc2-pycache-"))
    atexit.register(shutil.rmtree, _SOURCE_ONLY_CACHE_PREFIX, ignore_errors=True)
    sys.dont_write_bytecode = True
    sys.pycache_prefix = str(_SOURCE_ONLY_CACHE_PREFIX)
    sys.path[:] = [
        *sys.path,
        str(_BOOTSTRAP_ROOT),
        str(_BOOTSTRAP_ROOT / "src"),
        str(venv_site_packages),
    ]

from validation import diagnose_ida_coil_vacuum_grid_convergence as grid_diagnostic
from validation import ida_coil_vacuum_fixed_physical_contract as contract
from validation.ida_coil_vacuum_fixed_physical_response import (
    build_fixed_physical_convergence,
    build_fixed_physical_grid,
)

ROOT = grid_diagnostic.ROOT
REPORT_PATH = ROOT / contract.REPORT_PATH
MARKDOWN_PATH = ROOT / contract.MARKDOWN_PATH
_IMPORTED_SOURCE_SHA256 = {
    name: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    for name, path in contract.SOURCE_PATHS.items()
}


def _verify_project_source_loaders() -> bool:
    """Verify every imported CVGC2 module came from canonical source files."""
    if _SOURCE_ONLY_CACHE_PREFIX is None:
        return False
    for name, path in contract.SOURCE_PATHS.items():
        if name == "fixed_physical_diagnostic":
            continue
        module_name = f"validation.{Path(path).stem}"
        module = sys.modules.get(module_name)
        if (
            module is None
            or not isinstance(module.__loader__, importlib.machinery.SourceFileLoader)
            or module.__file__ is None
            or Path(module.__file__).resolve() != (ROOT / path).resolve()
        ):
            return False
        cached = getattr(module, "__cached__", None)
        if cached is not None and not Path(cached).is_relative_to(_SOURCE_ONLY_CACHE_PREFIX):
            return False
    return True


def _require_source_only_bootstrap() -> None:
    """Require direct-source execution with an empty isolated bytecode cache."""
    if (
        not _SOURCE_ONLY_BOOTSTRAP
        or not _SANITIZED_INTERPRETER
        or not _verify_project_source_loaders()
        or _SOURCE_ONLY_CACHE_PREFIX is None
        or sys.dont_write_bytecode is not True
        or sys.pycache_prefix != str(_SOURCE_ONLY_CACHE_PREFIX)
        or any(_SOURCE_ONLY_CACHE_PREFIX.iterdir())
    ):
        raise ValueError("CVGC2 requires the direct source-only diagnostic launcher")


def _source_artifacts() -> dict[str, dict[str, Any]]:
    """Bind every executed CVGC2 source file and the clean source commit."""
    return contract.execution_source_artifacts(ROOT)


def _verify_cvgc1_arrays(
    execution: grid_diagnostic.GridLadderExecution,
    upstream: dict[str, Any],
) -> None:
    """Require CVGC2 to reuse every upstream total forcing and response array."""
    if execution.anchor != upstream["anchor"]:
        raise ValueError("CVGC2 exact 129 anchor drifted from CVGC1")
    if execution.coil_manifest != upstream["coil_manifest"]:
        raise ValueError("CVGC2 coil manifest drifted from CVGC1")
    upstream_grids = {int(row["resolution"]): row for row in upstream["grids"]}
    for result in execution.results:
        expected = upstream_grids[result.resolution]
        if (
            result.report["forcing_partition"]["total"]["field_sha256"]
            != expected["forcing_partition"]["total"]["field_sha256"]
        ):
            raise ValueError(f"CVGC2 {result.resolution} total forcing drifted from CVGC1")
        if (
            result.report["response_partition"]["total"]["field_sha256"]
            != expected["response_partition"]["total"]["field_sha256"]
        ):
            raise ValueError(f"CVGC2 {result.resolution} total response drifted from CVGC1")
        radius = float(result.report["masks"]["fixed_physical_radius_m"])
        if radius != contract.FIXED_PHYSICAL_RADIUS_M:
            raise ValueError("CVGC2 fixed physical radius drifted from the frozen value")


def run_diagnostic(*, generated_at: str) -> dict[str, Any]:
    """Execute the exact CVGC1 ladder and fixed-physical inverse partitions."""
    _require_source_only_bootstrap()
    source_artifacts, source_snapshot = contract.execution_source_snapshot(ROOT)
    expected_imports = {
        name: str(source_artifacts[name]["sha256"]) for name in contract.SOURCE_PATHS
    }
    if expected_imports != _IMPORTED_SOURCE_SHA256:
        raise ValueError("CVGC2 imported source bytes do not match the clean execution commit")
    upstream = contract.load_upstream_report(ROOT)
    execution = grid_diagnostic.execute_grid_ladder()
    _verify_cvgc1_arrays(execution, upstream)
    fixed_results = [build_fixed_physical_grid(row) for row in execution.results]
    final_artifacts, final_snapshot = contract.execution_source_snapshot(ROOT)
    if final_artifacts != source_artifacts or final_snapshot != source_snapshot:
        raise ValueError("CVGC2 source provenance drifted during numerical execution")
    return contract.build_report(
        generated_at=generated_at,
        environment=execution.environment,
        execution_binding=contract.build_execution_binding(
            anchor=execution.anchor,
            coil_manifest=execution.coil_manifest,
            source_artifacts=source_artifacts,
        ),
        source_artifacts=source_artifacts,
        upstream=contract.upstream_binding(ROOT, upstream),
        grids=[row.report for row in fixed_results],
        convergence=build_fixed_physical_convergence(fixed_results),
    )


def write_report(
    report: dict[str, Any],
    *,
    output: Path = REPORT_PATH,
    markdown_output: Path = MARKDOWN_PATH,
) -> None:
    """Write validated JSON and Markdown evidence to explicit repository paths."""
    contract.validate_report(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(contract.render_markdown(report), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one written evidence file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Run, write, and optionally revalidate the CVGC2 report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--markdown-output", type=Path, default=MARKDOWN_PATH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    report = run_diagnostic(generated_at=args.generated_at)
    write_report(report, output=args.output, markdown_output=args.markdown_output)
    if args.check:
        loaded = json.loads(args.output.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("written CVGC2 report must remain a JSON object")
        contract.validate_report(loaded)
        if loaded["payload_sha256"] != report["payload_sha256"]:
            raise ValueError("written CVGC2 payload drifted after serialisation")
        if not args.markdown_output.read_text(encoding="utf-8").endswith("\n"):
            raise ValueError("written CVGC2 Markdown must end with a newline")
    sys.stdout.write(
        json.dumps(
            {
                "json_sha256": _file_sha256(args.output),
                "markdown_sha256": _file_sha256(args.markdown_output),
                "payload_sha256": report["payload_sha256"],
                "routing": report["routing"],
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
