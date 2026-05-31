#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Inventory public free-boundary machine metadata without accepting parity.

The cached FreeGSNKE machine configuration files are upstream public pickle
artifacts. This tool only loads known files from the repository's gitignored
public-source cache through a restricted unpickler and records geometry summary,
provenance, and checksums. It does not promote machine metadata to strict
free-boundary reconstruction parity; same-case native solver outputs and
FreeGS/FreeGSNKE comparisons are still required.
"""

from __future__ import annotations

import argparse
import hashlib
import json

# Restricted unpickler for fixed cached public metadata files.
import pickle  # nosec B403
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "data" / "external" / "full_fidelity_public_sources"
FREEGS_REPO = CACHE_ROOT / "repos" / "freegs"
FREEGSNKE_REPO = CACHE_ROOT / "repos" / "freegsnke"
MACHINE_CONFIG_ROOT = FREEGSNKE_REPO / "machine_configs"
ARTIFACT_DIR = ROOT / "validation" / "reference_data" / "full_fidelity_public_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "free_boundary_public_machine_metadata_inventory.json"
METADATA_PATH = ARTIFACT_DIR / "free_boundary_public_machine_metadata_inventory.metadata.json"
REPORT_DIR = ROOT / "validation" / "reports"
JSON_REPORT = REPORT_DIR / "free_boundary_public_machine_metadata_inventory.json"
MD_REPORT = REPORT_DIR / "free_boundary_public_machine_metadata_inventory.md"

ROLE_SUFFIXES = (
    "active_coils",
    "passive_coils",
    "magnetic_probes",
    "limiter",
    "wall",
)
_ALLOWED_NUMPY_GLOBALS = {
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
}


class RestrictedMachineConfigUnpickler(pickle.Unpickler):
    """Unpickler restricted to builtins and NumPy array/scalar constructors."""

    def find_class(self, module: str, name: str) -> object:
        """Resolve only the NumPy globals needed by upstream machine configs."""
        if (module, name) in _ALLOWED_NUMPY_GLOBALS:
            if name == "dtype":
                return np.dtype
            if name == "ndarray":
                return np.ndarray
            if name == "_reconstruct":
                multiarray = cast(Any, np.core.multiarray)
                return multiarray._reconstruct
            if name == "scalar":
                multiarray = cast(Any, np.core.multiarray)
                return multiarray.scalar
        raise pickle.UnpicklingError(f"disallowed pickle global: {module}.{name}")


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str | None:
    git_dir = repo / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if re.fullmatch(r"[0-9a-f]{40}", head):
        return head
    prefix = "ref: "
    if not head.startswith(prefix):
        return None
    ref = head.removeprefix(prefix)
    ref_path = git_dir / ref
    if ref_path.exists():
        candidate = ref_path.read_text(encoding="utf-8").strip()
        return candidate if re.fullmatch(r"[0-9a-f]{40}", candidate) else None
    packed_refs = git_dir / "packed-refs"
    if not packed_refs.exists():
        return None
    for line in packed_refs.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        try:
            candidate, packed_ref = line.split(" ", 1)
        except ValueError:
            continue
        if packed_ref == ref and re.fullmatch(r"[0-9a-f]{40}", candidate):
            return candidate
    return None


def _require_cached_path(path: Path) -> None:
    resolved = path.resolve()
    base = MACHINE_CONFIG_ROOT.resolve()
    if base not in resolved.parents:
        raise ValueError(f"machine config path is outside cache root: {path}")


def _load_machine_config(path: Path) -> Any:
    _require_cached_path(path)
    with path.open("rb") as handle:
        # Fixed cache-root path with a whitelist-only unpickler.
        return RestrictedMachineConfigUnpickler(handle).load()  # nosec B301


def _role_from_path(path: Path) -> str:
    stem = path.stem
    for suffix in ROLE_SUFFIXES:
        if stem.endswith(suffix):
            return suffix
    return "unknown_machine_metadata"


def _numeric_flat(value: Any) -> list[float]:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return []
    finite = array[np.isfinite(array)]
    return [float(item) for item in finite]


def _bounds(values: list[float]) -> list[float] | None:
    if not values:
        return None
    return [float(min(values)), float(max(values))]


def _sample_names(obj: Any, *, limit: int = 8) -> list[str]:
    names: list[str] = []
    if isinstance(obj, dict):
        names.extend(str(key) for key in obj)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict) and "name" in item:
                names.append(str(item["name"]))
    deduped = list(dict.fromkeys(names))
    return deduped[:limit]


def _top_level_count(obj: Any) -> int:
    if isinstance(obj, (dict, list, tuple)):
        return len(obj)
    return 1


def _collect_geometry(obj: Any) -> tuple[int, int, list[float], list[float], list[str]]:
    leaf_count = 0
    point_count = 0
    r_values: list[float] = []
    z_values: list[float] = []
    names: list[str] = []

    def visit(node: Any) -> None:
        nonlocal leaf_count, point_count
        if isinstance(node, dict):
            if "R" in node and "Z" in node:
                r_leaf = _numeric_flat(node["R"])
                z_leaf = _numeric_flat(node["Z"])
                if r_leaf and z_leaf:
                    leaf_count += 1
                    point_count += min(len(r_leaf), len(z_leaf))
                    r_values.extend(r_leaf)
                    z_values.extend(z_leaf)
                    if "name" in node:
                        names.append(str(node["name"]))
            for value in node.values():
                visit(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                visit(value)

    visit(obj)
    return leaf_count, point_count, r_values, z_values, list(dict.fromkeys(names))[:8]


def _probe_counts(obj: Any) -> dict[str, int]:
    if not isinstance(obj, dict):
        return {}
    out: dict[str, int] = {}
    for key in ("flux_loops", "pickups"):
        value = obj.get(key)
        if isinstance(value, list):
            out[key] = len(value)
    return out


def _machine_record(path: Path, commit: str | None) -> dict[str, Any]:
    obj = _load_machine_config(path)
    leaf_count, point_count, r_values, z_values, leaf_names = _collect_geometry(obj)
    role = _role_from_path(path)
    top_names = _sample_names(obj)
    return {
        "commit": commit,
        "config_name": path.stem,
        "deserialisation_guard": "restricted_numpy_builtin_unpickler",
        "geometry_element_count": leaf_count,
        "machine": path.parent.name,
        "path": _rel(path),
        "point_count": point_count,
        "probe_counts": _probe_counts(obj),
        "r_bounds_m": _bounds(r_values),
        "role": role,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "top_level_count": _top_level_count(obj),
        "top_level_names_sample": top_names or leaf_names,
        "z_bounds_m": _bounds(z_values),
    }


def _freegs_example_records(commit: str | None) -> list[dict[str, Any]]:
    if not FREEGS_REPO.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(FREEGS_REPO.glob("[0-9][0-9]-*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        records.append(
            {
                "commit": commit,
                "contains_coil_contract": bool(re.search(r"\b(Coils?|coil)\b", text)),
                "contains_limiter_or_wall_contract": bool(
                    re.search(r"\b(limiter|wall)\b", text, flags=re.IGNORECASE)
                ),
                "contains_machine_contract": "Machine" in text,
                "path": _rel(path),
                "sha256": _sha256(path),
            }
        )
    return records


def _machine_config_paths() -> Iterable[Path]:
    if not MACHINE_CONFIG_ROOT.exists():
        return ()
    return sorted(MACHINE_CONFIG_ROOT.glob("*/*.pickle"))


def _write_markdown(report: dict[str, Any], artifact: dict[str, Any]) -> None:
    lines = [
        "# Free-Boundary Public Machine Metadata Inventory",
        "",
        "This report indexes cached public FreeGSNKE machine metadata and FreeGS example",
        "scripts for the strict free-boundary reconstruction lane. It is a provenance",
        "and geometry inventory only; it is not accepted full-fidelity parity evidence.",
        "",
        f"- Schema: `{report['schema']}`",
        f"- Status: `{report['status']}`",
        f"- Accepted full fidelity: `{report['accepted_full_fidelity_ready']}`",
        f"- Machine metadata ready: `{report['machine_metadata_ready']}`",
        f"- Machine config count: `{report['machine_config_count']}`",
        f"- FreeGS example count: `{report['freegs_example_count']}`",
        f"- Artifact: `{report['artifact_path']}`",
        f"- Metadata: `{report['metadata_path']}`",
        f"- SHA256: `{report['sha256']}`",
        "",
        "## Machine configuration summaries",
        "",
        "| Machine | Role | Top-level count | Geometry elements | Points | R bounds (m) | Z bounds (m) |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for record in artifact["machine_configs"]:
        r_bounds = record["r_bounds_m"] or []
        z_bounds = record["z_bounds_m"] or []
        r_text = "none" if not r_bounds else f"{r_bounds[0]:.6g} to {r_bounds[1]:.6g}"
        z_text = "none" if not z_bounds else f"{z_bounds[0]:.6g} to {z_bounds[1]:.6g}"
        lines.append(
            "| {machine} | `{role}` | {top} | {geom} | {points} | {r} | {z} |".format(
                machine=record["machine"],
                role=record["role"],
                top=record["top_level_count"],
                geom=record["geometry_element_count"],
                points=record["point_count"],
                r=r_text,
                z=z_text,
            )
        )
    lines.extend(["", "## Missing full-fidelity requirements", ""])
    for item in report["missing_full_fidelity_requirements"]:
        lines.append(f"- {item}")
    lines.append("")
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


def build_free_boundary_machine_metadata_inventory(*, write: bool = True) -> dict[str, Any]:
    """Build and optionally write the free-boundary metadata inventory."""
    freegs_commit = _git_commit(FREEGS_REPO) if FREEGS_REPO.exists() else None
    freegsnke_commit = _git_commit(FREEGSNKE_REPO) if FREEGSNKE_REPO.exists() else None
    machine_configs = [_machine_record(path, freegsnke_commit) for path in _machine_config_paths()]
    freegs_examples = _freegs_example_records(freegs_commit)
    artifact = {
        "schema": "free-boundary-public-machine-metadata-inventory.v1",
        "surface": "free_boundary_equilibrium",
        "accepted_full_fidelity": False,
        "freegs_examples": freegs_examples,
        "machine_configs": machine_configs,
    }
    metadata = {
        "accepted_full_fidelity": False,
        "artifact_id": "free_boundary_public_machine_metadata_inventory",
        "artifact_path": _rel(ARTIFACT_PATH),
        "artifact_role": "partial_public_machine_metadata_inventory",
        "available_observables": [
            "machine_active_coil_geometry_summary",
            "machine_passive_conductor_geometry_summary",
            "limiter_wall_contour_summary",
            "magnetic_probe_inventory_summary",
            "public_freegs_example_script_checksums",
        ],
        "machine_config_count": len(machine_configs),
        "metadata_schema": "full-fidelity-public-output-artifact-metadata.v1",
        "missing_required_observables": [
            "same-case public equilibrium with linked external coil currents",
            "native free-boundary coil/vacuum reconstruction output",
            "strict FreeGS or FreeGSNKE same-case psi(R,Z) comparison",
            "axis_X-point_boundary_containment_thresholds",
            "grid convergence and current-closure evidence",
        ],
        "redistribution_license": "FreeGS LGPL-3.0-or-later; FreeGSNKE LGPL-3.0-or-later",
        "reference_family": "FreeGS/FreeGSNKE",
        "sha256": "",
        "solver_output_comparison_ready": False,
        "solver_output_comparison_status": (
            "blocked_machine_metadata_without_same_case_free_boundary_reconstruction"
        ),
        "surface": "free_boundary_equilibrium",
    }
    if write:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        metadata["sha256"] = _sha256(ARTIFACT_PATH)
        METADATA_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    elif ARTIFACT_PATH.exists():
        metadata["sha256"] = _sha256(ARTIFACT_PATH)
    machine_metadata_ready = bool(machine_configs)
    status = (
        "blocked_machine_metadata_indexed_missing_same_case_free_boundary_reconstruction"
        if machine_metadata_ready
        else "blocked_missing_free_boundary_machine_metadata_cache"
    )
    report = {
        "schema": "free-boundary-public-machine-metadata-inventory-report.v1",
        "accepted_full_fidelity_ready": False,
        "artifact_path": _rel(ARTIFACT_PATH),
        "freegs_commit": freegs_commit,
        "freegs_example_count": len(freegs_examples),
        "freegsnke_commit": freegsnke_commit,
        "machine_config_count": len(machine_configs),
        "machine_metadata_ready": machine_metadata_ready,
        "machines": sorted({record["machine"] for record in machine_configs}),
        "metadata_path": _rel(METADATA_PATH),
        "missing_full_fidelity_requirements": metadata["missing_required_observables"],
        "reference_output_ready": False,
        "sha256": metadata["sha256"],
        "status": status,
    }
    if write:
        JSON_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        _write_markdown(report, artifact)
    return report


def main(argv: list[str] | None = None) -> int:
    """Build the public free-boundary metadata inventory report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Build the report in memory without writing files.",
    )
    args = parser.parse_args(argv)
    report = build_free_boundary_machine_metadata_inventory(write=not args.check)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
