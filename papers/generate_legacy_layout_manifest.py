#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — legacy manuscript-layout retirement manifest.
"""Preserve exact Git-object custody for every retired manuscript path."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from typing import Any, Sequence


SOURCE_REVISION = "f960da9934167fcf95f1c3b8e29fa203f7361fab"
EXPECTED_RETIRED_PATHS = 58
LEGACY_PATHS = (
    "papers/arxiv_submission_b",
    "papers/figures",
    "papers/paper_a_equilibrium_solver.tex",
    "papers/paper_b_snn_controller.tex",
    "papers/scpn_fusion.bib",
)
PACKAGE_001 = "papers/submissions/001_hybrid_rust_python_grad_shafranov_equilibrium_solver"
PACKAGE_002 = "papers/submissions/002_neuromorphic_vertical_stability_control"
PACKAGE_003 = "papers/submissions/003_stochastic_petri_net_tokamak_control_conference_abstract"


def _git_bytes(repository: Path, arguments: Sequence[str]) -> bytes:
    """Run one fixed-argument Git query and return exact standard-output bytes."""
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout


def _tree_blobs(repository: Path) -> dict[str, str]:
    """Return every retired path and blob identity with one tree query."""
    payload = _git_bytes(
        repository,
        ["ls-tree", "-r", "-z", SOURCE_REVISION, "--", *LEGACY_PATHS],
    )
    blobs: dict[str, str] = {}
    for entry in payload.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", maxsplit=1)
        _, object_type, blob = metadata.decode("ascii").split()
        if object_type != "blob":
            raise ValueError(f"Unexpected legacy Git object type: {object_type}")
        blobs[raw_path.decode("utf-8")] = blob
    blobs = dict(sorted(blobs.items()))
    if len(blobs) != EXPECTED_RETIRED_PATHS:
        raise ValueError(
            f"Legacy inventory has {len(blobs)} paths; expected {EXPECTED_RETIRED_PATHS}."
        )
    return blobs


def _archive_files(repository: Path) -> dict[str, bytes]:
    """Read all retired bytes through one exact-revision Git archive."""
    payload = _git_bytes(
        repository,
        ["archive", "--format=tar", SOURCE_REVISION, "--", *LEGACY_PATHS],
    )
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"Could not read archived Git object: {member.name}")
            files[member.name] = handle.read()
    return files


def _successors(path: str) -> tuple[str, list[str]]:
    """Describe the explicit successor or historical-only disposition."""
    name = Path(path).name
    suffix = Path(path).suffix
    paper_001_figures = {"fig_inverse_reconstruction", "fig_sparc_equilibrium"}
    paper_002_figures = {
        "fig_compilation_pipeline",
        "fig_latency_comparison",
        "fig_lif_neuron",
        "fig_petri_net",
        "fig_radiation_tolerance",
        "fig_vertical_stability",
    }
    historical_only_figures = {
        "fig_gs_convergence",
        "fig_neural_surrogate",
        "fig_performance_scaling",
        "fig_validation_rmse",
    }

    if name == "paper_a_equilibrium_solver.tex":
        return "rewritten_as_evidence_bounded_manuscript", [f"{PACKAGE_001}/manuscript.tex"]
    if name == "paper_b_snn_controller.tex":
        return "rewritten_as_evidence_bounded_manuscript", [f"{PACKAGE_002}/manuscript.tex"]
    if name == "paper_b_snn_controller.bbl":
        return "generated_auxiliary_reproducible_from_package", []
    if name == "scpn_fusion.bib":
        destinations = [f"{PACKAGE_002}/references.bib"]
        if not path.startswith("papers/arxiv_submission_b/"):
            destinations.insert(0, f"{PACKAGE_001}/references.bib")
        return "split_into_package_specific_bibliographies", destinations
    if name == "style.py":
        return "split_into_package_specific_figure_styles", [
            f"{PACKAGE_001}/figures/style.py",
            f"{PACKAGE_002}/figures/style.py",
        ]
    if name == "generate_all_figures.py":
        return "replaced_by_package_specific_generators", [
            f"{PACKAGE_001}/figures/generate_figures.py",
            f"{PACKAGE_002}/figures/generate_figures.py",
        ]

    stem = Path(name).stem
    if stem in paper_001_figures:
        return "rewritten_and_regenerated_for_package_001", [
            f"{PACKAGE_001}/figures/{stem}{suffix}"
        ]
    if stem in paper_002_figures:
        return "rewritten_and_regenerated_for_package_002", [
            f"{PACKAGE_002}/figures/{stem}{suffix}"
        ]
    if stem in historical_only_figures:
        return "historical_only_not_used_by_current_manuscripts", []
    raise ValueError(f"No retirement disposition exists for {path}.")


def _build_manifest(repository: Path) -> dict[str, Any]:
    """Build the deterministic exact-object retirement manifest."""
    tree_blobs = _tree_blobs(repository)
    archive_files = _archive_files(repository)
    if set(archive_files) != set(tree_blobs):
        raise ValueError("Git tree and archive inventories differ.")
    files: dict[str, dict[str, Any]] = {}
    for path, blob in tree_blobs.items():
        object_spec = f"{SOURCE_REVISION}:{path}"
        payload = archive_files[path]
        disposition, successors = _successors(path)
        files[path] = {
            "git_blob_oid_sha1": blob,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_size": len(payload),
            "disposition": disposition,
            "successor_paths": successors,
            "retrieval_command": f"git show {object_spec}",
        }

    return {
        "SPDX-License-Identifier": "AGPL-3.0-or-later",
        "commercialLicense": "available",
        "conceptsCopyright": "1996-2026 Miroslav Sotek. All rights reserved.",
        "codeCopyright": "2020-2026 Miroslav Sotek. All rights reserved.",
        "orcid": "0009-0009-3560-0851",
        "contact": "www.anulum.li | protoscience@anulum.li",
        "projectDescription": "SCPN-FUSION-CORE legacy paper-layout custody manifest.",
        "schema_version": "1.0",
        "source_revision": SOURCE_REVISION,
        "retired_path_count": len(files),
        "claim_boundary": {
            "preserved": (
                "every retired byte remains addressable by source revision, Git blob, "
                "SHA-256 and retrieval command"
            ),
            "not_claimed": (
                "successor manuscripts, figures and bibliographies were corrected or "
                "regenerated; byte equivalence with legacy content is not claimed"
            ),
            "historical_only": (
                "unused legacy figures remain reproducible historical Git objects and are "
                "not evidence for the current manuscripts"
            ),
        },
        "current_packages": [PACKAGE_001, PACKAGE_002, PACKAGE_003],
        "files": files,
    }


def _render(manifest: dict[str, Any]) -> str:
    """Render stable UTF-8 JSON."""
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def _validate_current_layout(repository: Path, manifest: dict[str, Any]) -> None:
    """Require retired paths to be absent and every named successor to exist."""
    for path, record in manifest["files"].items():
        if (repository / path).exists():
            raise ValueError(f"Retired path still exists in the working tree: {path}")
        for successor in record["successor_paths"]:
            if not (repository / successor).is_file():
                raise FileNotFoundError(f"Migration successor is missing: {successor}")
    for package in manifest["current_packages"]:
        if not (repository / package / "submission_metadata.json").is_file():
            raise FileNotFoundError(f"Submission package is incomplete: {package}")


def main() -> int:
    """Write or check the exact legacy-layout custody manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    destination = repository / "papers/legacy_layout_manifest.json"
    manifest = _build_manifest(repository)
    rendered = _render(manifest)
    _validate_current_layout(repository, manifest)
    if args.check:
        if not destination.is_file() or destination.read_text(encoding="utf-8") != rendered:
            raise ValueError("Legacy-layout manifest is absent or stale.")
        print("[OK] 58 retired paper paths retain exact Git-object custody")
        return 0
    destination.write_text(rendered, encoding="utf-8")
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
