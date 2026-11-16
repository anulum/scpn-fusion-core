#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — runaway kinetic Rust-kernel paper evidence custody generator.
"""Assemble and validate the exact evidence cited by submission 007.

The benchmark report is a tracked file. Its package-local copy is
the tracked file with the seven structured public header fields placed first; the scientific
payload is unchanged. Every number the manuscript quotes is listed in ``quoted`` with the
format the manuscript prints it in; the generator recomputes each one from the evidence and
fails when one differs. It runs neither backend.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE_REVISION = "8bfedda56df3894f9bdf92bfa76c75a66e5c2460"
PACKAGING_TRANSFORM = "seven structured Tier-0 fields added first; scientific payload unchanged"
TIER0_HEADER: dict[str, str] = {
    "SPDX-License-Identifier": "AGPL-3.0-or-later",
    "commercialLicense": "available",
    "conceptsCopyright": "1996-2026 Miroslav Sotek. All rights reserved.",
    "codeCopyright": "2020-2026 Miroslav Sotek. All rights reserved.",
    "orcid": "0009-0009-3560-0851",
    "contact": "www.anulum.li | protoscience@anulum.li",
    "projectDescription": "SCPN-FUSION-CORE runaway kinetic Rust-kernel paper evidence.",
}
SOURCE_CUSTODY: dict[str, dict[str, str]] = {
    "runaway_kinetic_rust_benchmark.json": {
        "source_path": "validation/reports/runaway_kinetic_rust_benchmark.json",
        "source_revision": "d8a10e8fdb4a99132cf5cf2ce82536752b3c5659",
        "git_blob_oid_sha1": "ced1d2ce3ce98100bc87dfe3352dc1a6f38c564f",
        "source_sha256": "f9de44ee14a815db1d3c5d93bb1262e4e83afcc02a9e5429106cd88bda786c58",
        "packaging_transform": PACKAGING_TRANSFORM,
    },
}
IMPLEMENTATION_SOURCES: dict[str, dict[str, str]] = {
    "scpn-fusion-rs/crates/fusion-physics/src/runaway_kinetic/coefficients.rs": {
        "git_blob_oid_sha1": "1c82330c6547e4e499c7ac5bffc23078fd543275",
        "sha256": "9151bbd823368f14cfa0e3fc4575bfcd15f5c168fa6c9b1cab1d1b9aaf930dfd",
        "role": "Rust coefficient checks",
    },
    "scpn-fusion-rs/crates/fusion-physics/src/runaway_kinetic/grid.rs": {
        "git_blob_oid_sha1": "4f4fea12bb86ce263a8fda2f2b1f583941187f06",
        "sha256": "5f36fd096f56e0d8c22b3aeac6f240df2776818a3047d62f45a985f8404cb7f3",
        "role": "Rust grid",
    },
    "scpn-fusion-rs/crates/fusion-physics/src/runaway_kinetic/mod.rs": {
        "git_blob_oid_sha1": "d37b75f81a52706180c674293a10c688b26cbc2d",
        "sha256": "7a0ca2f0cf7cb50255b25cc70b9ba46a9c8aafbc31b0051df8b29fcf081a6f8b",
        "role": "Rust module root",
    },
    "scpn-fusion-rs/crates/fusion-physics/src/runaway_kinetic/operator.rs": {
        "git_blob_oid_sha1": "69c83a815604b76673398ce21b69f7ffc198c76f",
        "sha256": "baee9e75ea89fff01d873a968a198905605372ba37e1290ec14259312207aeda",
        "role": "Rust finite-volume operator",
    },
    "scpn-fusion-rs/crates/fusion-physics/src/runaway_kinetic/solver.rs": {
        "git_blob_oid_sha1": "44938b88ed4eb0a1369aef7a966493f21c0044e6",
        "sha256": "d87fbbcf113c86262e2fbd695be2c183c4d4d003d8d71358599dbb17ed2218b4",
        "role": "Rust SSPRK3 solver",
    },
    "scpn-fusion-rs/crates/fusion-python/src/bindings/runaway_kinetic.rs": {
        "git_blob_oid_sha1": "42d80c84af3c8cedaa60c508377070bd682a6d63",
        "sha256": "c3c46a6257780774fb6108d317c396ff27f13a1e9b00d31fdde600cdbdc30d2a",
        "role": "Python binding of the Rust kernel",
    },
    "src/scpn_fusion/core/runaway_kinetic_coefficients.py": {
        "git_blob_oid_sha1": "08a752da2bac5994ecdbc7aefd3c650b5a855407",
        "sha256": "32f32f49a2cbbc301754b7f2bcb53d505abca67f5872728671f27b1fb7062829",
        "role": "NumPy coefficient checks",
    },
    "src/scpn_fusion/core/runaway_kinetic_diagnostics.py": {
        "git_blob_oid_sha1": "fcc514e7ace3cb8e580b71c4d018c919187f03a0",
        "sha256": "d0f951d586ebb677e8e1b7331f49eeee8c102c9c21d8ae49e0191df76108f815",
        "role": "moments and diagnostics",
    },
    "src/scpn_fusion/core/runaway_kinetic_grid.py": {
        "git_blob_oid_sha1": "68d32c9f716596b7e73ac6173407f449f20addc7",
        "sha256": "6a6785c3cfdeba08070cd7ee01fed0e0f32a0b2683fdacd9f3a901aaf898cdf4",
        "role": "NumPy grid",
    },
    "src/scpn_fusion/core/runaway_kinetic_operator.py": {
        "git_blob_oid_sha1": "9105266f8f61edf178dccd30507220cbd0cc5e9a",
        "sha256": "77511f3ec134f142817861cde65b52e40bce155bb05de17d33b76b1892e8eb58",
        "role": "NumPy finite-volume operator",
    },
    "src/scpn_fusion/core/runaway_kinetic_solver.py": {
        "git_blob_oid_sha1": "103e61815d8ab16d8a0a62af7883d0cbde66a27a",
        "sha256": "16c4dcf91c844e69c49601a2d2b32e2a44ff5201f2cef8feca5567d6d2aa7de1",
        "role": "SSPRK3 solver and backend selection",
    },
    "validation/benchmark_runaway_kinetic_rust.py": {
        "git_blob_oid_sha1": "3c7f5bb9371978eb8d5a74418348b8d8200cc5e1",
        "sha256": "84fbc149b4ba98baf8efdd5a8bee8d3d07d18325f4444cdbc24e219516e631b5",
        "role": "parity case, gates and timing",
    },
}


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one exact file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object or reject an unsupported top level."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def packaged_bytes(source: bytes) -> bytes:
    """Return the package-local form of one tracked file: header fields, then the payload."""
    payload = json.loads(source)
    if not isinstance(payload, dict) or set(payload) & set(TIER0_HEADER):
        raise ValueError("A tracked file must be a JSON object without public header fields.")
    return (json.dumps({**TIER0_HEADER, **payload}, indent=2) + "\n").encode("utf-8")


def scientific_payload(packaged: dict[str, Any]) -> dict[str, Any]:
    """Remove only the structured public header fields from a package-local file."""
    return {key: value for key, value in packaged.items() if key not in TIER0_HEADER}


def _at(payload: dict[str, Any], dotted: str) -> Any:
    """Read one value by its dotted (or slash-separated) path; list items by index."""
    value: Any = payload
    # Keys that themselves contain dots (``moments.density_m3``) use "/" as the separator.
    for part in dotted.split("/") if "/" in dotted else dotted.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def quoted() -> list[tuple[str, str, str, str]]:
    """Return (evidence, dotted path, format, printed text) for every quoted number."""
    exact = (
        "distribution",
        "minimum_distribution",
        "times_s",
        "total_tendency",
        "radial_transport",
        "electric_acceleration",
        "collisional_drag_diffusion",
        "pitch_scattering",
        "cross_diffusion",
        "synchrotron_loss",
        "bremsstrahlung_loss",
        "avalanche_generation",
        "external_source",
        "runaway_density_m3",
        "runaway_density_avalanche_generation_m3_s",
        "runaway_density_external_source_m3_s",
    )
    rows = [("report", f"parity.{name}.relative_l2_error", "{}", "0.0") for name in exact]
    rows += [
        ("report", "parity/moments.current_density_a_m2/relative_l2_error", "{:.1e}", "5.4e-14"),
        ("report", "parity/moments.density_m3/relative_l2_error", "{:.1e}", "7.8e-16"),
        (
            "report",
            "parity/moments.kinetic_energy_density_j_m3/relative_l2_error",
            "{:.1e}",
            "5.6e-16",
        ),
        (
            "report",
            "parity.runaway_density_radial_transport_m3_s.relative_l2_error",
            "{:.1e}",
            "5.2e-16",
        ),
        ("report", "parity.runaway_density_tendency_m3_s.relative_l2_error", "{:.1e}", "2.4e-16"),
        ("report", "problem.grid_shape_radius_pitch_momentum.0", "{:d}", "8"),
        ("report", "problem.grid_shape_radius_pitch_momentum.1", "{:d}", "16"),
        ("report", "problem.grid_shape_radius_pitch_momentum.2", "{:d}", "32"),
        ("report", "problem.internal_ssprk3_steps", "{:d}", "4"),
        ("report", "timing_seconds.repeats", "{:d}", "11"),
        ("report", "timing_seconds.numpy_median", "{:.4f}", "0.0126"),
        ("report", "timing_seconds.rust_median", "{:.4f}", "0.0060"),
        ("report", "timing_seconds.rust_speedup_over_numpy", "{:.2f}", "2.10"),
        ("report", "host.logical_cpu_count", "{:d}", "12"),
        ("report", "host.load_average_1m_5m_15m.0", "{:.1f}", "7.6"),
        ("report", "host.load_average_1m_5m_15m.1", "{:.1f}", "14.6"),
        ("report", "host.load_average_1m_5m_15m.2", "{:.1f}", "14.6"),
        ("report", "runtime_provenance.numpy_version", "{}", "2.5.4"),
        ("report", "runtime_provenance.python_version", "{}", "3.12.3"),
        ("report", "source_revision", "{}", "98b6d4d2cfde1e8cf22129fa90e779b0291f8149"),
        ("report", "all_pass", "{}", "True"),
    ]
    rows += [
        ("report", f"gates.{gate}", "{}", "True")
        for gate in (
            "all_declared_outputs_compared",
            "full_output_relative_l2_within_2e_12",
            "internal_ssprk3_steps_exact",
            "rust_measured_faster_on_this_host_problem",
        )
    ]
    return rows


def validate_quoted(evidence: dict[str, dict[str, Any]]) -> None:
    """Recompute every quoted number and fail on the first that prints differently."""
    for source, path, fmt, text in quoted():
        if fmt.format(_at(evidence[source], path)) != text:
            raise ValueError(f"Quoted value drifted: {source}:{path}")


def _materialise_evidence(repository: Path, evidence: Path) -> None:
    """Write each package-local file from its tracked source, or check the existing copy."""
    evidence.mkdir(parents=True, exist_ok=True)
    for name, custody in SOURCE_CUSTODY.items():
        source = repository / custody["source_path"]
        destination = evidence / name
        if source.is_file():
            if _sha256(source) != custody["source_sha256"]:
                raise ValueError(f"Tracked file differs from the bound revision: {source}")
            expected = packaged_bytes(source.read_bytes())
            if not destination.is_file() or destination.read_bytes() != expected:
                destination.write_bytes(expected)
        if not destination.is_file():
            raise ValueError(f"Package-local evidence is unavailable: {name}")


def main() -> None:
    """Assemble the package-local evidence and write a deterministic manifest."""
    submission = Path(__file__).resolve().parent
    repository = submission.parents[2]
    evidence_dir = submission / "evidence"
    metadata = _load_json(submission / "submission_metadata.json")
    if metadata.get("repository_revision") != PACKAGE_REVISION:
        raise ValueError("Submission metadata does not name the package revision.")
    _materialise_evidence(repository, evidence_dir)
    loaded = {
        "report": scientific_payload(
            _load_json(evidence_dir / "runaway_kinetic_rust_benchmark.json")
        )
    }
    validate_quoted(loaded)
    for relative_path, custody in IMPLEMENTATION_SOURCES.items():
        path = repository / relative_path
        if not path.is_file() or _sha256(path) != custody["sha256"]:
            raise ValueError(f"Implementation source drifted: {relative_path}")
    roles = {
        "runaway_kinetic_rust_benchmark.json": "NumPy/Rust parity, gates and host-conditioned timing",
    }
    generator = Path(__file__).resolve()
    manifest = {
        **TIER0_HEADER,
        "projectDescription": "SCPN-FUSION-CORE runaway kinetic Rust-kernel paper evidence manifest.",
        "schema_version": "2.0",
        "repository_revision": PACKAGE_REVISION,
        "claim_boundary": {
            "scope": "NumPy/Rust equivalence of the operator on a synthetic parity case",
            "timing": "one loaded workstation and one small case; no portable speed-up",
            "excluded": "DREAM equivalence, physical runaway scenarios",
        },
        "source_custody": SOURCE_CUSTODY,
        "implementation_sources": IMPLEMENTATION_SOURCES,
        "quoted_values": len(quoted()),
        "generators": {
            generator.relative_to(repository).as_posix(): {"sha256": _sha256(generator)}
        },
        "files": {
            name: {"sha256": _sha256(evidence_dir / name), "role": role}
            for name, role in roles.items()
        },
    }
    destination = evidence_dir / "evidence_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("  [OK] submission 007 exact evidence and quoted values")


if __name__ == "__main__":
    main()
