#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — linear reduced-MHD paper evidence custody generator.
"""Assemble and validate the exact evidence cited by submission 008.

The acceptance report is a tracked file. Its package-local copy is
the tracked file with the seven structured public header fields placed first; the scientific
payload is unchanged. Every number the manuscript quotes is listed in ``quoted`` with the
format the manuscript prints it in; the generator recomputes each one from the evidence and
fails when one differs. It runs no solver.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE_REVISION = "f7a367a1a60f80c82a442a3ff9b55526582717ae"
PACKAGING_TRANSFORM = "seven structured Tier-0 fields added first; scientific payload unchanged"
TIER0_HEADER: dict[str, str] = {
    "SPDX-License-Identifier": "AGPL-3.0-or-later",
    "commercialLicense": "available",
    "conceptsCopyright": "1996-2026 Miroslav Sotek. All rights reserved.",
    "codeCopyright": "2020-2026 Miroslav Sotek. All rights reserved.",
    "orcid": "0009-0009-3560-0851",
    "contact": "www.anulum.li | protoscience@anulum.li",
    "projectDescription": "SCPN-FUSION-CORE linear reduced-MHD paper evidence.",
}
SOURCE_CUSTODY: dict[str, dict[str, str]] = {
    "linear_mhd_reduced_model_acceptance.json": {
        "source_path": "validation/reports/linear_mhd_reduced_model_acceptance.json",
        "source_revision": "44852feeedf2780f13d42ed5b90c8ee1b4f390a5",
        "git_blob_oid_sha1": "ab2a84765c79ebb007502d62370c945a60bd4a26",
        "source_sha256": "bcbbc302cfcc78747b7ed957a3d9de9842aae362587745b0a9c16e4c1ef60dd9",
        "packaging_transform": PACKAGING_TRANSFORM,
    },
}
IMPLEMENTATION_SOURCES: dict[str, dict[str, str]] = {
    "src/scpn_fusion/core/linear_mhd_1d.py": {
        "git_blob_oid_sha1": "33d8b7c3b11ffc911923851bd93fe21aeb648b4e",
        "sha256": "bb16a5957a0339dc4528e68ea8e71b75a7e3c6a4f338427dfd8953369d2e0ff2",
        "role": "Hermite-element slab and cylinder eigenproblem, growth rate",
    },
    "src/scpn_fusion/core/linear_mhd_delta_prime.py": {
        "git_blob_oid_sha1": "1cd790ae9267ea471397e03029c3fd3e2a07166a",
        "sha256": "8990e06a3752ffe98d0b2c41969ca3e4f7dc36cf0fdd375087e7e88ee6cf560d",
        "role": "independent shooting of the stability index",
    },
    "src/scpn_fusion/core/linear_mhd_spectrum.py": {
        "git_blob_oid_sha1": "954e1ac4c2f3cf59d4aeec2df060a8b8a712fdf7",
        "sha256": "4060453623aa2671954c54b240f5f951bf81c346249a5ee322af6c0a149d0d24",
        "role": "dense QZ and verified shift-and-invert search",
    },
    "src/scpn_fusion/core/tearing_mode_theory.py": {
        "git_blob_oid_sha1": "f038ca6370f8aaa4da547ad8557177f726fb2043",
        "sha256": "3ee63511991367468b5943bc6c18f76105a241773a04c9d9c786d3fd2bbdee4d",
        "role": "asymptotic FKR reference rate",
    },
    "validation/linear_mhd_reduced_model_acceptance.py": {
        "git_blob_oid_sha1": "db745213ebf8916c2d9466a2154d51c3be749d91",
        "sha256": "016354218ec44c91df7be5882c61242b8e6e296a04bab1da9ff45cdce3a10499",
        "role": "the nine fixed acceptance criteria",
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
    criteria = [
        ("harris_tearing_slope", "{:.3f}", "0.598", True),
        ("harris_tearing_fkr_ratio", "{:.3f}", "0.998", True),
        ("harris_negative_delta_prime_stable", "{:.1e}", "7.9e-09", False),
        ("harris_element_doubling_change", "{:.1e}", "1.3e-07", True),
        ("sparse_dense_agreement", "{}", "0.0", True),
        ("cylinder_tearing_slope", "{:.3f}", "0.554", False),
        ("cylinder_tearing_fkr_ratio", "{:.3f}", "0.951", True),
        ("cylinder_resistive_kink_slope", "{:.3f}", "0.306", True),
        ("cylinder_resistive_kink_ratio", "{:.3f}", "0.969", True),
    ]
    rows: list[tuple[str, str, str, str]] = []
    for name, fmt, text, passed in criteria:
        rows.append(("report", f"criteria.{name}.value", fmt, text))
        rows.append(("report", f"criteria.{name}.pass", "{}", str(passed)))
    rows += [
        ("report", "passed", "{:d}", "7"),
        ("report", "total", "{:d}", "9"),
        ("report", "inputs.cylinder_delta_prime", "{:.2f}", "9.69"),
        ("report", "inputs.cylinder_tearing_growth.0", "{:.2e}", "3.05e-04"),
        ("report", "inputs.cylinder_tearing_growth.1", "{:.2e}", "8.86e-05"),
        ("report", "inputs.cylinder_tearing_growth.2", "{:.2e}", "2.37e-05"),
        ("report", "inputs.slab_growth.0", "{:.2e}", "2.48e-04"),
        ("report", "inputs.etas.0", "{}", "1e-06"),
        ("report", "inputs.etas.2", "{}", "1e-08"),
        ("report", "criteria.harris_negative_delta_prime_stable.limit", "{}", "1e-09"),
        ("report", "criteria.cylinder_tearing_slope.band.0", "{}", "0.56"),
        ("report", "criteria.cylinder_tearing_slope.band.1", "{}", "0.64"),
        ("report", "provenance.numpy", "{}", "2.5.4"),
        ("report", "provenance.scipy", "{}", "1.17.1"),
        ("report", "provenance.python", "{}", "3.12.3"),
        ("report", "provenance.cpu_count", "{:d}", "24"),
        ("report", "provenance.threads.OPENBLAS_NUM_THREADS", "{}", "8"),
        ("report", "provenance.source_revision", "{}", "c1bc7f63a93b29e6f9cdf284b09a287775697292"),
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
            _load_json(evidence_dir / "linear_mhd_reduced_model_acceptance.json")
        )
    }
    validate_quoted(loaded)
    for relative_path, custody in IMPLEMENTATION_SOURCES.items():
        path = repository / relative_path
        if not path.is_file() or _sha256(path) != custody["sha256"]:
            raise ValueError(f"Implementation source drifted: {relative_path}")
    roles = {
        "linear_mhd_reduced_model_acceptance.json": "nine fixed acceptance criteria, seven passed",
    }
    generator = Path(__file__).resolve()
    manifest = {
        **TIER0_HEADER,
        "projectDescription": "SCPN-FUSION-CORE linear reduced-MHD paper evidence manifest.",
        "schema_version": "2.0",
        "repository_revision": PACKAGE_REVISION,
        "claim_boundary": {
            "scope": "linear resistive reduced MHD in a Harris slab and a periodic cylinder",
            "misses": "cylinder tearing slope and the negative-Delta-prime Harris root",
            "excluded": "toroidal 1/R physics, shaped equilibria, nonlinear MHD",
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
    print("  [OK] submission 008 exact evidence and quoted values")


if __name__ == "__main__":
    main()
