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


PACKAGE_REVISION = "576ce6f684cb2c5bb44a3877ef6887da006b3dc6"
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
    "linear_mhd_miss_study.json": {
        "source_path": "validation/reports/linear_mhd_miss_study.json",
        "source_revision": "0acee5884d0a7de889ce3e088bc67ac2eeca513c",
        "git_blob_oid_sha1": "f9d17e3b1705a1bf98ddb4f3e9a7039d3272c864",
        "source_sha256": "fe2315e0a5fb14a2bd7df6491bcc3d483f88c2b77736c0080cbdf3e008ef3db3",
        "packaging_transform": PACKAGING_TRANSFORM,
    },
    "linear_mhd_near_zero_spectrum.json": {
        "source_path": "validation/reports/linear_mhd_near_zero_spectrum.json",
        "source_revision": "576ce6f684cb2c5bb44a3877ef6887da006b3dc6",
        "git_blob_oid_sha1": "b236cd39ee6e6d92e6142e4c5f0cd83c2b1e2063",
        "source_sha256": "af6ffbcfefbbd97c38673f3a6243f3d8d291c397762e52716e0ec2944eeda7ed",
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
    "validation/linear_mhd_miss_study.py": {
        "git_blob_oid_sha1": "acb418a033dca4228e9696fb43cc78347e3b13fb",
        "sha256": "1dc3105b97f196b8e296af9467ef3affc500c67636df6d331fbcc0c9f44164fa",
        "role": "follow-up study of the two misses, dense QZ at every solve",
    },
    "validation/linear_mhd_near_zero_spectrum.py": {
        "git_blob_oid_sha1": "c4f0dbba14ba7f554e176b000bd6acd9713f8857",
        "sha256": "c31f4abab673667a05df02d1a4d39a6aae5f5085647aff5573ade1d706d0856c",
        "role": "conditioning of the near-zero eigenvalues of the Harris pencil",
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
    """Read one value by its dotted (or slash-separated) path; list items by index.

    ``ratio:<a>|<b>`` divides the values at two paths; ``min:<list>|<field>`` and
    ``max:<list>|<field>`` take the extreme of one field over a list of records.
    """
    if dotted.startswith(("min:", "max:")):
        operation, rest = dotted.split(":", 1)
        records, field = rest.split("|")
        values = [float(record[field]) for record in _at(payload, records)]
        return min(values) if operation == "min" else max(values)
    if dotted.startswith("ratio:"):
        numerator, denominator = dotted.removeprefix("ratio:").split("|")
        return float(_at(payload, numerator)) / float(_at(payload, denominator))
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
    slopes, roots = "slope_scan.", "root_scan."
    rows += [
        ("study", slopes + "etas.4", "{}", "1e-10"),
        ("study", slopes + "local_slopes.0", "{:.3f}", "0.537"),
        ("study", slopes + "local_slopes.1", "{:.3f}", "0.572"),
        ("study", slopes + "local_slopes.2", "{:.3f}", "0.588"),
        ("study", slopes + "local_slopes.3", "{:.3f}", "0.595"),
        ("study", slopes + "ratio_to_asymptotic.0", "{:.3f}", "0.772"),
        ("study", slopes + "ratio_to_asymptotic.2", "{:.3f}", "0.951"),
        ("study", slopes + "ratio_to_asymptotic.4", "{:.3f}", "0.991"),
        ("study", slopes + "growth_rates.4", "{:.2e}", "1.56e-06"),
        ("study", slopes + "smallest_eta_doubling_change", "{:.1e}", "1.7e-04"),
        ("study", roots + "roots_by_elements.200.0", "{:.2e}", "7.92e-09"),
        ("study", roots + "roots_by_elements.200.1", "{:.2e}", "1.05e-08"),
        ("study", roots + "roots_by_elements.200.2", "{:.2e}", "1.36e-08"),
        ("study", roots + "roots_by_elements.400.0", "{:.2e}", "1.04e-09"),
        ("study", roots + "roots_by_elements.400.1", "{:g}", "0"),
        ("study", roots + "roots_by_elements.400.2", "{:.2e}", "1.86e-07"),
        ("study", roots + "unstable_ka_0p5_growth.2", "{:.2e}", "1.58e-05"),
        (
            "study",
            "ratio:" + roots + "roots_by_elements.400.2|" + roots + "unstable_ka_0p5_growth.2",
            "{:.1%}",
            "1.2%",
        ),
        ("study", "checks.every_solve_complete_dense", "{}", "True"),
        ("study", "provenance.source_revision", "{}", "03fdaf1d158f43539656d4e2aea4b8c2272681ea"),
        ("study", "provenance.numpy", "{}", "2.5.4"),
        ("study", "provenance.scipy", "{}", "1.17.1"),
        ("study", "provenance.threads.OPENBLAS_NUM_THREADS", "{}", "8"),
    ]
    listed = "smallest_eigenvalues|"
    rows += [
        ("spectrum", "unknowns", "{:d}", "1600"),
        ("spectrum", "singular_values_a.smallest", "{:.1e}", "1.5e-17"),
        ("spectrum", "singular_values_a.largest", "{:.1e}", "1.3e+02"),
        ("spectrum", "near_zero_count", "{:d}", "158"),
        ("spectrum", "near_zero_limit", "{:.0e}", "1e-04"),
        ("spectrum", "min:" + listed + "condition", "{:.1e}", "1.6e+10"),
        ("spectrum", "max:" + listed + "condition", "{:.1e}", "1.9e+12"),
        ("spectrum", "min:" + listed + "error_bound", "{:.1e}", "4.6e-04"),
        ("spectrum", "max:" + listed + "error_bound", "{:.1e}", "5.6e-02"),
        ("spectrum", "max:" + listed + "relative_residual", "{:.1e}", "1.5e-14"),
        ("spectrum", "largest_real_part", "{:.2e}", "6.05e-10"),
        ("spectrum", "perturbation.relative_scale", "{:.0e}", "1e-15"),
        ("spectrum", "perturbation.trials.0.largest_real_part", "{:.1e}", "-8.9e-12"),
        ("spectrum", "perturbation.trials.1.largest_real_part", "{:.1e}", "3.8e-11"),
        ("spectrum", "perturbation.trials.2.largest_real_part", "{:.1e}", "6.2e-10"),
        ("spectrum", "positive_listed_inside_error_bound", "{}", "True"),
        (
            "spectrum",
            "provenance.source_revision",
            "{}",
            "d9c74c6c27b25cf4df9c6964d2c6d7eaafae4d2d",
        ),
        ("spectrum", "provenance.threads.OPENBLAS_NUM_THREADS", "{}", "4"),
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
        ),
        "study": scientific_payload(_load_json(evidence_dir / "linear_mhd_miss_study.json")),
        "spectrum": scientific_payload(
            _load_json(evidence_dir / "linear_mhd_near_zero_spectrum.json")
        ),
    }
    validate_quoted(loaded)
    for relative_path, custody in IMPLEMENTATION_SOURCES.items():
        path = repository / relative_path
        if not path.is_file() or _sha256(path) != custody["sha256"]:
            raise ValueError(f"Implementation source drifted: {relative_path}")
    roles = {
        "linear_mhd_reduced_model_acceptance.json": "nine fixed acceptance criteria, seven passed",
        "linear_mhd_miss_study.json": "follow-up study: slope to eta 1e-10, Harris root by grid",
        "linear_mhd_near_zero_spectrum.json": "conditioning of the near-zero Harris eigenvalues",
    }
    generator = Path(__file__).resolve()
    manifest = {
        **TIER0_HEADER,
        "projectDescription": "SCPN-FUSION-CORE linear reduced-MHD paper evidence manifest.",
        "schema_version": "2.0",
        "repository_revision": PACKAGE_REVISION,
        "claim_boundary": {
            "scope": "linear resistive reduced MHD in a Harris slab and a periodic cylinder",
            "misses": "cylinder tearing slope (explained by the follow-up scan) and the "
            "negative-Delta-prime Harris root (sign undetermined in double precision)",
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
