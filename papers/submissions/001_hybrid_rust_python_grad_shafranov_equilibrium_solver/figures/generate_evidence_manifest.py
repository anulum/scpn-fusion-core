#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — equilibrium-paper evidence manifest generator.
"""Validate Paper 001 evidence and bind it to exact sources and generators."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


EVIDENCE_REVISION = "476908debdd886d3a35bf0ae85216e684727adce"
DIIID_SOURCE_PATH = "artifacts/real_diiid_145419/real_145419_validation.json"
DIIID_SOURCE_SHA256 = "04c87959264291894976cbbec7b2c9bddbbb5eb3d6d2027dd6e8eeb49f1c8b23"
DIIID_SOURCE_BLOB = "41aaf842cdd71e892554aeec8eaf33889a9d1f48"
SPARC_SOURCE_PATH = "artifacts/sparc_geqdsk_rmse_benchmark.json"
SPARC_SOURCE_SHA256 = "c2eeeada47255ab3864b0d578e4740b60a30ac2f4a7cff80350aa8eac311dd4e"
SPARC_GENERATED_AT = "2026-08-26T16:44:27.724185+00:00"
TIER0_FIELDS = {
    "SPDX-License-Identifier",
    "commercialLicense",
    "conceptsCopyright",
    "codeCopyright",
    "orcid",
    "contact",
    "projectDescription",
}


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON evidence object or reject an unsupported top level."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one exact file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_diiid(payload: dict[str, Any]) -> None:
    """Reject drift in the DIII-D same-case results and disclosure."""
    if payload.get("disclosure") is None or "not blind prediction" not in payload["disclosure"]:
        raise ValueError("DIII-D evidence lost its warm-start disclosure.")
    provenance = payload["provenance"]
    if provenance.get("reference_file") != "g145419.02100":
        raise ValueError("DIII-D reference identity drifted.")
    if provenance.get("generator") != "validation/validate_real_diiid_145419.py":
        raise ValueError("DIII-D generator path drifted.")

    checks = (
        (payload["full_domain_reproduction"]["deep_rms_rel_span"], 0.019084943379848895),
        (payload["full_domain_reproduction"]["global_max_rel_span"], 0.039077316475147075),
        (payload["full_domain_reproduction"]["anderson_iterations"], 21),
        (payload["subcell_source_averaging"]["metrics"]["deep_rms_rel_span"], 0.018329789768440196),
        (payload["subcell_source_averaging"]["metrics"]["iterations"], 20),
        (payload["full_domain_cold_start"]["deep_rms_rel_span"], 1.2678812266175519),
        (payload["full_domain_cold_start"]["global_max_rel_span"], 1.7313910359868308),
        (payload["full_domain_cold_start"]["iterations"], 1),
        (
            payload["shell_pinning_attribution"]["metrics"]["deep_rms_rel_span"],
            0.0006971907802435192,
        ),
        (
            payload["shell_pinning_attribution"]["metrics"]["global_max_rel_span"],
            0.0017412164778905011,
        ),
        (payload["shell_pinning_attribution"]["metrics"]["iterations"], 14),
    )
    for actual, expected in checks:
        if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"DIII-D evidence value drifted: {actual!r} != {expected!r}")


def _validate_sparc(payload: dict[str, Any]) -> None:
    """Reject any widening or relabelling of the SPARC pointwise gate."""
    if payload.get("schema_version") != "sparc-geqdsk-rmse-benchmark.v2":
        raise ValueError("SPARC benchmark schema is unsupported.")
    if payload.get("generated_at_utc") != SPARC_GENERATED_AT:
        raise ValueError("SPARC benchmark generation time drifted.")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("SPARC benchmark cases are unavailable.")
    gated = [case for case in cases if case.get("gated") is True]
    if len(cases) != 36 or payload.get("reference_case_count") != 18:
        raise ValueError("SPARC benchmark inventory drifted.")
    if len(gated) != payload.get("gate_row_count") or len(gated) != 16:
        raise ValueError("SPARC gated-row inventory drifted.")
    if payload.get("passes") is not True:
        raise ValueError("SPARC aggregate gate no longer passes.")
    if any(case.get("machine") != "sparc" for case in gated):
        raise ValueError("A non-SPARC row entered the acceptance cohort.")
    if any(case.get("reference_role") != "gate" for case in gated):
        raise ValueError("A diagnostic row entered the acceptance cohort.")
    if any(case.get("surrogate_backend") != "neural_equilibrium" for case in gated):
        raise ValueError("A fallback row entered the neural acceptance cohort.")
    if any(case.get("passes") is not True for case in gated):
        raise ValueError("A gated SPARC row no longer passes the pointwise threshold.")
    adapted = sum(case.get("geqdsk_adapted_source_contract_pass") is True for case in gated)
    if adapted != 8 or payload.get("gate_adapted_source_contract_pass_count") != 8:
        raise ValueError("SPARC adapted-source-contract count drifted from 8 of 16.")
    if payload.get("all_cases_neural_backend") is not False:
        raise ValueError("Diagnostic fallback rows must remain visible.")


def main() -> None:
    """Validate the frozen payloads and write a deterministic custody manifest."""
    submission = Path(__file__).resolve().parent.parent
    evidence = submission / "evidence"
    repository = submission.parents[2]
    metadata = _load_json(submission / "submission_metadata.json")
    if metadata.get("repository_revision") != EVIDENCE_REVISION:
        raise ValueError("Submission metadata does not name the evidence revision.")

    diiid_path = evidence / "real_diiid_145419_validation.json"
    sparc_path = evidence / "sparc_geqdsk_rmse_benchmark.json"
    diiid = _load_json(diiid_path)
    sparc = _load_json(sparc_path)
    _validate_diiid(diiid)
    _validate_sparc(sparc)

    generator_paths = (
        repository / "validation/validate_real_diiid_145419.py",
        repository / "validation/benchmark_sparc_geqdsk_rmse.py",
        submission / "figures/fig_inverse_reconstruction.py",
        submission / "figures/fig_sparc_equilibrium.py",
        submission / "figures/generate_evidence_manifest.py",
    )
    generators = {
        path.relative_to(repository).as_posix(): {"sha256": _sha256(path)}
        for path in generator_paths
    }
    manifest = {
        "SPDX-License-Identifier": "AGPL-3.0-or-later",
        "commercialLicense": "available",
        "conceptsCopyright": "1996-2026 Miroslav Sotek. All rights reserved.",
        "codeCopyright": "2020-2026 Miroslav Sotek. All rights reserved.",
        "orcid": "0009-0009-3560-0851",
        "contact": "www.anulum.li | protoscience@anulum.li",
        "projectDescription": "SCPN-FUSION-CORE equilibrium-paper evidence manifest.",
        "schema_version": "2.0",
        "repository_revision": EVIDENCE_REVISION,
        "claim_boundary": {
            "diiid": "warm-started same-case reproduction, not blind prediction",
            "sparc": (
                "16 of 16 gated pointwise neural rows pass; 8 of 16 also pass the "
                "adapted source contract; not free-boundary or experimental accuracy"
            ),
        },
        "source_custody": {
            "real_diiid_145419_validation.json": {
                "source_path": DIIID_SOURCE_PATH,
                "source_revision": EVIDENCE_REVISION,
                "source_sha256": DIIID_SOURCE_SHA256,
                "git_blob_oid_sha1": DIIID_SOURCE_BLOB,
                "packaging_transform": "seven structured Tier-0 fields added; scientific payload unchanged",
            },
            "sparc_geqdsk_rmse_benchmark.json": {
                "source_path": SPARC_SOURCE_PATH,
                "source_revision_context": EVIDENCE_REVISION,
                "generated_at_utc": SPARC_GENERATED_AT,
                "source_sha256": SPARC_SOURCE_SHA256,
                "source_was_gitignored": True,
                "authoritative_public_custody": (
                    "this paper-local evidence file; the historical source artifact was not tracked"
                ),
                "packaging_transform": "seven structured Tier-0 fields added; scientific payload unchanged",
            },
        },
        "generators": generators,
        "files": {
            "real_diiid_145419_validation.json": {
                "sha256": _sha256(diiid_path),
                "role": "provenance-bound DIII-D same-case reproduction and honest negatives",
            },
            "sparc_geqdsk_rmse_benchmark.json": {
                "sha256": _sha256(sparc_path),
                "role": "pointwise SPARC neural gate plus diagnostic fallback and source-contract rows",
            },
        },
    }
    destination = evidence / "evidence_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("  [OK] evidence_manifest (DIII-D Git custody and SPARC package custody)")


if __name__ == "__main__":
    main()
