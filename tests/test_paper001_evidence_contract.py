# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Paper 001 evidence-custody tests.
"""Protect the public equilibrium paper's exact evidence and claim boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = (
    REPO_ROOT / "papers/submissions/001_hybrid_rust_python_grad_shafranov_equilibrium_solver"
)
EVIDENCE = SUBMISSION / "evidence"
EVIDENCE_REVISION = "476908debdd886d3a35bf0ae85216e684727adce"
DIIID_SOURCE_PATH = "artifacts/real_diiid_145419/real_145419_validation.json"
TIER0_FIELDS = {
    "SPDX-License-Identifier",
    "commercialLicense",
    "conceptsCopyright",
    "codeCopyright",
    "orcid",
    "contact",
    "projectDescription",
}


def _load(path: Path) -> dict[str, Any]:
    """Load one JSON object used by the paper contract."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest of exact evidence bytes."""
    return hashlib.sha256(payload).hexdigest()


def _scientific_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove only the structured public-package header fields."""
    return {key: value for key, value in payload.items() if key not in TIER0_FIELDS}


def test_diiid_payload_matches_the_exact_tracked_git_object() -> None:
    """The packaged DIII-D payload must be the immutable source plus headers."""
    packaged = _load(EVIDENCE / "real_diiid_145419_validation.json")
    source = subprocess.run(
        ["git", "show", f"{EVIDENCE_REVISION}:{DIIID_SOURCE_PATH}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    source_payload = json.loads(source)
    blob = (
        subprocess.run(
            ["git", "hash-object", "--stdin"],
            cwd=REPO_ROOT,
            check=True,
            input=source,
            capture_output=True,
        )
        .stdout.decode("ascii")
        .strip()
    )
    manifest = _load(EVIDENCE / "evidence_manifest.json")
    custody = manifest["source_custody"]["real_diiid_145419_validation.json"]

    assert _scientific_payload(packaged) == source_payload
    assert custody["source_sha256"] == _sha256_bytes(source)
    assert custody["git_blob_oid_sha1"] == blob
    assert custody["source_revision"] == EVIDENCE_REVISION
    assert "not blind prediction" in packaged["disclosure"]


def test_diiid_manuscript_numbers_are_exactly_evidence_backed() -> None:
    """Every selected DIII-D number in the table must retain its exact source."""
    payload = _load(EVIDENCE / "real_diiid_145419_validation.json")
    assert payload["full_domain_reproduction"]["deep_rms_rel_span"] == 0.019084943379848895
    assert payload["full_domain_reproduction"]["global_max_rel_span"] == 0.039077316475147075
    assert payload["full_domain_reproduction"]["anderson_iterations"] == 21
    assert (
        payload["subcell_source_averaging"]["metrics"]["deep_rms_rel_span"] == 0.018329789768440196
    )
    assert payload["subcell_source_averaging"]["metrics"]["iterations"] == 20
    assert payload["full_domain_cold_start"]["deep_rms_rel_span"] == 1.2678812266175519
    assert payload["full_domain_cold_start"]["global_max_rel_span"] == 1.7313910359868308
    assert payload["full_domain_cold_start"]["iterations"] == 1
    assert (
        payload["shell_pinning_attribution"]["metrics"]["deep_rms_rel_span"]
        == 0.0006971907802435192
    )
    assert (
        payload["shell_pinning_attribution"]["metrics"]["global_max_rel_span"]
        == 0.0017412164778905011
    )
    assert payload["shell_pinning_attribution"]["metrics"]["iterations"] == 14


def test_sparc_gate_keeps_backend_source_contract_and_diagnostics_separate() -> None:
    """Pointwise success must not erase partial source-contract or fallback results."""
    payload = _load(EVIDENCE / "sparc_geqdsk_rmse_benchmark.json")
    gated = [case for case in payload["cases"] if case["gated"]]
    diagnostic = [case for case in payload["cases"] if not case["gated"]]

    assert len(payload["cases"]) == 36
    assert payload["reference_case_count"] == 18
    assert len(gated) == payload["gate_row_count"] == 16
    assert payload["passes"] is True
    assert all(case["machine"] == "sparc" for case in gated)
    assert all(case["reference_role"] == "gate" for case in gated)
    assert all(case["surrogate_backend"] == "neural_equilibrium" for case in gated)
    assert all(case["passes"] for case in gated)
    assert sum(case["geqdsk_adapted_source_contract_pass"] for case in gated) == 8
    assert payload["gate_adapted_source_contract_pass_count"] == 8
    assert payload["all_cases_neural_backend"] is False
    assert any(case["surrogate_backend"] == "reduced_order_proxy" for case in diagnostic)


def test_manifest_records_asymmetric_source_custody_and_exact_generators() -> None:
    """Tracked DIII-D and ignored SPARC origins must never be conflated."""
    manifest = _load(EVIDENCE / "evidence_manifest.json")
    metadata = _load(SUBMISSION / "submission_metadata.json")
    sparc_custody = manifest["source_custody"]["sparc_geqdsk_rmse_benchmark.json"]

    assert manifest["repository_revision"] == metadata["repository_revision"]
    assert metadata["repository_revision_role"] == "evidence_revision"
    assert sparc_custody["source_was_gitignored"] is True
    assert "paper-local" in sparc_custody["authoritative_public_custody"]
    for relative_path, record in manifest["generators"].items():
        assert record["sha256"] == _sha256_bytes((REPO_ROOT / relative_path).read_bytes())
    for filename, record in manifest["files"].items():
        assert record["sha256"] == _sha256_bytes((EVIDENCE / filename).read_bytes())


def test_manuscript_and_figures_keep_manufactured_and_quantitative_claims_separate() -> None:
    """Public prose and graphics must not promote illustrations into results."""
    manuscript = (SUBMISSION / "manuscript.tex").read_text(encoding="utf-8")
    normalized_manuscript = " ".join(manuscript.split())
    figure_source = "\n".join(
        (SUBMISSION / "figures" / filename).read_text(encoding="utf-8")
        for filename in ("fig_inverse_reconstruction.py", "fig_sparc_equilibrium.py")
    )

    assert "Eight of those 16 rows" in manuscript
    assert "only 8 of 16 pass" in manuscript
    assert "not a Grad--Shafranov solution" in manuscript
    assert "does not claim exact-head hosted success" in normalized_manuscript
    assert "authoritative public custody copy" in manuscript
    assert "Solov" not in figure_source
    assert "X-point" not in figure_source
    assert "Wb/rad" not in figure_source
    assert "manufactured" in figure_source.lower()
