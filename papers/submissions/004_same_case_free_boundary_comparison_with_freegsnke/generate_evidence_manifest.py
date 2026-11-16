#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — same-case FreeGSNKE comparison evidence custody generator.
"""Assemble and validate the exact evidence cited by submission 004.

The two cited reports are tracked under ``validation/reports``. Each package-local copy is
the tracked report with the seven structured public header fields placed first; the
scientific payload is unchanged. The generator checks the tracked source digest, the
manuscript's claim-critical values and the implementation sources, then writes a
deterministic custody manifest. It runs no solver.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE_REVISION = "86649a05a15a0d3d919c35d560e02ebda6c8cc00"
PACKAGING_TRANSFORM = "seven structured Tier-0 fields added first; scientific payload unchanged"
TIER0_HEADER: dict[str, str] = {
    "SPDX-License-Identifier": "AGPL-3.0-or-later",
    "commercialLicense": "available",
    "conceptsCopyright": "1996-2026 Miroslav Sotek. All rights reserved.",
    "codeCopyright": "2020-2026 Miroslav Sotek. All rights reserved.",
    "orcid": "0009-0009-3560-0851",
    "contact": "www.anulum.li | protoscience@anulum.li",
    "projectDescription": "SCPN-FUSION-CORE same-case FreeGSNKE comparison evidence.",
}
SOURCE_CUSTODY: dict[str, dict[str, str]] = {
    "freegsnke_multimachine_parity.json": {
        "source_path": "validation/reports/freegsnke_multimachine_parity.json",
        "source_revision": "422e4c0b3d15629b4de0fa796a6f6ac3aac58df1",
        "git_blob_oid_sha1": "0a0f27336e0dd2f17be39e2a1c66bdf8aae97365",
        "source_sha256": "82e34d26567d5c1c2b652d06f459779aaf5163e6d67df86f4cd13cc762a575b7",
        "packaging_transform": PACKAGING_TRANSFORM,
    },
    "freegsnke_inverse_fit.json": {
        "source_path": "validation/reports/freegsnke_inverse_fit.json",
        "source_revision": "86649a05a15a0d3d919c35d560e02ebda6c8cc00",
        "git_blob_oid_sha1": "7ad702997128536c5626f867b68aa591492c95bb",
        "source_sha256": "abc4745331b4ed9d03cab47e022eb2f8a594f6f711c355951d25ef72f1ee5bb8",
        "packaging_transform": PACKAGING_TRANSFORM,
    },
}
IMPLEMENTATION_SOURCES: dict[str, dict[str, str]] = {
    "src/scpn_fusion/core/jax_free_boundary_predictive.py": {
        "git_blob_oid_sha1": "87dedb5642066e03ff5d02ff280ef0bcf650508a",
        "sha256": "6864163d4c4ccd43ee7b3bf289262400400d12587f35ce3858067d0dbfa600d9",
        "role": "SCPN predictive free-boundary solver",
    },
    "src/scpn_fusion/core/free_boundary_coil_fit.py": {
        "git_blob_oid_sha1": "32f7b891a816570627a8afcb3ac76f4e3feceb23",
        "sha256": "005447f78b837bfeb2c389d5b1789dd58630ce5a9d6491863e3ecb3dc18e37dd",
        "role": "Levenberg-Marquardt coil-current fit to isoflux and null-point targets",
    },
    "validation/benchmark_freegsnke_inverse.py": {
        "git_blob_oid_sha1": "6d813e88fa7f7d3cb599f0c63b34a620308f5cb6",
        "sha256": "35c1046f8570dd81dd76f72ba2d2b6ab71e30a6be5742516d3fd44fe310ca208",
        "role": "bridge audits shared by both benchmarks",
    },
    "validation/benchmark_freegsnke_multimachine.py": {
        "git_blob_oid_sha1": "52279ce22d65db1bb715515ebfeb53ad2c3be0ef",
        "sha256": "c6b1ae377b36eea618276d483e8a6174c414879c5474ca3c856559a772fd5b0f",
        "role": "forward same-case parity on the three public cases",
    },
    "validation/benchmark_freegsnke_inverse_fit.py": {
        "git_blob_oid_sha1": "5e435bce5aeee5ad6ef0575056d0b4a509e39df8",
        "sha256": "699d5e220bab2ec84c49d44c9c6ec2e4a2b1ce3821307d3c776021e823190106",
        "role": "inverse fits scored by both codes, and the merge of partial runs",
    },
}
FREEGSNKE_COMMIT = "00b79a08fbb6e642ac5ed2e440383cc5962e0358"
CASES = ("mastu_like", "sparc_like", "iter_like")
# Every forward-parity value the manuscript quotes, as the tracked report stores it.
FORWARD_VALUES: dict[str, dict[str, float]] = {
    "mastu_like": {
        "vacuum_max_abs_error_wb": 1.1102230246251565e-16,
        "source_relative_l2": 3.5121218106294314e-08,
        "warm_psi_n_rmse": 0.0035428935838590766,
        "frozen_psi_n_rmse": 0.00037768119984973175,
        "axis_position_error_m": 0.0013535296988508403,
        "worst_xpoint_error_m": 0.0021072270142722268,
        "cold_start_iterations": 600,
    },
    "sparc_like": {
        "vacuum_max_abs_error_wb": 1.7763568394002505e-14,
        "source_relative_l2": 9.1698813473659e-08,
        "warm_psi_n_rmse": 0.0004590541328966581,
        "frozen_psi_n_rmse": 0.00015184570236820223,
        "axis_position_error_m": 0.00012028400981143047,
        "worst_xpoint_error_m": 0.0011898216782683538,
        "cold_start_iterations": 153,
    },
    "iter_like": {
        "vacuum_max_abs_error_wb": 7.673861546209082e-13,
        "source_relative_l2": 2.985138085728447e-08,
        "warm_psi_n_rmse": 0.0006417804152384333,
        "frozen_psi_n_rmse": 0.00012160933317396045,
        "axis_position_error_m": 0.0012571865803061378,
        "worst_xpoint_error_m": 0.0014961681466249765,
        "cold_start_iterations": 270,
    },
}
COLD_START_OUTCOMES = {
    "mastu_like": "no_magnetic_axis_inside_limiter",
    "sparc_like": "same_fixed_point_as_warm_start",
    "iter_like": "same_fixed_point_as_warm_start",
}
# Every inverse value the manuscript quotes: (case, dotted path in the case, format, text).
INVERSE_QUOTED: tuple[tuple[str, str, str, str], ...] = (
    ("mastu_like", "freegsnke_loss.total_l2", "{:.2e}", "1.54e-03"),
    ("mastu_like", "scpn_evaluator_on_freegsnke_flux.total_l2", "{:.2e}", "1.55e-03"),
    ("mastu_like", "scpn_at_freegsnke_currents.loss.total_l2", "{:.2e}", "2.24e-03"),
    ("mastu_like", "starts.freegsnke.loss.total_l2", "{:.2e}", "1.92e-08"),
    (
        "mastu_like",
        "starts.freegsnke.freegsnke_at_scpn_currents.loss.total_l2",
        "{:.2e}",
        "2.11e-03",
    ),
    ("mastu_like", "starts.freegsnke.relative_l2_change_from_freegsnke", "{:.4f}", "0.0140"),
    ("mastu_like", "starts.perturbed.loss.total_l2", "{:.2e}", "1.96e-02"),
    ("mastu_like", "starts.perturbed.relative_l2_change_from_freegsnke", "{:.3f}", "0.642"),
    ("mastu_like", "starts.perturbed.continuation_steps", "{:d}", "387"),
    ("mastu_like", "starts.perturbed.outer_iterations", "{:d}", "7"),
    ("mastu_like", "starts.perturbed.stop_reason", "{}", "jacobian_solve_not_converged"),
    ("mastu_like", "starts.perturbed.freegsnke_at_scpn_currents.converged", "{}", "False"),
    (
        "mastu_like",
        "starts.perturbed.freegsnke_at_scpn_currents.relative_change",
        "{:.1e}",
        "2.6e-05",
    ),
    ("sparc_like", "freegsnke_loss.total_l2", "{:.2e}", "3.49e-02"),
    ("sparc_like", "scpn_evaluator_on_freegsnke_flux.total_l2", "{:.2e}", "3.48e-02"),
    ("sparc_like", "scpn_at_freegsnke_currents.loss.total_l2", "{:.2e}", "3.11e-02"),
    ("sparc_like", "starts.freegsnke.loss.total_l2", "{:.2e}", "4.92e-08"),
    (
        "sparc_like",
        "starts.freegsnke.freegsnke_at_scpn_currents.loss.total_l2",
        "{:.2e}",
        "7.78e-03",
    ),
    ("sparc_like", "starts.freegsnke.relative_l2_change_from_freegsnke", "{:.4f}", "0.0097"),
    ("sparc_like", "starts.perturbed.loss.total_l2", "{:.2e}", "1.21e-06"),
    (
        "sparc_like",
        "starts.perturbed.freegsnke_at_scpn_currents.loss.total_l2",
        "{:.2e}",
        "7.75e-03",
    ),
    ("sparc_like", "starts.perturbed.relative_l2_change_from_freegsnke", "{:.3f}", "0.127"),
    ("sparc_like", "starts.freegsnke.bounds_active", "{:d}", "0"),
    ("iter_like", "freegsnke_loss.total_l2", "{:.2f}", "1.71"),
    ("iter_like", "scpn_evaluator_on_freegsnke_flux.total_l2", "{:.2f}", "1.71"),
    ("iter_like", "scpn_at_freegsnke_currents.loss.total_l2", "{:.2f}", "1.81"),
    ("iter_like", "starts.freegsnke.loss.total_l2", "{:.2f}", "1.39"),
    ("iter_like", "starts.freegsnke.freegsnke_at_scpn_currents.loss.total_l2", "{:.2f}", "1.39"),
    ("iter_like", "starts.freegsnke.relative_l2_change_from_freegsnke", "{:.4f}", "0.0126"),
    ("iter_like", "starts.perturbed.loss.total_l2", "{:.2f}", "1.41"),
    ("iter_like", "starts.perturbed.freegsnke_at_scpn_currents.loss.total_l2", "{:.2f}", "1.41"),
    ("iter_like", "starts.perturbed.relative_l2_change_from_freegsnke", "{:.4f}", "0.0121"),
    ("iter_like", "starts.perturbed.continuation_steps", "{:d}", "17"),
    ("iter_like", "targets.isoflux_points", "{:d}", "24"),
    ("mastu_like", "targets.isoflux_points", "{:d}", "8"),
    ("sparc_like", "targets.isoflux_points", "{:d}", "6"),
)


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
    """Return the package-local form of one tracked report: header fields, then the payload."""
    payload = json.loads(source)
    if not isinstance(payload, dict) or set(payload) & set(TIER0_HEADER):
        raise ValueError("A tracked report must be a JSON object without public header fields.")
    packaged = {**TIER0_HEADER, **payload}
    return (json.dumps(packaged, indent=2) + "\n").encode("utf-8")


def scientific_payload(packaged: dict[str, Any]) -> dict[str, Any]:
    """Remove only the structured public header fields from a package-local report."""
    return {key: value for key, value in packaged.items() if key not in TIER0_HEADER}


def _materialise_evidence(repository: Path, evidence: Path) -> None:
    """Write each package-local report from its tracked source, or check the existing copy."""
    evidence.mkdir(parents=True, exist_ok=True)
    for name, custody in SOURCE_CUSTODY.items():
        source = repository / custody["source_path"]
        destination = evidence / name
        if source.is_file():
            if _sha256(source) != custody["source_sha256"]:
                raise ValueError(f"Tracked report differs from the bound revision: {source}")
            expected = packaged_bytes(source.read_bytes())
            if not destination.is_file() or destination.read_bytes() != expected:
                destination.write_bytes(expected)
        if not destination.is_file():
            raise ValueError(f"Package-local evidence is unavailable: {name}")
        packaged = _load_json(destination)
        if {key: packaged.get(key) for key in TIER0_HEADER} != TIER0_HEADER:
            raise ValueError(f"Package-local evidence lost its public header: {name}")


def validate_forward(report: dict[str, Any]) -> None:
    """Check the forward-parity status, boundary, provenance and every quoted value."""
    if report.get("status") != "pass":
        raise ValueError("The forward-parity report no longer passes.")
    boundary = report["claim_boundary"]
    if boundary.get("same_case_cross_solver_parity") is not True or any(
        boundary.get(key) is not False
        for key in (
            "control_admission",
            "experimental_validation",
            "facility_validation",
            "speed_comparison",
        )
    ):
        raise ValueError("The forward-parity claim boundary drifted.")
    if report["provenance"]["source_commit"] != FREEGSNKE_COMMIT:
        raise ValueError("The forward-parity report names another FreeGSNKE source.")
    for case in CASES:
        row = report["cases"][case]
        total = row["total_psi_comparison"]
        actual = {
            "vacuum_max_abs_error_wb": row["vacuum_bridge"]["max_abs_error_wb"],
            "source_relative_l2": row["profile_source_bridge"]["source_parity"][
                "relative_l2_error"
            ],
            "warm_psi_n_rmse": total["production_smooth"]["psi_n_rmse_inside_limiter"],
            "frozen_psi_n_rmse": total["frozen_topology_control"]["psi_n_rmse_inside_limiter"],
            "axis_position_error_m": total["production_smooth"]["axis_position_error_m"],
            "worst_xpoint_error_m": max(total["production_smooth"]["xpoint_position_errors_m"]),
            "cold_start_iterations": row["cold_start"]["iteration_count"],
        }
        if actual != FORWARD_VALUES[case]:
            raise ValueError(f"Forward-parity values drifted: {case}")
        if row["cold_start"]["outcome"] != COLD_START_OUTCOMES[case]:
            raise ValueError(f"Cold-start outcome drifted: {case}")
        if row["passed"] is not True:
            raise ValueError(f"Forward-parity case no longer passes: {case}")


def validate_inverse(report: dict[str, Any]) -> None:
    """Check the inverse-fit status, boundary, provenance and every quoted value."""
    # The report fails one completeness gate; the manuscript states which and why.
    if report.get("status") != "fail":
        raise ValueError("The inverse-fit report status differs from the manuscript.")
    boundary = report["claim_boundary"]
    if boundary.get("same_case_inverse_comparison") is not True or any(
        boundary.get(key) is not False
        for key in (
            "control_admission",
            "experimental_validation",
            "facility_validation",
            "speed_comparison",
        )
    ):
        raise ValueError("The inverse-fit claim boundary drifted.")
    provenance = report["provenance"]
    if provenance["freegsnke_source_commit"] != FREEGSNKE_COMMIT:
        raise ValueError("The inverse-fit report names another FreeGSNKE source.")
    for path, digest in provenance["scpn_source_sha256"].items():
        bound = IMPLEMENTATION_SOURCES.get(path)
        if bound is None or bound["sha256"] != digest:
            raise ValueError(f"The inverse-fit run used a source the package does not bind: {path}")
    for case, dotted, fmt, text in INVERSE_QUOTED:
        if fmt.format(_inverse_value(report["cases"][case], dotted)) != text:
            raise ValueError(f"Inverse-fit value drifted: {case}:{dotted}")
    failed = {
        (case, gate)
        for case, row in report["cases"].items()
        for gate, ok in row["checks"].items()
        if not ok
    }
    if failed != {("mastu_like", "freegsnke_refinements_at_scpn_currents_converged")}:
        raise ValueError(f"The inverse-fit gate outcome drifted: {sorted(failed)}")


def _inverse_value(row: dict[str, Any], key: str) -> Any:
    """Read one quoted inverse-fit value by its dotted path inside a case row."""
    value: Any = row
    for part in key.split("."):
        value = value[part]
    return value


def _validate_implementation(repository: Path) -> None:
    """Reject drift in every source file the manuscript describes."""
    for relative_path, custody in IMPLEMENTATION_SOURCES.items():
        path = repository / relative_path
        if not path.is_file() or _sha256(path) != custody["sha256"]:
            raise ValueError(f"Implementation source drifted: {relative_path}")


def main() -> None:
    """Assemble the package-local reports and write a deterministic evidence manifest."""
    submission = Path(__file__).resolve().parent
    repository = submission.parents[2]
    evidence = submission / "evidence"
    metadata = _load_json(submission / "submission_metadata.json")
    if metadata.get("repository_revision") != PACKAGE_REVISION:
        raise ValueError("Submission metadata does not name the package revision.")

    _materialise_evidence(repository, evidence)
    validate_forward(_load_json(evidence / "freegsnke_multimachine_parity.json"))
    validate_inverse(_load_json(evidence / "freegsnke_inverse_fit.json"))
    _validate_implementation(repository)

    roles = {
        "freegsnke_multimachine_parity.json": (
            "forward same-case parity: vacuum flux, profile source, warm and frozen-support "
            "equilibria, cold-start outcome"
        ),
        "freegsnke_inverse_fit.json": (
            "inverse coil-current fits to FreeGSNKE's targets, each solution scored by both codes"
        ),
    }
    files = {
        name: {"sha256": _sha256(evidence / name), "role": role} for name, role in roles.items()
    }
    generator = Path(__file__).resolve()
    manifest = {
        **{
            **TIER0_HEADER,
            "projectDescription": "SCPN-FUSION-CORE same-case FreeGSNKE evidence manifest.",
        },
        "schema_version": "2.0",
        "repository_revision": PACKAGE_REVISION,
        "claim_boundary": {
            "forward": (
                "same-case parity on three synthetic public machine descriptions shipped with "
                "FreeGSNKE 3.0.1; warm-started fixed points; no experimental comparison"
            ),
            "inverse": (
                "coil-current fits to FreeGSNKE's isoflux and null-point targets; each "
                "solution scored by both codes; agreement is reported, not gated"
            ),
            "excluded": "facility, control, safety and speed claims; measured discharges",
        },
        "source_custody": SOURCE_CUSTODY,
        "implementation_sources": IMPLEMENTATION_SOURCES,
        "generators": {
            generator.relative_to(repository).as_posix(): {"sha256": _sha256(generator)}
        },
        "files": files,
    }
    destination = evidence / "evidence_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("  [OK] submission 004 exact evidence and claim boundaries")


if __name__ == "__main__":
    main()
