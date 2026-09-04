#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — extended-abstract evidence custody generator.
"""Assemble and validate the exact evidence cited by submission 003."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


PACKAGE_REVISION = "02d45ec3f25b79e0a232bb44a327cbcc845ca133"
SOURCE_CUSTODY: dict[str, dict[str, str]] = {
    "vertical_stability_reduced_order.json": {
        "source_path": (
            "papers/submissions/002_neuromorphic_vertical_stability_control/"
            "evidence/vertical_stability_reduced_order.json"
        ),
        "source_revision": "9f675589ccf2c55862f99f84b9cd13582443b2dd",
        "git_blob_oid_sha1": "cd8577b11e03b769175c2148ac5f0e1f07b3c9e5",
        "source_sha256": ("185c36fa80443930fc3d0acb98c030eb6416a2ba0c20fcec027cbeac8bd460c1"),
        "packaging_transform": "exact byte copy",
    },
    "real_diiid_145419_validation.json": {
        "source_path": (
            "papers/submissions/001_hybrid_rust_python_grad_shafranov_"
            "equilibrium_solver/evidence/real_diiid_145419_validation.json"
        ),
        "source_revision": "0ac200a6be9b3c6a8e841ff1ca3062dc5cc6ecca",
        "git_blob_oid_sha1": "7a807a217f2e7f27c94004df70be54a4ec11154f",
        "source_sha256": ("444458af45e1ce2dd5d761c8baed718df28b44c21566583ee15a7ea724f3a5c6"),
        "packaging_transform": "exact byte copy",
    },
    "sparc_geqdsk_rmse_benchmark.json": {
        "source_path": (
            "papers/submissions/001_hybrid_rust_python_grad_shafranov_"
            "equilibrium_solver/evidence/sparc_geqdsk_rmse_benchmark.json"
        ),
        "source_revision": "0ac200a6be9b3c6a8e841ff1ca3062dc5cc6ecca",
        "git_blob_oid_sha1": "21ef1c8558e5adaa9202a7739d4bca7a8628fea0",
        "source_sha256": ("c45acd3fdb4a208ba86ab04f68e8ba5d5e994bcab3e911bcdcd1bea3a1fe8db8"),
        "packaging_transform": "exact byte copy",
    },
}
IMPLEMENTATION_SOURCES: dict[str, dict[str, str]] = {
    "src/scpn_fusion/scpn/structure.py": {
        "git_blob_oid_sha1": "1cc31f1ecf403378428d9494691c4d27ab17da60",
        "sha256": "06e367304bd00298bee2b60ee6754462e64997be309f79632a0c9884b671fd3a",
        "role": "public stochastic Petri-net topology and firing semantics",
    },
    "src/scpn_fusion/scpn/compiler.py": {
        "git_blob_oid_sha1": "dadf0611fc0831a4eb6bf32da4b7b2adc5db0212",
        "sha256": "3215fbdf44f49730e4bb31e0950e28096f47e524459a23e7626305a5c69c2f1b",
        "role": "one threshold-configured spiking unit per transition and float fallback",
    },
    "src/scpn_fusion/scpn/controller.py": {
        "git_blob_oid_sha1": "019086fec433611200cf3459f2127830a58250be",
        "sha256": "7c8df261d5cdbcce11e79925e51232c4ab8e491a33067348a68cc7279cf06504",
        "role": "public neuro-symbolic controller surface",
    },
    "src/scpn_fusion/scpn/controller_features_mixin.py": {
        "git_blob_oid_sha1": "3ef6c3d4dd62c52b302d2ee0d2c92f0689ee5287",
        "sha256": "0e84e62e5631e21a62e43fec6702c6f2d82aa0598bd7dc22bb5b91f0371ce603",
        "role": "marking clipping and bounded, slew-limited actuator decoding",
    },
    "src/scpn_fusion/scpn/contracts.py": {
        "git_blob_oid_sha1": "24502fb8d66a5a268333ff1849a9a2a6fc2d45b0",
        "sha256": "95f750a82c2b8089ce0c4330573e51a979b351d9ee0e45cf58304df66f8f6f3e",
        "role": "compiled-artifact and action-bound public contracts",
    },
    "src/scpn_fusion/control/fueling_mode.py": {
        "git_blob_oid_sha1": "a6010b61fbfbd7f72fda886d51746db63ee59ecb",
        "sha256": "74661b544fffaba65da61eb4362644644c1bbdb3b713d659693a0f61baaa5577",
        "role": "production Petri-to-controller construction surface used by the contract test",
    },
    "src/scpn_fusion/control/neuro_cybernetic_controller.py": {
        "git_blob_oid_sha1": "585e1871261cec3b02ce228f89879632664140b1",
        "sha256": "deac019d049c6444f4179f100e527de65da415cf30b981c30ed46d36ce123230",
        "role": "SpikingControllerPool used independently by the vertical benchmark",
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


def _close(actual: Any, expected: float) -> bool:
    """Compare one recorded scalar without widening the stored precision."""
    return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-15)


def _materialise_evidence(repository: Path, evidence: Path) -> None:
    """Copy exact companion bytes, or validate the package-local fallback."""
    evidence.mkdir(parents=True, exist_ok=True)
    for name, custody in SOURCE_CUSTODY.items():
        source = repository / custody["source_path"]
        destination = evidence / name
        expected = custody["source_sha256"]
        if source.is_file():
            if _sha256(source) != expected:
                raise ValueError(f"Companion evidence digest drifted: {source}")
            if not destination.is_file() or _sha256(destination) != expected:
                destination.write_bytes(source.read_bytes())
        if not destination.is_file() or _sha256(destination) != expected:
            raise ValueError(f"Exact package-local evidence is unavailable: {name}")


def _validate_vertical(payload: dict[str, Any]) -> None:
    """Validate the reduced-order vertical benchmark and negative SNN result."""
    plant = payload["plant"]
    expected_plant = {
        "gamma_per_s": 200.0,
        "kick_m": 0.005,
        "step_s": 0.001,
        "duration_s": 0.06,
        "current_limit_a": 10000.0,
    }
    for field, expected in expected_plant.items():
        if not _close(plant[field], expected):
            raise ValueError(f"Vertical-plant value drifted: {field}")
    controllers = payload["controllers"]
    if not _close(controllers["PID"]["settling_ms"], 25.0):
        raise ValueError("PID settling evidence drifted.")
    if not _close(controllers["LQR"]["settling_ms"], 39.0):
        raise ValueError("LQR settling evidence drifted.")
    if controllers["SNN"].get("outcome") != "diverges":
        raise ValueError("The negative SNN outcome must remain explicit.")
    if controllers["SNN"].get("settling_ms") is not None:
        raise ValueError("The divergent SNN must not acquire a settling time.")
    if not _close(controllers["SNN"]["peak_abs_displacement_mm"], 159.68272099657466):
        raise ValueError("SNN peak-displacement evidence drifted.")


def _validate_diiid(payload: dict[str, Any]) -> None:
    """Validate the DIII-D same-case reproduction and cold-start failure."""
    if "not blind prediction" not in str(payload.get("disclosure", "")):
        raise ValueError("DIII-D evidence lost its same-case disclosure.")
    checks = (
        (payload["full_domain_reproduction"]["deep_rms_rel_span"], 0.019084943379848895),
        (payload["full_domain_reproduction"]["anderson_iterations"], 21.0),
        (payload["full_domain_cold_start"]["deep_rms_rel_span"], 1.2678812266175519),
        (payload["full_domain_cold_start"]["iterations"], 1.0),
    )
    if any(not _close(actual, expected) for actual, expected in checks):
        raise ValueError("DIII-D result values drifted.")


def _validate_sparc(payload: dict[str, Any]) -> None:
    """Validate both the pointwise gate and the stricter source contract."""
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("SPARC case inventory is unavailable.")
    gated = [case for case in cases if case.get("gated") is True]
    adapted = sum(case.get("geqdsk_adapted_source_contract_pass") is True for case in gated)
    if len(gated) != 16 or payload.get("gate_row_count") != 16:
        raise ValueError("SPARC gated-row inventory drifted.")
    if payload.get("passes") is not True or any(case.get("passes") is not True for case in gated):
        raise ValueError("The 16-of-16 pointwise gate no longer passes.")
    if adapted != 8 or payload.get("gate_adapted_source_contract_pass_count") != 8:
        raise ValueError("The adapted source contract drifted from 8 of 16.")
    if payload.get("all_cases_neural_backend") is not False:
        raise ValueError("Diagnostic fallback rows must remain visible.")


def _validate_implementation(repository: Path) -> None:
    """Reject drift in every source file used to describe the control path."""
    for relative_path, custody in IMPLEMENTATION_SOURCES.items():
        path = repository / relative_path
        if not path.is_file() or _sha256(path) != custody["sha256"]:
            raise ValueError(f"Implementation source drifted: {relative_path}")


def main() -> None:
    """Assemble the frozen inputs and write a deterministic evidence manifest."""
    submission = Path(__file__).resolve().parent
    repository = submission.parents[2]
    evidence = submission / "evidence"
    metadata = _load_json(submission / "submission_metadata.json")
    if metadata.get("repository_revision") != PACKAGE_REVISION:
        raise ValueError("Submission metadata does not name the package revision.")

    _materialise_evidence(repository, evidence)
    vertical = _load_json(evidence / "vertical_stability_reduced_order.json")
    diiid = _load_json(evidence / "real_diiid_145419_validation.json")
    sparc = _load_json(evidence / "sparc_geqdsk_rmse_benchmark.json")
    _validate_vertical(vertical)
    _validate_diiid(diiid)
    _validate_sparc(sparc)
    _validate_implementation(repository)

    roles = {
        "vertical_stability_reduced_order.json": (
            "independent reduced-order PID, LQR and SpikingControllerPool outcomes"
        ),
        "real_diiid_145419_validation.json": (
            "DIII-D same-case warm-start reproduction and cold-start failure"
        ),
        "sparc_geqdsk_rmse_benchmark.json": (
            "SPARC pointwise neural gate, fallback disclosure and source-contract results"
        ),
    }
    files = {
        name: {"sha256": _sha256(evidence / name), "role": role} for name, role in roles.items()
    }
    generator = Path(__file__).resolve()
    generator_path = generator.relative_to(repository).as_posix()
    manifest = {
        "SPDX-License-Identifier": "AGPL-3.0-or-later",
        "commercialLicense": "available",
        "conceptsCopyright": "1996-2026 Miroslav Sotek. All rights reserved.",
        "codeCopyright": "2020-2026 Miroslav Sotek. All rights reserved.",
        "orcid": "0009-0009-3560-0851",
        "contact": "www.anulum.li | protoscience@anulum.li",
        "projectDescription": "SCPN-FUSION-CORE extended-abstract evidence manifest.",
        "schema_version": "2.0",
        "repository_revision": PACKAGE_REVISION,
        "claim_boundary": {
            "petri_compiler": (
                "source-level production contract; one threshold-configured spiking unit per "
                "transition when the optional backend is available, with a float fallback"
            ),
            "vertical_control": (
                "independent reduced-order ODE benchmark of PID, LQRController and "
                "SpikingControllerPool; it does not execute a compiled Petri artifact"
            ),
            "diiid": "warm-started same-case reproduction, not blind prediction",
            "sparc": (
                "16 of 16 gated pointwise neural rows pass; only 8 of 16 pass the adapted "
                "source contract; not free-boundary or experimental accuracy"
            ),
        },
        "source_custody": SOURCE_CUSTODY,
        "implementation_sources": IMPLEMENTATION_SOURCES,
        "generators": {generator_path: {"sha256": _sha256(generator)}},
        "files": files,
    }
    destination = evidence / "evidence_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("  [OK] submission 003 exact evidence and claim boundaries")


if __name__ == "__main__":
    main()
