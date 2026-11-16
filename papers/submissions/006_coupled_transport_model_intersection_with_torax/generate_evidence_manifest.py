#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — TORAX model-intersection paper evidence custody generator.
"""Assemble and validate the exact evidence cited by submission 006.

The two reports and the TORAX reference run are tracked files. Each package-local copy is
the tracked file with the seven structured public header fields placed first; the scientific
payload is unchanged. Every number the manuscript quotes is listed in ``quoted`` with the
format the manuscript prints it in; the generator recomputes each one from the evidence and
fails when one differs. It runs neither code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE_REVISION = "1445bb1405ed03341781d92afa6f62caa8fdb095"
PACKAGING_TRANSFORM = "seven structured Tier-0 fields added first; scientific payload unchanged"
TIER0_HEADER: dict[str, str] = {
    "SPDX-License-Identifier": "AGPL-3.0-or-later",
    "commercialLicense": "available",
    "conceptsCopyright": "1996-2026 Miroslav Sotek. All rights reserved.",
    "codeCopyright": "2020-2026 Miroslav Sotek. All rights reserved.",
    "orcid": "0009-0009-3560-0851",
    "contact": "www.anulum.li | protoscience@anulum.li",
    "projectDescription": "SCPN-FUSION-CORE TORAX model-intersection paper evidence.",
}
SOURCE_CUSTODY: dict[str, dict[str, str]] = {
    "torax_coupled_transport_parity.json": {
        "source_path": "validation/reports/torax_coupled_transport_parity.json",
        "source_revision": "849b9b2cfd1f8a41b898ddbc539734ad279e17dc",
        "git_blob_oid_sha1": "2288693f3bdd9480ce722f63cf2849dca1fcd1e0",
        "source_sha256": "846b19bef532db19ba232ccf216ffd4d720d132de2e968baf62591d9ef41b7c4",
        "packaging_transform": PACKAGING_TRANSFORM,
    },
    "torax_differentiable_transport.json": {
        "source_path": "validation/reports/torax_differentiable_transport.json",
        "source_revision": "d84af5398f4ad5eb211b7073c1dae8b290e4d336",
        "git_blob_oid_sha1": "93540f3167e8caaf556126d68c56c98569b2c8d9",
        "source_sha256": "bb8e4b70c06f706f195bad4091103b164b8d15aace33b7ae2e6fe9df90e21bb7",
        "packaging_transform": PACKAGING_TRANSFORM,
    },
    "torax_coupled_transport_reference.json": {
        "source_path": "validation/reference_data/torax/torax_coupled_transport_reference.json",
        "source_revision": "849b9b2cfd1f8a41b898ddbc539734ad279e17dc",
        "git_blob_oid_sha1": "09d53b22c7c4a0c25aceaf5aa70888a62ad21abc",
        "source_sha256": "c0e91c58f7bdb217832840af72bcbced21484275dd99d0e026dddc8ccdf07fbf",
        "packaging_transform": PACKAGING_TRANSFORM,
    },
}
IMPLEMENTATION_SOURCES: dict[str, dict[str, str]] = {
    "validation/reference_data/torax/coupled_transport_model_intersection_deck.py": {
        "git_blob_oid_sha1": "41e4cef33db4dfd514f072a9de17954f8fdfa24d",
        "sha256": "c10662a201afbeb996746d6a71360e5b6c32a83eea71cb9a05732cbdb3894340",
        "role": "declared model intersection and thresholds",
    },
    "validation/torax_coupled_reference_runner.py": {
        "git_blob_oid_sha1": "f7258404ebeb98dcfe543d1c12e046f1018c75f7",
        "sha256": "92926d291a0740932b9ab1f38eee343afdd4e6691638493c97a34f8fecb0a392",
        "role": "pinned TORAX 1.4.3 reference runner",
    },
    "validation/benchmark_torax_coupled_transport_parity.py": {
        "git_blob_oid_sha1": "52c3272e43c2bf2a00a5c1915e7ed8169e236471",
        "sha256": "61359b33d79a2a0ac26c9cf37fb0f0ade44ca480f5f63bd1fc3ebc0995f36937",
        "role": "parity benchmark and gates",
    },
    "validation/benchmark_torax_differentiable_transport.py": {
        "git_blob_oid_sha1": "418158b9e73fe2adb8dcffc2d3759a627d71bdb5",
        "sha256": "eb09241567e3a7d387379a12d708f0263b204846d013ab693b71d8c002d5b808",
        "role": "gradient and optimisation benchmark",
    },
    "src/scpn_fusion/core/integrated_transport_solver_coupled.py": {
        "git_blob_oid_sha1": "f75b5925d10d52251442e6e55cc943bca29dc43a",
        "sha256": "8563f89a62a00a7bdce1b312aadfda8189f9ab7c6da32f54c86d7772f7d524b4",
        "role": "coupled four-state step",
    },
    "src/scpn_fusion/core/integrated_transport_solver_coupled_contracts.py": {
        "git_blob_oid_sha1": "6e06685ce1d245c0a047877bf78508896d9e1330",
        "sha256": "d201cb72133e7f8e52e296e39b7e864255d78e4c8068f81729d1c0af6eb6898e",
        "role": "coupled step contracts",
    },
    "src/scpn_fusion/core/_integrated_transport_solver_coupled_numerics.py": {
        "git_blob_oid_sha1": "843f8db096c8f4b7a8bc7a89ed3295b7f122e4ae",
        "sha256": "bb2e4be040d7c3d07966ba78038cc3bf5cadbdfb2e541e1cb05aeb7544ffb4ea",
        "role": "coupled step numerics",
    },
    "src/scpn_fusion/core/current_diffusion.py": {
        "git_blob_oid_sha1": "944d8c65685d5d80cc9413cdc55b962c05d1ec73",
        "sha256": "7d4251347a961397b549cfcd46cdaffe607a6332f86f22ed9e9dc5351da1c82a",
        "role": "current diffusion",
    },
    "src/scpn_fusion/core/integrated_transport_solver_differentiable.py": {
        "git_blob_oid_sha1": "b264b0193503610c7e5e387ad81dd3ac0e736e6b",
        "sha256": "0ccbc2fc22b5190db7162ecf3d200246f82177e9ba6826a2429418719cd551c9",
        "role": "JAX step and bounded optimisation",
    },
    "src/scpn_fusion/core/_integrated_transport_solver_differentiable_numerics.py": {
        "git_blob_oid_sha1": "978e22a10e3df2ed7cb9445b426453e62fa7b746",
        "sha256": "d66a181c9634dfb44823af2231fbb2d021106fc3d116fb8f56cd4e4266582fd2",
        "role": "JAX tridiagonal numerics",
    },
    "src/scpn_fusion/core/integrated_transport_solver_differentiable_contracts.py": {
        "git_blob_oid_sha1": "ee7420f14f42e44905960fb12b9742e9400efb31",
        "sha256": "6c0503edffa82f0de99bae5818d6c2ded36a52cd39c3355e0f0842fe0e65ec7f",
        "role": "differentiable step contracts",
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
    """Read one value by its dotted path; list items by their index."""
    value: Any = payload
    for part in dotted.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def quoted() -> list[tuple[str, str, str, str]]:
    """Return (evidence, dotted path, format, printed text) for every quoted number."""
    rows: list[tuple[str, str, str, str]] = []
    parity = [
        ("state_metrics.electron_density_m3", "{:.2e}", "5.44e-04"),
        ("state_metrics.electron_temperature_kev", "{:.2e}", "3.88e-02"),
        ("state_metrics.ion_temperature_kev", "{:.2e}", "1.12e-02"),
        ("state_metrics.poloidal_flux_wb_per_rad", "{:.2e}", "6.94e-03"),
        ("refinement_metrics.native_primary_vs_refined.electron_density_m3", "{:.2e}", "3.29e-05"),
        (
            "refinement_metrics.native_primary_vs_refined.electron_temperature_kev",
            "{:.2e}",
            "2.64e-04",
        ),
        ("refinement_metrics.native_primary_vs_refined.ion_temperature_kev", "{:.2e}", "2.54e-04"),
        (
            "refinement_metrics.native_primary_vs_refined.poloidal_flux_wb_per_rad",
            "{:.2e}",
            "2.32e-05",
        ),
        ("refinement_metrics.torax_primary_vs_refined.electron_density_m3", "{:.2e}", "2.95e-04"),
        (
            "refinement_metrics.torax_primary_vs_refined.electron_temperature_kev",
            "{:.2e}",
            "1.09e-03",
        ),
        ("refinement_metrics.torax_primary_vs_refined.ion_temperature_kev", "{:.2e}", "1.72e-03"),
        (
            "refinement_metrics.torax_primary_vs_refined.poloidal_flux_wb_per_rad",
            "{:.2e}",
            "1.35e-04",
        ),
        ("source_budget_metrics.native_max_relative_error", "{:.1e}", "1.5e-16"),
        ("source_budget_metrics.torax_max_relative_error", "{:.1e}", "4.4e-16"),
        ("conservation_metrics.native_max_exchange_relative_closure", "{:.1e}", "1.6e-17"),
        ("conservation_metrics.native_max_linear_residual_linf", "{:.1e}", "7.1e-15"),
        ("runtime_seconds.torax_warm", "{:.4f}", "0.0934"),
        ("runtime_seconds.native_warm", "{:.2e}", "1.50e-03"),
        ("determinism.native_byte_identical", "{}", "True"),
        ("determinism.torax_byte_identical", "{}", "True"),
        ("passes_thresholds", "{}", "True"),
        ("physics_equivalence_claimed", "{}", "False"),
        ("performance_superiority_claimed", "{}", "False"),
    ]
    rows += [("parity", path, fmt, text) for path, fmt, text in parity]
    rows += [
        ("parity", f"gates.{gate}", "{}", "True")
        for gate in (
            "accuracy",
            "conservation",
            "cost_completeness",
            "nonlinear_convergence",
            "robustness",
            "source_budgets",
        )
    ]
    reference = [
        ("provenance.torax_version", "{}", "1.4.3"),
        ("provenance.precision", "{}", "float64"),
        ("provenance.runtime_backend", "{}", "cpu"),
        ("provenance.python", "{}", "3.12.3"),
        ("runtime_seconds.warm_primary", "{:.4f}", "0.0934"),
        ("runtime_samples_seconds.warm_primary.0", "{:.3f}", "0.110"),
        ("model_intersection.thresholds.electron_temperature_final_relative_l2", "{}", "0.12"),
        ("model_intersection.thresholds.ion_temperature_final_relative_l2", "{}", "0.12"),
        ("model_intersection.thresholds.electron_density_final_relative_l2", "{}", "0.08"),
        ("model_intersection.thresholds.poloidal_flux_final_relative_l2", "{}", "0.2"),
        ("model_intersection.thresholds.source_total_relative_error", "{}", "0.005"),
        ("model_intersection.thresholds.maximum_warm_cost_ratio", "{}", "100.0"),
        ("model_intersection.time.final_s", "{}", "0.02"),
        ("model_intersection.time.primary_dt_s", "{}", "0.01"),
        ("model_intersection.time.refined_dt_s", "{}", "0.005"),
        ("model_intersection.geometry.torax_cells", "{}", "16"),
        ("model_intersection.geometry.comparison_points", "{}", "17"),
    ]
    rows += [("reference", path, fmt, text) for path, fmt, text in reference]
    gradients = [
        ("forward_replay.maximum_absolute_error", "{:.1e}", "7.1e-15"),
        ("gradient_metrics.maximum_relative_error", "{:.1e}", "4.1e-08"),
        ("gradient_metrics.autodiff.0", "{:.2e}", "-4.92e-05"),
        ("gradient_metrics.autodiff.1", "{:.2e}", "3.63e-09"),
        ("gradient_metrics.autodiff.2", "{:.2e}", "3.15e-10"),
        ("perturbation_replay.relative_error", "{:.3f}", "0.010"),
        ("optimisation.initial_objective", "{:.2e}", "6.15e-06"),
        ("optimisation.final_objective", "{:.2e}", "3.53e-08"),
        ("optimisation.iterations", "{:d}", "12"),
        ("optimisation.final_controls.0", "{:.3f}", "1.269"),
        ("optimisation.final_controls.1", "{:.3f}", "0.872"),
        ("optimisation.final_controls.2", "{:.3f}", "0.999"),
        ("target_contract.target_controls.0", "{}", "1.25"),
        ("target_contract.target_controls.1", "{}", "0.8"),
        ("target_contract.target_controls.2", "{}", "1.3"),
        ("target_contract.finite_difference_step", "{}", "0.0001"),
        ("determinism.control_history_byte_identical", "{}", "True"),
        ("determinism.objective_history_byte_identical", "{}", "True"),
        ("general_transport_differentiability_claimed", "{}", "False"),
    ]
    rows += [("gradients", path, fmt, text) for path, fmt, text in gradients]
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
        key: scientific_payload(_load_json(evidence_dir / f"{name}.json"))
        for key, name in (
            ("parity", "torax_coupled_transport_parity"),
            ("gradients", "torax_differentiable_transport"),
            ("reference", "torax_coupled_transport_reference"),
        )
    }
    validate_quoted(loaded)
    for relative_path, custody in IMPLEMENTATION_SOURCES.items():
        path = repository / relative_path
        if not path.is_file() or _sha256(path) != custody["sha256"]:
            raise ValueError(f"Implementation source drifted: {relative_path}")
    roles = {
        "torax_coupled_transport_parity.json": "four-state parity, budgets, determinism, cost gate",
        "torax_differentiable_transport.json": "gradients, forward replay and bounded optimisation",
        "torax_coupled_transport_reference.json": "pinned TORAX 1.4.3 reference run and its deck",
    }
    generator = Path(__file__).resolve()
    manifest = {
        **TIER0_HEADER,
        "projectDescription": "SCPN-FUSION-CORE TORAX model-intersection paper evidence manifest.",
        "schema_version": "2.0",
        "repository_revision": PACKAGE_REVISION,
        "claim_boundary": {
            "scope": "the declared circular prescribed-coefficient model intersection only",
            "duration": "two 10 ms steps on 16 cells, refined with 5 ms steps",
            "gradients": "this coupled step only; the controls are not identified",
            "excluded": "TORAX equivalence outside the intersection, experiment, speed",
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
    print("  [OK] submission 006 exact evidence and quoted values")


if __name__ == "__main__":
    main()
