#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — conformal-coverage paper evidence custody generator.
"""Assemble and validate the exact evidence cited by submission 005.

The certificate and the five reports are tracked files. Each package-local copy is the
tracked file with the seven structured public header fields placed first; the scientific
payload is unchanged. Every number the manuscript quotes is listed in ``QUOTED`` with the
format the manuscript prints it in. The generator recomputes each one from the evidence,
derived values included, and fails when one differs. It runs no model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE_REVISION = "3a0c3f1451b0cb144364191710d057e7547bd9b1"
PACKAGING_TRANSFORM = "seven structured Tier-0 fields added first; scientific payload unchanged"
TIER0_HEADER: dict[str, str] = {
    "SPDX-License-Identifier": "AGPL-3.0-or-later",
    "commercialLicense": "available",
    "conceptsCopyright": "1996-2026 Miroslav Sotek. All rights reserved.",
    "codeCopyright": "2020-2026 Miroslav Sotek. All rights reserved.",
    "orcid": "0009-0009-3560-0851",
    "contact": "www.anulum.li | protoscience@anulum.li",
    "projectDescription": "SCPN-FUSION-CORE conformal-coverage paper evidence.",
}
SOURCE_CUSTODY: dict[str, dict[str, str]] = {
    "surrogate_conformal_certificates.json": {
        "source_path": "validation/surrogate_conformal_certificates.json",
        "source_revision": "019faf9c581ca0b5f78d55078971b4d2ae39f355",
        "git_blob_oid_sha1": "5d6217fdfa4ecac412261cdaf58cce58b109487f",
        "source_sha256": "bcc8ceef6837f0fb4cac328b85b5d9ba606c55bc3ae03df753ac4d9f4c53d744",
        "packaging_transform": PACKAGING_TRANSFORM,
    },
    "qlknn_conformal_regime_coverage.json": {
        "source_path": "validation/reports/qlknn_conformal_regime_coverage.json",
        "source_revision": "3a0c3f1451b0cb144364191710d057e7547bd9b1",
        "git_blob_oid_sha1": "8283d07e71ff95bb2d22592ebb0721048d1ba1fb",
        "source_sha256": "8220820f047f83dc91c1f11007ee0bb76d6a8be1d813b268a64d4465f69fca52",
        "packaging_transform": PACKAGING_TRANSFORM,
    },
    "qlknn_conformal_variants.json": {
        "source_path": "validation/reports/qlknn_conformal_variants.json",
        "source_revision": "3a0c3f1451b0cb144364191710d057e7547bd9b1",
        "git_blob_oid_sha1": "62243e5f307201b7db3224645c597938b0e94475",
        "source_sha256": "5d4dc473e69a8a6f51bef3fa5cb4720bb42595ed5d272092677b0085006ff110",
        "packaging_transform": PACKAGING_TRANSFORM,
    },
    "qlknn_conformal_stability_sets.json": {
        "source_path": "validation/reports/qlknn_conformal_stability_sets.json",
        "source_revision": "3a0c3f1451b0cb144364191710d057e7547bd9b1",
        "git_blob_oid_sha1": "c9aa34419189154d88f8bbf63cab9f9661792825",
        "source_sha256": "f54c751f8b3465cd6bee80451e8c6b7a0ea3790203fbc87b37edbdcbc9fd338c",
        "packaging_transform": PACKAGING_TRANSFORM,
    },
    "qlknn_conformal_profile_envelope.json": {
        "source_path": "validation/reports/qlknn_conformal_profile_envelope.json",
        "source_revision": "3a0c3f1451b0cb144364191710d057e7547bd9b1",
        "git_blob_oid_sha1": "53fc85cc40f560046eeb65d713f168e04672cf7e",
        "source_sha256": "7c7a3a69722a20285d4cca1e630b98ff0364bd9d03a6fb2576ba2460e68c47a7",
        "packaging_transform": PACKAGING_TRANSFORM,
    },
    "qlknn_conformal_self_consistent_envelope.json": {
        "source_path": "validation/reports/qlknn_conformal_self_consistent_envelope.json",
        "source_revision": "3a0c3f1451b0cb144364191710d057e7547bd9b1",
        "git_blob_oid_sha1": "298941c2be011c897483d82f7623b6e43a477ea1",
        "source_sha256": "bcfae360bba569a46c0aa101ba983ddc2cf9db67bd5208c0380fd866e8c6bfec",
        "packaging_transform": PACKAGING_TRANSFORM,
    },
    "neural_transport_qlknn.metrics.json": {
        "source_path": "weights/neural_transport_qlknn.metrics.json",
        "source_revision": "54af9b7c5c27b96781c7a760577384d1ed42373a",
        "git_blob_oid_sha1": "826d91ca032fe54dcda845eef971ff29022ea3a9",
        "source_sha256": "e761e2dc61bed856454a2b83a9a4006197e22b6ed072ccf438d6346bf47bcce8",
        "packaging_transform": PACKAGING_TRANSFORM,
    },
}
IMPLEMENTATION_SOURCES: dict[str, dict[str, str]] = {
    "tools/generate_surrogate_conformal_certificates.py": {
        "git_blob_oid_sha1": "f71257aec9e0d54618606aa3ff6519cd77c61abd",
        "sha256": "f79c844d31c26c9978325be09f111c6ad0b6ede2916068dd7102a2572b27036e",
        "role": "split-conformal certificate and the public residual scores",
    },
    "validation/qlknn_conformal_regime_coverage.py": {
        "git_blob_oid_sha1": "b96c2ab272df492861b4b0740b12616a23b05bb2",
        "sha256": "12fca8dd6e37f1da486eef6c35b3f28adbe5e0466bee79df4d16bba95d2b918a",
        "role": "coverage per target-defined regime and per input label",
    },
    "validation/qlknn_conformal_variants.py": {
        "git_blob_oid_sha1": "29de9433b57db50e8d77648d3154fd38c53b66a6",
        "sha256": "05d9716793b67b981fbbfc0d6f82eee2f9b690e24ce88f4bf339961900251282",
        "role": "four declared and two post hoc conformal variants, surrogate diagnostics",
    },
    "validation/qlknn_conformal_stability_sets.py": {
        "git_blob_oid_sha1": "96d917982512479cd5ffe6c2c66d583b1e771824",
        "sha256": "498beee8f154d2282124d45d0098ad16fbd2f9667145e7ec8213772a9b3dd61c",
        "role": "class-conditional stability sets, conditional flux intervals, re-splits",
    },
    "validation/qlknn_conformal_profile_envelope.py": {
        "git_blob_oid_sha1": "e3e14dd45c40a3bbaac0c66dfa078dbd7e66dcce",
        "sha256": "3f1933afa3521c2fdc914763b4c5dba3756fcceb6c8d37690ac37b192c6c2baf",
        "role": "frozen-coefficient propagation through the transport solver",
    },
    "validation/qlknn_conformal_self_consistent_envelope.py": {
        "git_blob_oid_sha1": "699957b2293c8ddcfc4ac26aae769e0120eef3f3",
        "sha256": "a7021707ac16c8464cecb14e7246b0b14085376d4fd1dda03a98791c159487e8",
        "role": "self-consistent bounded-closure propagation through the transport solver",
    },
}
CHANNELS = ("chi_e", "chi_i", "D_e")
VARIANTS = (
    "marginal_absolute",
    "mondrian_input_label",
    "normalised_gyro_bohm",
    "normalised_prediction",
    "mondrian_predicted_flux",
    "mondrian_predicted_flux_normalised_prediction",
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
    """Return the package-local form of one tracked file: header fields, then the payload."""
    payload = json.loads(source)
    if not isinstance(payload, dict) or set(payload) & set(TIER0_HEADER):
        raise ValueError("A tracked file must be a JSON object without public header fields.")
    return (json.dumps({**TIER0_HEADER, **payload}, indent=2) + "\n").encode("utf-8")


def scientific_payload(packaged: dict[str, Any]) -> dict[str, Any]:
    """Remove only the structured public header fields from a package-local file."""
    return {key: value for key, value in packaged.items() if key not in TIER0_HEADER}


def nonzero_coverage(target_regimes: dict[str, Any]) -> float:
    """Coverage of every nonzero-flux row: the near-threshold band and the rows above it."""
    near, above = target_regimes["near_threshold"], target_regimes["unstable"]
    return float(near["covered"] + above["covered"]) / float(near["n"] + above["n"])


def _at(payload: dict[str, Any], dotted: str) -> Any:
    """Read one value by its dotted path.

    ``nonzero:<path>`` derives the nonzero coverage; ``ratio:<a>|<b>`` divides two values.
    """
    if dotted.startswith("nonzero:"):
        return nonzero_coverage(_at(payload, dotted.removeprefix("nonzero:")))
    if dotted.startswith("ratio:"):
        numerator, denominator = dotted.removeprefix("ratio:").split("|")
        return float(_at(payload, numerator)) / float(_at(payload, denominator))
    value: Any = payload
    for part in dotted.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def _three(source: str, template: str, fmt: str, printed: str) -> list[tuple[str, str, str, str]]:
    """Expand one per-channel quote into three rows: chi_e, chi_i and D_e, in that order."""
    texts = printed.split(" / ")
    formats = fmt.split(" / ") if " / " in fmt else [fmt] * 3
    return [
        (source, template.format(channel=channel), formats[index], texts[index])
        for index, channel in enumerate(CHANNELS)
    ]


def quoted() -> list[tuple[str, str, str, str]]:
    """Return (evidence, dotted path, format, printed text) for every quoted number."""
    regime = "channels.{channel}"
    rows: list[tuple[str, str, str, str]] = []
    rows += _three("regime", regime + ".marginal_coverage", "{:.3f}", "0.901 / 0.900 / 0.903")
    rows += _three(
        "regime", regime + ".target_regimes.stable.coverage", "{:.3f}", "0.997 / 0.995 / 0.997"
    )
    rows += _three(
        "regime",
        regime + ".target_regimes.near_threshold.coverage",
        "{:.3f}",
        "0.999 / 0.991 / 0.993",
    )
    rows += _three(
        "regime", regime + ".target_regimes.unstable.coverage", "{:.3f}", "0.709 / 0.711 / 0.665"
    )
    rows += _three(
        "regime", "nonzero:" + regime + ".target_regimes", "{:.3f}", "0.737 / 0.738 / 0.697"
    )
    rows += _three(
        "regime",
        regime + ".target_regimes.near_threshold.median_relative_width",
        "{:.0f}",
        "216 / 12214 / 296",
    )
    rows += _three("regime", regime + ".radius", "{:.2f} / {:.2f} / {:.3f}", "2.92 / 3.30 / 0.394")
    table2 = {
        "marginal_absolute": (
            "0.709 / 0.711 / 0.665",
            "0.737 / 0.738 / 0.697",
            "2.92 / 3.30 / 0.394",
        ),
        "mondrian_input_label": (
            "0.713 / 0.716 / 0.671",
            "0.741 / 0.743 / 0.702",
            "3.28 / 5.73 / 0.548",
        ),
        "normalised_gyro_bohm": (
            "0.701 / 0.709 / 0.659",
            "0.730 / 0.737 / 0.691",
            "2.46 / 2.68 / 0.326",
        ),
        "normalised_prediction": (
            "0.706 / 0.723 / 0.664",
            "0.735 / 0.749 / 0.696",
            "1.24 / 1.14 / 0.196",
        ),
        "mondrian_predicted_flux": (
            "0.724 / 0.733 / 0.669",
            "0.751 / 0.758 / 0.700",
            "0.893 / 0.658 / 0.131",
        ),
        "mondrian_predicted_flux_normalised_prediction": (
            "0.719 / 0.735 / 0.666",
            "0.747 / 0.760 / 0.698",
            "0.716 / 0.541 / 0.110",
        ),
    }
    widths = {
        "marginal_absolute": "{:.2f} / {:.2f} / {:.3f}",
        "mondrian_input_label": "{:.2f} / {:.2f} / {:.3f}",
        "normalised_gyro_bohm": "{:.2f} / {:.2f} / {:.3f}",
        "normalised_prediction": "{:.2f} / {:.2f} / {:.3f}",
        "mondrian_predicted_flux": "{:.3f}",
        "mondrian_predicted_flux_normalised_prediction": "{:.3f}",
    }
    for variant, (above, nonzero, width) in table2.items():
        path = "channels.{channel}.variants." + variant
        rows += _three("variants", path + ".target_regimes.unstable.coverage", "{:.3f}", above)
        rows += _three("variants", "nonzero:" + path + ".target_regimes", "{:.3f}", nonzero)
        rows += _three("variants", path + ".median_half_width", widths[variant], width)
    diagnostics = "channels.{channel}.unstable_diagnostics."
    rows += _three(
        "variants", diagnostics + "share_predicted_at_most_r0", "{:.3f}", "0.683 / 0.645 / 0.738"
    )
    rows += _three(
        "variants", diagnostics + "median_prediction_over_target", "{:.3f}", "0.493 / 0.602 / 0.305"
    )
    sets = "channels.{channel}."
    rows += _three("sets", sets + "stable.set_coverage", "{:.3f}", "0.901 / 0.902 / 0.903")
    rows += _three("sets", sets + "unstable.set_coverage", "{:.3f}", "0.900 / 0.894 / 0.891")
    rows += _three("sets", sets + "stable.set_size_share.2", "{:.3f}", "0.113 / 0.205 / 0.081")
    rows += _three("sets", sets + "unstable.set_size_share.2", "{:.3f}", "0.101 / 0.156 / 0.071")
    rows += _three("sets", sets + "unstable.flux_coverage", "{:.3f}", "0.797 / 0.792 / 0.793")
    rows += _three(
        "sets",
        sets + "unstable.median_interval_width",
        "{:.1f} / {:.1f} / {:.2f}",
        "12.2 / 14.4 / 2.36",
    )
    rows += _three(
        "sets", sets + "unstable_flux_quantile", "{:.1f} / {:.1f} / {:.2f}", "11.6 / 13.8 / 2.32"
    )
    rows += _three("sets", sets + "abstention_share", "{:.1f}", "0.0 / 0.0 / 0.0")
    resplit = sets + "resplits."
    rows += _three("sets", resplit + "unstable_set.mean", "{:.3f}", "0.900 / 0.900 / 0.901")
    rows += _three("sets", resplit + "stable_set.mean", "{:.3f}", "0.900 / 0.900 / 0.900")
    rows += _three("sets", resplit + "unstable_set.min", "{:.3f}", "0.892 / 0.895 / 0.894")
    rows += _three("sets", resplit + "unstable_set.max", "{:.3f}", "0.907 / 0.903 / 0.909")
    rows += _three("sets", resplit + "stable_set.min", "{:.3f}", "0.895 / 0.895 / 0.896")
    rows += _three("sets", resplit + "stable_set.max", "{:.3f}", "0.905 / 0.904 / 0.905")
    rows += _three("sets", resplit + "unstable_flux.mean", "{:.3f}", "0.801 / 0.800 / 0.803")
    rows += _three("sets", resplit + "unstable_flux.min", "{:.3f}", "0.793 / 0.791 / 0.795")
    rows += _three("sets", resplit + "unstable_flux.max", "{:.3f}", "0.809 / 0.807 / 0.810")
    rows += _three(
        "envelope",
        "median_turbulent_diffusivity_m2_s.{channel}",
        "{:.3f} / {:.3f} / {:.4f}",
        "0.181 / 0.368 / 0.0152",
    )
    envelope = [
        ("neural_steady_state.T_core_keV", "{:.3f}", "1.702"),
        ("neural_steady_state.T_avg_keV", "{:.3f}", "0.957"),
        ("neural_steady_state.tau_e_s", "{:.3f}", "1.242"),
        ("envelopes.certificate.upper_turbulence.T_core_keV", "{:.3f}", "0.262"),
        ("envelopes.certificate.lower_turbulence.T_core_keV", "{:.3f}", "7.224"),
        ("envelopes.certificate.upper_turbulence.tau_e_s", "{:.3f}", "0.246"),
        ("envelopes.certificate.lower_turbulence.tau_e_s", "{:.3f}", "4.533"),
        ("envelopes.normalised_prediction.upper_turbulence.T_core_keV", "{:.3f}", "0.455"),
        ("envelopes.normalised_prediction.lower_turbulence.T_core_keV", "{:.3f}", "7.224"),
        ("case.p_aux_mw", "{:.0f}", "20"),
        ("case.steps", "{:d}", "1200"),
        (
            "ratio:envelopes.certificate.lower_turbulence.T_core_keV"
            "|neural_steady_state.T_core_keV",
            "{:.1f}",
            "4.2",
        ),
        (
            "ratio:neural_steady_state.T_core_keV"
            "|envelopes.certificate.upper_turbulence.T_core_keV",
            "{:.1f}",
            "6.5",
        ),
    ]
    rows += [("envelope", path, fmt, text) for path, fmt, text in envelope]
    lower, upper = (
        "envelopes.certificate.lower_turbulence.",
        "envelopes.certificate.upper_turbulence.",
    )
    scaled = "envelopes.normalised_prediction."
    self_consistent = [
        ("iteration.relaxation", "{:.3f}", "0.005"),
        ("iteration.check_relaxation", "{:.4f}", "0.0025"),
        ("iteration.residual_tolerance", "{:.0e}", "1e-10"),
        ("nominal.T_core_keV", "{:.3f}", "1.702"),
        ("nominal_reproduction_relative_error", "{:.1e}", "1.4e-04"),
        (upper + "T_core_keV", "{:.3f}", "0.281"),
        (lower + "T_core_keV", "{:.3f}", "4.707"),
        (lower + "at_check_relaxation.T_core_keV", "{:.3f}", "4.607"),
        (lower + "relaxation_relative_difference", "{:.1%}", "3.7%"),
        (upper + "tau_e_s", "{:.3f}", "0.257"),
        (lower + "tau_e_s", "{:.3f}", "3.270"),
        (scaled + "upper_turbulence.T_core_keV", "{:.3f}", "0.593"),
        (scaled + "lower_turbulence.T_core_keV", "{:.3f}", "3.523"),
        ("ratio:" + lower + "T_core_keV|nominal.T_core_keV", "{:.1f}", "2.8"),
        ("ratio:nominal.T_core_keV|" + upper + "T_core_keV", "{:.1f}", "6.1"),
    ]
    rows += [("self_consistent", path, fmt, text) for path, fmt, text in self_consistent]
    certificate = "certificates.0."
    rows += [
        ("certificate", certificate + "calibration.n", "{:d}", "47826"),
        ("certificate", certificate + "holdout.n", "{:d}", "47828"),
        ("certificate", certificate + "calibration.quantile_rank_one_based", "{:d}", "43045"),
        ("certificate", "method.alpha", "{:.1f}", "0.1"),
        ("metrics", "train_relative_l2", "{:.4f}", "0.0917"),
        ("metrics", "val_relative_l2", "{:.4f}", "0.0954"),
        ("metrics", "test_relative_l2", "{:.4f}", "0.0943"),
        ("metrics", "hidden_dims.0", "{:d}", "1024"),
        ("metrics", "hidden_dims.1", "{:d}", "512"),
        ("metrics", "hidden_dims.2", "{:d}", "256"),
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
        key: scientific_payload(_load_json(evidence_dir / f"{name}.json"))
        for key, name in (
            ("certificate", "surrogate_conformal_certificates"),
            ("regime", "qlknn_conformal_regime_coverage"),
            ("variants", "qlknn_conformal_variants"),
            ("sets", "qlknn_conformal_stability_sets"),
            ("envelope", "qlknn_conformal_profile_envelope"),
            ("self_consistent", "qlknn_conformal_self_consistent_envelope"),
            ("metrics", "neural_transport_qlknn.metrics"),
        )
    }
    validate_quoted(loaded)
    for relative_path, custody in IMPLEMENTATION_SOURCES.items():
        path = repository / relative_path
        if not path.is_file() or _sha256(path) != custody["sha256"]:
            raise ValueError(f"Implementation source drifted: {relative_path}")

    roles = {
        "surrogate_conformal_certificates.json": "90 % split-conformal certificate per channel",
        "qlknn_conformal_regime_coverage.json": "certificate coverage per regime and input label",
        "qlknn_conformal_variants.json": "conformal variants and surrogate false-stable diagnostics",
        "qlknn_conformal_stability_sets.json": "stability sets, conditional flux intervals, re-splits",
        "qlknn_conformal_profile_envelope.json": "frozen-coefficient transport-profile envelope",
        "qlknn_conformal_self_consistent_envelope.json": "self-consistent bounded-closure envelope",
        "neural_transport_qlknn.metrics.json": "surrogate training record: errors and hidden sizes",
    }
    generator = Path(__file__).resolve()
    manifest = {
        **TIER0_HEADER,
        "projectDescription": "SCPN-FUSION-CORE conformal-coverage paper evidence manifest.",
        "schema_version": "2.0",
        "repository_revision": PACKAGE_REVISION,
        "claim_boundary": {
            "coverage": "marginal within each class; exchangeability of the split assumed",
            "data": "QuaLiKiz-derived QLKNN-10D database values, not gyrokinetic or measured",
            "post_hoc": "two Mondrian-by-prediction variants and the stability sets came later",
            "profile": "frozen-coefficient sensitivity and self-consistent bounded closure, one L-mode case",
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
    print("  [OK] submission 005 exact evidence and quoted values")


if __name__ == "__main__":
    main()
