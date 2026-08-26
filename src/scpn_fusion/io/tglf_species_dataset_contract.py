# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Species-aware GACODE TGLF Dataset Contract
"""Fail-closed custody for ordered multi-species official GACODE TGLF runs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Final, cast

from scpn_fusion.core._tglf_interface_runtime import render_tglf_input
from scpn_fusion.core._tglf_interface_types import TGLFInputDeck, TGLFOutput, TGLFSpecies
from scpn_fusion.io.safe_loaders import checked_json_load
from scpn_fusion.io.tglf_dataset_contract import (
    MAX_TGLF_MANIFEST_BYTES,
    MAX_TGLF_RAW_FILE_BYTES,
    MAX_TGLF_RECORDS_BYTES,
    MAX_TGLF_SAMPLES,
    REQUIRED_TGLF_RAW_FILES,
    TGLF_REGIMES,
    TGLF_SAMPLING_STRATA,
    TGLF_SOURCE_LICENCE,
    TGLF_SOURCE_REPOSITORY,
    canonical_tglf_sample_id,
    deterministic_tglf_split,
    sha256_file,
)

TGLF_SPECIES_DATASET_SCHEMA_VERSION: Final = "scpn-fusion.tglf-gacode-dataset.v2"
TGLF_GBFLUX_LAYOUT: Final = "particle[NS],energy[NS],momentum[NS],exchange[NS]"
TGLF_PARTICLE_TRANSPORT_METHOD: Final = "paired-gradient-linear-least-squares.v1"
_REVISION_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_INPUT_SCALARS: Final = (
    "rho",
    "s_hat",
    "q",
    "q_prime_loc",
    "alpha_mhd",
    "p_prime_loc",
    "kappa",
    "delta",
    "s_kappa",
    "s_delta",
    "beta_e",
    "Z_eff",
    "xnue",
    "T_e_keV",
    "n_e_19",
    "R_major",
    "a_minor",
    "B_toroidal",
    "use_bper",
    "use_bpar",
)
_SPECIES_FIELDS: Final = tuple(asdict(TGLFSpecies("x", 1, 1, 1, 1, 1, 1)))
_FLUX_FIELDS: Final = (
    "species_index",
    "name",
    "charge_e",
    "particle_gb",
    "energy_gb",
    "momentum_gb",
    "exchange_gb",
)
_VIEW_FIELDS: Final = (
    "electron_species_index",
    "main_ion_species_index",
    "particle_e_gb",
    "particle_i_gb",
    "energy_e_gb",
    "energy_i_gb",
    "heat_effective_e_m2_s",
    "heat_effective_i_m2_s",
    "particle_effective_e_m2_s",
    "particle_effective_i_m2_s",
)


def tglf_species_input_payload(deck: TGLFInputDeck) -> dict[str, Any]:
    """Serialize one deck without the ambiguous legacy two-species fields."""
    render_tglf_input(deck)
    payload = {name: getattr(deck, name) for name in _INPUT_SCALARS}
    payload["species"] = [asdict(item) for item in deck.resolved_species()]
    return payload


def tglf_species_output_payload(output: TGLFOutput) -> dict[str, Any]:
    """Serialize canonical species fluxes plus the named compatibility view."""
    if len(output.species_fluxes) < 2:
        raise ValueError("Species-aware TGLF output requires canonical species_fluxes.")
    electron_indices = [item.species_index for item in output.species_fluxes if item.charge_e < 0]
    ion_indices = [item.species_index for item in output.species_fluxes if item.charge_e > 0]
    if electron_indices != [0] or not ion_indices:
        raise ValueError("TGLF output must contain one leading electron and at least one ion.")
    return {
        "rho": output.rho,
        "gamma_max": output.gamma_max,
        "species_fluxes": [asdict(item) for item in output.species_fluxes],
        "electron_main_ion_view": {
            "electron_species_index": 0,
            "main_ion_species_index": ion_indices[0],
            "particle_e_gb": output.particle_e,
            "particle_i_gb": output.particle_i,
            "energy_e_gb": output.q_e,
            "energy_i_gb": output.q_i,
            "heat_effective_e_m2_s": output.chi_e,
            "heat_effective_i_m2_s": output.chi_i,
            "particle_effective_e_m2_s": output.d_e,
            "particle_effective_i_m2_s": output.d_i,
        },
    }


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a portable identifier")
    return value


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _deck_from_payload(value: Any, label: str) -> TGLFInputDeck:
    payload = _object(value, label)
    expected = {*_INPUT_SCALARS, "species"}
    if set(payload) != expected:
        raise ValueError(f"{label} fields mismatch")
    species_raw = payload["species"]
    if not isinstance(species_raw, list):
        raise ValueError(f"{label}.species must be an array")
    for name in _INPUT_SCALARS:
        value = payload[name]
        if name in {"use_bper", "use_bpar"}:
            if not isinstance(value, bool):
                raise ValueError(f"{label}.{name} must be boolean")
        else:
            _finite_number(value, f"{label}.{name}")
    species: list[TGLFSpecies] = []
    for index, raw in enumerate(species_raw):
        item = _object(raw, f"{label}.species[{index}]")
        if set(item) != set(_SPECIES_FIELDS):
            raise ValueError(f"{label}.species[{index}] fields mismatch")
        try:
            species.append(TGLFSpecies(**item))
        except TypeError as exc:
            raise ValueError(f"{label}.species[{index}] is invalid: {exc}") from exc
    scalar = {name: payload[name] for name in _INPUT_SCALARS}
    try:
        deck = TGLFInputDeck(**scalar, species=tuple(species))
        render_tglf_input(deck)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid: {exc}") from exc
    return deck


def tglf_species_deck_from_payload(value: Mapping[str, Any]) -> TGLFInputDeck:
    """Deserialize and fully validate one canonical v2 input payload."""
    return _deck_from_payload(dict(value), "input")


def _validate_output(value: Any, deck: TGLFInputDeck, label: str) -> dict[str, Any]:
    payload = _object(value, label)
    if set(payload) != {"rho", "gamma_max", "species_fluxes", "electron_main_ion_view"}:
        raise ValueError(f"{label} fields mismatch")
    _finite_number(payload["rho"], f"{label}.rho")
    _finite_number(payload["gamma_max"], f"{label}.gamma_max")
    raw_fluxes = payload["species_fluxes"]
    species = deck.resolved_species()
    if not isinstance(raw_fluxes, list) or len(raw_fluxes) != len(species):
        raise ValueError(f"{label}.species_fluxes must contain exactly NS entries")
    for index, (raw, expected_species) in enumerate(zip(raw_fluxes, species, strict=True)):
        flux = _object(raw, f"{label}.species_fluxes[{index}]")
        if set(flux) != set(_FLUX_FIELDS):
            raise ValueError(f"{label}.species_fluxes[{index}] fields mismatch")
        if flux["species_index"] != index or flux["name"] != expected_species.name:
            raise ValueError(f"{label}.species_fluxes[{index}] species identity/order mismatch")
        if _finite_number(flux["charge_e"], f"{label}.species_fluxes[{index}].charge_e") != (
            expected_species.charge_e
        ):
            raise ValueError(f"{label}.species_fluxes[{index}] charge mismatch")
        for field_name in _FLUX_FIELDS[3:]:
            _finite_number(flux[field_name], f"{label}.species_fluxes[{index}].{field_name}")
    view = _object(payload["electron_main_ion_view"], f"{label}.electron_main_ion_view")
    if set(view) != set(_VIEW_FIELDS):
        raise ValueError(f"{label}.electron_main_ion_view fields mismatch")
    main_ion_index = next(index for index, item in enumerate(species) if item.charge_e > 0)
    if view["electron_species_index"] != 0 or view["main_ion_species_index"] != main_ion_index:
        raise ValueError(f"{label}.electron_main_ion_view index mismatch")
    for field_name in _VIEW_FIELDS[2:]:
        _finite_number(view[field_name], f"{label}.electron_main_ion_view.{field_name}")
    electron_flux = raw_fluxes[0]
    ion_flux = raw_fluxes[main_ion_index]
    exact_view = {
        "particle_e_gb": electron_flux["particle_gb"],
        "particle_i_gb": ion_flux["particle_gb"],
        "energy_e_gb": electron_flux["energy_gb"],
        "energy_i_gb": ion_flux["energy_gb"],
    }
    if any(view[name] != expected for name, expected in exact_view.items()):
        raise ValueError(f"{label}.electron_main_ion_view differs from canonical species fluxes")
    return payload


def _relative(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts or "\\" in value:
        raise ValueError(f"{label} must be a confined relative POSIX path")
    return path


def _file_contract(path: Path, *, with_path: str | None = None) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"not a regular non-symlink file: {path}")
    size = path.stat().st_size
    if size > MAX_TGLF_RAW_FILE_BYTES:
        raise ValueError(f"file exceeds {MAX_TGLF_RAW_FILE_BYTES} bytes: {path}")
    result = {"size_bytes": size, "sha256": sha256_file(path)}
    result["path" if with_path is not None else "name"] = with_path or path.name
    return result


def _raw_files(root: Path, sample_index: int, revision: str) -> tuple[str, list[dict[str, Any]]]:
    relative = PurePosixPath("runs") / f"sample_{sample_index:06d}"
    directory = root.joinpath(*relative.parts)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"missing regular sample directory: {relative}")
    files = sorted(
        (_file_contract(path) for path in directory.iterdir()), key=lambda row: row["name"]
    )
    names = {row["name"] for row in files}
    if not REQUIRED_TGLF_RAW_FILES.issubset(names):
        raise ValueError(f"sample {sample_index} is missing required GACODE raw files")
    version = (directory / "out.tglf.version").read_text(encoding="utf-8").splitlines()
    if not version or not version[0].startswith(revision[:8]):
        raise ValueError(f"sample {sample_index} GACODE version does not match revision")
    return version[0], files


def _paired_deck_signature(deck: TGLFInputDeck, species_index: int) -> tuple[Any, ...]:
    scalars = tuple(getattr(deck, name) for name in _INPUT_SCALARS)
    species = tuple(
        (
            item.name,
            item.mass_deuterium,
            item.charge_e,
            item.density_e_ratio,
            item.temperature_e_ratio,
            None if index == species_index else item.R_Ln,
            item.R_LT,
        )
        for index, item in enumerate(deck.resolved_species())
    )
    return scalars, species


def build_tglf_species_dataset_manifest(
    dataset_root: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    dataset_id: str,
    gacode_revision: str,
    seed: int,
    records_file: str = "dataset.json",
) -> dict[str, Any]:
    """Build a v2 manifest over retained ordered-species official runs."""
    root = Path(dataset_root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("dataset_root must be a regular directory")
    dataset_id = _identifier(dataset_id, "dataset_id")
    if _REVISION_RE.fullmatch(gacode_revision) is None:
        raise ValueError("gacode_revision must be an exact lower-case git object id")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not records or len(records) > MAX_TGLF_SAMPLES:
        raise ValueError("records must contain between 1 and MAX_TGLF_SAMPLES entries")
    records_relative = _relative(records_file, "records_file")
    records_path = root.joinpath(*records_relative.parts)
    records_contract = _file_contract(records_path, with_path=records_relative.as_posix())

    samples: list[dict[str, Any]] = []
    paired: dict[tuple[str, str], list[int]] = defaultdict(list)
    for position, raw_record in enumerate(records):
        record = dict(raw_record)
        index = record.get("sample_index")
        if index != position:
            raise ValueError("sample_index values must be contiguous and ordered")
        deck = _deck_from_payload(record.get("input"), f"record[{index}].input")
        output = _validate_output(record.get("output"), deck, f"record[{index}].output")
        group_id = _identifier(record.get("group_id"), f"record[{index}].group_id")
        regime = record.get("regime", "unclassified")
        stratum = record.get("sampling_stratum", "interior")
        if regime not in TGLF_REGIMES or stratum not in TGLF_SAMPLING_STRATA:
            raise ValueError(f"record[{index}] regime or sampling_stratum is invalid")
        paired_species = record.get("paired_gradient_species")
        if paired_species is not None:
            paired_species = _identifier(paired_species, f"record[{index}].paired_gradient_species")
            if paired_species not in {item.name for item in deck.resolved_species()}:
                raise ValueError(f"record[{index}] paired-gradient species is absent")
            paired[(group_id, paired_species)].append(index)
        sample_id = canonical_tglf_sample_id(
            cast(dict[str, object], record["input"]), gacode_revision
        )
        reported_version, files = _raw_files(root, index, gacode_revision)
        samples.append(
            {
                "sample_index": index,
                "sample_id": sample_id,
                "group_id": group_id,
                "split": deterministic_tglf_split(group_id, seed),
                "sampling_stratum": stratum,
                "regime": regime,
                "reported_version": reported_version,
                "run_directory": f"runs/sample_{index:06d}",
                "input": record["input"],
                "output": output,
                "raw_files": files,
            }
        )
    if len({sample["sample_id"] for sample in samples}) != len(samples):
        raise ValueError("duplicate TGLF input decks are forbidden")

    paired_groups: list[dict[str, Any]] = []
    for (group_id, species_name), indices in sorted(paired.items()):
        if len(indices) < 3:
            raise ValueError("paired-gradient groups require at least three samples")
        decks = [_deck_from_payload(samples[index]["input"], "sample.input") for index in indices]
        species_index = next(
            index
            for index, item in enumerate(decks[0].resolved_species())
            if item.name == species_name
        )
        signature = _paired_deck_signature(decks[0], species_index)
        if any(_paired_deck_signature(deck, species_index) != signature for deck in decks[1:]):
            raise ValueError("paired-gradient groups may change only the target species R/L_n")
        gradients = [
            deck.resolved_species()[species_index].R_Ln * deck.a_minor / deck.R_major
            for deck in decks
        ]
        if len(set(gradients)) < 3:
            raise ValueError("paired-gradient groups require at least three distinct a/L_n values")
        paired_groups.append(
            {
                "group_id": group_id,
                "species_name": species_name,
                "species_index": species_index,
                "sample_indices": indices,
                "gradients_a_over_l": gradients,
                "method": TGLF_PARTICLE_TRANSPORT_METHOD,
            }
        )
    return {
        "SPDX-License-Identifier": "AGPL-3.0-or-later",
        "schema_version": TGLF_SPECIES_DATASET_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "source": {
            "solver": "GACODE TGLF",
            "repository": TGLF_SOURCE_REPOSITORY,
            "revision": gacode_revision,
            "licence": TGLF_SOURCE_LICENCE,
            "executable": "tglf",
            "path_resolution_required": True,
            "reported_versions": sorted({sample["reported_version"] for sample in samples}),
        },
        "split_policy": {
            "method": "sha256-group-modulo-10000.v1",
            "seed": seed,
            "group_isolation_required": True,
        },
        "output_contract": {
            "gbflux_layout": TGLF_GBFLUX_LAYOUT,
            "species_order_preserved": True,
            "signed_fluxes_preserved": True,
            "legacy_scalar_view_is_derived": True,
            "one_point_particle_coefficients_are_effective_only": True,
        },
        "records": records_contract,
        "claims": {
            "official_gacode_outputs": True,
            "surrogate_promoted": False,
            "experimental_validation": False,
            "cross_solver_parity": False,
        },
        "paired_gradient_groups": paired_groups,
        "samples": samples,
    }


def write_tglf_species_dataset_manifest(
    dataset_root: str | Path, manifest: Mapping[str, Any]
) -> Path:
    """Atomically write one bounded v2 manifest."""
    root = Path(dataset_root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("dataset_root must be a regular directory")
    payload = json.dumps(dict(manifest), allow_nan=False, indent=2, sort_keys=True) + "\n"
    if len(payload.encode()) > MAX_TGLF_MANIFEST_BYTES:
        raise ValueError("manifest exceeds the hard byte limit")
    temporary = root / ".manifest.json.tmp"
    temporary.write_text(payload, encoding="utf-8")
    destination = root / "manifest.json"
    temporary.replace(destination)
    return destination


def _failure(failures: list[str], dataset_id: str | None = None) -> dict[str, Any]:
    return {"status": "failed", "dataset_id": dataset_id, "failures": failures}


def verify_tglf_species_dataset(dataset_root: str | Path) -> dict[str, Any]:
    """Rebuild and compare a v2 manifest, then verify its complete file inventory."""
    root = Path(dataset_root)
    manifest_path = root / "manifest.json"
    if not root.is_dir() or root.is_symlink() or not manifest_path.is_file():
        return _failure(["dataset root or manifest is missing or symlinked"])
    if manifest_path.stat().st_size > MAX_TGLF_MANIFEST_BYTES:
        return _failure(["manifest exceeds the hard byte limit"])
    try:
        manifest = checked_json_load(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _failure([f"manifest is unreadable: {exc}"])
    if not isinstance(manifest, dict):
        return _failure(["manifest must be an object"])
    dataset_id = manifest.get("dataset_id") if isinstance(manifest.get("dataset_id"), str) else None
    failures: list[str] = []
    if manifest.get("schema_version") != TGLF_SPECIES_DATASET_SCHEMA_VERSION:
        failures.append(f"schema_version must be {TGLF_SPECIES_DATASET_SCHEMA_VERSION}")
    records_spec = manifest.get("records")
    source = manifest.get("source")
    split = manifest.get("split_policy")
    samples = manifest.get("samples")
    if (
        not isinstance(records_spec, dict)
        or not isinstance(source, dict)
        or not isinstance(split, dict)
    ):
        return _failure(failures + ["manifest provenance objects are invalid"], dataset_id)
    if not isinstance(samples, list) or not samples or len(samples) > MAX_TGLF_SAMPLES:
        return _failure(failures + ["manifest samples are invalid"], dataset_id)
    try:
        records_relative = _relative(records_spec.get("path"), "records.path")
        records_path = root.joinpath(*records_relative.parts)
        if records_path.stat().st_size > MAX_TGLF_RECORDS_BYTES:
            raise ValueError("records file exceeds the hard byte limit")
        if records_spec.get("size_bytes") != records_path.stat().st_size:
            failures.append("records size mismatch")
        if records_spec.get("sha256") != sha256_file(records_path):
            failures.append("records SHA-256 mismatch")
        records = checked_json_load(records_path)
        if not isinstance(records, list):
            raise ValueError("records file must contain an array")
        rebuilt = build_tglf_species_dataset_manifest(
            root,
            records,
            dataset_id=_identifier(dataset_id, "dataset_id"),
            gacode_revision=cast(str, source.get("revision")),
            seed=cast(int, split.get("seed")),
            records_file=records_relative.as_posix(),
        )
        if rebuilt != manifest:
            failures.append("manifest differs from canonical rebuild")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        failures.append(str(exc))
    if failures:
        return _failure(failures, dataset_id)
    split_counts = {role: 0 for role in ("train", "calibration", "test")}
    for sample in samples:
        split_counts[sample["split"]] += 1
    return {
        "status": "passed",
        "dataset_id": dataset_id,
        "schema_version": TGLF_SPECIES_DATASET_SCHEMA_VERSION,
        "samples_verified": len(samples),
        "species_counts": sorted({len(sample["input"]["species"]) for sample in samples}),
        "paired_gradient_groups_verified": len(manifest["paired_gradient_groups"]),
        "split_counts": split_counts,
        "failures": [],
    }


__all__ = [
    "TGLF_GBFLUX_LAYOUT",
    "TGLF_PARTICLE_TRANSPORT_METHOD",
    "TGLF_SPECIES_DATASET_SCHEMA_VERSION",
    "build_tglf_species_dataset_manifest",
    "tglf_species_input_payload",
    "tglf_species_deck_from_payload",
    "tglf_species_output_payload",
    "verify_tglf_species_dataset",
    "write_tglf_species_dataset_manifest",
]
