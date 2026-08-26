# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TGLF Model-Selection Data Contract
"""Verified corpus loading and fixed-width TGLF selection representation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.io.safe_loaders import checked_json_load
from scpn_fusion.io.tglf_development_corpus import verify_tglf_development_corpus

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

SPLITS: Final = ("train", "calibration", "test")
STRATA: Final = ("interior", "boundary", "threshold")
COMPOSITIONS: Final = (
    "electron-deuterium",
    "electron-deuterium-tritium",
    "electron-deuterium-carbon",
)
GLOBAL_FIELDS: Final = (
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
SPECIES_FIELDS: Final = (
    "R_LT",
    "R_Ln",
    "charge_e",
    "mass_deuterium",
    "density_e_ratio",
    "temperature_e_ratio",
)
FLUX_FIELDS: Final = ("particle_gb", "energy_gb", "momentum_gb", "exchange_gb")
SPECIES_SLOTS: Final = 3


@dataclass(frozen=True)
class TGLFModelStudyData:
    """Validated matrices and immutable split metadata for one TGLF study."""

    features: FloatArray
    targets: FloatArray
    active_targets: BoolArray
    charges: FloatArray
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    splits: tuple[str, ...]
    strata: tuple[str, ...]
    compositions: tuple[str, ...]
    groups: tuple[str, ...]
    sample_indices: tuple[int, ...]
    verification: dict[str, Any]


def finite_number(value: Any, label: str) -> float:
    """Return a finite scalar while rejecting booleans and non-numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def object_value(value: Any, label: str) -> dict[str, Any]:
    """Return a typed JSON object or fail with its contract label."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def feature_names() -> tuple[str, ...]:
    """Return the frozen ordered 44-column feature vocabulary."""
    names = list(GLOBAL_FIELDS)
    names.extend(f"composition.{name}" for name in COMPOSITIONS)
    for slot in range(SPECIES_SLOTS):
        names.append(f"species_{slot}.present")
        names.extend(f"species_{slot}.{field}" for field in SPECIES_FIELDS)
    return tuple(names)


def target_names() -> tuple[str, ...]:
    """Return the frozen ordered 13-column target vocabulary."""
    names = ["gamma_max"]
    for slot in range(SPECIES_SLOTS):
        names.extend(f"species_{slot}.{field}" for field in FLUX_FIELDS)
    return tuple(names)


def record_row(
    record: dict[str, Any],
    *,
    row_index: int,
) -> tuple[list[float], list[float], list[bool], list[float]]:
    """Map one ordered-species record into fixed feature and target rows."""
    input_payload = object_value(record.get("input"), f"record[{row_index}].input")
    output_payload = object_value(record.get("output"), f"record[{row_index}].output")
    composition = record.get("composition")
    if composition not in COMPOSITIONS:
        raise ValueError(f"record[{row_index}].composition is invalid")
    row_x: list[float] = []
    for field in GLOBAL_FIELDS:
        value = input_payload.get(field)
        if field in {"use_bper", "use_bpar"}:
            if not isinstance(value, bool):
                raise ValueError(f"record[{row_index}].input.{field} must be boolean")
            row_x.append(float(value))
        else:
            row_x.append(finite_number(value, f"record[{row_index}].input.{field}"))
    row_x.extend(float(composition == name) for name in COMPOSITIONS)

    species_raw = input_payload.get("species")
    fluxes_raw = output_payload.get("species_fluxes")
    if not isinstance(species_raw, list) or not 2 <= len(species_raw) <= SPECIES_SLOTS:
        raise ValueError(f"record[{row_index}] must contain two or three species")
    if not isinstance(fluxes_raw, list) or len(fluxes_raw) != len(species_raw):
        raise ValueError(f"record[{row_index}] flux count differs from species count")

    row_y = [finite_number(output_payload.get("gamma_max"), f"record[{row_index}].gamma_max")]
    active = [True]
    charges: list[float] = []
    for slot in range(SPECIES_SLOTS):
        if slot < len(species_raw):
            species = object_value(species_raw[slot], f"record[{row_index}].species[{slot}]")
            flux = object_value(fluxes_raw[slot], f"record[{row_index}].flux[{slot}]")
            if flux.get("species_index") != slot or flux.get("name") != species.get("name"):
                raise ValueError(f"record[{row_index}] species/flux ordering differs")
            row_x.append(1.0)
            row_x.extend(
                finite_number(species.get(field), f"record[{row_index}].species[{slot}].{field}")
                for field in SPECIES_FIELDS
            )
            charges.append(
                finite_number(
                    species.get("charge_e"),
                    f"record[{row_index}].species[{slot}].charge_e",
                )
            )
            row_y.extend(
                finite_number(flux.get(field), f"record[{row_index}].flux[{slot}].{field}")
                for field in FLUX_FIELDS
            )
            active.extend([True] * len(FLUX_FIELDS))
        else:
            row_x.extend([0.0] * (1 + len(SPECIES_FIELDS)))
            charges.append(0.0)
            row_y.extend([0.0] * len(FLUX_FIELDS))
            active.extend([False] * len(FLUX_FIELDS))
    return row_x, row_y, active, charges


def load_tglf_model_study_data(dataset_root: str | Path) -> TGLFModelStudyData:
    """Verify an official development corpus and build fixed study matrices."""
    root = Path(dataset_root)
    verification = verify_tglf_development_corpus(root)
    if verification.get("status") != "passed" or verification.get("plan_replay") is not True:
        raise ValueError(f"TGLF development corpus verification failed: {verification}")
    records_raw = checked_json_load(root / "dataset.json")
    manifest_raw = checked_json_load(root / "manifest.json")
    if not isinstance(records_raw, list) or not isinstance(manifest_raw, dict):
        raise ValueError("TGLF records/manifest roots are invalid")
    samples_raw = manifest_raw.get("samples")
    if not isinstance(samples_raw, list) or len(samples_raw) != len(records_raw):
        raise ValueError("TGLF manifest samples differ from records")

    rows_x: list[list[float]] = []
    rows_y: list[list[float]] = []
    active_rows: list[list[bool]] = []
    charge_rows: list[list[float]] = []
    splits: list[str] = []
    strata: list[str] = []
    compositions: list[str] = []
    groups: list[str] = []
    group_splits: dict[str, str] = {}
    for row_index, (record_raw, sample_raw) in enumerate(
        zip(records_raw, samples_raw, strict=True)
    ):
        record = object_value(record_raw, f"record[{row_index}]")
        sample = object_value(sample_raw, f"manifest.samples[{row_index}]")
        if sample.get("sample_index") != row_index or record.get("sample_index") != row_index:
            raise ValueError("TGLF sample indices are not canonical and contiguous")
        for field in ("group_id", "sampling_stratum", "composition"):
            if record.get(field) != sample.get(field):
                raise ValueError(f"TGLF record/manifest {field} differs at row {row_index}")
        split = sample.get("split")
        stratum = sample.get("sampling_stratum")
        composition = sample.get("composition")
        group = sample.get("group_id")
        if split not in SPLITS or stratum not in STRATA or composition not in COMPOSITIONS:
            raise ValueError(f"TGLF categorical metadata is invalid at row {row_index}")
        if not isinstance(group, str) or not group:
            raise ValueError(f"TGLF group_id is invalid at row {row_index}")
        prior = group_splits.setdefault(group, split)
        if prior != split:
            raise ValueError(f"TGLF group {group} crosses split roles")
        row_x, row_y, active, row_charges = record_row(record, row_index=row_index)
        rows_x.append(row_x)
        rows_y.append(row_y)
        active_rows.append(active)
        charge_rows.append(row_charges)
        splits.append(split)
        strata.append(stratum)
        compositions.append(composition)
        groups.append(group)

    split_counts = {name: splits.count(name) for name in SPLITS}
    if any(count == 0 for count in split_counts.values()):
        raise ValueError(
            f"TGLF model selection requires non-empty train/calibration/test splits: {split_counts}"
        )
    features = np.asarray(rows_x, dtype=np.float64)
    targets = np.asarray(rows_y, dtype=np.float64)
    active_targets = np.asarray(active_rows, dtype=np.bool_)
    charge_matrix = np.asarray(charge_rows, dtype=np.float64)
    expected_shape = (len(records_raw), len(feature_names()))
    if features.shape != expected_shape:
        raise ValueError(f"TGLF feature matrix shape {features.shape} != {expected_shape}")
    if targets.shape != active_targets.shape or targets.shape[1] != len(target_names()):
        raise ValueError("TGLF target/mask matrix shape mismatch")
    if charge_matrix.shape != (len(records_raw), SPECIES_SLOTS):
        raise ValueError("TGLF charge matrix shape mismatch")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
        raise ValueError("TGLF study matrices contain non-finite values")
    return TGLFModelStudyData(
        features=features,
        targets=targets,
        active_targets=active_targets,
        charges=charge_matrix,
        feature_names=feature_names(),
        target_names=target_names(),
        splits=tuple(splits),
        strata=tuple(strata),
        compositions=tuple(compositions),
        groups=tuple(groups),
        sample_indices=tuple(range(len(records_raw))),
        verification=dict(verification),
    )


__all__ = ["TGLFModelStudyData", "load_tglf_model_study_data"]
