# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — GACODE TGLF Dataset Contract
"""Versioned custody and split validation for official GACODE TGLF runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Final, cast

from scpn_fusion.io.safe_loaders import checked_json_load

TGLF_DATASET_SCHEMA_VERSION: Final = "scpn-fusion.tglf-gacode-dataset.v1"
TGLF_SOURCE_REPOSITORY: Final = "https://github.com/gafusion/gacode"
TGLF_SOURCE_LICENCE: Final = "Apache-2.0"
TGLF_SPLIT_METHOD: Final = "sha256-group-modulo-10000.v1"
TGLF_SPLIT_FRACTIONS: Final[dict[str, int]] = {
    "train": 6000,
    "calibration": 2000,
    "test": 2000,
}
MAX_TGLF_MANIFEST_BYTES: Final = 32 * 1024 * 1024
MAX_TGLF_RECORDS_BYTES: Final = 32 * 1024 * 1024
MAX_TGLF_RAW_FILE_BYTES: Final = 256 * 1024 * 1024
MAX_TGLF_SAMPLES: Final = 1_000_000
TGLF_REGIMES: Final = frozenset({"stable", "ITG", "TEM", "ETG", "mixed", "unclassified"})
TGLF_SAMPLING_STRATA: Final = frozenset({"interior", "boundary", "threshold"})
TGLF_DATASET_PURPOSES: Final = frozenset({"pilot", "development"})
REQUIRED_TGLF_RAW_FILES: Final = frozenset(
    {
        "input.tglf",
        "out.tglf.gbflux",
        "out.tglf.eigenvalue_spectrum",
        "out.tglf.ky_spectrum",
        "out.tglf.run",
        "out.tglf.version",
    }
)
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _number_domain(
    unit: str,
    minimum: float,
    maximum: float,
    sampling: str,
) -> dict[str, Any]:
    return {
        "type": "number",
        "unit": unit,
        "minimum": minimum,
        "maximum": maximum,
        "sampling": sampling,
    }


def _boolean_domain(value: bool) -> dict[str, Any]:
    return {"type": "boolean", "unit": "1", "value": value, "sampling": "fixed"}


TGLF_INPUT_DOMAINS: Final[dict[str, dict[str, Any]]] = {
    "rho": _number_domain("1", 0.5, 0.5, "fixed"),
    "s_hat": _number_domain("1", 0.0, 3.0, "uniform"),
    "q": _number_domain("1", 1.0, 5.0, "uniform"),
    "q_prime_loc": _number_domain("1", 0.0, 0.0, "fixed"),
    "alpha_mhd": _number_domain("1", 0.0, 0.0, "fixed"),
    "p_prime_loc": _number_domain("1", 0.0, 0.0, "fixed"),
    "kappa": _number_domain("1", 1.7, 1.7, "fixed"),
    "delta": _number_domain("1", 0.3, 0.3, "fixed"),
    "s_kappa": _number_domain("1", 0.0, 0.0, "fixed"),
    "s_delta": _number_domain("1", 0.0, 0.0, "fixed"),
    "R_LTi": _number_domain("1", 0.0, 12.0, "uniform"),
    "R_LTe": _number_domain("1", 0.0, 12.0, "uniform"),
    "R_Lne": _number_domain("1", 0.0, 5.0, "uniform"),
    "R_Lni": _number_domain("1", 2.0, 2.0, "fixed"),
    "beta_e": _number_domain("1", 0.001, 0.05, "uniform"),
    "Z_eff": _number_domain("1", 1.0, 3.0, "uniform"),
    "xnue": _number_domain("1", 0.0, 0.0, "fixed"),
    "T_e_keV": _number_domain("keV", 10.0, 10.0, "fixed"),
    "T_i_keV": _number_domain("keV", 10.0, 10.0, "fixed"),
    "n_e_19": _number_domain("1e19 m^-3", 8.0, 8.0, "fixed"),
    "R_major": _number_domain("m", 6.2, 6.2, "fixed"),
    "a_minor": _number_domain("m", 2.0, 2.0, "fixed"),
    "B_toroidal": _number_domain("T", 5.3, 5.3, "fixed"),
    "use_bper": _boolean_domain(False),
    "use_bpar": _boolean_domain(False),
}

TGLF_OUTPUT_UNITS: Final[dict[str, str]] = {
    "rho": "1",
    "chi_i": "m^2 s^-1",
    "chi_e": "m^2 s^-1",
    "gamma_max": "c_s/a",
    "q_i": "gyro-Bohm",
    "q_e": "gyro-Bohm",
    "particle_e": "gyro-Bohm",
    "particle_i": "gyro-Bohm",
    "d_e": "m^2 s^-1",
    "d_i": "m^2 s^-1",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file without buffering it all."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_tglf_sample_id(input_payload: Mapping[str, object], revision: str) -> str:
    """Return the immutable identity of one input deck and GACODE revision.

    Parameters
    ----------
    input_payload : Mapping[str, object]
        Complete JSON-compatible ``TGLFInputDeck`` field map.
    revision : str
        Exact 40-character GACODE git revision.

    Returns
    -------
    str
        Lower-case SHA-256 digest over revision and canonical input JSON.

    Raises
    ------
    ValueError
        If the revision is not an exact lower-case git object identifier or the
        input cannot be represented as finite strict JSON.
    """
    if _REVISION_RE.fullmatch(revision) is None:
        raise ValueError("GACODE revision must be a 40-character lower-case hexadecimal digest")
    try:
        encoded = json.dumps(
            dict(input_payload),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"TGLF input is not finite canonical JSON: {exc}") from exc
    return hashlib.sha256(revision.encode("ascii") + b"\n" + encoded).hexdigest()


def deterministic_tglf_split(group_id: str, seed: int) -> str:
    """Assign one related-run group to a stable train/calibration/test role.

    Parameters
    ----------
    group_id : str
        Stable identifier shared by all perturbations that must remain together.
    seed : int
        Non-negative split seed recorded in the dataset manifest.

    Returns
    -------
    str
        One of ``train``, ``calibration`` or ``test``.
    """
    if _IDENTIFIER_RE.fullmatch(group_id) is None:
        raise ValueError("group_id must be a non-empty portable identifier")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("split seed must be a non-negative integer")
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big") % 10_000
    if bucket < TGLF_SPLIT_FRACTIONS["train"]:
        return "train"
    if bucket < TGLF_SPLIT_FRACTIONS["train"] + TGLF_SPLIT_FRACTIONS["calibration"]:
        return "calibration"
    return "test"


def _as_object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _require_identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a portable identifier")
    return value


def _validate_input(input_payload: dict[str, Any], failures: list[str], *, label: str) -> None:
    expected = set(TGLF_INPUT_DOMAINS)
    if set(input_payload) != expected:
        failures.append(
            f"{label} fields mismatch; extra={sorted(set(input_payload) - expected)}, "
            f"missing={sorted(expected - set(input_payload))}"
        )
        return
    for name, domain in TGLF_INPUT_DOMAINS.items():
        value = input_payload[name]
        if domain["type"] == "boolean":
            if not isinstance(value, bool) or value is not domain["value"]:
                failures.append(f"{label}.{name} must equal {domain['value']!r}")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            failures.append(f"{label}.{name} must be numeric")
            continue
        numeric = float(value)
        minimum = float(domain["minimum"])
        maximum = float(domain["maximum"])
        if not math.isfinite(numeric) or numeric < minimum or numeric > maximum:
            failures.append(f"{label}.{name} is outside [{minimum}, {maximum}]")


def _validate_output(output_payload: dict[str, Any], failures: list[str], *, label: str) -> None:
    expected = set(TGLF_OUTPUT_UNITS)
    if set(output_payload) != expected:
        failures.append(
            f"{label} fields mismatch; extra={sorted(set(output_payload) - expected)}, "
            f"missing={sorted(expected - set(output_payload))}"
        )
        return
    for name, value in output_payload.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            failures.append(f"{label}.{name} must be numeric")
        elif not math.isfinite(float(value)):
            failures.append(f"{label}.{name} must be finite")


def _safe_relative_path(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or "\\" in value:
        raise ValueError(f"{label} must be a confined relative POSIX path")
    return path


def _regular_file_contract(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"not a regular non-symlink file: {path}")
    size = path.stat().st_size
    if size > MAX_TGLF_RAW_FILE_BYTES:
        raise ValueError(f"raw file exceeds {MAX_TGLF_RAW_FILE_BYTES} bytes: {path}")
    return {"name": path.name, "size_bytes": size, "sha256": sha256_file(path)}


def _sample_manifest(
    root: Path,
    record: dict[str, Any],
    *,
    revision: str,
    split_seed: int,
) -> dict[str, Any]:
    index = record.get("sample_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("record.sample_index must be a non-negative integer")
    input_payload = _as_object(record.get("input"), label=f"record[{index}].input")
    output_payload = _as_object(record.get("output"), label=f"record[{index}].output")
    failures: list[str] = []
    _validate_input(input_payload, failures, label=f"record[{index}].input")
    _validate_output(output_payload, failures, label=f"record[{index}].output")
    if failures:
        raise ValueError("; ".join(failures))

    sample_id = canonical_tglf_sample_id(input_payload, revision)
    group_id = _require_identifier(record.get("group_id", sample_id), label="record.group_id")
    regime = record.get("regime", "unclassified")
    if regime not in TGLF_REGIMES:
        raise ValueError(f"record[{index}].regime is invalid")
    stratum = record.get("sampling_stratum", "interior")
    if stratum not in TGLF_SAMPLING_STRATA:
        raise ValueError(f"record[{index}].sampling_stratum is invalid")

    relative_dir = PurePosixPath("runs") / f"sample_{index:06d}"
    sample_dir = root.joinpath(*relative_dir.parts)
    if not sample_dir.is_dir() or sample_dir.is_symlink():
        raise ValueError(f"missing regular sample directory: {relative_dir.as_posix()}")
    files = sorted(
        (_regular_file_contract(path) for path in sample_dir.iterdir()),
        key=lambda item: cast(str, item["name"]),
    )
    actual_names = {cast(str, item["name"]) for item in files}
    missing = REQUIRED_TGLF_RAW_FILES - actual_names
    if missing:
        raise ValueError(f"sample {index} is missing required raw files: {sorted(missing)}")
    version_lines = (
        (sample_dir / "out.tglf.version").read_text(encoding="utf-8", errors="strict").splitlines()
    )
    if not version_lines or not version_lines[0].startswith(revision[:8]):
        raise ValueError(f"sample {index} GACODE version does not match revision {revision}")
    return {
        "sample_index": index,
        "sample_id": sample_id,
        "group_id": group_id,
        "split": deterministic_tglf_split(group_id, split_seed),
        "sampling_stratum": stratum,
        "regime": regime,
        "reported_version": version_lines[0],
        "run_directory": relative_dir.as_posix(),
        "input": input_payload,
        "output": output_payload,
        "raw_files": files,
    }


def build_tglf_dataset_manifest(
    dataset_root: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    dataset_id: str,
    gacode_revision: str,
    seed: int,
    purpose: str = "pilot",
    records_file: str = "dataset.json",
) -> dict[str, Any]:
    """Build a manifest from retained official-GACODE run directories.

    Parameters
    ----------
    dataset_root : str | Path
        Directory containing ``records_file`` and ``runs/sample_NNNNNN``.
    records : Sequence[Mapping[str, Any]]
        Parsed records produced by :class:`TGLFDatasetGenerator`.
    dataset_id : str
        Stable portable identifier chosen before generation.
    gacode_revision : str
        Exact 40-character GACODE git revision used for every run.
    seed : int
        Non-negative generation and deterministic split seed.
    purpose : str, optional
        ``pilot`` for bounded activation data or ``development`` for a corpus
        required to include interior, boundary and threshold sampling strata.
    records_file : str, optional
        Dataset-root-relative JSON file containing ``records``.

    Returns
    -------
    dict[str, Any]
        JSON-compatible TGLF dataset manifest.

    Raises
    ------
    ValueError
        If provenance, records, retained files, domains, strata or identifiers
        violate the versioned contract.
    """
    root = Path(dataset_root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"dataset root must be a regular directory: {root}")
    dataset_id = _require_identifier(dataset_id, label="dataset_id")
    if _REVISION_RE.fullmatch(gacode_revision) is None:
        raise ValueError("gacode_revision must be a 40-character lower-case hexadecimal digest")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if purpose not in TGLF_DATASET_PURPOSES:
        raise ValueError(f"purpose must be one of {sorted(TGLF_DATASET_PURPOSES)}")
    if not records or len(records) > MAX_TGLF_SAMPLES:
        raise ValueError(f"records must contain between 1 and {MAX_TGLF_SAMPLES} samples")
    records_relative = _safe_relative_path(records_file, label="records_file")
    records_path = root.joinpath(*records_relative.parts)
    records_contract = _regular_file_contract(records_path)
    records_contract["path"] = records_relative.as_posix()
    del records_contract["name"]

    samples = [
        _sample_manifest(root, dict(record), revision=gacode_revision, split_seed=seed)
        for record in records
    ]
    indices = [cast(int, item["sample_index"]) for item in samples]
    sample_ids = [cast(str, item["sample_id"]) for item in samples]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate sample_index values are forbidden")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate TGLF input decks are forbidden")
    if indices != list(range(len(samples))):
        raise ValueError("sample_index values must be contiguous and records must be ordered")
    present_strata = {cast(str, item["sampling_stratum"]) for item in samples}
    required_strata = {"interior"} if purpose == "pilot" else set(TGLF_SAMPLING_STRATA)
    if not required_strata.issubset(present_strata):
        raise ValueError(
            f"{purpose} dataset is missing sampling strata: {sorted(required_strata - present_strata)}"
        )
    versions = sorted({cast(str, item["reported_version"]) for item in samples})
    return {
        "SPDX-License-Identifier": "AGPL-3.0-or-later",
        "schema_version": TGLF_DATASET_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "purpose": purpose,
        "source": {
            "solver": "GACODE TGLF",
            "repository": TGLF_SOURCE_REPOSITORY,
            "revision": gacode_revision,
            "licence": TGLF_SOURCE_LICENCE,
            "executable": "tglf",
            "path_resolution_required": True,
            "reported_versions": versions,
        },
        "generation": {
            "seed": seed,
            "accepted_samples": len(samples),
            "failed_samples": 0,
            "input_domains": {name: dict(domain) for name, domain in TGLF_INPUT_DOMAINS.items()},
            "sampling_strata": sorted(present_strata),
            "required_strata": sorted(required_strata),
        },
        "split_policy": {
            "method": TGLF_SPLIT_METHOD,
            "seed": seed,
            "bucket_counts": dict(TGLF_SPLIT_FRACTIONS),
            "group_isolation_required": True,
        },
        "output_contract": {
            "units": dict(TGLF_OUTPUT_UNITS),
            "signed_fluxes_preserved": True,
        },
        "records": records_contract,
        "claims": {
            "official_gacode_outputs": True,
            "surrogate_promoted": False,
            "experimental_validation": False,
            "cross_solver_parity": False,
        },
        "samples": samples,
    }


def write_tglf_dataset_manifest(
    dataset_root: str | Path,
    manifest: Mapping[str, Any],
) -> Path:
    """Atomically write ``manifest.json`` inside one dataset root.

    Parameters
    ----------
    dataset_root : str | Path
        Existing non-symlink dataset directory.
    manifest : Mapping[str, Any]
        Manifest produced by :func:`build_tglf_dataset_manifest`.

    Returns
    -------
    Path
        Written ``manifest.json`` path.
    """
    root = Path(dataset_root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"dataset root must be a regular directory: {root}")
    path = root / "manifest.json"
    temporary = root / ".manifest.json.tmp"
    payload = json.dumps(dict(manifest), allow_nan=False, indent=2, sort_keys=True) + "\n"
    if len(payload.encode("utf-8")) > MAX_TGLF_MANIFEST_BYTES:
        raise ValueError(f"manifest exceeds {MAX_TGLF_MANIFEST_BYTES} bytes")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return path


def _verify_file_contract(
    path: Path,
    spec: dict[str, Any],
    failures: list[str],
    *,
    label: str,
) -> None:
    size = spec.get("size_bytes")
    digest = spec.get("sha256")
    if not path.is_file() or path.is_symlink():
        failures.append(f"{label} is missing or is not a regular file")
        return
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        failures.append(f"{label}.size_bytes is invalid")
        return
    if size > MAX_TGLF_RAW_FILE_BYTES or path.stat().st_size != size:
        failures.append(f"{label} size mismatch or hard-limit violation")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        failures.append(f"{label}.sha256 is invalid")
    elif sha256_file(path) != digest:
        failures.append(f"{label} SHA-256 mismatch")


def _failure_result(failures: list[str], *, dataset_id: str | None = None) -> dict[str, Any]:
    return {"status": "failed", "dataset_id": dataset_id, "failures": failures}


def verify_tglf_dataset(
    dataset_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify an official-GACODE TGLF dataset manifest and retained raw runs.

    Parameters
    ----------
    dataset_root : str | Path
        Dataset directory that confines records and all sample run directories.
    manifest_path : str | Path, optional
        Manifest path. Defaults to ``dataset_root/manifest.json`` and must remain
        inside ``dataset_root``.

    Returns
    -------
    dict[str, Any]
        Stable ``passed`` or ``failed`` envelope with every detected failure.
    """
    root = Path(dataset_root)
    if not root.is_dir() or root.is_symlink():
        return _failure_result([f"dataset root is missing or symlinked: {root}"])
    root_resolved = root.resolve()
    manifest_file = root / "manifest.json" if manifest_path is None else Path(manifest_path)
    try:
        if manifest_file.resolve().parent != root_resolved:
            return _failure_result(["manifest must be a direct child of dataset_root"])
    except OSError as exc:
        return _failure_result([f"cannot resolve manifest path: {exc}"])
    if not manifest_file.is_file() or manifest_file.is_symlink():
        return _failure_result([f"manifest is missing or symlinked: {manifest_file}"])
    try:
        manifest = _as_object(
            checked_json_load(manifest_file, max_bytes=MAX_TGLF_MANIFEST_BYTES),
            label="manifest",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _failure_result([f"cannot load manifest: {exc}"])

    failures: list[str] = []
    dataset_id_raw = manifest.get("dataset_id")
    dataset_id = dataset_id_raw if isinstance(dataset_id_raw, str) else None
    expected_top = {
        "SPDX-License-Identifier",
        "schema_version",
        "dataset_id",
        "purpose",
        "source",
        "generation",
        "split_policy",
        "output_contract",
        "records",
        "claims",
        "samples",
    }
    if set(manifest) != expected_top:
        failures.append("manifest top-level fields do not match the v1 contract")
    if manifest.get("SPDX-License-Identifier") != "AGPL-3.0-or-later":
        failures.append("manifest SPDX identifier mismatch")
    if manifest.get("schema_version") != TGLF_DATASET_SCHEMA_VERSION:
        failures.append(f"schema_version must be {TGLF_DATASET_SCHEMA_VERSION}")
    try:
        _require_identifier(dataset_id_raw, label="dataset_id")
    except ValueError as exc:
        failures.append(str(exc))
    purpose = manifest.get("purpose")
    if purpose not in TGLF_DATASET_PURPOSES:
        failures.append(f"purpose must be one of {sorted(TGLF_DATASET_PURPOSES)}")

    try:
        source = _as_object(manifest.get("source"), label="source")
        generation = _as_object(manifest.get("generation"), label="generation")
        split_policy = _as_object(manifest.get("split_policy"), label="split_policy")
        output_contract = _as_object(manifest.get("output_contract"), label="output_contract")
        records_spec = _as_object(manifest.get("records"), label="records")
        claims = _as_object(manifest.get("claims"), label="claims")
    except ValueError as exc:
        failures.append(str(exc))
        return _failure_result(failures, dataset_id=dataset_id)

    revision = source.get("revision")
    if (
        source.get("solver") != "GACODE TGLF"
        or source.get("repository") != TGLF_SOURCE_REPOSITORY
        or source.get("licence") != TGLF_SOURCE_LICENCE
        or source.get("executable") != "tglf"
        or source.get("path_resolution_required") is not True
        or not isinstance(revision, str)
        or _REVISION_RE.fullmatch(revision) is None
    ):
        failures.append("source provenance does not match the official PATH-resolved contract")
        revision = ""
    if generation.get("input_domains") != TGLF_INPUT_DOMAINS:
        failures.append("generation.input_domains differs from the frozen v1 domains")
    seed = generation.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        failures.append("generation.seed must be a non-negative integer")
        seed = -1
    if (
        split_policy.get("method") != TGLF_SPLIT_METHOD
        or split_policy.get("seed") != seed
        or split_policy.get("bucket_counts") != TGLF_SPLIT_FRACTIONS
        or split_policy.get("group_isolation_required") is not True
    ):
        failures.append("split_policy differs from the frozen group-isolated v1 policy")
    if output_contract != {"units": TGLF_OUTPUT_UNITS, "signed_fluxes_preserved": True}:
        failures.append("output_contract differs from the frozen signed-flux v1 contract")
    if claims != {
        "official_gacode_outputs": True,
        "surrogate_promoted": False,
        "experimental_validation": False,
        "cross_solver_parity": False,
    }:
        failures.append("claims exceed or differ from the admitted dataset boundary")

    try:
        records_relative = _safe_relative_path(records_spec.get("path"), label="records.path")
        records_path = root.joinpath(*records_relative.parts)
        _verify_file_contract(records_path, records_spec, failures, label="records")
        records_payload = checked_json_load(records_path, max_bytes=MAX_TGLF_RECORDS_BYTES)
        if not isinstance(records_payload, list):
            raise ValueError("records JSON must contain an array")
        records = records_payload
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        failures.append(f"cannot load records: {exc}")
        records = []

    raw_samples = manifest.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples or len(raw_samples) > MAX_TGLF_SAMPLES:
        failures.append(f"samples must contain between 1 and {MAX_TGLF_SAMPLES} objects")
        return _failure_result(failures, dataset_id=dataset_id)
    samples = raw_samples
    if generation.get("accepted_samples") != len(samples) or generation.get("failed_samples") != 0:
        failures.append("generation sample counts do not match a complete fail-closed dataset")
    if len(records) != len(samples):
        failures.append("records and manifest sample counts differ")

    seen_indices: set[int] = set()
    seen_sample_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    present_strata: set[str] = set()
    reported_versions: set[str] = set()
    for position, raw_sample in enumerate(samples):
        try:
            sample = _as_object(raw_sample, label=f"samples[{position}]")
        except ValueError as exc:
            failures.append(str(exc))
            continue
        index = sample.get("sample_index")
        if isinstance(index, bool) or not isinstance(index, int) or index != position:
            failures.append(f"samples[{position}].sample_index must equal its ordered position")
            continue
        if index in seen_indices:
            failures.append(f"duplicate sample_index: {index}")
        seen_indices.add(index)
        try:
            input_payload = _as_object(sample.get("input"), label=f"samples[{index}].input")
            output_payload = _as_object(sample.get("output"), label=f"samples[{index}].output")
        except ValueError as exc:
            failures.append(str(exc))
            continue
        _validate_input(input_payload, failures, label=f"samples[{index}].input")
        _validate_output(output_payload, failures, label=f"samples[{index}].output")
        if revision:
            try:
                expected_sample_id = canonical_tglf_sample_id(input_payload, revision)
            except ValueError as exc:
                failures.append(str(exc))
                expected_sample_id = ""
        else:
            expected_sample_id = ""
        sample_id = sample.get("sample_id")
        if sample_id != expected_sample_id:
            failures.append(f"samples[{index}].sample_id mismatch")
        if isinstance(sample_id, str):
            if sample_id in seen_sample_ids:
                failures.append(f"duplicate TGLF input deck/sample_id: {sample_id}")
            seen_sample_ids.add(sample_id)
        try:
            group_id = _require_identifier(
                sample.get("group_id"), label=f"samples[{index}].group_id"
            )
            declared_split = sample.get("split")
            expected_split = deterministic_tglf_split(group_id, seed)
            if declared_split != expected_split:
                failures.append(f"samples[{index}].split is not the deterministic group split")
            previous_split = group_splits.setdefault(group_id, cast(str, declared_split))
            if previous_split != declared_split:
                failures.append(f"group {group_id} leaks across split roles")
        except ValueError as exc:
            failures.append(str(exc))
        regime = sample.get("regime")
        if regime not in TGLF_REGIMES:
            failures.append(f"samples[{index}].regime is invalid")
        stratum = sample.get("sampling_stratum")
        if stratum not in TGLF_SAMPLING_STRATA:
            failures.append(f"samples[{index}].sampling_stratum is invalid")
        else:
            present_strata.add(cast(str, stratum))
        reported_version = sample.get("reported_version")
        if not isinstance(reported_version, str) or not reported_version.startswith(revision[:8]):
            failures.append(f"samples[{index}].reported_version does not match source revision")
        else:
            reported_versions.add(reported_version)

        try:
            relative_dir = _safe_relative_path(
                sample.get("run_directory"), label=f"samples[{index}].run_directory"
            )
            expected_dir = PurePosixPath("runs") / f"sample_{index:06d}"
            if relative_dir != expected_dir:
                failures.append(f"samples[{index}].run_directory must be {expected_dir}")
            sample_dir = root.joinpath(*relative_dir.parts)
            if not sample_dir.is_dir() or sample_dir.is_symlink():
                failures.append(f"samples[{index}] run directory is missing or symlinked")
                continue
        except ValueError as exc:
            failures.append(str(exc))
            continue
        raw_files = sample.get("raw_files")
        if not isinstance(raw_files, list) or not raw_files:
            failures.append(f"samples[{index}].raw_files must be a non-empty array")
            continue
        declared_names: set[str] = set()
        for file_position, raw_file in enumerate(raw_files):
            try:
                file_spec = _as_object(
                    raw_file, label=f"samples[{index}].raw_files[{file_position}]"
                )
                name_path = _safe_relative_path(
                    file_spec.get("name"), label=f"samples[{index}].raw_files[{file_position}].name"
                )
                if len(name_path.parts) != 1:
                    raise ValueError("raw file names must be basenames")
                name = name_path.name
            except ValueError as exc:
                failures.append(str(exc))
                continue
            if name in declared_names:
                failures.append(f"samples[{index}] declares duplicate raw file {name}")
            declared_names.add(name)
            _verify_file_contract(
                sample_dir / name, file_spec, failures, label=f"sample {index}/{name}"
            )
        actual_names = {
            path.name for path in sample_dir.iterdir() if path.is_file() or path.is_symlink()
        }
        if declared_names != actual_names:
            failures.append(
                f"samples[{index}] raw inventory mismatch; "
                f"extra={sorted(actual_names - declared_names)}, "
                f"missing={sorted(declared_names - actual_names)}"
            )
        missing_required = REQUIRED_TGLF_RAW_FILES - declared_names
        if missing_required:
            failures.append(
                f"samples[{index}] missing required raw files: {sorted(missing_required)}"
            )
        if position < len(records):
            try:
                record = _as_object(records[position], label=f"records[{position}]")
                if record.get("sample_index") != index:
                    failures.append(f"records[{position}].sample_index mismatch")
                if record.get("input") != input_payload or record.get("output") != output_payload:
                    failures.append(f"records[{position}] payload differs from manifest sample")
            except ValueError as exc:
                failures.append(str(exc))

    required_strata = {"interior"} if purpose == "pilot" else set(TGLF_SAMPLING_STRATA)
    if not required_strata.issubset(present_strata):
        failures.append(
            f"dataset is missing required sampling strata: {sorted(required_strata - present_strata)}"
        )
    if generation.get("sampling_strata") != sorted(present_strata):
        failures.append("generation.sampling_strata does not match samples")
    if generation.get("required_strata") != sorted(required_strata):
        failures.append("generation.required_strata does not match purpose")
    if source.get("reported_versions") != sorted(reported_versions):
        failures.append("source.reported_versions does not match sample version files")
    return {
        "status": "passed" if not failures else "failed",
        "dataset_id": dataset_id,
        "schema_version": manifest.get("schema_version"),
        "samples_verified": len(samples),
        "split_counts": {
            role: sum(sample.get("split") == role for sample in samples if isinstance(sample, dict))
            for role in TGLF_SPLIT_FRACTIONS
        },
        "sampling_strata": sorted(present_strata),
        "reported_versions": sorted(reported_versions),
        "failures": failures,
    }


__all__ = [
    "MAX_TGLF_MANIFEST_BYTES",
    "MAX_TGLF_RAW_FILE_BYTES",
    "REQUIRED_TGLF_RAW_FILES",
    "TGLF_DATASET_SCHEMA_VERSION",
    "TGLF_INPUT_DOMAINS",
    "TGLF_OUTPUT_UNITS",
    "TGLF_REGIMES",
    "TGLF_SAMPLING_STRATA",
    "build_tglf_dataset_manifest",
    "canonical_tglf_sample_id",
    "deterministic_tglf_split",
    "sha256_file",
    "verify_tglf_dataset",
    "write_tglf_dataset_manifest",
]
