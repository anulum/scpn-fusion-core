# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — GACODE TGLF Dataset Contract Tests
"""Real-format API and CLI tests for the official-GACODE dataset contract."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, cast

import pytest

from scpn_fusion.core import tglf_interface
from scpn_fusion.io.tglf_dataset_contract import (
    TGLF_DATASET_SCHEMA_VERSION,
    build_tglf_dataset_manifest,
    canonical_tglf_sample_id,
    deterministic_tglf_split,
    sha256_file,
    verify_tglf_dataset,
    write_tglf_dataset_manifest,
)
from tools import tglf_dataset_contract as contract_cli

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "validation" / "reference_data" / "tglf_gacode_dataset_v1_fixture"
SCHEMA = ROOT / "schemas" / "tglf_gacode_dataset.schema.json"
CLI = ROOT / "tools" / "tglf_dataset_contract.py"
REVISION = "b49339750a4aa4cf2b089fa9ff3afe098005f0f8"
DATASET_ID = "gacode-b4933975-authentic-fixture-v1"
SEED = 20260826
jsonschema = cast(Any, importlib.import_module("jsonschema"))


def _copy_fixture(tmp_path: Path) -> Path:
    """Copy the retained official-GACODE run fixture into a mutable directory."""
    root = tmp_path / "dataset"
    shutil.copytree(FIXTURE, root)
    return root


def _load_records(root: Path) -> list[dict[str, Any]]:
    """Load the authentic fixture records with an explicit object-array type."""
    payload = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    return [cast(dict[str, Any], item) for item in payload]


def _build_manifest(root: Path) -> dict[str, Any]:
    """Build the pilot contract for the authentic retained GACODE run."""
    return build_tglf_dataset_manifest(
        root,
        _load_records(root),
        dataset_id=DATASET_ID,
        gacode_revision=REVISION,
        seed=SEED,
    )


def _write_manifest(root: Path, manifest: dict[str, Any]) -> Path:
    """Write a mutable manifest through the production atomic writer."""
    return write_tglf_dataset_manifest(root, manifest)


def _set_nested(payload: dict[str, Any], path: tuple[str | int, ...], value: Any) -> None:
    """Replace one nested manifest field for a fail-closed contract case."""
    target: Any = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _refresh_records_contract(root: Path, manifest: dict[str, Any]) -> None:
    """Rebind the records file after intentionally changing its valid bytes."""
    records = root / "dataset.json"
    manifest["records"]["size_bytes"] = records.stat().st_size
    manifest["records"]["sha256"] = sha256_file(records)


def test_public_api_builds_schema_valid_authentic_manifest(tmp_path: Path) -> None:
    """The public TGLF API binds and verifies an authentic official run."""
    root = _copy_fixture(tmp_path)
    manifest = tglf_interface.build_tglf_dataset_manifest(
        root,
        _load_records(root),
        dataset_id=DATASET_ID,
        gacode_revision=REVISION,
        seed=SEED,
    )
    manifest_path = tglf_interface.write_tglf_dataset_manifest(root, manifest)
    result = tglf_interface.verify_tglf_dataset(root, manifest_path=manifest_path)

    assert tglf_interface.TGLF_DATASET_SCHEMA_VERSION == TGLF_DATASET_SCHEMA_VERSION
    assert result == {
        "status": "passed",
        "dataset_id": DATASET_ID,
        "schema_version": TGLF_DATASET_SCHEMA_VERSION,
        "samples_verified": 1,
        "split_counts": {
            "train": int(manifest["samples"][0]["split"] == "train"),
            "calibration": int(manifest["samples"][0]["split"] == "calibration"),
            "test": int(manifest["samples"][0]["split"] == "test"),
        },
        "sampling_strata": ["interior"],
        "reported_versions": ["b4933975 [2026-08-20]"],
        "failures": [],
    }
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(manifest)
    assert manifest["claims"]["surrogate_promoted"] is False
    assert manifest["samples"][0]["regime"] == "unclassified"


def test_cli_build_and_verify_crosses_the_real_file_boundary(tmp_path: Path) -> None:
    """The executable CLI builds and re-verifies the authentic repository format."""
    root = _copy_fixture(tmp_path)
    build = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "build",
            "--dataset-root",
            str(root),
            "--dataset-id",
            DATASET_ID,
            "--gacode-revision",
            REVISION,
            "--seed",
            str(SEED),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    assert json.loads(build.stdout)["status"] == "passed"
    verify = subprocess.run(
        [sys.executable, str(CLI), "verify", "--dataset-root", str(root)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert verify.returncode == 0, verify.stderr
    assert json.loads(verify.stdout)["samples_verified"] == 1


def test_validator_rejects_raw_corruption_and_undeclared_file(tmp_path: Path) -> None:
    """Hash drift and an undeclared raw output both fail the production validator."""
    root = _copy_fixture(tmp_path)
    _write_manifest(root, _build_manifest(root))
    raw_dir = root / "runs" / "sample_000000"
    flux_path = raw_dir / "out.tglf.gbflux"
    flux_path.write_text(flux_path.read_text(encoding="utf-8") + "0\n", encoding="utf-8")
    (raw_dir / "out.tglf.undeclared").write_text("unexpected\n", encoding="utf-8")

    result = verify_tglf_dataset(root)

    assert result["status"] == "failed"
    failures = "\n".join(result["failures"])
    assert "SHA-256 mismatch" in failures
    assert "raw inventory mismatch" in failures


def test_validator_rejects_symlink_and_path_traversal(tmp_path: Path) -> None:
    """Symlinked raw custody and traversal declarations fail closed."""
    root = _copy_fixture(tmp_path)
    manifest = _build_manifest(root)
    _write_manifest(root, manifest)
    raw_dir = root / "runs" / "sample_000000"
    version_path = raw_dir / "out.tglf.version"
    target = root / "outside-version.txt"
    target.write_text(version_path.read_text(encoding="utf-8"), encoding="utf-8")
    version_path.unlink()
    version_path.symlink_to(target)
    first_file = manifest["samples"][0]["raw_files"][0]
    first_file["name"] = "../" + first_file["name"]
    _write_manifest(root, manifest)

    result = verify_tglf_dataset(root)

    failures = "\n".join(result["failures"])
    assert result["status"] == "failed"
    assert "confined relative POSIX path" in failures
    assert "not a regular file" in failures or "raw inventory mismatch" in failures


def test_validator_rejects_revision_and_deterministic_split_drift(tmp_path: Path) -> None:
    """A forged source revision or reassigned split cannot pass validation."""
    root = _copy_fixture(tmp_path)
    manifest = _build_manifest(root)
    manifest["source"]["revision"] = "0" * 40
    manifest["samples"][0]["split"] = {
        "train": "test",
        "calibration": "train",
        "test": "calibration",
    }[manifest["samples"][0]["split"]]
    _write_manifest(root, manifest)

    result = verify_tglf_dataset(root)

    failures = "\n".join(result["failures"])
    assert "sample_id mismatch" in failures
    assert "deterministic group split" in failures
    assert "reported_version does not match" in failures


def test_builder_rejects_duplicate_and_out_of_domain_records(tmp_path: Path) -> None:
    """Repeated decks and inputs outside the frozen domain are rejected before writing."""
    root = _copy_fixture(tmp_path)
    records = _load_records(root)
    duplicate = deepcopy(records[0])
    duplicate["sample_index"] = 1
    shutil.copytree(root / "runs" / "sample_000000", root / "runs" / "sample_000001")
    with pytest.raises(ValueError, match="duplicate TGLF input decks"):
        build_tglf_dataset_manifest(
            root,
            [records[0], duplicate],
            dataset_id=DATASET_ID,
            gacode_revision=REVISION,
            seed=SEED,
        )

    records[0]["input"]["R_LTi"] = 12.0001
    with pytest.raises(ValueError, match="outside"):
        build_tglf_dataset_manifest(
            root,
            records,
            dataset_id=DATASET_ID,
            gacode_revision=REVISION,
            seed=SEED,
        )


def test_development_purpose_requires_all_frozen_sampling_strata(tmp_path: Path) -> None:
    """A development corpus cannot masquerade as an interior-only pilot."""
    root = _copy_fixture(tmp_path)
    with pytest.raises(ValueError, match="missing sampling strata"):
        build_tglf_dataset_manifest(
            root,
            _load_records(root),
            dataset_id=DATASET_ID,
            gacode_revision=REVISION,
            seed=SEED,
            purpose="development",
        )


def test_group_split_is_stable_and_keeps_perturbations_together() -> None:
    """The published group splitter is deterministic and isolates related runs."""
    group_id = "gradient-perturbation-pair-0042"
    first = deterministic_tglf_split(group_id, SEED)
    assert first == deterministic_tglf_split(group_id, SEED)
    assert first in {"train", "calibration", "test"}
    roles = {deterministic_tglf_split(f"device-group-{index}", SEED) for index in range(128)}
    assert roles == {"train", "calibration", "test"}


def test_sample_identity_binds_revision_and_input() -> None:
    """Input or source-revision drift changes the public immutable sample identity."""
    input_payload = cast(dict[str, Any], _load_records(FIXTURE)[0]["input"])
    baseline = canonical_tglf_sample_id(input_payload, REVISION)
    changed = dict(input_payload)
    changed["R_LTi"] = float(changed["R_LTi"]) + 0.01

    assert len(baseline) == 64
    assert canonical_tglf_sample_id(changed, REVISION) != baseline
    assert canonical_tglf_sample_id(input_payload, "0" * 40) != baseline


def test_records_hash_and_manifest_path_are_confined(tmp_path: Path) -> None:
    """Records hash drift and a manifest outside the dataset root are rejected."""
    root = _copy_fixture(tmp_path)
    manifest = _build_manifest(root)
    manifest_path = _write_manifest(root, manifest)
    records_path = root / "dataset.json"
    assert manifest["records"]["sha256"] == sha256_file(records_path)
    records_path.write_text(records_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    result = verify_tglf_dataset(root)
    assert any("records SHA-256 mismatch" in failure for failure in result["failures"])

    outside = tmp_path / "outside-manifest.json"
    shutil.copyfile(manifest_path, outside)
    outside_result = verify_tglf_dataset(root, manifest_path=outside)
    assert outside_result["failures"] == ["manifest must be a direct child of dataset_root"]


def test_cli_returns_failure_for_missing_dataset(tmp_path: Path) -> None:
    """The real CLI emits a machine-readable non-zero failure for a missing root."""
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "verify",
            "--dataset-root",
            str(tmp_path / "missing"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "failed"
    assert "dataset root" in payload["failures"][0]


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("SPDX-License-Identifier",), "MIT", "SPDX identifier mismatch"),
        (("schema_version",), "future", "schema_version must be"),
        (("dataset_id",), "../escape", "dataset_id must be"),
        (("purpose",), "training", "purpose must be"),
        (("source", "repository"), "https://invalid.example", "source provenance"),
        (("source", "executable"), "./tglf", "source provenance"),
        (("generation", "input_domains"), {}, "input_domains differs"),
        (("generation", "seed"), True, "generation.seed"),
        (("generation", "accepted_samples"), 2, "sample counts"),
        (("generation", "failed_samples"), 1, "sample counts"),
        (("split_policy", "method"), "random", "split_policy differs"),
        (("split_policy", "group_isolation_required"), False, "split_policy differs"),
        (("output_contract", "signed_fluxes_preserved"), False, "output_contract differs"),
        (("claims", "surrogate_promoted"), True, "claims exceed"),
        (("samples", 0, "sample_id"), "0" * 64, "sample_id mismatch"),
        (("samples", 0, "group_id"), "../group", "group_id must be"),
        (("samples", 0, "regime"), "heuristic", "regime is invalid"),
        (("samples", 0, "sampling_stratum"), "random", "sampling_stratum is invalid"),
        (("samples", 0, "reported_version"), "deadbeef", "reported_version does not match"),
        (("samples", 0, "run_directory"), "runs/other", "run_directory must be"),
        (("samples", 0, "run_directory"), "../escape", "confined relative POSIX path"),
        (("samples", 0, "input", "R_LTi"), 12.1, "outside"),
        (("samples", 0, "input", "use_bper"), True, "use_bper must equal"),
        (("samples", 0, "input", "q"), "1.4", "q must be numeric"),
        (("samples", 0, "output", "gamma_max"), "2.5", "gamma_max must be numeric"),
        (("samples", 0, "raw_files", 0, "size_bytes"), True, "size_bytes is invalid"),
        (("samples", 0, "raw_files", 0, "sha256"), "invalid", "sha256 is invalid"),
        (("samples", 0, "raw_files", 0, "name"), "nested/file", "must be basenames"),
    ],
)
def test_validator_rejects_manifest_contract_mutations(
    tmp_path: Path,
    path: tuple[str | int, ...],
    value: Any,
    expected: str,
) -> None:
    """Every frozen provenance, domain, split and raw-custody field fails closed."""
    root = _copy_fixture(tmp_path)
    manifest = _build_manifest(root)
    _set_nested(manifest, path, value)
    _write_manifest(root, manifest)

    result = verify_tglf_dataset(root)

    assert result["status"] == "failed"
    assert expected in "\n".join(result["failures"])


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda manifest: manifest.__setitem__("unexpected", True), "top-level fields"),
        (lambda manifest: manifest.__setitem__("source", []), "source must be a JSON object"),
        (lambda manifest: manifest.__setitem__("samples", []), "samples must contain"),
        (
            lambda manifest: manifest["samples"].__setitem__(0, []),
            "samples[0] must be a JSON object",
        ),
        (
            lambda manifest: manifest["samples"][0].__setitem__("sample_index", 4),
            "sample_index must equal",
        ),
        (
            lambda manifest: manifest["samples"][0].__setitem__("input", []),
            "input must be a JSON object",
        ),
        (
            lambda manifest: manifest["samples"][0]["input"].pop("q"),
            "input fields mismatch",
        ),
        (
            lambda manifest: manifest["samples"][0]["output"].pop("q_i"),
            "output fields mismatch",
        ),
        (
            lambda manifest: manifest["samples"][0].__setitem__("raw_files", []),
            "raw_files must be a non-empty array",
        ),
        (
            lambda manifest: manifest["samples"][0]["raw_files"].append(
                dict(manifest["samples"][0]["raw_files"][0])
            ),
            "declares duplicate raw file",
        ),
    ],
)
def test_validator_rejects_structural_manifest_mutations(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], object],
    expected: str,
) -> None:
    """Malformed container shapes cannot bypass semantic validation."""
    root = _copy_fixture(tmp_path)
    manifest = _build_manifest(root)
    mutate(manifest)
    _write_manifest(root, manifest)

    result = verify_tglf_dataset(root)

    assert result["status"] == "failed"
    assert expected in "\n".join(result["failures"])


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"dataset_id": "../bad"}, "dataset_id"),
        ({"gacode_revision": "deadbeef"}, "gacode_revision"),
        ({"seed": -1}, "seed must be"),
        ({"purpose": "training"}, "purpose must be"),
        ({"records_file": "../dataset.json"}, "confined relative POSIX path"),
    ],
)
def test_builder_rejects_invalid_dataset_metadata(
    tmp_path: Path,
    kwargs: dict[str, Any],
    expected: str,
) -> None:
    """Dataset identity, revision, seed, purpose and records path are fail closed."""
    root = _copy_fixture(tmp_path)
    arguments: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "gacode_revision": REVISION,
        "seed": SEED,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=expected):
        build_tglf_dataset_manifest(root, _load_records(root), **arguments)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("sample_index", True, "sample_index"),
        ("group_id", "../group", "group_id"),
        ("regime", "heuristic", "regime"),
        ("sampling_stratum", "random", "sampling_stratum"),
    ],
)
def test_builder_rejects_invalid_sample_metadata(
    tmp_path: Path,
    field: str,
    value: Any,
    expected: str,
) -> None:
    """Invalid per-run identity and labelling fail before manifest creation."""
    root = _copy_fixture(tmp_path)
    records = _load_records(root)
    records[0][field] = value
    with pytest.raises(ValueError, match=expected):
        build_tglf_dataset_manifest(
            root,
            records,
            dataset_id=DATASET_ID,
            gacode_revision=REVISION,
            seed=SEED,
        )


def test_builder_rejects_missing_run_raw_file_and_version_drift(tmp_path: Path) -> None:
    """Missing run custody, incomplete output and wrong solver version each stop building."""
    root = _copy_fixture(tmp_path)
    records = _load_records(root)
    run_dir = root / "runs" / "sample_000000"
    moved = root / "sample_saved"
    run_dir.rename(moved)
    with pytest.raises(ValueError, match="missing regular sample directory"):
        build_tglf_dataset_manifest(
            root, records, dataset_id=DATASET_ID, gacode_revision=REVISION, seed=SEED
        )
    moved.rename(run_dir)

    flux = run_dir / "out.tglf.gbflux"
    saved_flux = root / "saved.gbflux"
    flux.rename(saved_flux)
    with pytest.raises(ValueError, match="missing required raw files"):
        build_tglf_dataset_manifest(
            root, records, dataset_id=DATASET_ID, gacode_revision=REVISION, seed=SEED
        )
    saved_flux.rename(flux)

    (run_dir / "out.tglf.version").write_text("deadbeef [unknown]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="version does not match"):
        build_tglf_dataset_manifest(
            root, records, dataset_id=DATASET_ID, gacode_revision=REVISION, seed=SEED
        )


def test_builder_rejects_missing_root_empty_records_and_bad_records_file(tmp_path: Path) -> None:
    """The builder requires an existing root, records, and a regular confined file."""
    with pytest.raises(ValueError, match="dataset root"):
        build_tglf_dataset_manifest(
            tmp_path / "missing",
            [{}],
            dataset_id=DATASET_ID,
            gacode_revision=REVISION,
            seed=SEED,
        )
    root = _copy_fixture(tmp_path)
    with pytest.raises(ValueError, match="records must contain"):
        build_tglf_dataset_manifest(
            root, [], dataset_id=DATASET_ID, gacode_revision=REVISION, seed=SEED
        )
    with pytest.raises(ValueError, match="non-empty relative POSIX path"):
        build_tglf_dataset_manifest(
            root,
            _load_records(root),
            dataset_id=DATASET_ID,
            gacode_revision=REVISION,
            seed=SEED,
            records_file="",
        )
    records_path = root / "dataset.json"
    target = root / "records-target.json"
    records_path.rename(target)
    records_path.symlink_to(target)
    with pytest.raises(ValueError, match="regular non-symlink"):
        build_tglf_dataset_manifest(
            root,
            _load_records(root),
            dataset_id=DATASET_ID,
            gacode_revision=REVISION,
            seed=SEED,
        )


def test_builder_rejects_duplicate_and_noncontiguous_indices(tmp_path: Path) -> None:
    """Record order and sample indices remain a unique contiguous sequence."""
    root = _copy_fixture(tmp_path)
    records = _load_records(root)
    with pytest.raises(ValueError, match="duplicate sample_index"):
        build_tglf_dataset_manifest(
            root,
            [records[0], deepcopy(records[0])],
            dataset_id=DATASET_ID,
            gacode_revision=REVISION,
            seed=SEED,
        )

    noncontiguous = deepcopy(records[0])
    noncontiguous["sample_index"] = 2
    shutil.copytree(root / "runs" / "sample_000000", root / "runs" / "sample_000002")
    with pytest.raises(ValueError, match="contiguous"):
        build_tglf_dataset_manifest(
            root,
            [noncontiguous],
            dataset_id=DATASET_ID,
            gacode_revision=REVISION,
            seed=SEED,
        )


def test_identity_split_and_writer_reject_invalid_inputs(tmp_path: Path) -> None:
    """Public identity, split and writer APIs reject malformed direct calls."""
    with pytest.raises(ValueError, match="revision"):
        canonical_tglf_sample_id({}, "short")
    with pytest.raises(ValueError, match="finite canonical JSON"):
        canonical_tglf_sample_id({"bad": float("nan")}, REVISION)
    with pytest.raises(ValueError, match="group_id"):
        deterministic_tglf_split("../group", SEED)
    with pytest.raises(ValueError, match="split seed"):
        deterministic_tglf_split("group", -1)
    with pytest.raises(ValueError, match="dataset root"):
        write_tglf_dataset_manifest(tmp_path / "missing", {})


def test_validator_rejects_missing_malformed_and_nonobject_inputs(tmp_path: Path) -> None:
    """Missing roots/manifests and malformed JSON return stable failure envelopes."""
    missing_root = verify_tglf_dataset(tmp_path / "missing")
    assert missing_root["status"] == "failed"

    root = tmp_path / "dataset"
    root.mkdir()
    assert "manifest is missing" in verify_tglf_dataset(root)["failures"][0]
    manifest_path = root / "manifest.json"
    manifest_path.write_text("{", encoding="utf-8")
    assert "cannot load manifest" in verify_tglf_dataset(root)["failures"][0]
    manifest_path.write_text("[]\n", encoding="utf-8")
    assert "manifest must be a JSON object" in verify_tglf_dataset(root)["failures"][0]


def test_validator_rejects_nonarray_records_and_records_payload_drift(tmp_path: Path) -> None:
    """Records must remain an authenticated array identical to manifest samples."""
    root = _copy_fixture(tmp_path)
    manifest = _build_manifest(root)
    records_path = root / "dataset.json"
    records_path.write_text("{}\n", encoding="utf-8")
    _refresh_records_contract(root, manifest)
    _write_manifest(root, manifest)
    result = verify_tglf_dataset(root)
    assert "records JSON must contain an array" in "\n".join(result["failures"])

    root = _copy_fixture(tmp_path / "second")
    manifest = _build_manifest(root)
    records = _load_records(root)
    records[0]["output"]["q_i"] = 0.0
    (root / "dataset.json").write_text(json.dumps(records) + "\n", encoding="utf-8")
    _refresh_records_contract(root, manifest)
    _write_manifest(root, manifest)
    result = verify_tglf_dataset(root)
    assert "payload differs from manifest sample" in "\n".join(result["failures"])

    root = _copy_fixture(tmp_path / "third")
    manifest = _build_manifest(root)
    (root / "dataset.json").write_text("[[]]\n", encoding="utf-8")
    _refresh_records_contract(root, manifest)
    _write_manifest(root, manifest)
    result = verify_tglf_dataset(root)
    assert "records[0] must be a JSON object" in "\n".join(result["failures"])

    root = _copy_fixture(tmp_path / "fourth")
    manifest = _build_manifest(root)
    records = _load_records(root)
    records[0]["sample_index"] = 9
    (root / "dataset.json").write_text(json.dumps(records) + "\n", encoding="utf-8")
    _refresh_records_contract(root, manifest)
    _write_manifest(root, manifest)
    result = verify_tglf_dataset(root)
    assert "sample_index mismatch" in "\n".join(result["failures"])


def test_validator_rejects_nonfinite_output_duplicate_sample_and_missing_raw(
    tmp_path: Path,
) -> None:
    """Non-finite outputs, duplicate identities, split leakage and missing files fail."""
    root = _copy_fixture(tmp_path)
    manifest = _build_manifest(root)
    manifest["samples"][0]["output"]["gamma_max"] = float("nan")
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    result = verify_tglf_dataset(root)
    assert "gamma_max must be finite" in "\n".join(result["failures"])

    root = _copy_fixture(tmp_path / "duplicate")
    manifest = _build_manifest(root)
    duplicate = deepcopy(manifest["samples"][0])
    duplicate["sample_index"] = 1
    duplicate["run_directory"] = "runs/sample_000001"
    duplicate["split"] = {
        "train": "test",
        "calibration": "train",
        "test": "calibration",
    }[duplicate["split"]]
    manifest["samples"].append(duplicate)
    manifest["generation"]["accepted_samples"] = 2
    records = _load_records(root)
    second_record = deepcopy(records[0])
    second_record["sample_index"] = 1
    records.append(second_record)
    (root / "dataset.json").write_text(json.dumps(records) + "\n", encoding="utf-8")
    _refresh_records_contract(root, manifest)
    shutil.copytree(root / "runs" / "sample_000000", root / "runs" / "sample_000001")
    _write_manifest(root, manifest)
    failures = "\n".join(verify_tglf_dataset(root)["failures"])
    assert "duplicate TGLF input deck" in failures
    assert "leaks across split roles" in failures

    root = _copy_fixture(tmp_path / "missing_raw")
    manifest = _build_manifest(root)
    manifest["samples"][0]["raw_files"][0]["name"] = "missing.raw"
    _write_manifest(root, manifest)
    failures = "\n".join(verify_tglf_dataset(root)["failures"])
    assert "is missing or is not a regular file" in failures


def test_cli_main_covers_build_failure_and_explicit_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The callable CLI surface reports build errors and verifies an explicit manifest."""
    root = _copy_fixture(tmp_path)
    rc = contract_cli.main(
        [
            "build",
            "--dataset-root",
            str(root),
            "--dataset-id",
            DATASET_ID,
            "--gacode-revision",
            "deadbeef",
            "--seed",
            str(SEED),
        ]
    )
    assert rc == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"

    assert (
        contract_cli.main(
            [
                "build",
                "--dataset-root",
                str(root),
                "--dataset-id",
                DATASET_ID,
                "--gacode-revision",
                REVISION,
                "--seed",
                str(SEED),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "passed"
    path = root / "manifest.json"
    assert contract_cli.main(["verify", "--dataset-root", str(root), "--manifest", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"

    (root / "dataset.json").write_text("{}\n", encoding="utf-8")
    assert (
        contract_cli.main(
            [
                "build",
                "--dataset-root",
                str(root),
                "--dataset-id",
                DATASET_ID,
                "--gacode-revision",
                REVISION,
                "--seed",
                str(SEED),
            ]
        )
        == 1
    )
    assert "non-empty array" in capsys.readouterr().out

    (root / "dataset.json").write_text("[1]\n", encoding="utf-8")
    assert (
        contract_cli.main(
            [
                "build",
                "--dataset-root",
                str(root),
                "--dataset-id",
                DATASET_ID,
                "--gacode-revision",
                REVISION,
                "--seed",
                str(SEED),
            ]
        )
        == 1
    )
    assert "records[0] must be an object" in capsys.readouterr().out
