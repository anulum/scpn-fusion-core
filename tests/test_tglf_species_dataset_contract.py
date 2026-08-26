# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Species-aware GACODE TGLF Dataset Tests
"""Real-file tests for the ordered multi-species v2 dataset boundary."""

from __future__ import annotations

from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil

import pytest
from typing import Any, cast

from scpn_fusion.core._tglf_interface_runtime import (
    _parse_gacode_tglf_output,
    identify_tglf_particle_transport,
)
from scpn_fusion.io.tglf_species_dataset_contract import (
    TGLF_SPECIES_DATASET_SCHEMA_VERSION,
    build_tglf_species_dataset_manifest,
    tglf_species_deck_from_payload,
    verify_tglf_species_dataset,
    write_tglf_species_dataset_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "validation" / "reference_data" / "tglf_gacode_species_v2_fixture"
DEVELOPMENT_FIXTURE = ROOT / "validation" / "reference_data" / "tglf_gacode_development_v2_fixture"
SCHEMA = ROOT / "schemas" / "tglf_gacode_species_dataset_v2.schema.json"
REVISION = "b49339750a4aa4cf2b089fa9ff3afe098005f0f8"
DATASET_ID = "gacode-b4933975-species-carbon-isotope-v2"
SEED = 20260826
jsonschema = cast(Any, importlib.import_module("jsonschema"))


def _records(root: Path) -> list[dict[str, Any]]:
    value = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    assert isinstance(value, list)
    return cast(list[dict[str, Any]], value)


def _copy_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "fixture"
    shutil.copytree(FIXTURE, root)
    return root


def _set_nested(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = payload
    for key in path[:-1]:
        target = cast(dict[str, Any], target[key])
    target[path[-1]] = value


def test_authentic_species_fixture_verifies_and_matches_schema() -> None:
    result = verify_tglf_species_dataset(FIXTURE)
    assert result == {
        "status": "passed",
        "dataset_id": DATASET_ID,
        "schema_version": TGLF_SPECIES_DATASET_SCHEMA_VERSION,
        "samples_verified": 4,
        "species_counts": [3],
        "paired_gradient_groups_verified": 1,
        "split_counts": {"train": 1, "calibration": 0, "test": 3},
        "failures": [],
    }
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(manifest)
    assert manifest["output_contract"]["legacy_scalar_view_is_derived"] is True
    assert manifest["claims"]["surrogate_promoted"] is False


def test_authentic_raw_fluxes_round_trip_and_identify_carbon_transport() -> None:
    records = _records(FIXTURE)
    decks = []
    outputs = []
    for record in records:
        deck = tglf_species_deck_from_payload(record["input"])
        run_dir = FIXTURE / "runs" / f"sample_{record['sample_index']:06d}"
        output = _parse_gacode_tglf_output(run_dir, deck)
        expected_fluxes = record["output"]["species_fluxes"]
        assert [item.name for item in output.species_fluxes] == [
            item["name"] for item in expected_fluxes
        ]
        assert [item.particle_gb for item in output.species_fluxes] == pytest.approx(
            [item["particle_gb"] for item in expected_fluxes]
        )
        if record.get("paired_gradient_species") == "carbon":
            decks.append(deck)
            outputs.append(output)

    fitted = identify_tglf_particle_transport(decks, outputs, species_name="carbon")
    assert fitted.gradients_a_over_l == pytest.approx((0.8, 1.0, 1.2))
    assert fitted.density_e_ratio == pytest.approx(0.1)
    assert fitted.diffusion_gb == pytest.approx(11.051075)
    assert fitted.pinch_gb == pytest.approx(-7.982518333333339)
    assert fitted.diffusion_m2_s == pytest.approx(56.84751442866729)
    assert fitted.pinch_m_s == pytest.approx(-41.06264106716025)
    assert fitted.residual_max_abs_gb_per_density == pytest.approx(0.05274333333333359)


def test_builder_rejects_gradient_group_state_drift(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    records = _records(root)
    records[2] = deepcopy(records[2])
    records[2]["input"]["species"][1]["temperature_e_ratio"] = 1.1
    with pytest.raises(ValueError, match="may change only"):
        build_tglf_species_dataset_manifest(
            root,
            records,
            dataset_id=DATASET_ID,
            gacode_revision=REVISION,
            seed=SEED,
        )


def test_verifier_rejects_species_order_and_raw_flux_corruption(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    records = _records(root)
    records[0]["output"]["species_fluxes"][2]["name"] = "deuterium"
    (root / "dataset.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = verify_tglf_species_dataset(root)
    assert result["status"] == "failed"
    assert "species identity/order mismatch" in "\n".join(result["failures"])

    root = _copy_fixture(tmp_path / "second")
    flux_path = root / "runs" / "sample_000000" / "out.tglf.gbflux"
    flux_path.write_text(flux_path.read_text(encoding="utf-8") + "0\n", encoding="utf-8")
    result = verify_tglf_species_dataset(root)
    assert result["status"] == "failed"
    assert "canonical rebuild" in "\n".join(result["failures"])


def test_public_builder_round_trip_preserves_gradient_group_split(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    records = _records(root)
    manifest = build_tglf_species_dataset_manifest(
        root,
        records,
        dataset_id=DATASET_ID,
        gacode_revision=REVISION,
        seed=SEED,
    )
    write_tglf_species_dataset_manifest(root, manifest)
    assert verify_tglf_species_dataset(root)["status"] == "passed"
    carbon_splits = {
        sample["split"]
        for sample in manifest["samples"]
        if sample["group_id"] == "official-tglf07-carbon-gradient"
    }
    assert len(carbon_splits) == 1


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("extra",), 1, "fields mismatch"),
        (("schema_version",), "wrong", "schema version"),
        (("design_method",), "", "design_method"),
        (("plan_sha256",), "bad", "plan_sha256"),
        (("accepted_runs",), 8, "accepted_runs"),
        (("base_groups",), 2, "base_groups"),
        (("samples_per_group",), 4, "three samples"),
        (("sampling_strata_counts", "interior"), 2, "sampling-strata counts"),
        (("composition_counts", "electron-deuterium"), 2, "composition counts"),
        (("command_policy", "extra"), 1, "command_policy fields"),
        (("command_policy", "command"), "other", "PATH name tglf"),
        (("command_policy", "timeout_s"), 0.0, "timeout/retry"),
        (("command_policy", "max_retries"), -1, "timeout/retry"),
        (("storage_contract", "extra"), 1, "storage_contract fields"),
        (("storage_contract", "working_location"), "remote", "working_location"),
        (("storage_contract", "local_max_bytes"), 0, "local_max_bytes"),
        (("storage_contract", "local_max_runs"), True, "local_max_runs"),
        (("storage_contract", "large_artifact_policy"), "git", "large_artifact_policy"),
    ],
)
def test_development_metadata_guards_reject_semantic_drift(
    path: tuple[str, ...], value: Any, message: str
) -> None:
    """Every frozen development metadata field is checked by the public builder."""
    manifest = cast(dict[str, Any], json.loads((DEVELOPMENT_FIXTURE / "manifest.json").read_text()))
    development = deepcopy(cast(dict[str, Any], manifest["development"]))
    _set_nested(development, path, value)
    records = _records(DEVELOPMENT_FIXTURE)
    with pytest.raises(ValueError, match=message):
        build_tglf_species_dataset_manifest(
            DEVELOPMENT_FIXTURE,
            records,
            dataset_id=cast(str, manifest["dataset_id"]),
            gacode_revision=REVISION,
            seed=SEED,
            development=development,
            plan_file="plan.json",
            rejections_file="rejections.json",
        )


def test_development_builder_requires_complete_sidecars_and_composition() -> None:
    """Development-only composition and sidecars cannot leak into fixture manifests."""
    manifest = cast(dict[str, Any], json.loads((DEVELOPMENT_FIXTURE / "manifest.json").read_text()))
    records = _records(DEVELOPMENT_FIXTURE)
    with pytest.raises(ValueError, match="plan_file and rejections_file"):
        build_tglf_species_dataset_manifest(
            DEVELOPMENT_FIXTURE,
            records,
            dataset_id=cast(str, manifest["dataset_id"]),
            gacode_revision=REVISION,
            seed=SEED,
            development=cast(dict[str, Any], manifest["development"]),
        )
    without_composition = deepcopy(records)
    without_composition[0].pop("composition")
    with pytest.raises(ValueError, match="composition is invalid"):
        build_tglf_species_dataset_manifest(
            DEVELOPMENT_FIXTURE,
            without_composition,
            dataset_id=cast(str, manifest["dataset_id"]),
            gacode_revision=REVISION,
            seed=SEED,
            development=cast(dict[str, Any], manifest["development"]),
            plan_file="plan.json",
            rejections_file="rejections.json",
        )
    with pytest.raises(ValueError, match="admitted only"):
        build_tglf_species_dataset_manifest(
            DEVELOPMENT_FIXTURE,
            records,
            dataset_id=cast(str, manifest["dataset_id"]),
            gacode_revision=REVISION,
            seed=SEED,
        )
    pilot_records = _records(FIXTURE)
    with pytest.raises(ValueError, match="require development metadata"):
        build_tglf_species_dataset_manifest(
            FIXTURE,
            pilot_records,
            dataset_id=DATASET_ID,
            gacode_revision=REVISION,
            seed=SEED,
            plan_file="plan.json",
        )
