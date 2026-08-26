# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Official TGLF Development Corpus Tests
"""Real-plan, retained-output, recovery and CLI tests for TGLF-DATA-03."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, cast

import pytest

from scpn_fusion.io.tglf_development_corpus import (
    generate_tglf_development_corpus,
    verify_tglf_development_corpus,
)
from scpn_fusion.io.tglf_dataset_contract import sha256_file
from scpn_fusion.io.tglf_species_dataset_contract import (
    build_tglf_species_dataset_manifest,
    write_tglf_species_dataset_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "validation" / "reference_data" / "tglf_gacode_development_v2_fixture"
SCHEMA = ROOT / "schemas" / "tglf_gacode_species_dataset_v2.schema.json"
CLI = ROOT / "tools" / "generate_tglf_development_corpus.py"
FIXTURE_TREE_SHA256 = "8600ac44853417b6e1cd5f3fabee60ce6769296893263324e2ce80355a0f590e"
jsonschema = cast(Any, importlib.import_module("jsonschema"))


def _copy_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "fixture"
    shutil.copytree(FIXTURE, root)
    return root


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _refresh_file_contract(manifest: dict[str, Any], name: str, path: Path) -> None:
    manifest[name]["size_bytes"] = path.stat().st_size
    manifest[name]["sha256"] = sha256_file(path)


def _rewrite_manifest(root: Path, manifest: dict[str, Any]) -> None:
    (root / "manifest.json").write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _new_partial(tmp_path: Path, name: str = "partial") -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    output = tmp_path / name
    generate_tglf_development_corpus(output, profile="fixture", max_runs=0)
    return output, output.with_name(f"{name}.partial"), output.with_name(f"{name}.recovery.json")


def test_authentic_fixture_passes_schema_plan_and_complete_hash_audit() -> None:
    """Nine official retained runs pass the public schema and independent verifier."""
    result = verify_tglf_development_corpus(FIXTURE)
    assert result["status"] == "passed"
    assert result["samples_verified"] == 9
    assert result["paired_gradient_groups_verified"] == 3
    assert result["sampling_strata_counts"] == {"boundary": 3, "interior": 3, "threshold": 3}
    assert result["composition_counts"] == {
        "electron-deuterium": 3,
        "electron-deuterium-carbon": 3,
        "electron-deuterium-tritium": 3,
    }
    assert result["negative_controls_verified"] == 3
    assert result["tree_sha256"] == FIXTURE_TREE_SHA256
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(manifest)


def test_cli_plan_and_verify_cross_real_file_boundaries(tmp_path: Path) -> None:
    """The executable CLI writes the frozen plan and verifies retained official files."""
    plan_path = tmp_path / "plan.json"
    planned = subprocess.run(
        [sys.executable, str(CLI), "plan", "--output", str(plan_path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert planned.returncode == 0, planned.stderr
    assert json.loads(planned.stdout)["accepted_runs"] == 72
    assert json.loads(plan_path.read_text(encoding="utf-8"))["plan_sha256"] == (
        "a004de110868bfd5cd33fb545dd917eb8416f4e0654a2b541daaa39f5e5856f5"
    )
    verified = subprocess.run(
        [sys.executable, str(CLI), "verify", "--dataset-root", str(FIXTURE)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert verified.returncode == 0, verified.stderr
    assert json.loads(verified.stdout)["tree_sha256"] == FIXTURE_TREE_SHA256


def test_verifier_rejects_plan_and_raw_output_tamper(tmp_path: Path) -> None:
    """Plan drift and official raw-byte drift both fail the public verifier."""
    root = _copy_fixture(tmp_path)
    plan_path = root / "plan.json"
    plan = cast(dict[str, Any], json.loads(plan_path.read_text(encoding="utf-8")))
    plan["seed"] = 7
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    result = verify_tglf_development_corpus(root)
    assert result["status"] == "failed"
    assert "canonical rebuild" in "\n".join(result["failures"])


def test_development_verifier_rejects_semantic_sidecar_and_root_drift(tmp_path: Path) -> None:
    """Sidecar semantics, plan replay, root inventory and storage bound are fail-closed."""
    root = _copy_fixture(tmp_path / "rejections")
    manifest = cast(dict[str, Any], json.loads((root / "manifest.json").read_text()))
    rejections_path = root / "rejections.json"
    rejections = cast(list[dict[str, Any]], json.loads(rejections_path.read_text()))
    rejections[0]["reason"] = "changed"
    rejections_path.write_text(json.dumps(rejections), encoding="utf-8")
    _refresh_file_contract(manifest, "rejections", rejections_path)
    _rewrite_manifest(root, manifest)
    result = verify_tglf_development_corpus(root)
    assert result["status"] == "failed"
    assert "negative controls" in "\n".join(result["failures"])

    root = _copy_fixture(tmp_path / "plan-root")
    manifest = cast(dict[str, Any], json.loads((root / "manifest.json").read_text()))
    plan_path = root / "plan.json"
    plan = cast(dict[str, Any], json.loads(plan_path.read_text()))
    plan["seed"] = True
    without_digest = {key: value for key, value in plan.items() if key != "plan_sha256"}
    plan["plan_sha256"] = _canonical_digest(without_digest)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    _refresh_file_contract(manifest, "plan", plan_path)
    _rewrite_manifest(root, manifest)
    result = verify_tglf_development_corpus(root)
    assert result["status"] == "failed"
    assert "seed/profile" in "\n".join(result["failures"])

    root = _copy_fixture(tmp_path / "extra-root")
    (root / "undeclared.txt").write_text("extra\n", encoding="utf-8")
    result = verify_tglf_development_corpus(root)
    assert result["status"] == "failed"
    assert "root inventory" in "\n".join(result["failures"])

    root = _copy_fixture(tmp_path / "storage")
    manifest = cast(dict[str, Any], json.loads((root / "manifest.json").read_text()))
    manifest["development"]["storage_contract"]["local_max_bytes"] = 1
    _rewrite_manifest(root, manifest)
    result = verify_tglf_development_corpus(root)
    assert result["status"] == "failed"
    assert "storage boundary" in "\n".join(result["failures"])


def test_development_verifier_rejects_record_metadata_drift_against_plan(tmp_path: Path) -> None:
    """A canonically rebuilt v2 manifest still cannot detach records from its plan."""
    root = _copy_fixture(tmp_path)
    manifest = cast(dict[str, Any], json.loads((root / "manifest.json").read_text()))
    records = cast(list[dict[str, Any]], json.loads((root / "dataset.json").read_text()))
    records[0]["regime"] = "stable"
    (root / "dataset.json").write_text(json.dumps(records), encoding="utf-8")
    rebuilt = build_tglf_species_dataset_manifest(
        root,
        records,
        dataset_id=manifest["dataset_id"],
        gacode_revision=manifest["source"]["revision"],
        seed=manifest["split_policy"]["seed"],
        development=manifest["development"],
        plan_file="plan.json",
        rejections_file="rejections.json",
    )
    write_tglf_species_dataset_manifest(root, rebuilt)
    result = verify_tglf_development_corpus(root)
    assert result["status"] == "failed"
    assert "metadata differs from plan" in "\n".join(result["failures"])

    root = _copy_fixture(tmp_path / "raw")
    raw = root / "runs" / "sample_000000" / "out.tglf.gbflux"
    raw.write_text(raw.read_text(encoding="utf-8") + "0\n", encoding="utf-8")
    result = verify_tglf_development_corpus(root)
    assert result["status"] == "failed"
    assert "canonical rebuild" in "\n".join(result["failures"])


def test_resume_adopts_one_authentic_completed_run_and_authenticates_it(tmp_path: Path) -> None:
    """Recovery adopts an atomic official run, then refuses changed retained bytes."""
    output = tmp_path / "resume-fixture"
    initial = generate_tglf_development_corpus(output, profile="fixture", max_runs=0)
    assert initial["status"] == "partial"
    partial = output.with_name("resume-fixture.partial")
    shutil.copytree(
        FIXTURE / "runs" / "sample_000000",
        partial / "runs" / "sample_000000",
    )
    resumed = generate_tglf_development_corpus(output, profile="fixture", resume=True, max_runs=0)
    assert resumed["status"] == "partial"
    assert resumed["accepted_runs"] == 1
    records = json.loads((partial / "dataset.json").read_text(encoding="utf-8"))
    assert records[0] == json.loads((FIXTURE / "dataset.json").read_text(encoding="utf-8"))[0]

    raw = partial / "runs" / "sample_000000" / "out.tglf.gbflux"
    raw.write_text(raw.read_text(encoding="utf-8") + "0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="committed run SHA-256"):
        generate_tglf_development_corpus(output, profile="fixture", resume=True, max_runs=0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("recovery-root", "recovery checkpoint"),
        ("next-index", "next_sample_index"),
        ("committed-prefix", "committed-run prefix"),
        ("records-root", "records must be an array"),
        ("records-length", "recoverable committed prefix"),
        ("records-hash", "records SHA-256"),
    ],
)
def test_resume_rejects_malformed_checkpoint_state(
    tmp_path: Path, mutation: str, message: str
) -> None:
    """Malformed durable cursors and records fail before an external run."""
    output, partial, recovery_path = _new_partial(tmp_path, mutation)
    recovery: Any = json.loads(recovery_path.read_text(encoding="utf-8"))
    if mutation == "recovery-root":
        recovery = []
    elif mutation == "next-index":
        recovery["next_sample_index"] = True
    elif mutation == "committed-prefix":
        recovery["committed_runs"] = [{}]
    elif mutation == "records-root":
        (partial / "dataset.json").write_text("{}", encoding="utf-8")
    elif mutation == "records-length":
        (partial / "dataset.json").write_text("[{}, {}]", encoding="utf-8")
    else:
        recovery["records_sha256"] = "0" * 64
    recovery_path.write_text(json.dumps(recovery), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        generate_tglf_development_corpus(output, profile="fixture", resume=True, max_runs=0)


@pytest.mark.parametrize("unsafe", ["symlink", "missing-required"])
def test_resume_rejects_unsafe_authentic_run_inventory(tmp_path: Path, unsafe: str) -> None:
    """A symlink or incomplete retained run cannot enter the durable prefix."""
    output, partial, _ = _new_partial(tmp_path, unsafe)
    target = partial / "runs" / "sample_000000"
    shutil.copytree(FIXTURE / "runs" / "sample_000000", target)
    if unsafe == "symlink":
        (target / "unsafe-link").symlink_to(target / "input.tglf")
        match = "unsafe run entry"
    else:
        (target / "out.tglf.gbflux").unlink()
        match = "missing required GACODE files"
    with pytest.raises(ValueError, match=match):
        generate_tglf_development_corpus(output, profile="fixture", resume=True, max_runs=0)


def test_resume_rejects_record_drift_on_authentic_prefix_and_adoption(tmp_path: Path) -> None:
    """Both committed-prefix and crash-adoption record drift are authenticated."""
    output, partial, recovery_path = _new_partial(tmp_path / "committed", "case")
    shutil.copytree(FIXTURE / "runs" / "sample_000000", partial / "runs" / "sample_000000")
    generate_tglf_development_corpus(output, profile="fixture", resume=True, max_runs=0)
    records = cast(list[dict[str, Any]], json.loads((partial / "dataset.json").read_text()))
    records[0]["regime"] = "stable"
    (partial / "dataset.json").write_text(json.dumps(records), encoding="utf-8")
    recovery = cast(dict[str, Any], json.loads(recovery_path.read_text()))
    recovery["records_sha256"] = _canonical_digest(records)
    recovery_path.write_text(json.dumps(recovery), encoding="utf-8")
    with pytest.raises(ValueError, match="record differs"):
        generate_tglf_development_corpus(output, profile="fixture", resume=True, max_runs=0)

    output, partial, _ = _new_partial(tmp_path / "adoption", "case")
    shutil.copytree(FIXTURE / "runs" / "sample_000000", partial / "runs" / "sample_000000")
    records = cast(list[dict[str, Any]], json.loads((FIXTURE / "dataset.json").read_text()))
    records[0]["regime"] = "stable"
    (partial / "dataset.json").write_text(json.dumps(records[:1]), encoding="utf-8")
    with pytest.raises(ValueError, match="recoverable completed record"):
        generate_tglf_development_corpus(output, profile="fixture", resume=True, max_runs=0)


@pytest.mark.skipif(shutil.which("tglf") is None, reason="official PATH TGLF is unavailable")
def test_official_generator_completes_after_a_real_checkpoint(tmp_path: Path) -> None:
    """A real PATH TGLF run checkpoints, resumes and finalises all nine fixture cases."""
    output = tmp_path / "official-fixture"
    first = generate_tglf_development_corpus(output, profile="fixture", max_runs=1)
    assert first["status"] == "partial"
    assert first["accepted_runs"] == 1
    final = generate_tglf_development_corpus(output, profile="fixture", resume=True)
    assert final["status"] == "passed"
    assert final["samples_verified"] == 9
    assert not output.with_name("official-fixture.recovery.json").exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seed": -1}, "seed"),
        ({"profile": "promotion"}, "profile"),
        ({"command": "./tglf"}, "PATH command"),
        ({"timeout_s": 0.0}, "timeout_s"),
        ({"max_retries": -1}, "max_retries"),
        ({"max_runs": -1}, "max_runs"),
    ],
)
def test_generator_rejects_changed_or_unbounded_invocation(
    tmp_path: Path, kwargs: dict[str, Any], message: str
) -> None:
    """Invalid plan and process policies fail before any corpus execution."""
    with pytest.raises(ValueError, match=message):
        generate_tglf_development_corpus(tmp_path / "invalid", **kwargs)


def test_existing_output_and_changed_resume_contract_fail_closed(tmp_path: Path) -> None:
    """The generator never overwrites a final path or resumes under changed policy."""
    output = tmp_path / "corpus"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        generate_tglf_development_corpus(output, profile="fixture", max_runs=0)
    output.rmdir()
    generate_tglf_development_corpus(output, profile="fixture", max_runs=0)
    with pytest.raises(ValueError, match="command_policy"):
        generate_tglf_development_corpus(
            output,
            profile="fixture",
            resume=True,
            timeout_s=60.0,
            max_runs=0,
        )


def test_resume_requires_exact_partial_plan_and_committed_directory(tmp_path: Path) -> None:
    """Missing recovery state, plan drift and lost committed runs all fail closed."""
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="resume requires"):
        generate_tglf_development_corpus(missing, profile="fixture", resume=True, max_runs=0)

    occupied = tmp_path / "occupied"
    occupied.with_name("occupied.partial").mkdir()
    with pytest.raises(FileExistsError, match="partial corpus"):
        generate_tglf_development_corpus(occupied, profile="fixture", max_runs=0)

    output, partial, _ = _new_partial(tmp_path / "plan-drift", "case")
    plan_path = partial / "plan.json"
    plan = cast(dict[str, Any], json.loads(plan_path.read_text()))
    plan["design_method"] = "changed"
    without_digest = {key: value for key, value in plan.items() if key != "plan_sha256"}
    plan["plan_sha256"] = _canonical_digest(without_digest)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="deterministic regeneration"):
        generate_tglf_development_corpus(output, profile="fixture", resume=True, max_runs=0)

    output, partial, _ = _new_partial(tmp_path / "lost-run", "case")
    shutil.copytree(FIXTURE / "runs" / "sample_000000", partial / "runs" / "sample_000000")
    generate_tglf_development_corpus(output, profile="fixture", resume=True, max_runs=0)
    shutil.rmtree(partial / "runs" / "sample_000000")
    with pytest.raises(ValueError, match="unsafe or missing run directory"):
        generate_tglf_development_corpus(output, profile="fixture", resume=True, max_runs=0)


def test_development_manifest_rejects_composition_metadata_drift(tmp_path: Path) -> None:
    """A composition-count change cannot be hidden by rewriting only the records hash."""
    root = _copy_fixture(tmp_path)
    records_path = root / "dataset.json"
    records = cast(list[dict[str, Any]], json.loads(records_path.read_text(encoding="utf-8")))
    records[0] = deepcopy(records[0])
    records[0]["composition"] = "electron-deuterium-carbon"
    records_path.write_text(json.dumps(records), encoding="utf-8")
    result = verify_tglf_development_corpus(root)
    assert result["status"] == "failed"
    assert "composition counts" in "\n".join(result["failures"])
