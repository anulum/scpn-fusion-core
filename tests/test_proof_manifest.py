# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Lean Proof-Contract Manifest Tests
"""Tests for the Lean proof-contract checksum and its ``.scpnctl.json`` export."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from scpn_fusion.scpn.artifact import (
    ActionReadout,
    Artifact,
    ArtifactMeta,
    CompilerInfo,
    FixedPoint,
    InitialState,
    PlaceInjection,
    PlaceSpec,
    Readout,
    SeedPolicy,
    Topology,
    TransitionSpec,
    WeightMatrix,
    Weights,
    load_artifact,
    save_artifact,
    stamp_proof_contract,
)
from scpn_fusion.scpn.artifact_validation import validate_artifact
from scpn_fusion.scpn.proof_manifest import (
    CONTRACT_FILES,
    CONTRACT_THEOREMS,
    PROOF_SYSTEM,
    compute_proof_checksum,
    default_lean_dir,
    proof_contract_manifest,
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _build_artifact() -> Artifact:
    return Artifact(
        meta=ArtifactMeta(
            artifact_version="1.0.0",
            name="proof-test",
            dt_control_s=1.0e-4,
            stream_length=128,
            fixed_point=FixedPoint(data_width=16, fraction_bits=8, signed=True),
            firing_mode="binary",
            seed_policy=SeedPolicy(id="seed", hash_fn="sha256", rng_family="pcg64"),
            created_utc="2026-07-14T00:00:00Z",
            compiler=CompilerInfo(name="scpn", version="3.10.1", git_sha="deadbee"),
            notes=None,
        ),
        topology=Topology(
            places=[PlaceSpec(id=0, name="p0")],
            transitions=[TransitionSpec(id=0, name="t0", threshold=0.5, margin=0.0, delay_ticks=0)],
        ),
        weights=Weights(
            w_in=WeightMatrix(shape=[1, 1], data=[0.0]),
            w_out=WeightMatrix(shape=[1, 1], data=[0.75]),
            packed=None,
        ),
        readout=Readout(
            actions=[ActionReadout(id=0, name="a0", pos_place=0, neg_place=0)],
            gains=[1.0],
            abs_max=[1.0],
            slew_per_s=[10.0],
        ),
        initial_state=InitialState(
            marking=[0.1],
            place_injections=[
                PlaceInjection(place_id=0, source="sensor_x", scale=1.0, offset=0.0, clamp_0_1=True)
            ],
        ),
    )


def test_checksum_is_deterministic_hex64() -> None:
    """The checksum is a stable 64-char hex digest across repeated calls."""
    first = compute_proof_checksum()
    second = compute_proof_checksum()
    assert first == second
    assert _HEX64.match(first)


def test_checksum_depends_on_source(tmp_path: Path) -> None:
    """Editing a contract source changes the checksum (tamper-evidence)."""
    for name in CONTRACT_FILES:
        (tmp_path / name).write_text("-- placeholder\n", encoding="utf-8")
    baseline = compute_proof_checksum(tmp_path)
    (tmp_path / CONTRACT_FILES[0]).write_text("-- placeholder edited\n", encoding="utf-8")
    assert compute_proof_checksum(tmp_path) != baseline


def test_checksum_missing_file_raises(tmp_path: Path) -> None:
    """A broken proof bundle (missing file) must not silently produce a checksum."""
    (tmp_path / CONTRACT_FILES[0]).write_text("-- only one file\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Lean contract proof file missing"):
        compute_proof_checksum(tmp_path)


def test_contract_files_exist_in_repo() -> None:
    """The declared contract files are present in the repository's Lean package."""
    lean_dir = default_lean_dir()
    for name in CONTRACT_FILES:
        assert (lean_dir / name).is_file(), name


def test_manifest_structure() -> None:
    """The manifest is self-describing: system, files, theorems, checksum."""
    manifest = proof_contract_manifest()
    assert manifest["proof_system"] == PROOF_SYSTEM == "lean4"
    assert manifest["files"] == list(CONTRACT_FILES)
    assert manifest["theorems"] == list(CONTRACT_THEOREMS)
    assert _HEX64.match(manifest["checksum"])
    # The dynamic interlock + replay theorems are the M-1 additions.
    for theorem in (
        "compile_preserves_interlock_block",
        "compile_replay_commutes",
        "replay_keeps_guard_clear",
    ):
        assert theorem in manifest["theorems"]


def test_stamp_proof_contract_sets_compiler_fields() -> None:
    """Stamping fills the compiler proof identity with the live checksum."""
    artifact = _build_artifact()
    assert artifact.meta.compiler.proof_checksum is None
    stamp_proof_contract(artifact)
    assert artifact.meta.compiler.proof_system == "lean4"
    assert artifact.meta.compiler.proof_checksum == compute_proof_checksum()


def test_stamped_artifact_roundtrips(tmp_path: Path) -> None:
    """The proof checksum survives save/load and passes strict validation."""
    artifact = stamp_proof_contract(_build_artifact())
    path = tmp_path / "stamped.scpnctl.json"
    save_artifact(artifact, path)
    validate_artifact(artifact, error_type=ValueError)
    loaded = load_artifact(path)
    assert loaded.meta.compiler.proof_system == "lean4"
    assert loaded.meta.compiler.proof_checksum == artifact.meta.compiler.proof_checksum


def test_unstamped_artifact_omits_proof_fields(tmp_path: Path) -> None:
    """An unstamped artifact stays backwards compatible (no proof keys emitted)."""
    import json

    artifact = _build_artifact()
    path = tmp_path / "plain.scpnctl.json"
    save_artifact(artifact, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "proof_checksum" not in raw["meta"]["compiler"]
    assert "proof_system" not in raw["meta"]["compiler"]
    loaded = load_artifact(path)
    assert loaded.meta.compiler.proof_checksum is None
