# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Real Studio consumer admission contracts
"""Exercise the installed SDK on freshly reproduced FUSION producer bytes."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from scpn_studio_platform.evidence import EvidenceBundle

pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")


@pytest.fixture
def source_checked_bundle(tmp_path: Path) -> EvidenceBundle:
    """Bind new evidence to a real emission checked against its canonical producer.

    The claim covers manifest byte reproduction only. It does not qualify the
    advertised solvers, certify a facility, or transform historical evidence.
    """
    from scpn_studio_platform.evidence import (
        AdmissionDecision,
        ClaimBoundary,
        ClaimStatus,
        EvidenceBundle,
        EvidenceKind,
        EvidenceLevel,
        Freshness,
        ProvActivity,
        ProvAgent,
        ProvEntity,
        ValidityDomain,
    )

    from scpn_fusion import studio
    from scpn_fusion.studio.manifest import STUDIO_VERSION

    started = datetime.now(timezone.utc).isoformat()
    emitted = studio.write_federation_document(tmp_path).read_bytes()
    canonical = (
        Path(__file__).resolve().parents[1] / "docs/_generated/studio_manifest.json"
    ).read_bytes()
    digest = "sha256:" + hashlib.sha256(emitted).hexdigest()
    reference_digest = "sha256:" + hashlib.sha256(canonical).hexdigest()
    assert studio.compare_bit_exact(digest, reference_digest).reproduced
    return EvidenceBundle(
        schema="studio.evidence-replay.v1",
        entity=ProvEntity("fusion-manifest-byte-reproduction", digest),
        activity=ProvActivity(
            verb="replay",
            studio=studio.STUDIO_ID,
            started=started,
            ended=datetime.now(timezone.utc).isoformat(),
            regenerated_by="scpn-emit-studio-manifest",
        ),
        agent=ProvAgent(STUDIO_VERSION, "fusion-consumer-contract-test"),
        evidence_level=EvidenceLevel.ENGINEERING_VERIFIED,
        evidence_kind=EvidenceKind.MEASURED,
        freshness=Freshness.VERIFIED_AT_SOURCE,
        claim_boundary=ClaimBoundary(
            ClaimStatus.REFERENCE_VALIDATED,
            AdmissionDecision.ADMITTED,
            validity_domain=ValidityDomain(
                note="Byte reproduction of the canonical manifest; no solver or facility admission."
            ),
        ),
    )


def test_consumer_admits_fresh_source_checked_producer(
    source_checked_bundle: EvidenceBundle,
) -> None:
    """The SDK ingests the wire form and preserves the real producer digest."""
    from scpn_studio_platform.evidence import validate_studio_bundle

    wire = json.loads(json.dumps(source_checked_bundle.to_dict()))
    verdict = validate_studio_bundle(wire, era="v2")
    assert verdict.admitted and verdict.mode == "validated"
    assert verdict.rejections == ()
    assert wire["prov"]["entity"]["digest"] == source_checked_bundle.entity.digest
    assert wire["freshness"] == "verified-at-source"


@pytest.mark.parametrize("freshness", [None, "traceable-unchecked", "untraceable"])
def test_consumer_refuses_missing_or_stale_freshness(
    source_checked_bundle: EvidenceBundle, freshness: str | None
) -> None:
    """New reference-validated admissions cannot omit or falsify recency."""
    from scpn_studio_platform.evidence import validate_studio_bundle

    wire = source_checked_bundle.to_dict()
    wire["freshness"] = freshness
    before = json.dumps(wire, sort_keys=True)
    verdict = validate_studio_bundle(wire, era="v2")
    assert not verdict.admitted and verdict.mode == "boundary"
    assert any("freshness" in reason for reason in verdict.rejections)
    assert json.dumps(wire, sort_keys=True) == before


@pytest.mark.parametrize("freshness", [None, "traceable-unchecked", "untraceable"])
def test_producer_refuses_unchecked_validated_claim(
    source_checked_bundle: EvidenceBundle, freshness: str | None
) -> None:
    """Producer construction enforces the same freshness rule as real ingest."""
    from scpn_studio_platform.evidence import Freshness

    with pytest.raises(ValueError, match="freshness"):
        replace(
            source_checked_bundle,
            freshness=None if freshness is None else Freshness(freshness),
        )


def test_historical_v1_wire_is_preserved_and_refused_for_new_admission(
    source_checked_bundle: EvidenceBundle,
) -> None:
    """Legacy replay may retain its verdict; a new v2 boundary cannot relabel it."""
    from scpn_studio_platform.evidence import validate_studio_bundle

    historical = source_checked_bundle.to_dict()
    del historical["freshness"]
    original = json.dumps(historical, sort_keys=True).encode()
    assert validate_studio_bundle(historical, era="v1").mode == "validated"
    assert not validate_studio_bundle(historical, era="v2").admitted
    assert json.dumps(historical, sort_keys=True).encode() == original


def test_bounded_evidence_stays_at_its_boundary(source_checked_bundle: EvidenceBundle) -> None:
    """An admitted reduced-model record remains bounded with unchecked freshness."""
    from scpn_studio_platform.evidence import (
        AdmissionDecision,
        ClaimBoundary,
        ClaimStatus,
        Freshness,
        validate_studio_bundle,
    )

    bounded = replace(
        source_checked_bundle,
        freshness=Freshness.TRACEABLE_UNCHECKED,
        claim_boundary=ClaimBoundary(ClaimStatus.BOUNDED_MODEL, AdmissionDecision.ADMITTED),
    )
    verdict = validate_studio_bundle(bounded.to_dict(), era="v2")
    assert verdict.admitted and verdict.mode == "boundary"


def test_real_manifest_requires_a_shared_era() -> None:
    """The same actual producer is accepted on v2 and refused by a v1 consumer."""
    from scpn_studio_platform.manifest import validate_studio_manifest

    from scpn_fusion.studio import build_federation_document

    wire = build_federation_document()["schema_a"]
    assert validate_studio_manifest(wire, supported_eras={"v2"}).admitted
    refused = validate_studio_manifest(wire, supported_eras={"v1"})
    assert not refused.admitted
    assert any("no common era" in reason for reason in refused.rejections)


def test_real_keeper_retains_its_default_era_policy() -> None:
    """Installing a compatible SDK does not grant a v1 keeper authority over v2."""
    from scpn_studio_platform.manifest.aggregate import aggregate_federation

    from scpn_fusion.studio import build_federation_document

    with pytest.raises(ValueError, match="no common era"):
        aggregate_federation([build_federation_document()])


def test_registered_emitter_and_drift_guard(tmp_path: Path) -> None:
    """Run the installed console entry point with the actual optional SDK."""
    from scpn_studio_platform.manifest import validate_studio_manifest

    emitter = Path(sys.executable).with_name("scpn-emit-studio-manifest")
    emitted = subprocess.run([str(emitter)], cwd=tmp_path, capture_output=True, text=True)
    assert emitted.returncode == 0, emitted.stderr
    wire = json.loads((tmp_path / "docs/_generated/studio_manifest.json").read_text())
    assert validate_studio_manifest(wire["schema_a"], supported_eras={"v2"}).admitted
    checked = subprocess.run(
        [str(emitter), "--check"], cwd=tmp_path, capture_output=True, text=True
    )
    assert checked.returncode == 0, checked.stderr
    (tmp_path / "docs/_generated/studio_manifest.json").write_text("{}\n")
    drift = subprocess.run([str(emitter), "--check"], cwd=tmp_path, capture_output=True, text=True)
    assert drift.returncode == 1 and "stale" in drift.stderr


def test_new_producer_seal_survives_ingest_and_detects_tampering(
    source_checked_bundle: EvidenceBundle,
) -> None:
    """A real temporary Ed25519 signature binds this new source check only."""
    from scpn_studio_platform.evidence import validate_studio_bundle
    from scpn_studio_platform.manifest.aggregate import aggregate_federation
    from scpn_studio_platform.seal import Ed25519Signer, Keyring, Verdict, seal, verify

    from scpn_fusion.studio import build_federation_document

    signer = Ed25519Signer.generate("fusion:consumer-contract-test")
    keyring = Keyring()
    keyring.add(signer.key_id, signer.verifier())
    envelope = seal(
        source_checked_bundle.to_dict(),
        signer=signer,
        grader={"name": "studio-bundle-federation", "version": "v2"},
        verifiability_mode="recompute",
        exactness_class="bit-exact",
    ).to_dict()
    assert (
        verify(
            envelope,
            "validated",
            keyring=keyring,
            regrade=lambda unit: validate_studio_bundle(unit).mode,
        )
        is Verdict.VERIFIED
    )
    document = build_federation_document()
    entry = {
        "id": source_checked_bundle.entity.entity_id,
        "bundle": source_checked_bundle.to_dict(),
        "seal": envelope,
    }
    snapshot = aggregate_federation([document], evidence=[entry], supported_eras={"v2"})
    assert snapshot["studios"] == [document]
    assert snapshot["evidence"] == [entry]
    stale = json.loads(json.dumps(entry))
    stale["bundle"]["freshness"] = "traceable-unchecked"
    with pytest.raises(ValueError, match="stale freshness"):
        aggregate_federation([document], evidence=[stale], supported_eras={"v2"})
    original = json.dumps(envelope, sort_keys=True)
    altered = json.loads(original)
    altered["unit"]["freshness"] = "traceable-unchecked"
    assert (
        verify(
            altered,
            "validated",
            keyring=keyring,
            regrade=lambda unit: validate_studio_bundle(unit).mode,
        )
        is Verdict.FORGED
    )
    assert json.dumps(envelope, sort_keys=True) == original


def test_keeper_cli_negotiates_and_preserves_output_on_refusal(tmp_path: Path) -> None:
    """The public keeper CLI ingests the real emission and refuses a mismatched era."""
    from scpn_fusion.studio import write_federation_document

    manifest_path = write_federation_document(tmp_path)
    output = tmp_path / "federation.json"
    command = [
        sys.executable,
        "-m",
        "scpn_studio_platform.manifest.aggregate",
        "--root",
        str(tmp_path),
        str(manifest_path),
        "--out",
        str(output),
    ]
    accepted = subprocess.run([*command, "--supported-era", "v2"], capture_output=True, text=True)
    assert accepted.returncode == 0, accepted.stderr
    assert json.loads(output.read_text())["studios"] == [json.loads(manifest_path.read_text())]
    before = output.read_bytes()
    refused = subprocess.run(command, capture_output=True, text=True)
    assert refused.returncode != 0 and "no common era" in refused.stderr
    assert output.read_bytes() == before
