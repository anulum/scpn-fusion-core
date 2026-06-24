# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Studio schema-A manifest tests
"""Tests for the FUSION studio federation manifest (schema A + extension)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")

from scpn_fusion.studio import federation, manifest, verbs  # noqa: E402


def test_manifest_builds_with_studio_identity() -> None:
    """The schema-A manifest carries the FUSION studio identity and all verbs."""
    data = manifest.build_manifest().to_dict()
    assert data["studio"] == "scpn-fusion-core"
    assert len(data["verbs"]) == len(verbs.FUSION_VERBS) == 8
    assert data["content_digest"].startswith("sha256:")
    assert data["contract_era"] == "v1"
    assert data["platform_sdk"] == ">=0.8,<0.9"
    assert data["transport_profile"] == "local-first"


def test_studio_version_is_a_non_empty_string() -> None:
    """The manifest stamps the installed distribution version, never a blank."""
    resolved = manifest._resolve_studio_version()
    assert isinstance(resolved, str)
    assert resolved
    assert manifest.build_manifest().to_dict()["studio_version"] == resolved


def test_content_digest_is_reproducible() -> None:
    """The content digest depends on the declared surface, not git state."""
    first = manifest.build_manifest().to_dict()["content_digest"]
    second = manifest.build_manifest().to_dict()["content_digest"]
    assert first == second


def test_declared_surface_covers_every_verb_and_evidence() -> None:
    """The hashed surface has one canonical-JSON entry per verb plus the evidence list."""
    surface = manifest.declared_surface()
    expected = {f"verb/{v.name}" for v in verbs.FUSION_VERBS} | {"evidence/schemas"}
    assert set(surface) == expected
    assert all(isinstance(payload, bytes) for payload in surface.values())


def test_evidence_schemas_match_verb_outputs() -> None:
    """Every evidence schema a verb produces is declared in evidence_schemas()."""
    declared = set(verbs.evidence_schemas())
    produced = {schema for verb in verbs.FUSION_VERBS for schema in verb.produces}
    assert produced == declared
    assert all(name.startswith("studio.") and name.endswith(".v1") for name in declared)


def test_verb_vocabulary_uses_locked_enums() -> None:
    """Verb attributes serialise to the locked SDK enum value strings."""
    data = manifest.build_manifest().to_dict()
    by_name = {v["verb"]: v for v in data["verbs"]}
    assert by_name["reconstruct"]["side_effect"] == "read-only"
    assert by_name["reconstruct"]["fidelity"] == "first-principles"
    assert by_name["simulate"]["side_effect"] == "simulated"
    assert by_name["validate"]["fidelity"] == "analytic"


def test_no_verb_touches_live_hardware() -> None:
    """FUSION never drives a real machine: no live-hardware verb, every tier research."""
    data = manifest.build_manifest().to_dict()
    assert not [v["verb"] for v in data["verbs"] if v["side_effect"] == "live-hardware"]
    assert {v["safety_tier"] for v in data["verbs"]} == {"research"}


def test_control_verb_is_realtime_with_the_10khz_deadline() -> None:
    """The closed-loop control verb is real-time with a 100 microsecond (10 kHz) deadline."""
    data = manifest.build_manifest().to_dict()
    control = next(v for v in data["verbs"] if v["verb"] == "control")
    assert control["timing"]["class"] == "realtime"
    assert control["timing"]["deadline_us"] == 100.0
    assert control["fidelity"] == "reduced-order"


def test_predict_verb_is_a_machine_learning_surrogate() -> None:
    """Disruption forecasting is advertised as an ML-surrogate, read-only verb."""
    data = manifest.build_manifest().to_dict()
    predict = next(v for v in data["verbs"] if v["verb"] == "predict")
    assert predict["fidelity"] == "ml-surrogate"
    assert predict["side_effect"] == "read-only"


def test_federation_document_has_both_blocks() -> None:
    """The federation document is schema_a core + architecture_map extension."""
    doc = federation.build_federation_document()
    assert set(doc) == {"schema_a", "architecture_map"}
    assert doc["schema_a"]["studio"] == "scpn-fusion-core"


def test_architecture_map_extension_shape() -> None:
    """The v2 extension block carries the fleet-aligned field set."""
    ext = federation.build_architecture_map_extension()
    assert ext["version"] == "architecture-map.v2"
    assert {
        "pipeline_stages",
        "capabilities",
        "backends",
        "interfaces",
        "wire_formats",
        "cross_repo",
        "boundaries",
    } <= set(ext)
    stages = [s["stage"] for s in ext["pipeline_stages"]]
    assert stages == [
        "configuration",
        "equilibrium",
        "transport",
        "stability",
        "control",
        "disruption",
        "evidence",
    ]
    assert {b["name"] for b in ext["backends"]} >= {"rust", "julia", "go", "python"}
    assert {b["status"] for b in ext["backends"]} <= {
        "runtime-active",
        "build-available",
        "declared",
    }
    assert {c["status"] for c in ext["capabilities"]} <= {
        "wired",
        "library-only",
        "stub",
        "build-available",
        "feasibility-only",
    }
    assert all({"kind", "entry"} <= set(i) for i in ext["interfaces"])
    assert all({"name", "schema_ref"} <= set(w) for w in ext["wire_formats"])
    assert all({"sibling", "adapter", "wire_format"} <= set(c) for c in ext["cross_repo"])
    assert {"executed", "bounded", "feasibility_only", "closed"} <= set(ext["boundaries"])


def test_backend_dispatch_order_is_fastest_first() -> None:
    """The dispatch order is strictly increasing with the Rust hot path first."""
    backends = federation.build_architecture_map_extension()["backends"]
    orders = [b["dispatch_order"] for b in backends]
    assert orders == sorted(orders)
    assert backends[0]["name"] == "rust"
    assert backends[-1]["name"] == "python"


def test_federation_document_is_json_serialisable() -> None:
    """The whole document round-trips through JSON."""
    doc = federation.build_federation_document()
    assert json.loads(json.dumps(doc)) == doc


def test_write_federation_document(tmp_path: Path) -> None:
    """The emitter writes a parseable federation document to the studio path."""
    out = federation.write_federation_document(tmp_path)
    assert out == tmp_path / federation.STUDIO_MANIFEST_PATH
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["schema_a"]["studio"] == "scpn-fusion-core"


def test_manifest_passes_studio_conformance_gate() -> None:
    """The CapabilityManifest is admitted by the platform ``validate_studio_manifest`` gate.

    Keeps the federation contract honest in CI: any schema-A drift (bad digest form,
    duplicate verb, unversioned evidence schema, unknown contract era) reds the build.
    """
    from scpn_studio_platform import manifest as platform_manifest

    validate = getattr(platform_manifest, "validate_studio_manifest", None)
    if validate is None:  # pragma: no cover - only on SDK < 0.8
        pytest.skip("validate_studio_manifest unavailable (scpn-studio-platform < 0.8)")
    verdict = validate(manifest.build_manifest().to_dict())
    assert verdict.admitted, f"manifest rejected: {verdict.rejections}"
    assert verdict.rejections == ()
    assert verdict.warnings == ()


def test_committed_studio_manifest_is_current() -> None:
    """The committed studio_manifest.json byte-matches the generator (no stale drift).

    Mirrors the capability-manifest drift guard: an edit to verbs/evidence/architecture-map
    that forgets to re-emit cannot ship a stale federation artefact for the keeper gate.
    """
    repo_root = Path(__file__).resolve().parents[1]
    drift = federation.studio_manifest_drift(repo_root)
    assert drift is None, drift


def test_studio_manifest_drift_reports_missing_file(tmp_path: Path) -> None:
    """An absent artefact is reported as missing, not silently treated as current."""
    drift = federation.studio_manifest_drift(tmp_path)
    assert drift is not None
    assert "missing" in drift


def test_studio_manifest_drift_reports_stale_file(tmp_path: Path) -> None:
    """A divergent artefact is reported as stale."""
    out = federation.write_federation_document(tmp_path)
    out.write_text("{}\n", encoding="utf-8")
    drift = federation.studio_manifest_drift(tmp_path)
    assert drift is not None
    assert "stale" in drift


def test_main_emits_then_check_passes(tmp_path: Path, monkeypatch, capsys) -> None:
    """``main`` writes the document, and a following ``--check`` sees it as current."""
    monkeypatch.chdir(tmp_path)
    assert federation.main([]) == 0
    assert "content_digest=sha256:" in capsys.readouterr().out
    assert federation.main(["--check"]) == 0
    assert "is current" in capsys.readouterr().out


def test_main_check_fails_when_artifact_missing(tmp_path: Path, monkeypatch, capsys) -> None:
    """``--check`` fails closed (exit 1) when no artefact has been emitted."""
    monkeypatch.chdir(tmp_path)
    assert federation.main(["--check"]) == 1
    assert "drift" in capsys.readouterr().err


def test_studio_package_reexports_federation_surface() -> None:
    """The package namespace re-exports the federation entry points."""
    import scpn_fusion.studio as studio_pkg

    for name in (
        "FUSION_VERBS",
        "STUDIO_ID",
        "build_manifest",
        "build_federation_document",
        "build_architecture_map_extension",
        "declared_surface",
        "evidence_schemas",
        "write_federation_document",
    ):
        assert hasattr(studio_pkg, name), name
    assert studio_pkg.STUDIO_ID == "scpn-fusion-core"
