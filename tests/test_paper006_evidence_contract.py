# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — submission 006 exact-evidence contract tests.
"""Verify submission 006 against its tracked evidence and the sources it describes."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "papers" / "submissions" / "006_coupled_transport_model_intersection_with_torax"
EVIDENCE = SUBMISSION / "evidence"


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object used by the contract."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _generator() -> ModuleType:
    """Import the package's evidence generator as a module, without running it."""
    spec = importlib.util.spec_from_file_location(
        "paper006_generator", SUBMISSION / "generate_evidence_manifest.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(*arguments: str) -> bytes:
    """Return the standard output of one Git command in the repository."""
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, timeout=30
    ).stdout


def _evidence(generator: ModuleType) -> dict[str, dict[str, Any]]:
    """Load the package-local evidence under the keys the generator uses."""
    names = {
        "parity": "torax_coupled_transport_parity",
        "gradients": "torax_differentiable_transport",
        "reference": "torax_coupled_transport_reference",
    }
    return {
        key: generator.scientific_payload(_load_json(EVIDENCE / f"{name}.json"))
        for key, name in names.items()
    }


def test_packaged_files_are_the_tracked_files_plus_the_public_header() -> None:
    """Each packaged file must be its historical Git object with the header placed first."""
    generator = _generator()
    manifest = _load_json(EVIDENCE / "evidence_manifest.json")
    assert set(manifest["source_custody"]) == set(manifest["files"])
    for name, custody in manifest["source_custody"].items():
        spec = f"{custody['source_revision']}:{custody['source_path']}"
        source = _git("show", spec)

        assert _git("rev-parse", spec).decode("ascii").strip() == custody["git_blob_oid_sha1"]
        assert hashlib.sha256(source).hexdigest() == custody["source_sha256"]
        assert (EVIDENCE / name).read_bytes() == generator.packaged_bytes(source)
        assert generator.scientific_payload(_load_json(EVIDENCE / name)) == json.loads(source)


def test_described_sources_match_the_bound_revision() -> None:
    """Bind every analysis module to the package revision and to the working tree."""
    manifest = _load_json(EVIDENCE / "evidence_manifest.json")
    revision = manifest["repository_revision"]
    assert len(manifest["implementation_sources"]) == 11
    for relative_path, custody in manifest["implementation_sources"].items():
        spec = f"{revision}:{relative_path}"
        source = _git("show", spec)

        assert _git("rev-parse", spec).decode("ascii").strip() == custody["git_blob_oid_sha1"]
        assert hashlib.sha256(source).hexdigest() == custody["sha256"]
        assert (ROOT / relative_path).read_bytes() == source


def test_every_quoted_value_is_recomputed_from_the_evidence() -> None:
    """The generator's table recomputes each quoted number; a changed digit must fail."""
    generator = _generator()
    evidence = _evidence(generator)
    generator.validate_quoted(evidence)
    assert (
        len(generator.quoted()) == _load_json(EVIDENCE / "evidence_manifest.json")["quoted_values"]
    )

    evidence["parity"]["state_metrics"]["electron_temperature_kev"] = 0.0391
    with pytest.raises(ValueError, match="electron_temperature_kev"):
        generator.validate_quoted(evidence)


def test_generator_rejects_a_tracked_file_with_a_header() -> None:
    """A source that already carries public header fields is not a tracked file."""
    generator = _generator()
    with pytest.raises(ValueError, match="without public header fields"):
        generator.packaged_bytes(json.dumps({"orcid": "x"}).encode("utf-8"))


def test_manuscript_names_its_revision_and_quotes_the_key_results() -> None:
    """Keep the manuscript's headline numbers and boundaries tied to the evidence."""
    text = " ".join((SUBMISSION / "manuscript.tex").read_text(encoding="utf-8").split())
    manifest = _load_json(EVIDENCE / "evidence_manifest.json")

    assert manifest["repository_revision"][:12] in text
    for phrase in (
        "two steps of 10\\,ms on 16 cells",
        "$3.9 \\times 10^{-2}$ in electron temperature",
        "does not recover the scales",
        "1.269, 0.872 and 0.999",
        "a factor of 174",
        "no speed comparison",
        "Nothing is claimed for the production transport models",
    ):
        assert phrase in text, phrase


def test_the_cited_reports_record_the_bound_sources() -> None:
    """Each report's own source digests equal the digests the package binds."""
    manifest = _load_json(EVIDENCE / "evidence_manifest.json")
    bound = {
        Path(path).name: custody["sha256"]
        for path, custody in manifest["implementation_sources"].items()
    }
    parity = _load_json(EVIDENCE / "torax_coupled_transport_parity.json")["source_provenance"]
    assert parity["coupled_runtime_sha256"] == bound["integrated_transport_solver_coupled.py"]
    assert parity["deck_sha256"] == bound["coupled_transport_model_intersection_deck.py"]
    gradients = _load_json(EVIDENCE / "torax_differentiable_transport.json")["source_provenance"]
    assert gradients["runtime_sha256"] == bound["integrated_transport_solver_differentiable.py"]
    assert gradients["benchmark_sha256"] == bound["benchmark_torax_differentiable_transport.py"]


def test_evidence_generator_is_byte_deterministic(tmp_path: Path) -> None:
    """Regenerate the evidence in a real checkout without a repository-local venv."""
    checkout = tmp_path / "checkout"
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", "--no-checkout", str(ROOT), str(checkout)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    manifest = _load_json(EVIDENCE / "evidence_manifest.json")
    paths = [
        SUBMISSION.relative_to(ROOT).as_posix(),
        *(custody["source_path"] for custody in manifest["source_custody"].values()),
        *manifest["implementation_sources"],
    ]
    subprocess.run(
        ["git", "checkout", "HEAD", "--", *paths],
        cwd=checkout,
        check=True,
        capture_output=True,
        timeout=120,
    )
    assert not (checkout / ".venv").exists()
    submission = checkout / SUBMISSION.relative_to(ROOT)
    tracked = sorted((submission / "evidence").glob("*.json"))
    before = {path: path.read_bytes() for path in tracked}

    subprocess.run(
        [sys.executable, str(submission / "generate_evidence_manifest.py")],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert {path: path.read_bytes() for path in tracked} == before
