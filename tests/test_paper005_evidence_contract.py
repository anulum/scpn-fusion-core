# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — submission 005 exact-evidence contract tests.
"""Verify submission 005 against its tracked evidence and the sources it describes."""

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
SUBMISSION = (
    ROOT
    / "papers"
    / "submissions"
    / "005_regime_conditional_conformal_coverage_of_a_neural_transport_surrogate"
)
EVIDENCE = SUBMISSION / "evidence"


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object used by the contract."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _generator() -> ModuleType:
    """Import the package's evidence generator as a module, without running it."""
    spec = importlib.util.spec_from_file_location(
        "paper005_generator", SUBMISSION / "generate_evidence_manifest.py"
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
        "certificate": "surrogate_conformal_certificates",
        "regime": "qlknn_conformal_regime_coverage",
        "variants": "qlknn_conformal_variants",
        "sets": "qlknn_conformal_stability_sets",
        "envelope": "qlknn_conformal_profile_envelope",
        "self_consistent": "qlknn_conformal_self_consistent_envelope",
        "metrics": "neural_transport_qlknn.metrics",
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
    assert len(manifest["implementation_sources"]) == 6
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

    stability = evidence["sets"]["channels"]["chi_i"]["unstable"]
    stability["flux_coverage"] = 0.7926
    with pytest.raises(ValueError, match="unstable.flux_coverage"):
        generator.validate_quoted(evidence)


def test_nonzero_coverage_is_the_pooled_band_and_above_band_coverage() -> None:
    """The derived nonzero-flux coverage pools the two positive regimes by their counts."""
    generator = _generator()
    regimes = {
        "near_threshold": {"covered": 9, "n": 10},
        "unstable": {"covered": 21, "n": 30},
    }
    assert generator.nonzero_coverage(regimes) == pytest.approx(0.75)


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
        "0.697--0.738",
        "0.665--0.711",
        "0.792--0.797",
        "0.891--0.903",
        "0.262--7.224",
        "0.281--4.707",
        "change of 3.7\\%",
        "64.5--73.8\\%",
        "0.737 / 0.738 / 0.697",
        "0.797 (0.737)",
        "post hoc",
        "assume exchangeability",
        "not QLKNN itself",
    ):
        assert phrase in text, phrase


def test_weights_flags_match_the_manuscript_description() -> None:
    """The certificate's weights file stores no gate and no log transform."""
    np = pytest.importorskip("numpy")
    certificate = _load_json(EVIDENCE / "surrogate_conformal_certificates.json")
    model = certificate["certificates"][0]["model"]
    weights = ROOT / model["weights"]
    if hashlib.sha256(weights.read_bytes()).hexdigest() != model["weights_file_sha256"]:
        pytest.skip("the weights file is not the certified one (for example an LFS pointer)")
    archive = np.load(weights)
    assert int(archive["gated"]) == 0
    assert int(archive["log_transform"]) == 0
    assert [archive[f"w{index}"].shape[1] for index in (1, 2, 3, 4)] == [1024, 512, 256, 3]


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
