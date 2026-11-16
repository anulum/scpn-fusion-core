# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — submission 004 exact-evidence contract tests.
"""Verify submission 004 against its tracked reports and the sources it describes."""

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
    ROOT / "papers" / "submissions" / "004_same_case_free_boundary_comparison_with_freegsnke"
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
        "paper004_generator", SUBMISSION / "generate_evidence_manifest.py"
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


def test_packaged_reports_are_the_tracked_reports_plus_the_public_header() -> None:
    """Each packaged report must be its historical Git object with the header placed first."""
    generator = _generator()
    manifest = _load_json(EVIDENCE / "evidence_manifest.json")
    assert set(manifest["source_custody"]) == {
        "freegsnke_multimachine_parity.json",
        "freegsnke_inverse_fit.json",
        "freegsnke_grid_convergence.json",
    }
    for name, custody in manifest["source_custody"].items():
        spec = f"{custody['source_revision']}:{custody['source_path']}"
        source = _git("show", spec)

        assert _git("rev-parse", spec).decode("ascii").strip() == custody["git_blob_oid_sha1"]
        assert hashlib.sha256(source).hexdigest() == custody["source_sha256"]
        assert (EVIDENCE / name).read_bytes() == generator.packaged_bytes(source)
        packaged = _load_json(EVIDENCE / name)
        assert list(packaged)[:7] == list(generator.TIER0_HEADER)
        assert generator.scientific_payload(packaged) == json.loads(source)


def test_described_sources_match_the_bound_revision_and_the_inverse_run() -> None:
    """Bind every described source to the package revision and to the inverse run's digests."""
    manifest = _load_json(EVIDENCE / "evidence_manifest.json")
    revision = manifest["repository_revision"]
    for relative_path, custody in manifest["implementation_sources"].items():
        spec = f"{revision}:{relative_path}"
        source = _git("show", spec)

        assert _git("rev-parse", spec).decode("ascii").strip() == custody["git_blob_oid_sha1"]
        assert hashlib.sha256(source).hexdigest() == custody["sha256"]
        assert (ROOT / relative_path).read_bytes() == source
    inverse = _load_json(EVIDENCE / "freegsnke_inverse_fit.json")
    for relative_path, digest in inverse["provenance"]["scpn_source_sha256"].items():
        assert manifest["implementation_sources"][relative_path]["sha256"] == digest
    report_revision = manifest["source_custody"]["freegsnke_inverse_fit.json"]["source_revision"]
    assert _is_ancestor(inverse["provenance"]["scpn_source_revision"], report_revision)
    assert _is_ancestor(report_revision, revision)


def _is_ancestor(older: str, newer: str) -> bool:
    """Return whether ``older`` is an ancestor of ``newer`` in the repository history."""
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", older, newer],
        cwd=ROOT,
        check=False,
        capture_output=True,
        timeout=30,
    )
    return completed.returncode == 0


def test_generator_rejects_a_changed_quoted_value() -> None:
    """A one-digit change in a quoted forward value must fail the generator's check."""
    generator = _generator()
    report = generator.scientific_payload(
        _load_json(EVIDENCE / "freegsnke_multimachine_parity.json")
    )
    generator.validate_forward(report)
    report["cases"]["sparc_like"]["vacuum_bridge"]["max_abs_error_wb"] *= 1.0000001
    with pytest.raises(ValueError, match="Forward-parity values drifted: sparc_like"):
        generator.validate_forward(report)


def test_generator_rejects_a_tracked_report_with_a_header() -> None:
    """A source that already carries public header fields is not a tracked report."""
    generator = _generator()
    with pytest.raises(ValueError, match="without public header fields"):
        generator.packaged_bytes(json.dumps({"orcid": "x", "status": "pass"}).encode("utf-8"))


def test_manuscript_names_its_revision_and_quotes_the_reports() -> None:
    """Keep the manuscript's numbers and boundary sentences tied to the evidence."""
    manuscript = " ".join((SUBMISSION / "manuscript.tex").read_text(encoding="utf-8").split())
    manifest = _load_json(EVIDENCE / "evidence_manifest.json")
    forward = _load_json(EVIDENCE / "freegsnke_multimachine_parity.json")

    assert manifest["repository_revision"][:12] in manuscript
    assert "@" not in manuscript.split("\\begin{document}", 1)[1].replace(
        "protoscience@anulum.li", ""
    )
    mastu = forward["cases"]["mastu_like"]
    assert mastu["cold_start"]["outcome"] == "no_magnetic_axis_inside_limiter"
    assert "no magnetic axis inside the limiter" in manuscript
    assert "$3.54 \\times 10^{-3}$ & $3.78 \\times 10^{-4}$" in manuscript
    assert "does not validate either code against experiment" in manuscript
    assert "lowers the frozen-support difference fourfold" in manuscript
    assert "Grid convergence is checked on the MAST-U-like case only, at two grids." in manuscript


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
        timeout=60,
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


def test_generator_rejects_a_changed_grid_value() -> None:
    """A changed grid-study ratio must fail the generator's check."""
    generator = _generator()
    report = generator.scientific_payload(_load_json(EVIDENCE / "freegsnke_grid_convergence.json"))
    generator.validate_grid(report)
    report["changes_against_base_grid"][0]["warm_rmse_ratio"] = 0.83
    with pytest.raises(ValueError, match="warm_rmse_ratio"):
        generator.validate_grid(report)
