# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — exact legacy manuscript-layout retirement tests.
"""Verify that retired paper paths remain exact, retrievable Git objects."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "papers" / "legacy_layout_manifest.json"
GENERATOR = ROOT / "papers" / "generate_legacy_layout_manifest.py"


def _load_manifest() -> dict[str, Any]:
    """Load the migration manifest as one JSON object."""
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_retirement_manifest_regenerates_from_exact_git_objects() -> None:
    """Require every retired byte and successor mapping to regenerate exactly."""
    result = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), str(GENERATOR), "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "[OK] 58 retired paper paths retain exact Git-object custody"


def test_manifest_retires_only_declared_layout_and_denies_equivalence() -> None:
    """Keep the retired inventory exact and the successor claims conservative."""
    manifest = _load_manifest()
    files = manifest["files"]

    assert manifest["retired_path_count"] == 58
    assert len(files) == 58
    assert set(manifest["current_packages"]) == {
        "papers/submissions/001_hybrid_rust_python_grad_shafranov_equilibrium_solver",
        "papers/submissions/002_neuromorphic_vertical_stability_control",
        "papers/submissions/003_stochastic_petri_net_tokamak_control_conference_abstract",
    }
    assert (
        "byte equivalence with legacy content is not claimed"
        in manifest["claim_boundary"]["not_claimed"]
    )
    assert all(
        path.startswith("papers/arxiv_submission_b/")
        or path.startswith("papers/figures/")
        or path
        in {
            "papers/paper_a_equilibrium_solver.tex",
            "papers/paper_b_snn_controller.tex",
            "papers/scpn_fusion.bib",
        }
        for path in files
    )
    assert all(not (ROOT / path).exists() for path in files)


def test_each_successor_is_present_or_explicitly_historical() -> None:
    """Require a real package successor or a named history-only disposition."""
    files = _load_manifest()["files"]
    for record in files.values():
        successors = record["successor_paths"]
        if successors:
            assert all((ROOT / path).is_file() for path in successors)
        else:
            assert record["disposition"] in {
                "generated_auxiliary_reproducible_from_package",
                "historical_only_not_used_by_current_manuscripts",
            }
