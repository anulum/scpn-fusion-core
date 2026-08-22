# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Benchmark Result Collection Tests
"""Artifact-boundary tests for the benchmark result collector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from validation import collect_results


def test_real_shot_loader_reads_json_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public real-shot loader preserves a valid object-root artifact."""
    payload = {"overall_pass": False, "disruption": {"recall": 0.75}}
    artifact_path = tmp_path / "real_shot_validation.json"
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(collect_results, "ARTIFACTS", tmp_path)

    assert collect_results.load_real_shot_validation() == payload


def test_real_shot_loader_rejects_non_object_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A list-root artifact fails closed before report rendering."""
    artifact_path = tmp_path / "real_shot_validation.json"
    artifact_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(collect_results, "ARTIFACTS", tmp_path)

    with pytest.raises(ValueError, match="JSON object with string keys"):
        collect_results.load_real_shot_validation()
