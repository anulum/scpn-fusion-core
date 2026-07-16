# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Ballooning Second-Stability Reference Tests
from __future__ import annotations

import json

import pytest

from scpn_fusion.core.ballooning_second_stability import (
    ARTIFACT_SCHEMA,
    BallooningSecondStabilityReference,
)


def _payload(rows=None):
    """A schema-valid synthetic artifact payload (test double, not real physics)."""
    if rows is None:
        rows = [
            {"shat": 2.0, "alpha_crit": 1.35, "second_stability_access": False, "alpha_max": 8.0},
            {"shat": 0.5, "alpha_crit": None, "second_stability_access": True, "alpha_max": 8.0},
            {"shat": 1.0, "alpha_crit": 1.23, "second_stability_access": False, "alpha_max": 8.0},
        ]
    return {
        "schema": ARTIFACT_SCHEMA,
        "provenance": {"code": "pyrokinetics", "licence": "LGPL-3.0-or-later"},
        "diiid_shaping": {"kappa": 1.74, "delta": 0.3},
        "diiid_alpha_crit": rows,
    }


def test_from_payload_sorts_by_shear_and_fills_second_stability_ceiling():
    ref = BallooningSecondStabilityReference.from_payload(_payload())
    assert ref.shear == (0.5, 1.0, 2.0)
    # the shat=0.5 second-stability row is filled to the alpha_max ceiling
    assert ref.alpha_crit == (8.0, 1.23, 1.35)
    assert ref.second_stability_access == (True, False, False)
    assert ref.shaping["kappa"] == 1.74
    assert ref.provenance["licence"] == "LGPL-3.0-or-later"


def test_from_payload_rejects_foreign_schema():
    with pytest.raises(ValueError, match="schema"):
        BallooningSecondStabilityReference.from_payload({"schema": "other", "diiid_alpha_crit": []})


def test_from_payload_rejects_empty_table():
    with pytest.raises(ValueError, match="non-empty"):
        BallooningSecondStabilityReference.from_payload(_payload(rows=[]))


def test_from_payload_rejects_first_stability_row_without_alpha_crit():
    bad = [{"shat": 1.0, "alpha_crit": None, "second_stability_access": False, "alpha_max": 8.0}]
    with pytest.raises(ValueError, match="finite alpha_crit"):
        BallooningSecondStabilityReference.from_payload(_payload(rows=bad))


def test_from_artifact_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        BallooningSecondStabilityReference.from_artifact(tmp_path / "absent.json")


def test_from_artifact_reads_committed_json(tmp_path):
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    ref = BallooningSecondStabilityReference.from_artifact(path)
    assert ref.shear == (0.5, 1.0, 2.0)


def test_alpha_crit_at_interpolates_and_clamps():
    ref = BallooningSecondStabilityReference.from_payload(_payload())
    # midpoint between shat=1.0 (1.23) and shat=2.0 (1.35)
    assert ref.alpha_crit_at(1.5) == pytest.approx(1.29, abs=1e-6)
    # clamp below the grid -> first value (second-stability ceiling)
    assert ref.alpha_crit_at(0.1) == pytest.approx(8.0)
    # clamp above the grid -> last value
    assert ref.alpha_crit_at(9.0) == pytest.approx(1.35)


def test_has_second_stability_at_uses_nearest_grid_row():
    ref = BallooningSecondStabilityReference.from_payload(_payload())
    assert ref.has_second_stability_at(0.4) is True  # nearest 0.5 -> access
    assert ref.has_second_stability_at(1.1) is False  # nearest 1.0 -> no access
    assert ref.has_second_stability_at(2.2) is False  # nearest 2.0 -> no access
