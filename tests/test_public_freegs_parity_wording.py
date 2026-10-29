# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core public FreeGS strict-parity wording tests

"""Bind the public FreeGS strict-parity wording to the tracked evidence reports.

The public README, benchmark guide, validation status page and landing page
once called the FreeGS strict-parity lane accepted while the tracked rollup and
the strict-parity classifier reported it as blocked. These tests read the
tracked reports and refuse public prose that claims more than they admit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ROLLUP = ROOT / "validation" / "reports" / "external_parity_score.json"
STRICT = ROOT / "validation" / "reports" / "free_boundary_strict_parity_benchmark.json"
FREEGS_LANE = "freegsnke_free_boundary"
PUBLIC_DOCS = (
    "README.md",
    "docs/BENCHMARKS.md",
    "docs/PHYSICS_VALIDATION_STATUS.md",
    "docs/pages/index.html",
)
ACCEPTANCE_PHRASES = (
    "strict parity is accepted",
    "accepted free-boundary strict parity",
    "free-boundary strict parity is accepted",
    "freegs strict-parity lane are accepted",
    "freegs/freegsnke is accepted",
    "freegs/freegsnke free-boundary lane is accepted",
    "strict-parity gate is accepted",
    "accepted strict gate",
    "accepted public freegs lane",
    "current local result is\n`accepted_full_fidelity_free_boundary_parity`",
)


def _load(path: Path) -> dict[str, Any]:
    """Return a tracked JSON report as a mapping."""
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path} is not a JSON object"
    return data


def _freegs_lane(rollup: dict[str, Any]) -> dict[str, Any]:
    """Return the single FreeGS lane row of the external parity rollup."""
    lanes: list[dict[str, Any]] = [
        lane
        for lane in rollup["lanes"]
        if isinstance(lane, dict) and lane.get("lane") == FREEGS_LANE
    ]
    assert len(lanes) == 1, f"expected one {FREEGS_LANE} lane, found {len(lanes)}"
    return lanes[0]


def _public_text(name: str) -> str:
    """Return a public document lower-cased, with line breaks kept."""
    return (ROOT / name).read_text(encoding="utf-8").lower()


def test_blocked_freegs_lane_is_not_called_accepted_in_public_docs() -> None:
    """A blocked FreeGS lane must not appear as accepted in any public document."""
    lane = _freegs_lane(_load(ROLLUP))
    strict = _load(STRICT)
    blocked = not str(lane["status"]).startswith("accepted") or not str(
        strict["status"]
    ).startswith("accepted")
    if not blocked:
        return
    for name in PUBLIC_DOCS:
        text = _public_text(name)
        for phrase in ACCEPTANCE_PHRASES:
            assert phrase not in text, f"{name} still says {phrase!r} for a blocked lane"


def test_readme_quotes_the_tracked_rollup_scores() -> None:
    """The README sentence about the rollup quotes its current scores."""
    rollup = _load(ROLLUP)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    expected = (
        f"reproducibility score `{rollup['reproducibility_score']}`\n"
        f"  and parity score `{rollup['parity_score']}`"
    )
    assert expected in readme


def test_benchmarks_guide_quotes_the_tracked_rollup_scores() -> None:
    """The benchmark guide row for the rollup quotes its current scores and lane."""
    rollup = _load(ROLLUP)
    lane = _freegs_lane(rollup)
    guide = (ROOT / "docs" / "BENCHMARKS.md").read_text(encoding="utf-8")
    assert f"reproducibility score is `{rollup['reproducibility_score']}`" in guide
    assert f"parity score is `{rollup['parity_score']}`" in guide
    assert f"`{lane['status']}`\nwith a parity score of `{lane['parity_score']}`" in guide


def test_phrase_guard_detects_the_old_wording() -> None:
    """The guard phrases match the wording this test file was written to remove."""
    old = (
        "the rollup keeps the overall status blocked with reproducibility score `1.0`\n"
        "  and parity score `0.6`: freegs/freegsnke strict parity is accepted, while"
    )
    assert any(phrase in old for phrase in ACCEPTANCE_PHRASES)
