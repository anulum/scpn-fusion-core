# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Serialisation-contract test for the disruption structural-response report.

The subsystem fault-model tests exercise the evaluation path but never call the
report's ``to_dict`` serialiser; this closes that helper, which JSON-normalises
the frozen report and converts the failure-reason tuple to a list.
"""

from __future__ import annotations

from scpn_fusion.core.disruption_structural_response import (
    evaluate_disruption_structural_response,
)


def test_report_to_dict_is_json_normalised() -> None:
    """``to_dict`` returns a plain dict with the reason tuple flattened to a list."""
    report = evaluate_disruption_structural_response()
    payload = report.to_dict()

    assert isinstance(payload, dict)
    assert isinstance(payload["failure_reasons"], list)
    assert list(payload["failure_reasons"]) == list(report.failure_reasons)
    # The serialised payload mirrors the dataclass scalar fields verbatim.
    assert payload["status"] == report.status
    assert payload["passes_thresholds"] == report.passes_thresholds
    assert payload["claim_boundary"] == report.claim_boundary
