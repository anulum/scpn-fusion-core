# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Direct contract tests for the IDA fixed-reference source ablation."""

from __future__ import annotations

import validation.ida_fixed_reference_source_contract as contract


def test_contract_keeps_all_claims_false_and_binds_both_modules() -> None:
    """The frozen contract must bind both implementation surfaces and no admission."""
    assert set(contract.SOURCE_PATHS) >= {"ablation", "contract", "solver"}
    assert contract.CLAIM_FIELDS == (
        "control_admission",
        "facility_validation",
        "pcs_deployment",
        "scientific_validation",
        "safety_admission",
    )
    assert contract.ROUTING_THRESHOLDS["fixed_reference_tv_max"] == 0.02
