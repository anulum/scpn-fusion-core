# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Direct invariants for the fixed-reference source-mechanism contract."""

from __future__ import annotations

import validation.ida_fixed_reference_source_mechanism_contract as contract


def test_contract_binds_cross_repo_mechanisms_and_keeps_claims_false() -> None:
    assert contract.CURRENT_FIELDS == (
        "freegs_hard_romberg",
        "freegs_hard_rectangular_normalised",
        "fusion_smooth_unscaled",
        "fusion_smooth_rectangular_normalised",
    )
    assert contract.MECHANISM_COMPONENTS == (
        "hard_rectangular_normalisation",
        "smooth_cutoff",
        "smooth_ip_normalisation",
    )
    assert contract.SOURCE_PATHS["control_profile_source"].startswith("../SCPN-CONTROL/")
    assert contract.CLAIM_FIELDS == (
        "control_admission",
        "facility_validation",
        "pcs_deployment",
        "safety_admission",
        "scientific_validation",
    )
