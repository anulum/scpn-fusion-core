# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Official TGLF Development Plan Tests
"""Public-surface tests for the deterministic TGLF-DATA-03 sampling plan."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, cast

import pytest

from scpn_fusion.core import tglf_interface
from scpn_fusion.io.tglf_development_plan import (
    TGLF_DEVELOPMENT_GACODE_REVISION,
    TGLF_DEVELOPMENT_PLAN_VERSION,
    TGLF_DEVELOPMENT_SEED,
    build_tglf_development_plan,
    canonical_tglf_development_digest,
    validate_tglf_development_plan,
)


def _rehash(plan: dict[str, Any]) -> None:
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    plan["plan_sha256"] = canonical_tglf_development_digest(payload)


def test_development_plan_is_deterministic_and_covers_frozen_cells() -> None:
    """The public plan replays byte-for-byte with every frozen coverage cell."""
    first = build_tglf_development_plan()
    second = tglf_interface.build_tglf_development_plan()
    assert first == second
    assert validate_tglf_development_plan(first) == first
    assert first["schema_version"] == TGLF_DEVELOPMENT_PLAN_VERSION
    assert first["gacode_revision"] == TGLF_DEVELOPMENT_GACODE_REVISION
    assert first["seed"] == TGLF_DEVELOPMENT_SEED
    assert first["base_groups"] == 24
    assert first["accepted_runs"] == 72
    assert (
        first["plan_sha256"] == "a004de110868bfd5cd33fb545dd917eb8416f4e0654a2b541daaa39f5e5856f5"
    )
    runs = cast(list[dict[str, Any]], first["runs"])
    assert Counter(run["sampling_stratum"] for run in runs) == {
        "interior": 36,
        "boundary": 18,
        "threshold": 18,
    }
    assert Counter(run["composition"] for run in runs) == {
        "electron-deuterium": 24,
        "electron-deuterium-tritium": 24,
        "electron-deuterium-carbon": 24,
    }
    assert [item["reason"] for item in first["negative_controls"]] == [
        "non_quasineutral_composition",
        "minor_radius_not_less_than_major_radius",
        "electromagnetic_switch_requires_positive_beta",
    ]


def test_fixture_plan_preserves_gradient_pairs_and_physical_invariants() -> None:
    """Every compact fixture group varies only its declared density gradient."""
    plan = build_tglf_development_plan(profile="fixture")
    assert plan["plan_sha256"] == (
        "102ba43f1f9a495d99e291bef36cdf6acde8e31809d0ba4a2d03392390080047"
    )
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in cast(list[dict[str, Any]], plan["runs"]):
        groups[cast(str, run["group_id"])].append(run)
        deck = cast(dict[str, Any], run["input"])
        species = cast(list[dict[str, Any]], deck["species"])
        charge_density = sum(
            float(item["charge_e"]) * float(item["density_e_ratio"]) for item in species
        )
        assert charge_density == pytest.approx(0.0, abs=1.0e-12)
        assert 0.0 < float(deck["a_minor"]) < float(deck["R_major"])
        if deck["use_bper"] or deck["use_bpar"]:
            assert float(deck["beta_e"]) > 0.0
    assert len(groups) == 3
    for runs in groups.values():
        assert len(runs) == 3
        target = runs[0]["paired_gradient_species"]
        gradients: list[float] = []
        scrubbed: list[dict[str, Any]] = []
        for run in runs:
            copied = deepcopy(run["input"])
            for species in copied["species"]:
                if species["name"] == target:
                    gradients.append(float(species.pop("R_Ln")))
            scrubbed.append(copied)
        assert gradients[1] - gradients[0] == pytest.approx(0.5)
        assert gradients[2] - gradients[1] == pytest.approx(0.5)
        assert scrubbed[0] == scrubbed[1] == scrubbed[2]


@pytest.mark.parametrize(
    ("seed", "profile", "message"),
    [(-1, "development", "seed"), (True, "development", "seed"), (1, "promotion", "profile")],
)
def test_plan_builder_rejects_invalid_public_inputs(seed: Any, profile: str, message: str) -> None:
    """Invalid seed and profile selections fail before sampling."""
    with pytest.raises(ValueError, match=message):
        build_tglf_development_plan(seed=seed, profile=profile)


def test_plan_validator_rejects_hash_identity_and_profile_drift() -> None:
    """Digest-valid plan mutations still fail exact deterministic regeneration."""
    plan = build_tglf_development_plan(profile="fixture")
    plan["runs"][0]["sampling_stratum"] = "changed"
    with pytest.raises(ValueError, match="SHA-256"):
        validate_tglf_development_plan(plan)

    plan = build_tglf_development_plan(profile="fixture")
    plan["design_method"] = "changed"
    _rehash(plan)
    with pytest.raises(ValueError, match="deterministic regeneration"):
        validate_tglf_development_plan(plan)

    plan = build_tglf_development_plan(profile="fixture")
    plan["seed"] = True
    _rehash(plan)
    with pytest.raises(ValueError, match="seed/profile"):
        validate_tglf_development_plan(plan)
