# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — RustBCA public request tests
"""Exercise the declared request against the actual retained D/W case."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from validation.rustbca_request import validate_rustbca_request

REQUEST = Path(__file__).resolve().parents[2] / "validation/reference_data/rustbca_request.json"


def test_normalization_detaches_caller_mutable_values() -> None:
    """Keep future caller changes from altering the accepted sampling request."""
    original = json.loads(REQUEST.read_bytes())
    result = validate_rustbca_request(original)
    original["ion"]["m"] = 3.0
    original["seeds"].append(9)
    original["energies_eV"][0] = 200.0
    assert result["ion"]["m"] == 2.0
    assert result["seeds"] == [0, 1, 2, 3]
    assert result["energies_eV"] == [100.0, 1000.0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "unknown"),
        ("extra", True),
        ("threads", True),
        ("threads", 0),
        ("samples_per_seed", 0),
        ("samples_per_seed", 100001),
        ("seeds", [1]),
        ("seeds", [1, 1]),
        ("seeds", [-1, 0]),
        ("seeds", [0, 2**64]),
        ("seeds", [False, 1]),
        ("seeds", "01"),
        ("energies_eV", []),
        ("energies_eV", [100] * 9),
        ("energies_eV", [100, 100]),
        ("energies_eV", [0.1]),
        ("energies_eV", [100000]),
        ("energies_eV", [float("nan")]),
        ("energies_eV", [10**1000]),
        ("angles_deg", [90]),
        ("angles_deg", [-1]),
        ("angles_deg", [True]),
        ("target", {}),
        ("ion", None),
    ],
)
def test_invalid_sampling_contract_is_refused(field: str, value: Any) -> None:
    """Reject invalid variants of the real request before starting a simulation."""
    request = json.loads(REQUEST.read_bytes())
    request[field] = value
    with pytest.raises(ValueError):
        validate_rustbca_request(request)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("Z", 74.0),
        ("Z", 0),
        ("m", 0),
        ("n", -1),
        ("Ec", 0),
        ("Es", -1),
        ("Eb", float("inf")),
        ("m", "183.84"),
        ("label", " "),
        ("provenance", ""),
        ("provenance", "x" * 4097),
    ],
)
def test_material_values_cannot_be_silently_repaired(field: str, value: Any) -> None:
    """Bad material data are refused without clipping or an inferred alternative."""
    request = json.loads(REQUEST.read_bytes())
    request["target"][field] = value
    with pytest.raises(ValueError):
        validate_rustbca_request(request)


def test_total_histories_are_bounded_across_grid_and_seeds() -> None:
    """A individually valid batch cannot bypass the aggregate work limit."""
    request = json.loads(REQUEST.read_bytes())
    request["samples_per_seed"] = 100_000
    with pytest.raises(ValueError, match="million histories"):
        validate_rustbca_request(request)
