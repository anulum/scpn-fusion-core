# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests: IMAS equilibrium payload validation branches

from __future__ import annotations

import copy
from typing import Any, cast

import numpy as np
import pytest

from scpn_fusion.core.eqdsk import GEqdsk
from scpn_fusion.io.imas_connector_equilibrium import (
    geqdsk_to_imas_equilibrium,
    validate_imas_equilibrium_payload,
)


def _sample_geqdsk(nw: int = 4, nh: int = 4) -> GEqdsk:
    """Minimal valid GEqdsk with a strictly-increasing rectangular grid."""
    return GEqdsk(
        description="test equilibrium",
        nw=nw,
        nh=nh,
        rdim=2.0,
        zdim=3.0,
        rcentr=1.7,
        rleft=1.0,
        zmid=0.0,
        rmaxis=1.65,
        zmaxis=0.02,
        simag=-1.5,
        sibry=-0.2,
        bcentr=5.3,
        current=15e6,
        fpol=np.linspace(4.0, 5.0, nw),
        pres=np.linspace(3e4, 0.0, nw),
        ffprime=np.zeros(nw, dtype=np.float64),
        pprime=np.zeros(nw, dtype=np.float64),
        qpsi=np.linspace(1.0, 6.0, nw),
        psirz=np.random.default_rng(42).standard_normal((nh, nw)),
        rbdry=np.array([1.2, 1.8, 2.5, 1.2], dtype=np.float64),
        zbdry=np.array([-1.0, -1.2, 0.5, 1.0], dtype=np.float64),
        rlim=np.array([], dtype=np.float64),
        zlim=np.array([], dtype=np.float64),
    )


def _valid_payload() -> dict[str, Any]:
    """A schema-valid IMAS equilibrium IDS to be mutated per invalid case."""
    return geqdsk_to_imas_equilibrium(_sample_geqdsk())


def _first_profile(payload: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", payload["time_slice"][0]["profiles_2d"][0])


class TestValidatePayloadRejectsMalformedSchema:
    """validate_imas_equilibrium_payload rejects each malformed field independently."""

    def test_missing_required_keys(self) -> None:
        payload = _valid_payload()
        del payload["time"]
        with pytest.raises(ValueError, match="missing keys"):
            validate_imas_equilibrium_payload(payload)

    def test_time_slice_not_sequence(self) -> None:
        payload = _valid_payload()
        payload["time_slice"] = "not-a-sequence"
        with pytest.raises(ValueError, match="time_slice must be a non-empty sequence"):
            validate_imas_equilibrium_payload(payload)

    def test_time_slice_empty(self) -> None:
        payload = _valid_payload()
        payload["time_slice"] = []
        with pytest.raises(ValueError, match="at least one time_slice"):
            validate_imas_equilibrium_payload(payload)

    def test_time_slice_exceeds_safety_limit(self) -> None:
        payload = _valid_payload()
        slice0 = payload["time_slice"][0]
        payload["time_slice"] = [slice0] * 1025
        with pytest.raises(ValueError, match="exceeds safety limit 1024"):
            validate_imas_equilibrium_payload(payload)

    def test_time_slice_first_entry_not_mapping(self) -> None:
        payload = _valid_payload()
        payload["time_slice"] = [42]
        with pytest.raises(ValueError, match="time_slice\\[0\\] must be a mapping"):
            validate_imas_equilibrium_payload(payload)

    def test_profiles_2d_not_sequence(self) -> None:
        payload = _valid_payload()
        payload["time_slice"][0]["profiles_2d"] = "not-a-sequence"
        with pytest.raises(ValueError, match="profiles_2d must be a sequence"):
            validate_imas_equilibrium_payload(payload)

    def test_profiles_2d_empty(self) -> None:
        payload = _valid_payload()
        payload["time_slice"][0]["profiles_2d"] = []
        with pytest.raises(ValueError, match="at least one profiles_2d entry"):
            validate_imas_equilibrium_payload(payload)

    def test_profiles_2d_exceeds_safety_limit(self) -> None:
        payload = _valid_payload()
        profile0 = _first_profile(payload)
        payload["time_slice"][0]["profiles_2d"] = [profile0] * 65
        with pytest.raises(ValueError, match="exceeds safety limit 64"):
            validate_imas_equilibrium_payload(payload)

    def test_profiles_2d_first_entry_not_mapping(self) -> None:
        payload = _valid_payload()
        payload["time_slice"][0]["profiles_2d"] = [42]
        with pytest.raises(ValueError, match="profiles_2d\\[0\\] must be a mapping"):
            validate_imas_equilibrium_payload(payload)

    def test_grid_not_mapping(self) -> None:
        payload = _valid_payload()
        _first_profile(payload)["grid"] = "not-a-mapping"
        with pytest.raises(ValueError, match="grid must be a mapping"):
            validate_imas_equilibrium_payload(payload)

    def test_psi_not_sequence(self) -> None:
        payload = _valid_payload()
        _first_profile(payload)["psi"] = "not-a-sequence"
        with pytest.raises(ValueError, match="psi must be a 2-D sequence"):
            validate_imas_equilibrium_payload(payload)

    def test_psi_row_count_mismatch(self) -> None:
        payload = _valid_payload()
        profile = _first_profile(payload)
        nw = len(profile["grid"]["dim1"])
        profile["psi"] = [[0.0] * nw]  # one row, but grid dim2 has several
        with pytest.raises(ValueError, match="psi row count must match grid dim2"):
            validate_imas_equilibrium_payload(payload)

    def test_psi_column_count_mismatch(self) -> None:
        payload = _valid_payload()
        profile = _first_profile(payload)
        nw = len(profile["grid"]["dim1"])
        nh = len(profile["grid"]["dim2"])
        profile["psi"] = [[0.0] * (nw + 1) for _ in range(nh)]
        with pytest.raises(ValueError, match="psi column count must match grid dim1"):
            validate_imas_equilibrium_payload(payload)


class TestGeqdskToImasGuards:
    """geqdsk_to_imas_equilibrium rejects a degenerate flux map."""

    def test_empty_psirz_is_rejected(self) -> None:
        eq = GEqdsk(nw=2, nh=2)
        assert eq.psirz.size == 0
        with pytest.raises(ValueError, match="psirz must be non-empty"):
            geqdsk_to_imas_equilibrium(eq)


class TestRoundTripUnaffected:
    """The deep-copy mutation helpers do not perturb the shared valid payload."""

    def test_valid_payload_still_validates(self) -> None:
        payload = _valid_payload()
        mutated = copy.deepcopy(payload)
        del mutated["time"]
        # Original remains schema-valid after an independent mutation of a copy.
        validate_imas_equilibrium_payload(payload)
