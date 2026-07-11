# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests: IMAS core_transport converter validation branches

from __future__ import annotations

from typing import Any

import pytest

from scpn_fusion.io.imas_connector_transport import (
    imas_core_transport_to_state,
    state_to_imas_core_transport,
)

_RHO = [0.0, 0.5, 1.0]


class TestStateToImasCoreTransport:
    """state_to_imas_core_transport validates its input state contract."""

    def test_state_must_be_mapping(self) -> None:
        with pytest.raises(ValueError, match="state must be a mapping"):
            state_to_imas_core_transport(True)  # type: ignore[arg-type]

    def test_chi_e_length_must_match_rho(self) -> None:
        state: dict[str, Any] = {"rho_norm": _RHO, "chi_e": [1.0, 2.0]}
        with pytest.raises(ValueError, match="state.chi_e length must match"):
            state_to_imas_core_transport(state)

    def test_chi_i_length_must_match_rho(self) -> None:
        state: dict[str, Any] = {"rho_norm": _RHO, "chi_i": [1.0, 2.0]}
        with pytest.raises(ValueError, match="state.chi_i length must match"):
            state_to_imas_core_transport(state)

    def test_full_state_round_trips_all_channels(self) -> None:
        state: dict[str, Any] = {
            "rho_norm": _RHO,
            "chi_e": [1.0, 1.5, 2.0],
            "d_e": [0.1, 0.2, 0.3],
            "chi_i": [2.0, 2.5, 3.0],
        }
        ids = state_to_imas_core_transport(state, time_s=0.25)
        restored = imas_core_transport_to_state(ids)

        assert restored["rho_norm"] == _RHO
        assert restored["chi_e"] == [1.0, 1.5, 2.0]
        assert restored["d_e"] == [0.1, 0.2, 0.3]
        assert restored["chi_i"] == [2.0, 2.5, 3.0]


class TestImasCoreTransportToState:
    """imas_core_transport_to_state guards each malformed IDS layer."""

    def test_ids_must_be_mapping(self) -> None:
        with pytest.raises(ValueError, match="ids must be a mapping"):
            imas_core_transport_to_state([1, 2, 3])  # type: ignore[arg-type]

    def test_model_must_be_non_empty_sequence(self) -> None:
        with pytest.raises(ValueError, match="at least one model"):
            imas_core_transport_to_state({"model": []})

    def test_profiles_1d_required(self) -> None:
        ids: dict[str, Any] = {"model": [{"identifier": {}}]}
        with pytest.raises(ValueError, match="must have profiles_1d"):
            imas_core_transport_to_state(ids)

    def test_grid_rho_tor_norm_required(self) -> None:
        ids: dict[str, Any] = {"model": [{"profiles_1d": [{"grid_d": {}}]}]}
        with pytest.raises(ValueError, match="grid_d.rho_tor_norm"):
            imas_core_transport_to_state(ids)
