# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Safety Interlocks Tests
"""Tests for inhibitor-arc safety interlocks."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.scpn.contracts import SafetyContract, verify_safety_contracts
from scpn_fusion.scpn.safety_interlocks import (
    CONTROL_TRANSITIONS,
    SAFETY_CHANNELS,
    SafetyInterlockRuntime,
    build_safety_net,
    default_safety_contracts,
    evaluate_transition_enablement,
)
from scpn_fusion.scpn.structure import StochasticPetriNet


def test_safety_net_compiles_with_inhibitor_support() -> None:
    net = build_safety_net()
    assert net.is_compiled
    assert net.W_in is not None
    assert np.any(net.W_in.toarray() < 0.0)


def test_all_actions_allowed_when_no_limits_violated() -> None:
    runtime = SafetyInterlockRuntime()
    runtime.set_safety_tokens({k: 0.0 for k in SAFETY_CHANNELS})
    allowed = runtime.allowed_actions()
    assert set(allowed) == set(CONTROL_TRANSITIONS)
    assert all(allowed.values())
    assert runtime.last_contract_violations == []


def test_each_safety_channel_inhibits_paired_transition() -> None:
    runtime = SafetyInterlockRuntime()

    for safety_place, transition in SAFETY_CHANNELS.items():
        runtime.set_safety_tokens({k: 0.0 for k in SAFETY_CHANNELS})
        runtime.set_safety_tokens({safety_place: 1.0})
        allowed = runtime.allowed_actions()

        assert allowed[transition] is False
        for other in CONTROL_TRANSITIONS:
            if other != transition:
                assert allowed[other] is True


def test_combined_safety_violations_disable_all_control_actions() -> None:
    runtime = SafetyInterlockRuntime()
    runtime.set_safety_tokens({k: 1.0 for k in SAFETY_CHANNELS})
    allowed = runtime.allowed_actions()
    assert not any(allowed.values())


def test_runtime_state_update_flags_expected_channels() -> None:
    runtime = SafetyInterlockRuntime()
    state = {
        "T_e": 30.0,
        "n_e": 20.0,
        "beta_N": 3.5,
        "I_p": 18.0,
        "dZ_dt": 2.0,
    }
    allowed = runtime.update_from_state(state)
    assert not any(allowed.values())


def test_formal_contracts_detect_inconsistent_enablement() -> None:
    violations = verify_safety_contracts(
        safety_tokens={"thermal_limit": 1.0},
        transition_enabled={"heat_ramp": True},
    )
    assert violations == ["thermal_limit inhibits heat_ramp"]


def test_evaluate_transition_enablement_requires_compiled_net() -> None:
    uncompiled = StochasticPetriNet()
    assert not uncompiled.is_compiled
    with pytest.raises(RuntimeError, match="must be compiled before evaluation"):
        evaluate_transition_enablement(uncompiled, np.zeros(1, dtype=np.float64))


def test_evaluate_transition_enablement_rejects_wrong_marking_shape() -> None:
    net = build_safety_net()
    with pytest.raises(ValueError, match=r"marking must have shape"):
        evaluate_transition_enablement(net, np.zeros(3, dtype=np.float64))


def test_default_safety_contracts_returns_contract_tuple() -> None:
    contracts = default_safety_contracts()
    assert isinstance(contracts, tuple)
    assert len(contracts) > 0
    assert all(isinstance(contract, SafetyContract) for contract in contracts)


def test_runtime_compiles_an_uncompiled_net() -> None:
    net = StochasticPetriNet()
    net.add_place("p", initial_tokens=1.0)
    net.add_transition("t", threshold=0.5)
    net.add_arc("p", "t", weight=1.0)
    assert not net.is_compiled

    runtime = SafetyInterlockRuntime(net=net)
    assert runtime.net.is_compiled


def test_marking_property_returns_independent_copy() -> None:
    runtime = SafetyInterlockRuntime()
    marking = runtime.marking
    marking[:] = 999.0
    # Mutating the returned array must not disturb the runtime's state.
    assert not np.array_equal(runtime.marking, marking)
