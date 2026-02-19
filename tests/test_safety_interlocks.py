# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Safety Interlocks Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for inhibitor-arc safety interlocks."""

from __future__ import annotations

import numpy as np

from scpn_fusion.scpn.contracts import verify_safety_contracts
from scpn_fusion.scpn.safety_interlocks import (
    CONTROL_TRANSITIONS,
    SAFETY_CHANNELS,
    SafetyInterlockRuntime,
    build_safety_net,
)


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

