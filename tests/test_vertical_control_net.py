# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Vertical Control Net Tests
# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Unit tests for the vertical position control Petri net.

Topology under test: 8 places, 7 transitions encoding error-sign
routing and error-magnitude classification for tokamak plasma
vertical stabilization.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.scpn.vertical_control_net import (
    VerticalControlNet,
    PLACES,
    TRANSITIONS,
)
from scpn_fusion.scpn.structure import StochasticPetriNet
from scpn_fusion.scpn.compiler import CompiledNet


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def vcn() -> VerticalControlNet:
    """Default vertical control net builder."""
    return VerticalControlNet()


@pytest.fixture
def net(vcn: VerticalControlNet) -> StochasticPetriNet:
    """Built (not compiled) vertical control net."""
    return vcn.create_net()


@pytest.fixture
def compiled_net(vcn: VerticalControlNet) -> CompiledNet:
    """Compiled vertical control net."""
    vcn.create_net()
    return vcn.compile()


# ── Topology Tests ───────────────────────────────────────────────────────────


class TestTopology:
    """Verify the net structure: 8 places, 7 transitions, correct names."""

    def test_net_has_8_places(self, net: StochasticPetriNet) -> None:
        assert net.n_places == 8

    def test_net_has_7_transitions(self, net: StochasticPetriNet) -> None:
        assert net.n_transitions == 7

    def test_place_names_match(self, net: StochasticPetriNet) -> None:
        assert net.place_names == PLACES

    def test_transition_names_match(self, net: StochasticPetriNet) -> None:
        assert net.transition_names == TRANSITIONS

    def test_initial_marking_idle(self, net: StochasticPetriNet) -> None:
        """P_idle starts with token=1.0, all others 0."""
        marking = net.get_initial_marking()
        assert marking[0] == 1.0, "P_idle must start with token=1.0"
        for i in range(1, 8):
            assert marking[i] == 0.0, f"{PLACES[i]} must start at 0.0"


# ── Compilation Tests ────────────────────────────────────────────────────────


class TestCompilation:
    """Verify that compilation produces valid matrix shapes."""

    def test_compile_produces_matrices(
        self, compiled_net: CompiledNet
    ) -> None:
        """W_in and W_out have correct shapes (nT x nP) and (nP x nT)."""
        assert compiled_net.W_in.shape == (7, 8)
        assert compiled_net.W_out.shape == (8, 7)

    def test_compile_place_count(self, compiled_net: CompiledNet) -> None:
        assert compiled_net.n_places == 8

    def test_compile_transition_count(self, compiled_net: CompiledNet) -> None:
        assert compiled_net.n_transitions == 7

    def test_compile_initial_marking_length(
        self, compiled_net: CompiledNet
    ) -> None:
        assert len(compiled_net.initial_marking) == 8

    def test_compile_thresholds_length(
        self, compiled_net: CompiledNet
    ) -> None:
        assert len(compiled_net.thresholds) == 7


# ── Encode Measurement Tests ────────────────────────────────────────────────


class TestEncodeMeasurement:
    """Verify measurement encoding for sign and magnitude routing."""

    def test_encode_positive_error(self, vcn: VerticalControlNet) -> None:
        """Z > 0 produces P_error_pos marking, P_error_neg stays 0."""
        m = vcn.encode_measurement(0.005)  # 5 mm upward
        assert m["P_error_pos"] > 0.0
        assert m["P_error_neg"] == 0.0

    def test_encode_negative_error(self, vcn: VerticalControlNet) -> None:
        """Z < 0 produces P_error_neg marking, P_error_pos stays 0."""
        m = vcn.encode_measurement(-0.003)  # 3 mm downward
        assert m["P_error_neg"] > 0.0
        assert m["P_error_pos"] == 0.0

    def test_encode_large_error(self, vcn: VerticalControlNet) -> None:
        """|Z| > 2 mm produces P_error_large marking."""
        m = vcn.encode_measurement(0.005)  # 5 mm > 2 mm threshold
        assert m["P_error_large"] > 0.0
        assert m["P_error_small"] == 0.0

    def test_encode_small_error(self, vcn: VerticalControlNet) -> None:
        """|Z| < 2 mm produces P_error_small marking."""
        m = vcn.encode_measurement(0.001)  # 1 mm < 2 mm threshold
        assert m["P_error_small"] > 0.0
        assert m["P_error_large"] == 0.0

    def test_encode_zero_displacement(
        self, vcn: VerticalControlNet
    ) -> None:
        """Z == 0 produces zero tokens in both sign places."""
        m = vcn.encode_measurement(0.0)
        # z=0 routes to neg branch by convention, but density is 0.
        assert m["P_error_pos"] == 0.0
        assert m["P_error_neg"] == 0.0

    def test_encode_values_bounded(self, vcn: VerticalControlNet) -> None:
        """All encoded token densities are in [0, 1]."""
        for z in [-0.1, -0.01, -0.001, 0.0, 0.001, 0.01, 0.1]:
            m = vcn.encode_measurement(z)
            for place, val in m.items():
                assert 0.0 <= val <= 1.0, (
                    f"z={z}, {place}={val} out of [0, 1]"
                )

    def test_encode_sign_mutex(self, vcn: VerticalControlNet) -> None:
        """P_error_pos and P_error_neg are never both nonzero."""
        for z in [-0.01, -0.001, 0.0, 0.001, 0.01]:
            m = vcn.encode_measurement(z)
            pos = m["P_error_pos"]
            neg = m["P_error_neg"]
            assert not (pos > 0.0 and neg > 0.0), (
                f"z={z}: both P_error_pos={pos} and P_error_neg={neg} nonzero"
            )


# ── Decode Output Tests ─────────────────────────────────────────────────────


class TestDecodeOutput:
    """Verify control-signal decoding opposes displacement."""

    def test_decode_positive_gives_negative_u(
        self, vcn: VerticalControlNet
    ) -> None:
        """Positive Z should produce negative control (oppose displacement)."""
        marking = {
            "P_error_large": 0.8,
            "P_error_small": 0.0,
            "P_actuating": 0.5,
            "P_applied": 0.0,
        }
        u = vcn.decode_output(marking, z_positive=True)
        assert u < 0.0, f"Expected u < 0 for positive Z, got {u}"

    def test_decode_negative_gives_positive_u(
        self, vcn: VerticalControlNet
    ) -> None:
        """Negative Z should produce positive control (oppose displacement)."""
        marking = {
            "P_error_large": 0.8,
            "P_error_small": 0.0,
            "P_actuating": 0.5,
            "P_applied": 0.0,
        }
        u = vcn.decode_output(marking, z_positive=False)
        assert u > 0.0, f"Expected u > 0 for negative Z, got {u}"

    def test_larger_error_stronger_output(
        self, vcn: VerticalControlNet
    ) -> None:
        """|Z|=5 mm gives larger |u| than |Z|=1 mm."""
        # Encode both measurements and compare resulting control magnitudes.
        m_large = vcn.encode_measurement(0.005)  # 5 mm
        m_small = vcn.encode_measurement(0.001)  # 1 mm

        u_large = abs(vcn.decode_output(m_large, z_positive=True))
        u_small = abs(vcn.decode_output(m_small, z_positive=True))
        assert u_large > u_small, (
            f"|u(5mm)|={u_large} should be > |u(1mm)|={u_small}"
        )

    def test_zero_marking_gives_zero_output(
        self, vcn: VerticalControlNet
    ) -> None:
        """All-zero marking produces zero control signal."""
        marking = {p: 0.0 for p in PLACES}
        u = vcn.decode_output(marking, z_positive=True)
        assert u == 0.0

    def test_gain_scale_multiplies_output(self) -> None:
        """Doubling gain_scale doubles the control magnitude."""
        vcn1 = VerticalControlNet(gain_scale=1.0)
        vcn2 = VerticalControlNet(gain_scale=2.0)
        marking = {
            "P_error_large": 0.5,
            "P_error_small": 0.0,
            "P_actuating": 0.3,
            "P_applied": 0.0,
        }
        u1 = abs(vcn1.decode_output(marking, z_positive=True))
        u2 = abs(vcn2.decode_output(marking, z_positive=True))
        assert abs(u2 - 2.0 * u1) < 1e-12, (
            f"gain_scale=2 should double output: u1={u1}, u2={u2}"
        )


# ── Integration Tests ────────────────────────────────────────────────────────


class TestIntegration:
    """End-to-end: create -> compile -> encode -> decode pipeline."""

    def test_full_pipeline_positive(self) -> None:
        """Positive displacement through full pipeline gives negative u."""
        vcn = VerticalControlNet()
        vcn.create_net()
        compiled = vcn.compile()

        z = 0.004  # 4 mm upward
        marking = vcn.encode_measurement(z)
        u = vcn.decode_output(marking, z_positive=True)
        assert u < 0.0

    def test_full_pipeline_negative(self) -> None:
        """Negative displacement through full pipeline gives positive u."""
        vcn = VerticalControlNet()
        vcn.create_net()
        compiled = vcn.compile()

        z = -0.004  # 4 mm downward
        marking = vcn.encode_measurement(z)
        u = vcn.decode_output(marking, z_positive=False)
        assert u > 0.0

    def test_create_net_is_idempotent(self) -> None:
        """Calling create_net() twice returns a fresh net each time."""
        vcn = VerticalControlNet()
        net1 = vcn.create_net()
        net2 = vcn.create_net()
        assert net1 is not net2
        assert net2.n_places == 8

    def test_compile_without_explicit_create(self) -> None:
        """compile() auto-creates the net if not yet built."""
        vcn = VerticalControlNet()
        compiled = vcn.compile()
        assert compiled.n_places == 8
        assert compiled.n_transitions == 7
