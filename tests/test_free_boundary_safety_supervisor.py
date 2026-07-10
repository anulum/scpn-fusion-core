# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Direct guard tests for the free-boundary safety supervisor.

Covers the constructor consistency checks (margin fraction, disruption ceiling,
ordered risk thresholds) and the ``filter_action`` shape guards (action/current
length, predicted-next-state finiteness), plus a nominal filtered action.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control._free_boundary_safety_supervisor import FreeBoundarySafetySupervisor
from scpn_fusion.control._free_boundary_supervisory_types import SafetyFilterResult


def _supervisor() -> FreeBoundarySafetySupervisor:
    return FreeBoundarySafetySupervisor()


class TestConstructorGuards:
    """Constructor rejects inconsistent safety configuration."""

    def test_margin_fraction_out_of_range_raises(self) -> None:
        """A current-margin fraction outside ``(0, 1]`` is rejected."""
        with pytest.raises(ValueError, match="current_margin_fraction must be finite"):
            FreeBoundarySafetySupervisor(current_margin_fraction=1.5)

    def test_disruption_ceiling_at_or_above_one_raises(self) -> None:
        """A disruption risk ceiling of 1.0 or greater is rejected."""
        with pytest.raises(ValueError, match="disruption_risk_ceiling must be < 1.0"):
            FreeBoundarySafetySupervisor(disruption_risk_ceiling=1.0)

    def test_unordered_risk_thresholds_raise(self) -> None:
        """A guarded threshold not above the warning threshold is rejected."""
        with pytest.raises(ValueError, match="guarded_risk_score_threshold must be >"):
            FreeBoundarySafetySupervisor(
                warning_risk_score_threshold=0.88,
                guarded_risk_score_threshold=0.5,
            )


class TestFilterActionGuards:
    """``filter_action`` validates array shapes before filtering."""

    def test_action_current_length_mismatch_raises(self) -> None:
        """A proposed action and coil currents of unequal length are rejected."""
        with pytest.raises(ValueError, match="identical length"):
            _supervisor().filter_action(
                np.array([1.0, 2.0, 3.0]),
                corrected_state=np.zeros(4),
                target_state=np.zeros(4),
                bias_hat=np.zeros(4),
                coil_currents=np.array([1.0, 2.0]),
                target_ip_ma=8.0,
            )

    def test_non_finite_predicted_next_state_raises(self) -> None:
        """A predicted next state that is not a finite 4-vector is rejected."""
        with pytest.raises(ValueError, match="predicted_next_state must be a finite 4-vector"):
            _supervisor().filter_action(
                np.array([0.5, -0.5, 0.3, -0.3]),
                corrected_state=np.array([6.0, 0.0, 5.0, -3.48]),
                target_state=np.array([6.0, 0.0, 5.02, -3.48]),
                bias_hat=np.array([0.01, 0.01, 0.01, 0.01]),
                coil_currents=np.array([0.5, -0.5, 0.4, -0.4]),
                target_ip_ma=8.0,
                predicted_next_state=np.array([1.0, 2.0, 3.0]),
            )


def test_filter_action_nominal_returns_result() -> None:
    """A well-formed action passes the guards and returns a filter result."""
    result = _supervisor().filter_action(
        np.array([0.2, -0.2, 0.1, -0.1]),
        corrected_state=np.array([6.0, 0.0, 5.02, -3.48]),
        target_state=np.array([6.0, 0.0, 5.02, -3.48]),
        bias_hat=np.array([0.01, 0.01, 0.01, 0.01]),
        coil_currents=np.array([0.5, -0.5, 0.4, -0.4]),
        target_ip_ma=8.0,
    )
    assert isinstance(result, SafetyFilterResult)
    assert result.action.size == 4
    assert np.all(np.isfinite(result.action))
