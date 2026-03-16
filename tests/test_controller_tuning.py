# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Controller Tuning Tests
"""
Tests for automated controller tuning logic.
"""

from __future__ import annotations

import pytest

from scpn_fusion.control.controller_tuning import HAS_OPTUNA, tune_pid


def test_optuna_guard_behavior():
    """Verify that tune_pid behaves correctly when Optuna is missing."""
    if not HAS_OPTUNA:
        # Should return default gains and not crash
        gains = tune_pid(None, n_trials=1)
        assert "Kp" in gains
        assert "Ki" in gains
        assert "Kd" in gains
    else:
        # If available, we could test a mock env
        pass


@pytest.mark.skipif(not HAS_OPTUNA, reason="Optuna not installed")
def test_tune_pid_with_optuna():
    """Placeholder for full tuning test if optuna available."""
    # This would require a mock gymnasium environment
    pass
