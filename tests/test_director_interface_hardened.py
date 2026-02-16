# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Director Interface Hardening Tests
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pytest

from scpn_fusion.control.director_interface import DirectorInterface, _RuleBasedDirector

def test_rule_based_director_finite_init() -> None:
    """Ensure _RuleBasedDirector rejects non-finite entropy thresholds."""
    with pytest.raises(ValueError, match="entropy_threshold must be finite"):
        _RuleBasedDirector(entropy_threshold=np.nan)
    with pytest.raises(ValueError, match="entropy_threshold must be finite"):
        _RuleBasedDirector(entropy_threshold=np.inf)
    with pytest.raises(ValueError, match="entropy_threshold must be finite"):
        _RuleBasedDirector(entropy_threshold=-1.0)

def test_rule_based_director_history_window_init() -> None:
    """Ensure _RuleBasedDirector rejects invalid history windows."""
    with pytest.raises(ValueError, match="history_window must be >= 1"):
        _RuleBasedDirector(history_window=0)
    with pytest.raises(ValueError, match="history_window must be >= 1"):
        _RuleBasedDirector(history_window=-5)

def test_director_format_state_finite_guards() -> None:
    """Ensure format_state_for_director rejects non-finite telemetry."""
    # Mock the controller factory to avoid loading real config
    mock_factory = MagicMock()
    di = DirectorInterface("fake_config.json", allow_fallback=True, controller_factory=mock_factory)
    
    # Test Ip non-finite
    with pytest.raises(ValueError, match="ip must be finite"):
        di.format_state_for_director(1, np.nan, 0.0, 0.0, [0.1])
    
    # Test err_r non-finite
    with pytest.raises(ValueError, match="err_r must be finite"):
        di.format_state_for_director(1, 15.0, np.inf, 0.0, [0.1])
        
    # Test brain_activity non-finite
    with pytest.raises(ValueError, match="brain_activity must contain finite values"):
        di.format_state_for_director(1, 15.0, 0.0, 0.0, [np.nan])

def test_run_directed_mission_envelope_guards() -> None:
    """Ensure run_directed_mission rejects non-finite glitch parameters."""
    mock_factory = MagicMock()
    di = DirectorInterface("fake_config.json", allow_fallback=True, controller_factory=mock_factory)
    
    with pytest.raises(ValueError, match="glitch_std must be finite"):
        di.run_directed_mission(duration=10, glitch_std=np.nan)
        
    with pytest.raises(ValueError, match="duration must be >= 1"):
        di.run_directed_mission(duration=0)

def test_rule_based_director_review_action_types() -> None:
    """Ensure review_action handles basic strings."""
    rb = _RuleBasedDirector()
    rb.review_action("State: Stable, BrainEntropy=0.1", "Proposal: Action")
