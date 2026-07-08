# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Direct contract tests for disruption replay extraction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.control.advanced_soc_fusion_learning import FusionAIAgent
from scpn_fusion.control import disruption_replay_contracts as replay_mod
from scpn_fusion.control.disruption_replay_contracts import run_real_shot_replay

FloatArray = NDArray[np.float64]


def _shot_payload(
    n: int = 160,
    *,
    is_disruption: bool = True,
    disruption_time_idx: int | None = None,
) -> dict[str, FloatArray | bool | int]:
    """Return a finite real-shot-style replay payload."""
    t = np.linspace(0.0, 0.16, n, dtype=np.float64)
    d_bdt = 0.35 + 0.08 * np.sin(2.0 * np.pi * 4.0 * t)
    n1 = 0.10 + 0.20 * np.exp(-(((t - 0.12) / 0.020) ** 2))
    n2 = 0.05 + 0.08 * np.exp(-(((t - 0.12) / 0.025) ** 2))
    return {
        "time_s": t,
        "Ip_MA": np.full(n, 12.0, dtype=np.float64),
        "beta_N": np.full(n, 2.1, dtype=np.float64),
        "n1_amp": n1,
        "n2_amp": n2,
        "dBdt_gauss_per_s": d_bdt,
        "is_disruption": is_disruption,
        "disruption_time_idx": n - 12 if disruption_time_idx is None else disruption_time_idx,
    }


def _agent() -> FusionAIAgent:
    """Return the replay contract's required agent dependency."""
    return FusionAIAgent(epsilon=0.05)


def test_run_real_shot_replay_no_spi_non_disruption_contract() -> None:
    """Non-disruption shots without high risk are marked prevented."""
    out = run_real_shot_replay(
        shot_data=_shot_payload(is_disruption=False, disruption_time_idx=-1),
        rl_agent=_agent(),
        risk_threshold=0.95,
        spi_trigger_risk=0.98,
        window_size=96,
        replay_pipeline={"sensor_preprocess_enabled": False, "actuator_lag_enabled": False},
    )
    assert out["pipeline"]["sensor_preprocess_enabled"] is False
    assert out["pipeline"]["actuator_lag_enabled"] is False
    assert out["spi_triggered"] is False
    assert out["spi_trigger_idx"] == -1
    assert out["prevented"] is True
    assert out["final_current_MA"] == pytest.approx(12.0)
    assert out["z_eff"] == pytest.approx(1.0)
    assert len(out["risk_series"]) == 160


def test_run_real_shot_replay_reports_detection_lead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disruption shots with alarm but no SPI report lead time."""

    def fixed_risk(_signal: FloatArray, _toroidal: dict[str, float]) -> float:
        return 0.85

    monkeypatch.setattr(replay_mod, "predict_disruption_risk", fixed_risk)
    out = run_real_shot_replay(
        shot_data=_shot_payload(is_disruption=True, disruption_time_idx=120),
        rl_agent=_agent(),
        risk_threshold=0.50,
        spi_trigger_risk=1.0,
        window_size=16,
        replay_pipeline={"sensor_preprocess_enabled": False, "actuator_lag_enabled": False},
    )
    assert out["first_alarm_idx"] == 16
    assert out["spi_triggered"] is False
    assert out["detection_lead_ms"] > 0.0
    assert out["prevented"] is False
    assert out["peak_risk_raw"] == pytest.approx(0.85)


def test_run_real_shot_replay_triggers_spi_and_marks_prevented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """High-risk disruption shots trigger SPI before disruption time."""

    def fixed_risk(_signal: FloatArray, _toroidal: dict[str, float]) -> float:
        return 0.85

    monkeypatch.setattr(replay_mod, "predict_disruption_risk", fixed_risk)
    out = run_real_shot_replay(
        shot_data=_shot_payload(is_disruption=True, disruption_time_idx=120),
        rl_agent=_agent(),
        risk_threshold=0.50,
        spi_trigger_risk=0.70,
        window_size=16,
        replay_pipeline={"sensor_preprocess_enabled": False, "actuator_lag_enabled": False},
    )
    assert out["spi_triggered"] is True
    assert 0 <= int(out["spi_trigger_idx"]) < 120
    assert out["prevented"] is True
    assert out["neon_mol"] > 0.0
    assert out["total_impurity_mol"] > 0.0
    assert out["tau_cq_ms"] > 0.0
    assert out["final_current_MA"] >= 0.0
    assert out["z_eff"] >= 1.0


@pytest.mark.parametrize(
    ("shot_data", "window_size", "risk_threshold", "spi_trigger_risk", "match"),
    [
        (_shot_payload(), 16, 0.70, 0.60, "spi_trigger_risk"),
        (
            {**_shot_payload(), "time_s": np.r_[np.linspace(0.0, 0.1, 159), 0.1]},
            16,
            0.65,
            0.80,
            "strictly increasing",
        ),
        (_shot_payload(n=16), 32, 0.65, 0.80, "window_size"),
        (_shot_payload(disruption_time_idx=999), 16, 0.65, 0.80, "disruption_time_idx"),
        (_shot_payload(disruption_time_idx=-2), 16, 0.65, 0.80, "disruption_time_idx"),
    ],
)
def test_run_real_shot_replay_rejects_invalid_contracts(
    shot_data: dict[str, Any],
    window_size: int,
    risk_threshold: float,
    spi_trigger_risk: float,
    match: str,
) -> None:
    """Replay contract validation rejects inconsistent shot settings."""
    with pytest.raises(ValueError, match=match):
        run_real_shot_replay(
            shot_data=shot_data,
            rl_agent=_agent(),
            window_size=window_size,
            risk_threshold=risk_threshold,
            spi_trigger_risk=spi_trigger_risk,
        )
