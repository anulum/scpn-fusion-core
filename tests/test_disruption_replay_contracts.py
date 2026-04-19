# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Direct linkage tests for disruption replay contract extraction."""

from __future__ import annotations

import numpy as np

from scpn_fusion.control.advanced_soc_fusion_learning import FusionAIAgent
from scpn_fusion.control.disruption_replay_contracts import run_real_shot_replay


def _shot_payload(n: int = 160) -> dict[str, np.ndarray | bool | int]:
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
        "is_disruption": True,
        "disruption_time_idx": n - 12,
    }


def test_run_real_shot_replay_smoke_from_extracted_module() -> None:
    out = run_real_shot_replay(
        shot_data=_shot_payload(),
        rl_agent=FusionAIAgent(epsilon=0.05),
        risk_threshold=0.55,
        spi_trigger_risk=0.72,
        window_size=96,
    )
    assert "pipeline" in out
    assert "risk_series" in out
    assert int(out["n_steps"]) == 160
