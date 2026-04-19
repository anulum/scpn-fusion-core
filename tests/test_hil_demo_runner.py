# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Direct tests for ``scpn_fusion.control.hil_demo_runner`` linkage and behavior."""

from __future__ import annotations

import numpy as np

from scpn_fusion.control.hil_demo_runner import HILDemoRunner


def test_q16_roundtrip_preserves_scale() -> None:
    value = 0.375
    raw = HILDemoRunner.float_to_q16_16(value)
    restored = HILDemoRunner.q16_16_to_float(raw)
    assert abs(restored - value) < 1.0 / HILDemoRunner.Q16_SCALE


def test_step_updates_register_latency_and_shape() -> None:
    runner = HILDemoRunner(n_neurons=6, n_inputs=3, n_outputs=2)
    out = runner.step(np.array([0.1, -0.2, 0.05], dtype=np.float64))
    assert out.shape == (2,)
    assert runner.total_steps == 1
    assert runner.registers[0x200 // 4] > 0


def test_run_episode_reports_expected_keys() -> None:
    runner = HILDemoRunner(n_neurons=4, n_inputs=2, n_outputs=2)
    report = runner.run_episode(n_steps=32, inject_faults=True)
    assert report["total_steps"] == 32
    assert set(report).issuperset(
        {
            "tmr_mismatches",
            "tmr_mismatch_rate",
            "latency_mean_cycles",
            "latency_p95_cycles",
            "latency_max_cycles",
        }
    )
