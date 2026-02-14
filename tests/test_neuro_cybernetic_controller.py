# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Neuro Cybernetic Controller Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Deterministic tests for reduced spiking-controller pool behavior."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_fusion.control.neuro_cybernetic_controller as controller_mod
from scpn_fusion.control.neuro_cybernetic_controller import (
    SpikingControllerPool,
    run_neuro_cybernetic_control,
)


class _DummyKernel:
    def __init__(self, _config_file: str) -> None:
        self.cfg = {
            "physics": {"plasma_current_target": 5.0},
            "coils": [{"current": 0.0} for _ in range(5)],
        }
        self.R = np.linspace(5.9, 6.5, 25)
        self.Z = np.linspace(-0.3, 0.3, 25)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((25, 25), dtype=np.float64)
        self.solve_equilibrium()

    def solve_equilibrium(self) -> None:
        radial_drive = float(self.cfg["coils"][2]["current"])
        vertical_drive = float(self.cfg["coils"][4]["current"]) - float(
            self.cfg["coils"][0]["current"]
        )
        center_r = 6.2 + 0.07 * np.tanh(radial_drive / 20.0)
        center_z = 0.0 + 0.05 * np.tanh(vertical_drive / 20.0)
        self.Psi = 1.0 - (
            (self.RR - center_r) ** 2 + ((self.ZZ - center_z) / 1.4) ** 2
        )


@pytest.fixture(autouse=True)
def _force_numpy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    # CI py3.11 lane does not bootstrap sc-neurocore; force same path locally.
    monkeypatch.setattr(controller_mod, "SC_NEUROCORE_AVAILABLE", False)


def test_spiking_pool_is_deterministic_for_same_seed() -> None:
    kwargs = dict(
        n_neurons=24,
        gain=2.0,
        tau_window=8,
        seed=19,
        use_quantum=False,
    )
    p1 = SpikingControllerPool(**kwargs)
    p2 = SpikingControllerPool(**kwargs)
    o1 = np.asarray([p1.step(0.15) for _ in range(24)], dtype=np.float64)
    o2 = np.asarray([p2.step(0.15) for _ in range(24)], dtype=np.float64)
    np.testing.assert_allclose(o1, o2, atol=0.0, rtol=0.0)


def test_spiking_pool_push_pull_sign_response() -> None:
    pos_pool = SpikingControllerPool(
        n_neurons=20,
        gain=3.0,
        tau_window=6,
        seed=31,
        use_quantum=False,
    )
    neg_pool = SpikingControllerPool(
        n_neurons=20,
        gain=3.0,
        tau_window=6,
        seed=31,
        use_quantum=False,
    )

    pos = [pos_pool.step(0.2) for _ in range(32)]
    neg = [neg_pool.step(-0.2) for _ in range(32)]
    assert float(np.mean(pos[-8:])) > 0.0
    assert float(np.mean(neg[-8:])) < 0.0


def test_spiking_pool_exposes_backend_name() -> None:
    pool = SpikingControllerPool(n_neurons=8, gain=1.0, tau_window=4, seed=11)
    assert pool.backend == "numpy_lif"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_neurons": 0}, "n_neurons"),
        ({"tau_window": 0}, "tau_window"),
        ({"gain": float("nan")}, "gain"),
        ({"dt_s": 0.0}, "dt_s"),
        ({"tau_mem_s": 0.0}, "tau_mem_s"),
        ({"noise_std": -1.0e-3}, "noise_std"),
    ],
)
def test_spiking_pool_rejects_invalid_constructor_inputs(
    kwargs: dict[str, float | int], match: str
) -> None:
    params: dict[str, float | int | bool] = {
        "n_neurons": 8,
        "gain": 1.0,
        "tau_window": 4,
        "seed": 11,
        "use_quantum": False,
    }
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        SpikingControllerPool(**params)


def test_run_neuro_cybernetic_control_returns_finite_summary_without_plot() -> None:
    summary = run_neuro_cybernetic_control(
        config_file="dummy.json",
        shot_duration=14,
        seed=101,
        quantum=False,
        save_plot=False,
        verbose=False,
        kernel_factory=_DummyKernel,
    )
    for key in (
        "seed",
        "steps",
        "mode",
        "backend_r",
        "backend_z",
        "final_r",
        "final_z",
        "mean_abs_err_r",
        "mean_abs_err_z",
        "max_abs_control_r",
        "max_abs_control_z",
        "mean_spike_imbalance",
        "plot_saved",
    ):
        assert key in summary
    assert summary["seed"] == 101
    assert summary["steps"] == 14
    assert summary["mode"] == "classical"
    assert summary["backend_r"] == "numpy_lif"
    assert summary["backend_z"] == "numpy_lif"
    assert summary["plot_saved"] is False
    assert np.isfinite(summary["final_r"])
    assert np.isfinite(summary["final_z"])


def test_run_neuro_cybernetic_control_is_deterministic_for_seed() -> None:
    kwargs = dict(
        config_file="dummy.json",
        shot_duration=12,
        seed=77,
        quantum=False,
        save_plot=False,
        verbose=False,
        kernel_factory=_DummyKernel,
    )
    a = run_neuro_cybernetic_control(**kwargs)
    b = run_neuro_cybernetic_control(**kwargs)
    for key in (
        "final_r",
        "final_z",
        "mean_abs_err_r",
        "mean_abs_err_z",
        "max_abs_control_r",
        "max_abs_control_z",
        "mean_spike_imbalance",
    ):
        assert a[key] == pytest.approx(b[key], rel=0.0, abs=0.0)


def test_run_neuro_cybernetic_control_rejects_nonpositive_duration() -> None:
    with pytest.raises(ValueError, match="shot_duration"):
        run_neuro_cybernetic_control(
            config_file="dummy.json",
            shot_duration=0,
            save_plot=False,
            verbose=False,
            kernel_factory=_DummyKernel,
        )
