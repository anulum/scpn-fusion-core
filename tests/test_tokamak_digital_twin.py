# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Tokamak Digital Twin Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Deterministic smoke tests for tokamak_digital_twin runtime entry points."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.tokamak_digital_twin import (
    run_digital_twin,
    run_digital_twin_ids_history,
    run_digital_twin_ids_pulse,
)
from scpn_fusion.io.imas_connector import (
    digital_twin_summary_to_ids,
    ids_pulse_to_digital_twin_history,
    ids_to_digital_twin_history,
    ids_to_digital_twin_summary,
    validate_ids_pulse_payload,
    validate_ids_payload_sequence,
)


def test_run_digital_twin_returns_finite_summary_without_plot() -> None:
    summary = run_digital_twin(
        time_steps=24,
        seed=123,
        save_plot=False,
        verbose=False,
    )
    for key in (
        "seed",
        "steps",
        "final_avg_temp",
        "final_reward",
        "final_action",
        "final_islands_px",
        "reward_mean_last_50",
        "chaos_monkey",
        "sensor_dropout_prob",
        "sensor_noise_std",
        "sensor_dropouts_total",
        "sensor_dropout_rate",
        "plot_saved",
    ):
        assert key in summary
    assert summary["steps"] == 24
    assert summary["plot_saved"] is False
    assert np.isfinite(summary["final_avg_temp"])
    assert np.isfinite(summary["final_reward"])
    assert np.isfinite(summary["final_action"])
    assert np.isfinite(summary["reward_mean_last_50"])
    assert summary["chaos_monkey"] is False
    assert summary["sensor_dropouts_total"] == 0


def test_run_digital_twin_chaos_monkey_is_deterministic_for_fixed_seed() -> None:
    kwargs = dict(
        time_steps=20,
        seed=77,
        save_plot=False,
        verbose=False,
        chaos_monkey=True,
        sensor_dropout_prob=0.25,
        sensor_noise_std=0.05,
    )
    a = run_digital_twin(**kwargs)
    b = run_digital_twin(**kwargs)
    assert a == b


def test_run_digital_twin_chaos_monkey_full_dropout_counts_all_channels() -> None:
    steps = 7
    summary = run_digital_twin(
        time_steps=steps,
        seed=9,
        save_plot=False,
        verbose=False,
        chaos_monkey=True,
        sensor_dropout_prob=1.0,
        sensor_noise_std=0.0,
    )
    assert summary["sensor_dropouts_total"] == steps * 40
    assert summary["sensor_dropout_rate"] == 1.0


@pytest.mark.parametrize(
    ("kwargs", "msg"),
    [
        ({"sensor_dropout_prob": -0.1}, "sensor_dropout_prob"),
        ({"sensor_dropout_prob": 1.1}, "sensor_dropout_prob"),
        ({"sensor_dropout_prob": float("nan")}, "sensor_dropout_prob"),
        ({"sensor_noise_std": -0.01}, "sensor_noise_std"),
        ({"sensor_noise_std": float("inf")}, "sensor_noise_std"),
    ],
)
def test_run_digital_twin_rejects_invalid_chaos_monkey_controls(
    kwargs: dict[str, object], msg: str
) -> None:
    with pytest.raises(ValueError, match=msg):
        run_digital_twin(
            time_steps=4,
            seed=3,
            save_plot=False,
            verbose=False,
            chaos_monkey=True,
            **kwargs,
        )


def test_run_digital_twin_is_deterministic_for_fixed_seed() -> None:
    a = run_digital_twin(time_steps=20, seed=77, save_plot=False, verbose=False)
    b = run_digital_twin(time_steps=20, seed=77, save_plot=False, verbose=False)
    assert a["final_avg_temp"] == b["final_avg_temp"]
    assert a["final_reward"] == b["final_reward"]
    assert a["final_action"] == b["final_action"]
    assert a["final_islands_px"] == b["final_islands_px"]
    assert a["reward_mean_last_50"] == b["reward_mean_last_50"]


def test_ids_roundtrip_preserves_core_digital_twin_fields() -> None:
    summary = run_digital_twin(time_steps=18, seed=5, save_plot=False, verbose=False)
    ids_payload = digital_twin_summary_to_ids(summary, machine="ITER", shot=101, run=2)
    recovered = ids_to_digital_twin_summary(ids_payload)

    assert recovered["steps"] == summary["steps"]
    assert recovered["final_islands_px"] == summary["final_islands_px"]
    assert np.isfinite(recovered["final_avg_temp"])
    assert np.isfinite(recovered["final_reward"])


@pytest.mark.parametrize("bad_time_s", [-0.001, float("nan"), float("inf")])
def test_ids_to_digital_twin_summary_rejects_invalid_time_slice_time(
    bad_time_s: float,
) -> None:
    summary = run_digital_twin(time_steps=12, seed=3, save_plot=False, verbose=False)
    ids_payload = digital_twin_summary_to_ids(summary, machine="ITER", shot=101, run=2)
    ids_payload["time_slice"]["time_s"] = bad_time_s
    with pytest.raises(ValueError, match="time_slice.time_s"):
        ids_to_digital_twin_summary(ids_payload)


@pytest.mark.parametrize(
    ("patch", "msg"),
    [
        ({"machine": ""}, "machine"),
        ({"shot": -1}, "shot"),
        ({"shot": 1.5}, "shot"),
        ({"run": True}, "run"),
    ],
)
def test_digital_twin_summary_to_ids_rejects_invalid_metadata(
    patch: dict[str, object],
    msg: str,
) -> None:
    summary = run_digital_twin(time_steps=12, seed=31, save_plot=False, verbose=False)
    kwargs: dict[str, object] = {"machine": "ITER", "shot": 1, "run": 2}
    kwargs.update(patch)
    with pytest.raises(ValueError, match=msg):
        digital_twin_summary_to_ids(summary, **kwargs)


@pytest.mark.parametrize(
    ("summary_patch", "msg"),
    [
        ({"steps": 12.5}, "summary.steps"),
        ({"steps": -1}, "summary.steps"),
        ({"final_axis_r": float("nan")}, "summary.final_axis_r"),
        ({"final_islands_px": 2.2}, "summary.final_islands_px"),
        ({"final_reward": "bad"}, "summary.final_reward"),
    ],
)
def test_digital_twin_summary_to_ids_rejects_invalid_summary_fields(
    summary_patch: dict[str, object],
    msg: str,
) -> None:
    summary = run_digital_twin(time_steps=12, seed=32, save_plot=False, verbose=False)
    summary.update(summary_patch)
    with pytest.raises(ValueError, match=msg):
        digital_twin_summary_to_ids(summary, machine="ITER", shot=101, run=2)


def test_ids_to_digital_twin_summary_rejects_non_ms_time_slice_time() -> None:
    summary = run_digital_twin(time_steps=12, seed=33, save_plot=False, verbose=False)
    ids_payload = digital_twin_summary_to_ids(summary, machine="ITER", shot=101, run=2)
    ids_payload["time_slice"]["time_s"] = 0.0125
    with pytest.raises(ValueError, match="integer millisecond"):
        ids_to_digital_twin_summary(ids_payload)


@pytest.mark.parametrize(
    ("patch_path", "value", "msg"),
    [
        (("equilibrium", "axis", "r_m"), float("inf"), "equilibrium.axis.r_m"),
        (("equilibrium", "islands_px"), -1, "equilibrium.islands_px"),
        (("performance", "final_reward"), float("nan"), "performance.final_reward"),
        (("time_slice", "index"), 1.2, "time_slice.index"),
    ],
)
def test_ids_to_digital_twin_summary_rejects_invalid_field_types(
    patch_path: tuple[str, ...],
    value: object,
    msg: str,
) -> None:
    summary = run_digital_twin(time_steps=14, seed=34, save_plot=False, verbose=False)
    ids_payload = digital_twin_summary_to_ids(summary, machine="ITER", shot=101, run=2)
    node: dict[str, object] = ids_payload
    for key in patch_path[:-1]:
        next_node = node[key]
        assert isinstance(next_node, dict)
        node = next_node
    node[patch_path[-1]] = value
    with pytest.raises(ValueError, match=msg):
        ids_to_digital_twin_summary(ids_payload)


def test_run_digital_twin_supports_deterministic_gyro_surrogate() -> None:
    def surrogate(temp_map: np.ndarray, q_map: np.ndarray, danger: np.ndarray) -> np.ndarray:
        _ = q_map
        return np.where(danger, 1.35, 0.95 + 0.0005 * temp_map)

    a = run_digital_twin(
        time_steps=20,
        seed=13,
        save_plot=False,
        verbose=False,
        gyro_surrogate=surrogate,
    )
    b = run_digital_twin(
        time_steps=20,
        seed=13,
        save_plot=False,
        verbose=False,
        gyro_surrogate=surrogate,
    )
    assert a["final_avg_temp"] == b["final_avg_temp"]
    assert a["final_reward"] == b["final_reward"]
    assert a["final_action"] == b["final_action"]


def test_run_digital_twin_validates_gyro_surrogate_shape() -> None:
    def bad_surrogate(temp_map: np.ndarray, q_map: np.ndarray, danger: np.ndarray) -> np.ndarray:
        _ = q_map
        _ = danger
        return np.ones((temp_map.shape[0],), dtype=float)

    with pytest.raises(ValueError, match="gyro_surrogate correction shape"):
        run_digital_twin(
            time_steps=4,
            seed=7,
            save_plot=False,
            verbose=False,
            gyro_surrogate=bad_surrogate,
        )


def test_run_digital_twin_does_not_mutate_global_numpy_rng_state() -> None:
    np.random.seed(4321)
    state = np.random.get_state()

    run_digital_twin(time_steps=8, seed=2, save_plot=False, verbose=False)

    observed = float(np.random.random())
    np.random.set_state(state)
    expected = float(np.random.random())
    assert observed == expected


def test_run_digital_twin_accepts_injected_rng_and_replays_deterministically() -> None:
    a = run_digital_twin(
        time_steps=20,
        seed=1,
        save_plot=False,
        verbose=False,
        rng=np.random.default_rng(2026),
    )
    b = run_digital_twin(
        time_steps=20,
        seed=999,
        save_plot=False,
        verbose=False,
        rng=np.random.default_rng(2026),
    )
    assert a["final_avg_temp"] == b["final_avg_temp"]
    assert a["final_reward"] == b["final_reward"]
    assert a["final_action"] == b["final_action"]


def test_run_digital_twin_rejects_invalid_time_steps() -> None:
    with pytest.raises(ValueError, match="time_steps"):
        run_digital_twin(time_steps=0, seed=2, save_plot=False, verbose=False)


def test_run_digital_twin_ids_history_returns_valid_sequence() -> None:
    payloads = run_digital_twin_ids_history(
        [6, 12, 18],
        machine="ITER",
        shot=42,
        run=7,
        seed=21,
        save_plot=False,
        verbose=False,
    )
    validate_ids_payload_sequence(payloads)
    recovered = ids_to_digital_twin_history(payloads)
    assert len(payloads) == 3
    assert len(recovered) == 3
    indices = [int(payload["time_slice"]["index"]) for payload in payloads]
    times = [float(payload["time_slice"]["time_s"]) for payload in payloads]
    assert indices == [0, 1, 2]
    assert times[0] < times[1] < times[2]


def test_run_digital_twin_ids_history_is_deterministic() -> None:
    kwargs = dict(
        history_steps=[5, 10],
        machine="ITER",
        shot=1,
        run=2,
        seed=31,
        save_plot=False,
        verbose=False,
    )
    a = run_digital_twin_ids_history(**kwargs)
    b = run_digital_twin_ids_history(**kwargs)
    assert a == b


def test_run_digital_twin_ids_pulse_returns_valid_container() -> None:
    pulse = run_digital_twin_ids_pulse(
        [6, 12, 18],
        machine="ITER",
        shot=42,
        run=7,
        seed=21,
        save_plot=False,
        verbose=False,
    )
    validate_ids_pulse_payload(pulse)
    recovered = ids_pulse_to_digital_twin_history(pulse)
    assert pulse["schema"] == "ids_equilibrium_pulse_v1"
    assert pulse["machine"] == "ITER"
    assert pulse["shot"] == 42
    assert pulse["run"] == 7
    assert len(pulse["time_slices"]) == 3
    assert len(recovered) == 3


def test_run_digital_twin_ids_pulse_is_deterministic() -> None:
    kwargs = dict(
        history_steps=[5, 10],
        machine="ITER",
        shot=1,
        run=2,
        seed=31,
        save_plot=False,
        verbose=False,
    )
    a = run_digital_twin_ids_pulse(**kwargs)
    b = run_digital_twin_ids_pulse(**kwargs)
    assert a == b


@pytest.mark.parametrize("history_steps", [[], [0, 2], [3.0, 6]])  # type: ignore[list-item]
def test_run_digital_twin_ids_history_rejects_invalid_history_steps(history_steps) -> None:
    with pytest.raises(ValueError, match="history_steps"):
        run_digital_twin_ids_history(
            history_steps,
            machine="ITER",
            shot=1,
            run=2,
            save_plot=False,
            verbose=False,
        )


@pytest.mark.parametrize("history_steps", [[], [0, 2], [3.0, 6]])  # type: ignore[list-item]
def test_run_digital_twin_ids_pulse_rejects_invalid_history_steps(history_steps) -> None:
    with pytest.raises(ValueError, match="history_steps"):
        run_digital_twin_ids_pulse(
            history_steps,
            machine="ITER",
            shot=1,
            run=2,
            save_plot=False,
            verbose=False,
        )
