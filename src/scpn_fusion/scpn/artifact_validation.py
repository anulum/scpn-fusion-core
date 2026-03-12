# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Artifact Validation Helpers
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math
from typing import Any, Type


def validate_artifact(
    artifact: Any,
    *,
    error_type: Type[Exception],
) -> None:
    """Lightweight checks: required fields, ranges, and shape consistency."""
    meta = artifact.meta

    if meta.firing_mode not in ("binary", "fractional"):
        raise error_type(f"firing_mode must be 'binary' or 'fractional', got '{meta.firing_mode}'")

    if isinstance(meta.fixed_point.data_width, bool) or not isinstance(
        meta.fixed_point.data_width, int
    ):
        raise error_type("fixed_point.data_width must be an integer >= 1")
    if meta.fixed_point.data_width < 1:
        raise error_type("fixed_point.data_width must be >= 1")
    if isinstance(meta.fixed_point.fraction_bits, bool) or not isinstance(
        meta.fixed_point.fraction_bits, int
    ):
        raise error_type("fixed_point.fraction_bits must be an integer >= 0")
    if meta.fixed_point.fraction_bits < 0:
        raise error_type("fixed_point.fraction_bits must be >= 0")
    if meta.fixed_point.fraction_bits >= meta.fixed_point.data_width:
        raise error_type("fixed_point.fraction_bits must be < fixed_point.data_width")
    if not isinstance(meta.fixed_point.signed, bool):
        raise error_type("fixed_point.signed must be a boolean")

    if isinstance(meta.stream_length, bool) or not isinstance(meta.stream_length, int):
        raise error_type("stream_length must be an integer >= 1")
    if meta.stream_length < 1:
        raise error_type("stream_length must be >= 1")

    if isinstance(meta.dt_control_s, bool) or not isinstance(meta.dt_control_s, (int, float)):
        raise error_type("dt_control_s must be finite and > 0")
    if not math.isfinite(meta.dt_control_s):
        raise error_type("dt_control_s must be finite and > 0")
    if meta.dt_control_s <= 0:
        raise error_type("dt_control_s must be > 0")

    for val in artifact.weights.w_in.data:
        if not (-1.0 <= val <= 1.0):
            raise error_type(f"w_in weight {val} outside [-1, 1]")
    for val in artifact.weights.w_out.data:
        if not (0.0 <= val <= 1.0):
            raise error_type(f"w_out weight {val} outside [0, 1]")

    for transition in artifact.topology.transitions:
        if isinstance(transition.threshold, bool) or not isinstance(
            transition.threshold, (int, float)
        ):
            raise error_type(
                f"threshold {transition.threshold} for '{transition.name}' must be finite and in [0, 1]"
            )
        if not math.isfinite(transition.threshold):
            raise error_type(
                f"threshold {transition.threshold} for '{transition.name}' must be finite and in [0, 1]"
            )
        if not (0.0 <= transition.threshold <= 1.0):
            raise error_type(
                f"threshold {transition.threshold} for '{transition.name}' outside [0, 1]"
            )
        if transition.margin is not None:
            if isinstance(transition.margin, bool) or not isinstance(
                transition.margin, (int, float)
            ):
                raise error_type(
                    f"margin {transition.margin} for '{transition.name}' must be finite and >= 0"
                )
            if not math.isfinite(transition.margin) or transition.margin < 0.0:
                raise error_type(
                    f"margin {transition.margin} for '{transition.name}' must be finite and >= 0"
                )
        if isinstance(transition.delay_ticks, bool) or not isinstance(transition.delay_ticks, int):
            raise error_type(
                f"delay_ticks {transition.delay_ticks} for '{transition.name}' must be an integer >= 0"
            )
        if transition.delay_ticks < 0:
            raise error_type(
                f"delay_ticks {transition.delay_ticks} for '{transition.name}' must be >= 0"
            )

    nP = artifact.nP
    nT = artifact.nT
    expected_w_in = nT * nP
    expected_w_out = nP * nT

    if len(artifact.weights.w_in.data) != expected_w_in:
        raise error_type(
            f"w_in data length {len(artifact.weights.w_in.data)} != nT*nP={expected_w_in}"
        )
    if len(artifact.weights.w_out.data) != expected_w_out:
        raise error_type(
            f"w_out data length {len(artifact.weights.w_out.data)} != nP*nT={expected_w_out}"
        )

    if len(artifact.initial_state.marking) != nP:
        raise error_type(f"marking length {len(artifact.initial_state.marking)} != nP={nP}")
    for val in artifact.initial_state.marking:
        if not (0.0 <= val <= 1.0):
            raise error_type(f"initial marking {val} outside [0, 1]")
    for injection in artifact.initial_state.place_injections:
        if isinstance(injection.place_id, bool) or not isinstance(injection.place_id, int):
            raise error_type("place_injections.place_id must be an integer")
        if injection.place_id < 0 or injection.place_id >= nP:
            raise error_type(
                f"place_injections.place_id {injection.place_id} out of bounds for nP={nP}"
            )
        if not isinstance(injection.source, str) or not injection.source:
            raise error_type("place_injections.source must be a non-empty string")
        if (
            isinstance(injection.scale, bool)
            or not isinstance(injection.scale, (int, float))
            or not math.isfinite(injection.scale)
        ):
            raise error_type("place_injections.scale must be finite numeric")
        if (
            isinstance(injection.offset, bool)
            or not isinstance(injection.offset, (int, float))
            or not math.isfinite(injection.offset)
        ):
            raise error_type("place_injections.offset must be finite numeric")
        if not isinstance(injection.clamp_0_1, bool):
            raise error_type("place_injections.clamp_0_1 must be a boolean")

    n_actions = len(artifact.readout.actions)
    for action in artifact.readout.actions:
        if isinstance(action.id, bool) or not isinstance(action.id, int) or action.id < 0:
            raise error_type("readout.actions.id must be an integer >= 0")
        if not isinstance(action.name, str) or not action.name:
            raise error_type("readout.actions.name must be a non-empty string")
        if isinstance(action.pos_place, bool) or not isinstance(action.pos_place, int):
            raise error_type("readout.actions.pos_place must be an integer")
        if isinstance(action.neg_place, bool) or not isinstance(action.neg_place, int):
            raise error_type("readout.actions.neg_place must be an integer")
        if action.pos_place < 0 or action.pos_place >= nP:
            raise error_type(
                f"readout.actions.pos_place {action.pos_place} out of bounds for nP={nP}"
            )
        if action.neg_place < 0 or action.neg_place >= nP:
            raise error_type(
                f"readout.actions.neg_place {action.neg_place} out of bounds for nP={nP}"
            )
    if len(artifact.readout.gains) != n_actions:
        raise error_type("readout.gains length must equal number of actions")
    if len(artifact.readout.abs_max) != n_actions:
        raise error_type("readout.abs_max length must equal number of actions")
    if len(artifact.readout.slew_per_s) != n_actions:
        raise error_type("readout.slew_per_s length must equal number of actions")
    for val in artifact.readout.gains:
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise error_type("readout.gains must contain finite numeric values")
    for val in artifact.readout.abs_max:
        if (
            isinstance(val, bool)
            or not isinstance(val, (int, float))
            or not math.isfinite(val)
            or val < 0.0
        ):
            raise error_type("readout.abs_max must contain finite numeric values >= 0")
    for val in artifact.readout.slew_per_s:
        if (
            isinstance(val, bool)
            or not isinstance(val, (int, float))
            or not math.isfinite(val)
            or val < 0.0
        ):
            raise error_type("readout.slew_per_s must contain finite numeric values >= 0")


__all__ = ["validate_artifact"]
