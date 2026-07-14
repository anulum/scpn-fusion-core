# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Controller Artifact Schema Guard
"""Schema validator for raw ``.scpnctl.json`` artifact payloads."""

from __future__ import annotations

from typing import Any, Type


def _require_object(value: Any, path: str, *, error_type: Type[Exception]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise error_type(f"{path} must be an object")
    return value


def _require_array(value: Any, path: str, *, error_type: Type[Exception]) -> list[Any]:
    if not isinstance(value, list):
        raise error_type(f"{path} must be an array")
    return value


def _validate_object_keys(
    obj: dict[str, Any],
    *,
    path: str,
    required: set[str],
    allowed: set[str],
    error_type: Type[Exception],
) -> None:
    for key in sorted(required - obj.keys()):
        raise error_type(f"missing required property {path}.{key}")
    for key in sorted(obj.keys() - allowed):
        raise error_type(f"unexpected property {path}.{key}")


def _validate_weight_matrix_schema(obj: Any, path: str, *, error_type: Type[Exception]) -> None:
    wm = _require_object(obj, path, error_type=error_type)
    _validate_object_keys(
        wm,
        path=path,
        required={"shape", "data"},
        allowed={"shape", "data"},
        error_type=error_type,
    )
    _require_array(wm["shape"], f"{path}.shape", error_type=error_type)
    _require_array(wm["data"], f"{path}.data", error_type=error_type)


def _validate_packed_weight_schema(obj: Any, path: str, *, error_type: Type[Exception]) -> None:
    pw = _require_object(obj, path, error_type=error_type)
    common = {"shape"}
    expanded = common | {"data_u64"}
    compact = common | {"encoding", "count", "data_u64_b64_zlib"}
    keys = set(pw)
    if keys >= expanded and "data_u64" in keys:
        _validate_object_keys(
            pw, path=path, required=expanded, allowed=expanded, error_type=error_type
        )
        _require_array(pw["data_u64"], f"{path}.data_u64", error_type=error_type)
    elif keys >= compact and "data_u64_b64_zlib" in keys:
        _validate_object_keys(
            pw, path=path, required=compact, allowed=compact, error_type=error_type
        )
    else:
        raise error_type(
            f"{path} must contain either data_u64 or compact data_u64_b64_zlib payload"
        )
    _require_array(pw["shape"], f"{path}.shape", error_type=error_type)


def validate_raw_artifact_schema(obj: Any, *, error_type: Type[Exception]) -> None:
    """Validate raw ``.scpnctl.json`` structure before dataclass construction."""
    root = _require_object(obj, "root", error_type=error_type)
    _validate_object_keys(
        root,
        path="root",
        required={"meta", "topology", "weights", "readout", "initial_state"},
        allowed={"meta", "topology", "weights", "readout", "initial_state"},
        error_type=error_type,
    )

    meta = _require_object(root["meta"], "meta", error_type=error_type)
    _validate_object_keys(
        meta,
        path="meta",
        required={
            "artifact_version",
            "name",
            "dt_control_s",
            "stream_length",
            "fixed_point",
            "firing_mode",
            "seed_policy",
            "created_utc",
            "compiler",
        },
        allowed={
            "artifact_version",
            "name",
            "dt_control_s",
            "stream_length",
            "fixed_point",
            "firing_mode",
            "seed_policy",
            "created_utc",
            "compiler",
            "notes",
        },
        error_type=error_type,
    )
    _validate_object_keys(
        _require_object(meta["fixed_point"], "meta.fixed_point", error_type=error_type),
        path="meta.fixed_point",
        required={"data_width", "fraction_bits", "signed"},
        allowed={"data_width", "fraction_bits", "signed"},
        error_type=error_type,
    )
    _validate_object_keys(
        _require_object(meta["seed_policy"], "meta.seed_policy", error_type=error_type),
        path="meta.seed_policy",
        required={"id", "hash_fn", "rng_family"},
        allowed={"id", "hash_fn", "rng_family"},
        error_type=error_type,
    )
    _validate_object_keys(
        _require_object(meta["compiler"], "meta.compiler", error_type=error_type),
        path="meta.compiler",
        required={"name", "version", "git_sha"},
        allowed={"name", "version", "git_sha", "proof_system", "proof_checksum"},
        error_type=error_type,
    )

    topology = _require_object(root["topology"], "topology", error_type=error_type)
    _validate_object_keys(
        topology,
        path="topology",
        required={"places", "transitions"},
        allowed={"places", "transitions"},
        error_type=error_type,
    )
    for idx, place in enumerate(
        _require_array(topology["places"], "topology.places", error_type=error_type)
    ):
        _validate_object_keys(
            _require_object(place, f"topology.places[{idx}]", error_type=error_type),
            path=f"topology.places[{idx}]",
            required={"id", "name"},
            allowed={"id", "name"},
            error_type=error_type,
        )
    for idx, transition in enumerate(
        _require_array(topology["transitions"], "topology.transitions", error_type=error_type)
    ):
        _validate_object_keys(
            _require_object(transition, f"topology.transitions[{idx}]", error_type=error_type),
            path=f"topology.transitions[{idx}]",
            required={"id", "name", "threshold"},
            allowed={"id", "name", "threshold", "margin", "delay_ticks"},
            error_type=error_type,
        )

    weights = _require_object(root["weights"], "weights", error_type=error_type)
    _validate_object_keys(
        weights,
        path="weights",
        required={"w_in", "w_out"},
        allowed={"w_in", "w_out", "packed"},
        error_type=error_type,
    )
    _validate_weight_matrix_schema(weights["w_in"], "weights.w_in", error_type=error_type)
    _validate_weight_matrix_schema(weights["w_out"], "weights.w_out", error_type=error_type)
    if "packed" in weights:
        packed = _require_object(weights["packed"], "weights.packed", error_type=error_type)
        _validate_object_keys(
            packed,
            path="weights.packed",
            required={"words_per_stream", "w_in_packed"},
            allowed={"words_per_stream", "w_in_packed", "w_out_packed"},
            error_type=error_type,
        )
        _validate_packed_weight_schema(
            packed["w_in_packed"], "weights.packed.w_in_packed", error_type=error_type
        )
        if "w_out_packed" in packed:
            _validate_packed_weight_schema(
                packed["w_out_packed"],
                "weights.packed.w_out_packed",
                error_type=error_type,
            )

    readout = _require_object(root["readout"], "readout", error_type=error_type)
    _validate_object_keys(
        readout,
        path="readout",
        required={"actions", "gains", "limits"},
        allowed={"actions", "gains", "limits"},
        error_type=error_type,
    )
    for idx, action in enumerate(
        _require_array(readout["actions"], "readout.actions", error_type=error_type)
    ):
        _validate_object_keys(
            _require_object(action, f"readout.actions[{idx}]", error_type=error_type),
            path=f"readout.actions[{idx}]",
            required={"id", "name", "pos_place", "neg_place"},
            allowed={"id", "name", "pos_place", "neg_place"},
            error_type=error_type,
        )
    _validate_object_keys(
        _require_object(readout["gains"], "readout.gains", error_type=error_type),
        path="readout.gains",
        required={"per_action"},
        allowed={"per_action"},
        error_type=error_type,
    )
    _require_array(
        readout["gains"]["per_action"], "readout.gains.per_action", error_type=error_type
    )
    _validate_object_keys(
        _require_object(readout["limits"], "readout.limits", error_type=error_type),
        path="readout.limits",
        required={"per_action_abs_max", "slew_per_s"},
        allowed={"per_action_abs_max", "slew_per_s"},
        error_type=error_type,
    )
    _require_array(
        readout["limits"]["per_action_abs_max"],
        "readout.limits.per_action_abs_max",
        error_type=error_type,
    )
    _require_array(
        readout["limits"]["slew_per_s"], "readout.limits.slew_per_s", error_type=error_type
    )

    initial_state = _require_object(root["initial_state"], "initial_state", error_type=error_type)
    _validate_object_keys(
        initial_state,
        path="initial_state",
        required={"marking", "place_injections"},
        allowed={"marking", "place_injections"},
        error_type=error_type,
    )
    _require_array(initial_state["marking"], "initial_state.marking", error_type=error_type)
    for idx, injection in enumerate(
        _require_array(
            initial_state["place_injections"],
            "initial_state.place_injections",
            error_type=error_type,
        )
    ):
        _validate_object_keys(
            _require_object(
                injection, f"initial_state.place_injections[{idx}]", error_type=error_type
            ),
            path=f"initial_state.place_injections[{idx}]",
            required={"place_id", "source", "scale", "offset", "clamp_0_1"},
            allowed={"place_id", "source", "scale", "offset", "clamp_0_1"},
            error_type=error_type,
        )


__all__ = ["validate_raw_artifact_schema"]
