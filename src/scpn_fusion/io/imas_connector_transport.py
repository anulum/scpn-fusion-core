# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — IMAS Connector Transport/Core Profiles
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""IMAS DD v3 converters for core_profiles, summary, and core_transport."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

from scpn_fusion.io.imas_connector_common import (
    _coerce_finite_real_sequence,
    _coerce_profiles_1d,
)

IMAS_DD_CORE_PROFILES_KEYS = (
    "ids_properties",
    "time",
    "profiles_1d",
)
IMAS_DD_SUMMARY_KEYS = (
    "ids_properties",
    "time",
    "global_quantities",
)


def state_to_imas_core_profiles(
    state: Mapping[str, Any],
    *,
    time_s: float = 0.0,
) -> dict[str, Any]:
    """Convert a plasma state dict to an IMAS ``core_profiles`` IDS."""
    profiles = _coerce_profiles_1d(state, name="state")
    rho = profiles["rho_norm"]
    te_kev = profiles["electron_temp_keV"]
    ne_1e20 = profiles["electron_density_1e20_m3"]

    te_ev = [v * 1.0e3 for v in te_kev]
    ne_m3 = [v * 1.0e20 for v in ne_1e20]

    return {
        "ids_properties": {
            "homogeneous_time": 1,
            "comment": "SCPN Fusion Core core_profiles export",
        },
        "time": [float(time_s)],
        "profiles_1d": [
            {
                "time": float(time_s),
                "grid": {
                    "rho_tor_norm": rho,
                },
                "electrons": {
                    "temperature": te_ev,
                    "density": ne_m3,
                },
            }
        ],
    }


def state_to_imas_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a performance/state dict to an IMAS ``summary`` IDS."""
    if isinstance(state, bool) or not isinstance(state, Mapping):
        raise ValueError("state must be a mapping.")

    gq: dict[str, Any] = {}
    for key, imas_key in [
        ("power_fusion_MW", "power_fusion"),
        ("q_sci", "q"),
        ("beta_n", "beta_n"),
        ("li", "li"),
        ("plasma_current_MA", "ip"),
        ("confinement_time_s", "tau_e"),
    ]:
        val = state.get(key)
        if val is not None:
            gq[imas_key] = float(val)

    return {
        "ids_properties": {
            "homogeneous_time": 0,
            "comment": "SCPN Fusion Core summary export",
        },
        "time": [0.0],
        "global_quantities": gq,
    }


def state_to_imas_core_transport(
    state: Mapping[str, Any],
    *,
    time_s: float = 0.0,
) -> dict[str, Any]:
    """Convert a plasma state dict to an IMAS ``core_transport`` IDS."""
    if isinstance(state, bool) or not isinstance(state, Mapping):
        raise ValueError("state must be a mapping.")

    rho_raw = state.get("rho_norm")
    if rho_raw is None:
        raise ValueError("state must contain 'rho_norm'.")
    rho = _coerce_finite_real_sequence(
        "state.rho_norm",
        rho_raw,
        minimum_len=2,
        minimum=0.0,
        maximum=1.0,
        strictly_increasing=True,
    )

    transport_1d: dict[str, Any] = {
        "grid_d": {"rho_tor_norm": rho},
    }

    def _set_nested(root: dict[str, Any], path: str, value: list[float]) -> None:
        parts = path.split(".")
        current: dict[str, Any] = root
        for part in parts[:-1]:
            nested = current.setdefault(part, {})
            if not isinstance(nested, dict):
                raise ValueError(f"Unexpected non-dict node while setting {path}: {part}")
            current = nested
        current[parts[-1]] = value

    for key, setter in [
        ("chi_e", lambda v: _set_nested(transport_1d, "electrons.energy.d", v)),
        ("d_e", lambda v: _set_nested(transport_1d, "electrons.particles.d", v)),
    ]:
        val = state.get(key)
        if val is not None:
            vals = _coerce_finite_real_sequence(
                f"state.{key}",
                val,
                minimum_len=2,
                minimum=0.0,
            )
            if len(vals) != len(rho):
                raise ValueError(f"state.{key} length must match state.rho_norm.")
            setter(vals)

    chi_i_raw = state.get("chi_i")
    if chi_i_raw is not None:
        chi_i_vals = _coerce_finite_real_sequence(
            "state.chi_i",
            chi_i_raw,
            minimum_len=2,
            minimum=0.0,
        )
        if len(chi_i_vals) != len(rho):
            raise ValueError("state.chi_i length must match state.rho_norm.")
        transport_1d["ion"] = [{"energy": {"d": chi_i_vals}}]

    return {
        "ids_properties": {
            "homogeneous_time": 1,
            "comment": "SCPN Fusion Core core_transport export",
        },
        "time": [float(time_s)],
        "model": [
            {
                "time": float(time_s),
                "identifier": {
                    "name": "scpn_fusion_transport",
                    "description": "SCPN Fusion Core reduced-order transport",
                },
                "profiles_1d": [transport_1d],
            }
        ],
    }


def imas_core_transport_to_state(ids: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an IMAS ``core_transport`` IDS back to a state dict."""
    if not isinstance(ids, Mapping):
        raise ValueError("ids must be a mapping.")
    models = ids.get("model")
    if not isinstance(models, Sequence) or len(models) == 0:
        raise ValueError("core_transport IDS must contain at least one model.")

    model0 = models[0]
    profiles = model0.get("profiles_1d")
    if not isinstance(profiles, Sequence) or len(profiles) == 0:
        raise ValueError("core_transport model must have profiles_1d.")
    p1d = profiles[0]

    grid = p1d.get("grid_d", {})
    rho = grid.get("rho_tor_norm")
    if rho is None:
        raise ValueError("core_transport profiles_1d must contain grid_d.rho_tor_norm.")

    out: dict[str, Any] = {"rho_norm": list(rho)}

    electrons = p1d.get("electrons", {})
    chi_e = electrons.get("energy", {}).get("d")
    if chi_e is not None:
        out["chi_e"] = list(chi_e)

    d_e = electrons.get("particles", {}).get("d")
    if d_e is not None:
        out["d_e"] = list(d_e)

    ions = p1d.get("ion", [])
    if isinstance(ions, Sequence) and len(ions) > 0:
        ion0 = ions[0]
        chi_i = ion0.get("energy", {}).get("d") if isinstance(ion0, Mapping) else None
        if chi_i is not None:
            out["chi_i"] = list(chi_i)

    return out


__all__ = [
    "IMAS_DD_CORE_PROFILES_KEYS",
    "IMAS_DD_SUMMARY_KEYS",
    "state_to_imas_core_profiles",
    "state_to_imas_summary",
    "state_to_imas_core_transport",
    "imas_core_transport_to_state",
]

