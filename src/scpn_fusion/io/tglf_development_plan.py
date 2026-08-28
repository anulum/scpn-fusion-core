# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Official TGLF Development Sampling Plan
"""Deterministic physical-domain plans for official multi-species TGLF runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
import math
from typing import Any, Final, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core._tglf_interface_types import TGLFInputDeck, TGLFSpecies
from scpn_fusion.io.tglf_species_dataset_contract import tglf_species_input_payload

TGLF_DEVELOPMENT_PLAN_VERSION: Final = "scpn-fusion.tglf-development-plan.v1"
TGLF_DEVELOPMENT_DESIGN_METHOD: Final = (
    "seeded-stratified-latin-hypercube-with-frozen-boundary-and-transition-probes.v1"
)
TGLF_DEVELOPMENT_GACODE_REVISION: Final = "b49339750a4aa4cf2b089fa9ff3afe098005f0f8"
TGLF_DEVELOPMENT_SEED: Final = 20260826
TGLF_EXPANDED_SELECTION_SEED: Final = 20260828
_FLOAT_RANGES: Final[dict[str, tuple[float, float]]] = {
    "rho": (0.20, 0.85),
    "s_hat": (0.10, 2.50),
    "q": (1.10, 4.50),
    "alpha_mhd": (0.0, 1.50),
    "kappa": (1.0, 2.20),
    "delta": (-0.35, 0.55),
    "s_kappa": (-0.80, 0.80),
    "s_delta": (-0.80, 0.80),
    "beta_e": (1.0e-5, 0.04),
    "xnue": (0.0, 0.50),
    "T_e_keV": (0.5, 20.0),
    "n_e_19": (0.5, 20.0),
    "R_major": (1.5, 7.0),
    "inverse_aspect_ratio": (0.20, 0.40),
    "B_toroidal": (1.5, 8.0),
    "ion_temperature_ratio": (0.30, 2.0),
    "R_LTe": (0.0, 15.0),
    "R_LTi": (0.0, 15.0),
    "R_Lne": (0.0, 6.0),
    "target_R_Ln_center": (0.5, 5.5),
    "carbon_density_fraction": (0.005, 0.03),
    "deuterium_fraction_dt": (0.10, 0.90),
}
_COMPOSITIONS: Final = (
    "electron-deuterium",
    "electron-deuterium-tritium",
    "electron-deuterium-carbon",
)
_REJECTION_REASONS: Final = (
    "non_quasineutral_composition",
    "minor_radius_not_less_than_major_radius",
    "electromagnetic_switch_requires_positive_beta",
)
FloatArray: TypeAlias = NDArray[np.float64]


def canonical_tglf_development_digest(value: Any) -> str:
    """Return SHA-256 over canonical finite JSON bytes for plan/recovery custody."""
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _strict_json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(dict(value), allow_nan=False, separators=(",", ":"), sort_keys=True)
    return cast(dict[str, Any], json.loads(encoded))


def _latin_hypercube(samples: int, dimensions: int, seed: int) -> FloatArray:
    rng = np.random.default_rng(seed)
    design: FloatArray = np.empty((samples, dimensions), dtype=np.float64)
    for column in range(dimensions):
        strata = rng.permutation(samples)
        design[:, column] = (strata + rng.random(samples)) / samples
    return design


def _allocation(profile: str) -> list[tuple[str, str]]:
    if profile == "development":
        repetitions = {"interior": 4, "boundary": 2, "threshold": 2}
        return [
            (stratum, composition)
            for stratum, count in repetitions.items()
            for _ in range(count)
            for composition in _COMPOSITIONS
        ]
    if profile == "fixture":
        return list(zip(("interior", "boundary", "threshold"), _COMPOSITIONS, strict=True))
    if profile == "expanded":
        repetitions = {"interior": 12, "boundary": 6, "threshold": 6}
        return [
            (stratum, composition)
            for stratum, count in repetitions.items()
            for _ in range(count)
            for composition in _COMPOSITIONS
        ]
    raise ValueError("profile must be 'development', 'expanded' or 'fixture'")


def _sampled_rows(groups: int, seed: int) -> list[dict[str, float]]:
    names = tuple(_FLOAT_RANGES)
    unit = _latin_hypercube(groups, len(names), seed)
    rows: list[dict[str, float]] = []
    for row in unit:
        rows.append(
            {
                name: float(lower + value * (upper - lower))
                for name, value in zip(names, row, strict=True)
                for lower, upper in [_FLOAT_RANGES[name]]
            }
        )
    return rows


def _negative_control_reason(payload: Mapping[str, Any]) -> str | None:
    if not math.isclose(
        float(payload["electron_charge_density"]),
        float(payload["ion_charge_density"]),
        rel_tol=1.0e-9,
        abs_tol=1.0e-12,
    ):
        return _REJECTION_REASONS[0]
    if float(payload["a_minor"]) >= float(payload["R_major"]):
        return _REJECTION_REASONS[1]
    if bool(payload["use_bper"] or payload["use_bpar"]) and float(payload["beta_e"]) <= 0.0:
        return _REJECTION_REASONS[2]
    return None


def _negative_controls() -> list[dict[str, Any]]:
    valid = {
        "electron_charge_density": 1.0,
        "ion_charge_density": 1.0,
        "R_major": 3.0,
        "a_minor": 1.0,
        "beta_e": 0.01,
        "use_bper": False,
        "use_bpar": False,
    }
    candidates = [
        {**valid, "ion_charge_density": 0.9},
        {**valid, "a_minor": 3.1},
        {**valid, "beta_e": 0.0, "use_bper": True},
    ]
    rejected: list[dict[str, Any]] = []
    for candidate_index, payload in enumerate(candidates):
        reason = _negative_control_reason(payload)
        if reason != _REJECTION_REASONS[candidate_index]:
            raise RuntimeError("frozen negative control was not rejected by its declared gate")
        rejected.append(
            {
                "candidate_index": candidate_index,
                "reason": reason,
                "payload": payload,
                "payload_sha256": canonical_tglf_development_digest(payload),
            }
        )
    return rejected


def _electromagnetic_flags(group_index: int) -> tuple[bool, bool]:
    mode = group_index % 3
    return (mode >= 1, mode == 2)


def _species_for_group(
    composition: str, row: Mapping[str, float]
) -> tuple[tuple[TGLFSpecies, ...], str]:
    electron = TGLFSpecies("electron", 2.723e-4, -1.0, 1.0, 1.0, row["R_Lne"], row["R_LTe"])
    ion_temperature = row["ion_temperature_ratio"]
    if composition == "electron-deuterium":
        return (
            electron,
            TGLFSpecies("deuterium", 1.0, 1.0, 1.0, ion_temperature, 2.0, row["R_LTi"]),
        ), "deuterium"
    if composition == "electron-deuterium-tritium":
        deuterium_fraction = row["deuterium_fraction_dt"]
        return (
            electron,
            TGLFSpecies(
                "deuterium",
                1.0,
                1.0,
                deuterium_fraction,
                ion_temperature,
                2.0,
                row["R_LTi"],
            ),
            TGLFSpecies(
                "tritium",
                1.5,
                1.0,
                1.0 - deuterium_fraction,
                ion_temperature,
                2.0,
                row["R_LTi"],
            ),
        ), "tritium"
    carbon_fraction = row["carbon_density_fraction"]
    return (
        electron,
        TGLFSpecies(
            "deuterium",
            1.0,
            1.0,
            1.0 - 6.0 * carbon_fraction,
            ion_temperature,
            2.0,
            row["R_LTi"],
        ),
        TGLFSpecies(
            "carbon",
            6.0,
            6.0,
            carbon_fraction,
            0.8 * ion_temperature,
            2.0,
            row["R_LTi"],
        ),
    ), "carbon"


def _pin_stratum(row: dict[str, float], stratum: str, group_index: int) -> None:
    if stratum == "boundary":
        names = (
            "rho",
            "s_hat",
            "q",
            "kappa",
            "delta",
            "beta_e",
            "xnue",
            "T_e_keV",
            "n_e_19",
            "inverse_aspect_ratio",
            "B_toroidal",
        )
        name = names[group_index % len(names)]
        row[name] = _FLOAT_RANGES[name][group_index % 2]
    elif stratum == "threshold":
        probes = ((2.5, 3.5, 1.0), (3.5, 2.5, 1.5), (4.0, 4.0, 2.0))
        row["R_LTe"], row["R_LTi"], row["target_R_Ln_center"] = probes[group_index % len(probes)]


def _base_deck(
    row: Mapping[str, float], species: tuple[TGLFSpecies, ...], group: int
) -> TGLFInputDeck:
    use_bper, use_bpar = _electromagnetic_flags(group)
    carbon = next((item for item in species if item.name == "carbon"), None)
    z_eff = 1.0 if carbon is None else 1.0 + 30.0 * carbon.density_e_ratio
    return TGLFInputDeck(
        rho=row["rho"],
        s_hat=row["s_hat"],
        q=row["q"],
        q_prime_loc=0.0,
        alpha_mhd=row["alpha_mhd"],
        p_prime_loc=0.0,
        kappa=row["kappa"],
        delta=row["delta"],
        s_kappa=row["s_kappa"],
        s_delta=row["s_delta"],
        beta_e=row["beta_e"],
        Z_eff=z_eff,
        xnue=row["xnue"],
        T_e_keV=row["T_e_keV"],
        T_i_keV=row["T_e_keV"] * row["ion_temperature_ratio"],
        n_e_19=row["n_e_19"],
        R_major=row["R_major"],
        a_minor=row["R_major"] * row["inverse_aspect_ratio"],
        B_toroidal=row["B_toroidal"],
        use_bper=use_bper,
        use_bpar=use_bpar,
        species=species,
    )


def build_tglf_development_plan(
    *, seed: int = TGLF_DEVELOPMENT_SEED, profile: str = "development"
) -> dict[str, Any]:
    """Build the byte-stable official-TGLF sampling plan before execution.

    Parameters
    ----------
    seed : int, optional
        Non-negative deterministic sampling seed.
    profile : {"development", "expanded", "fixture"}, optional
        Frozen 72-run development design, 216-run expanded selection design,
        or a nine-run authentic contract fixture.

    Returns
    -------
    dict[str, Any]
        Strict JSON-compatible plan with its own canonical SHA-256.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    allocation = _allocation(profile)
    rows = _sampled_rows(len(allocation), seed)
    runs: list[dict[str, Any]] = []
    for group_index, ((stratum, composition), sampled) in enumerate(
        zip(allocation, rows, strict=True)
    ):
        row = dict(sampled)
        _pin_stratum(row, stratum, group_index)
        species, target = _species_for_group(composition, row)
        base = _base_deck(row, species, group_index)
        group_prefix = "expanded-official" if profile == "expanded" else f"data03-{profile}"
        group_id = f"{group_prefix}-{stratum}-{composition}-{group_index:03d}"
        for gradient in (
            row["target_R_Ln_center"] - 0.5,
            row["target_R_Ln_center"],
            row["target_R_Ln_center"] + 0.5,
        ):
            varied_species = tuple(
                replace(item, R_Ln=gradient) if item.name == target else item
                for item in base.species
            )
            runs.append(
                {
                    "sample_index": len(runs),
                    "group_id": group_id,
                    "sampling_stratum": stratum,
                    "composition": composition,
                    "paired_gradient_species": target,
                    "regime": "unclassified",
                    "input": tglf_species_input_payload(replace(base, species=varied_species)),
                }
            )
    payload: dict[str, Any] = {
        "SPDX-License-Identifier": "AGPL-3.0-or-later",
        "schema_version": TGLF_DEVELOPMENT_PLAN_VERSION,
        "gacode_revision": TGLF_DEVELOPMENT_GACODE_REVISION,
        "seed": seed,
        "profile": profile,
        "design_method": TGLF_DEVELOPMENT_DESIGN_METHOD,
        "domains": {
            name: {"minimum": bounds[0], "maximum": bounds[1]}
            for name, bounds in _FLOAT_RANGES.items()
        },
        "base_groups": len(allocation),
        "samples_per_group": 3,
        "accepted_runs": len(runs),
        "runs": runs,
        "negative_controls": _negative_controls(),
    }
    payload["plan_sha256"] = canonical_tglf_development_digest(payload)
    return payload


def validate_tglf_development_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate a plan and require exact deterministic regeneration."""
    payload = _strict_json_copy(plan)
    digest = payload.pop("plan_sha256", None)
    if not isinstance(digest, str) or digest != canonical_tglf_development_digest(payload):
        raise ValueError("development plan SHA-256 mismatch")
    payload["plan_sha256"] = digest
    seed = payload.get("seed")
    profile = payload.get("profile")
    if isinstance(seed, bool) or not isinstance(seed, int) or not isinstance(profile, str):
        raise ValueError("development plan seed/profile is invalid")
    rebuilt = build_tglf_development_plan(seed=seed, profile=profile)
    if payload != rebuilt:
        raise ValueError("development plan differs from deterministic regeneration")
    return payload


__all__ = [
    "TGLF_DEVELOPMENT_DESIGN_METHOD",
    "TGLF_DEVELOPMENT_GACODE_REVISION",
    "TGLF_DEVELOPMENT_PLAN_VERSION",
    "TGLF_DEVELOPMENT_SEED",
    "TGLF_EXPANDED_SELECTION_SEED",
    "build_tglf_development_plan",
    "canonical_tglf_development_digest",
    "validate_tglf_development_plan",
]
