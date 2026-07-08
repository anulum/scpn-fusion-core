#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TORAX IMAS Interchange Fixture
"""Generate the TORAX reference ``core_profiles`` IMAS interchange fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REFERENCE = ROOT / "validation" / "reference_data" / "torax" / "torax_basic_config_profiles.json"
IDS_OUTPUT = (
    ROOT / "validation" / "reference_data" / "torax" / "torax_basic_config_core_profiles_ids.json"
)
REPORT_JSON = ROOT / "validation" / "reports" / "torax_imas_interchange.json"
REPORT_MD = ROOT / "validation" / "reports" / "torax_imas_interchange.md"
SCHEMA = "scpn-fusion-core.torax-imas-interchange.v1"

sys.path.insert(0, str(SRC))

from scpn_fusion.io.imas_connector_omas import (  # noqa: E402
    HAS_OMAS,
    ids_to_omas_core_profiles,
    omas_core_profiles_to_ids,
)
from scpn_fusion.io.imas_connector_storage import read_ids, write_ids  # noqa: E402
from scpn_fusion.io.imas_connector_transport import state_to_imas_core_profiles  # noqa: E402


def _canonical_checksum(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 checksum for a JSON-like mapping."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _as_float_list(name: str, raw: object) -> list[float]:
    """Coerce a reference profile vector to finite floats."""
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(f"{name} must be a sequence.")
    values = [float(value) for value in raw]
    if len(values) < 2:
        raise ValueError(f"{name} must contain at least 2 values.")
    if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
        raise ValueError(f"{name} must contain only finite values.")
    return values


def load_torax_reference(path: Path = REFERENCE) -> dict[str, Any]:
    """Load and validate the tracked TORAX profile reference artifact.

    Parameters
    ----------
    path:
        JSON artifact produced by ``validation/torax_reference_runner.py``.

    Returns
    -------
    dict[str, Any]
        Validated TORAX reference payload.
    """
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if payload.get("schema") != "scpn-fusion-core.torax-reference-profiles.v1":
        raise ValueError("TORAX reference artifact has an unexpected schema.")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("code") != "TORAX":
        raise ValueError("TORAX reference artifact must carry TORAX provenance.")
    profiles = payload.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("TORAX reference artifact must contain profiles.")
    expected = ("rho_norm", "T_e_keV", "n_e_m3")
    missing = [key for key in expected if key not in profiles]
    if missing:
        raise ValueError(f"TORAX reference profiles missing keys: {missing}")
    lengths = {key: len(_as_float_list(f"profiles.{key}", profiles[key])) for key in expected}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"TORAX reference profile lengths are misaligned: {lengths}")
    rho = _as_float_list("profiles.rho_norm", profiles["rho_norm"])
    if any(next_value <= value for value, next_value in zip(rho, rho[1:])):
        raise ValueError("profiles.rho_norm must be strictly increasing.")
    return payload


def torax_reference_to_state(reference: Mapping[str, Any]) -> dict[str, list[float]]:
    """Convert TORAX reference profiles to the bounded internal state shape."""
    profiles = cast(Mapping[str, Any], reference["profiles"])
    rho = _as_float_list("profiles.rho_norm", profiles["rho_norm"])
    te_kev = _as_float_list("profiles.T_e_keV", profiles["T_e_keV"])
    ne_m3 = _as_float_list("profiles.n_e_m3", profiles["n_e_m3"])
    return {
        "rho_norm": rho,
        "electron_temp_keV": te_kev,
        "electron_density_1e20_m3": [value / 1.0e20 for value in ne_m3],
    }


def build_core_profiles_ids(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Build an IMAS ``core_profiles`` IDS from a TORAX reference artifact."""
    ids = state_to_imas_core_profiles(
        torax_reference_to_state(reference),
        time_s=float(reference["final_time_s"]),
    )
    ids["ids_properties"]["comment"] = (
        "SCPN Fusion Core TORAX basic_config core_profiles interchange fixture"
    )
    ids["ids_properties"]["provenance"] = {
        "source": "validation/reference_data/torax/torax_basic_config_profiles.json",
        "source_schema": reference["schema"],
        "source_code": cast(Mapping[str, Any], reference["provenance"])["code"],
        "source_config_sha256": cast(Mapping[str, Any], reference["provenance"])["config_sha256"],
        "source_profile_checksum_sha256": _canonical_checksum(
            cast(Mapping[str, Any], reference["profiles"])
        ),
    }
    return ids


def _validate_ids_against_reference(ids: Mapping[str, Any], reference: Mapping[str, Any]) -> None:
    """Verify that the IMAS fixture preserves the TORAX profile arrays."""
    state = torax_reference_to_state(reference)
    profiles = ids.get("profiles_1d")
    if not isinstance(profiles, list) or len(profiles) != 1:
        raise ValueError("core_profiles IDS must contain exactly one profiles_1d entry.")
    profile = cast(Mapping[str, Any], profiles[0])
    grid = cast(Mapping[str, Any], profile["grid"])
    electrons = cast(Mapping[str, Any], profile["electrons"])
    rho = np.asarray(grid["rho_tor_norm"], dtype=np.float64)
    te_ev = np.asarray(electrons["temperature"], dtype=np.float64)
    ne_m3 = np.asarray(electrons["density"], dtype=np.float64)
    if not np.allclose(rho, np.asarray(state["rho_norm"], dtype=np.float64), rtol=0.0, atol=0.0):
        raise ValueError("core_profiles rho_tor_norm does not match TORAX rho_norm.")
    if not np.allclose(
        te_ev,
        np.asarray(state["electron_temp_keV"], dtype=np.float64) * 1.0e3,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("core_profiles electron temperature does not match TORAX Te.")
    if not np.allclose(ne_m3, np.asarray(cast(Mapping[str, Any], reference["profiles"])["n_e_m3"])):
        raise ValueError("core_profiles electron density does not match TORAX ne.")


def _omas_roundtrip_status(ids: Mapping[str, Any]) -> dict[str, Any]:
    """Run the optional OMAS roundtrip when the runtime dependency is available."""
    if not HAS_OMAS:
        return {
            "available": False,
            "roundtrip_executed": False,
            "status": "not_executed_optional_dependency_missing",
        }
    ods = ids_to_omas_core_profiles(ids)
    roundtrip = omas_core_profiles_to_ids(ods)
    return {
        "available": True,
        "roundtrip_executed": True,
        "status": "roundtrip_passed",
        "roundtrip_checksum_sha256": _canonical_checksum(roundtrip),
    }


def build_report(ids: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    """Build the TORAX IMAS interchange report payload."""
    _validate_ids_against_reference(ids, reference)
    profiles = cast(Mapping[str, Any], reference["profiles"])
    profile_count = len(cast(Sequence[Any], profiles["rho_norm"]))
    return {
        "schema": SCHEMA,
        "status": "torax_core_profiles_imas_fixture_ready",
        "passes_thresholds": True,
        "physics_equivalence_claimed": False,
        "claim_boundary": (
            "This artifact validates deterministic TORAX profile interchange into "
            "IMAS core_profiles. It does not tighten TORAX-vs-native physics "
            "equivalence thresholds."
        ),
        "reference": {
            "artifact": str(REFERENCE.relative_to(ROOT)),
            "schema": reference["schema"],
            "profile_checksum_sha256": _canonical_checksum(profiles),
            "final_time_s": float(reference["final_time_s"]),
            "provenance": reference["provenance"],
        },
        "imas_fixture": {
            "artifact": str(IDS_OUTPUT.relative_to(ROOT)),
            "ids_type": "core_profiles",
            "profile_count": profile_count,
            "ids_checksum_sha256": _canonical_checksum(ids),
            "unit_conversions": {
                "electron_temperature": "TORAX T_e_keV * 1000 -> IMAS electrons.temperature eV",
                "electron_density": "TORAX n_e_m3 -> IMAS electrons.density m^-3",
                "radial_grid": "TORAX rho_norm -> IMAS grid.rho_tor_norm",
            },
        },
        "omas_bridge": _omas_roundtrip_status(ids),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact Markdown report for the interchange fixture."""
    imas = cast(Mapping[str, Any], report["imas_fixture"])
    omas_status = cast(Mapping[str, Any], report["omas_bridge"])
    lines = [
        "# TORAX IMAS Interchange",
        "",
        f"Status: `{report['status']}`",
        "",
        f"Physics equivalence claimed: `{report['physics_equivalence_claimed']}`",
        "",
        str(report["claim_boundary"]),
        "",
        "## Artifacts",
        "",
        f"- Reference: `{cast(Mapping[str, Any], report['reference'])['artifact']}`",
        f"- IMAS fixture: `{imas['artifact']}`",
        f"- IDS type: `{imas['ids_type']}`",
        f"- Profile points: `{imas['profile_count']}`",
        f"- IDS checksum: `{imas['ids_checksum_sha256']}`",
        "",
        "## OMAS Bridge",
        "",
        f"- Runtime available: `{omas_status['available']}`",
        f"- Roundtrip executed: `{omas_status['roundtrip_executed']}`",
        f"- Status: `{omas_status['status']}`",
        "",
    ]
    return "\n".join(lines)


def write_artifacts(
    *,
    reference_path: Path = REFERENCE,
    ids_output: Path = IDS_OUTPUT,
    report_json: Path = REPORT_JSON,
    report_md: Path = REPORT_MD,
) -> dict[str, Any]:
    """Generate and write the IMAS fixture plus reports."""
    reference = load_torax_reference(reference_path)
    ids = build_core_profiles_ids(reference)
    _validate_ids_against_reference(ids, reference)

    ids_output.parent.mkdir(parents=True, exist_ok=True)
    write_ids(ids, ids_output)
    stored_ids = read_ids(ids_output)
    report = build_report(stored_ids, reference)

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_md.write_text(render_markdown(report), encoding="utf-8")
    return report


def check_artifacts(
    *,
    reference_path: Path = REFERENCE,
    ids_output: Path = IDS_OUTPUT,
    report_json: Path = REPORT_JSON,
    report_md: Path = REPORT_MD,
) -> list[str]:
    """Return drift errors for tracked TORAX IMAS interchange artifacts."""
    reference = load_torax_reference(reference_path)
    expected_ids = build_core_profiles_ids(reference)
    expected_report = build_report(expected_ids, reference)
    expected_md = render_markdown(expected_report)
    errors: list[str] = []

    if not ids_output.exists():
        errors.append(f"missing IMAS fixture: {ids_output}")
    else:
        observed_ids = read_ids(ids_output)
        if _canonical_checksum(observed_ids) != _canonical_checksum(expected_ids):
            errors.append("tracked IMAS fixture is stale")

    if not report_json.exists():
        errors.append(f"missing interchange report: {report_json}")
    else:
        observed_report = cast(
            Mapping[str, Any], json.loads(report_json.read_text(encoding="utf-8"))
        )
        if _canonical_checksum(observed_report) != _canonical_checksum(expected_report):
            errors.append("tracked interchange JSON report is stale")

    if not report_md.exists():
        errors.append(f"missing interchange Markdown report: {report_md}")
    elif report_md.read_text(encoding="utf-8") != expected_md:
        errors.append("tracked interchange Markdown report is stale")

    return errors


def main(argv: list[str] | None = None) -> int:
    """Run or drift-check the TORAX IMAS interchange fixture generator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--ids-output", type=Path, default=IDS_OUTPUT)
    parser.add_argument("--report-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    if args.check:
        errors = check_artifacts(
            reference_path=args.reference,
            ids_output=args.ids_output,
            report_json=args.report_json,
            report_md=args.report_md,
        )
        for error in errors:
            print(f"TORAX IMAS INTERCHANGE ERROR: {error}", file=sys.stderr)
        if errors:
            return 1
        print(f"TORAX IMAS interchange artifacts are up to date: {args.ids_output}")
        return 0

    report = write_artifacts(
        reference_path=args.reference,
        ids_output=args.ids_output,
        report_json=args.report_json,
        report_md=args.report_md,
    )
    print(json.dumps(report["imas_fixture"], indent=2, sort_keys=True))
    print(f"status: {report['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
