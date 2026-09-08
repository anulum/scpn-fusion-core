# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — cfspopcon operating-map integration
"""Run a source-pinned, uniform-DT scoping map through public cfspopcon APIs.

Equal D/T, Ti=Te and uniform density/temperature are prescribed. ITER98y2
sets total loss power; alpha heating and prescribed ohmic heating determine
required auxiliary power. This is code comparison, not experimental evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, cast


SOURCE_COMMIT = "b9ed8c3fd973bd2ad3d8226acbf77aa0e4caf7d1"
SOURCE_MANIFEST = Path(__file__).parent / "reference_data/cfspopcon_source.json"
REQUEST_SCHEMA = "scpn-fusion.cfspopcon-uniform-dt-request.v1"
RESULT_SCHEMA = "scpn-fusion.cfspopcon-uniform-dt-map.v1"
_SCALARS = {
    "major_radius_m",
    "minor_radius_m",
    "elongation",
    "magnetic_field_t",
    "plasma_current_ma",
    "h98",
    "ohmic_power_mw",
    "coupling_fraction",
    "auxiliary_limit_mw",
    "greenwald_limit",
}


def verify_upstream(package_root: Path) -> dict[str, str]:
    """Verify installed Python/YAML bytes against the reviewed upstream manifest.

    Parameters
    ----------
    package_root : Path
        Root of the cfspopcon package being used.

    Returns
    -------
    dict[str, str]
        Exact source commit and manifest digest.

    Raises
    ------
    ValueError
        If a source file is absent, added or changed.
    """
    raw = SOURCE_MANIFEST.read_bytes()
    manifest = json.loads(raw)
    expected = manifest["files"]
    actual = {
        str(path.relative_to(package_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in package_root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml"}
    }
    if manifest["commit"] != SOURCE_COMMIT or actual != expected:
        raise ValueError("cfspopcon source does not match the pinned manifest")
    return {"commit": SOURCE_COMMIT, "manifest_sha256": hashlib.sha256(raw).hexdigest()}


def validate_request(request: Mapping[str, object]) -> None:
    """Reject unsupported modes, nonfinite values and grids above 256 points.

    Parameters
    ----------
    request : Mapping[str, object]
        Exact schema plus SI geometry, MA/MW powers and keV temperature axes.

    Raises
    ------
    ValueError
        If keys, numeric domains or bounded grid dimensions are invalid.
    """
    keys = _SCALARS | {"schema", "density_m3", "temperature_kev"}
    if (
        not isinstance(request, Mapping)
        or set(request) != keys
        or request["schema"] != REQUEST_SCHEMA
    ):
        raise ValueError("unsupported operating-map request schema or fields")
    for key in _SCALARS:
        value = request[key]
        if type(value) not in (int, float) or not math.isfinite(float(cast(Any, value))):
            raise ValueError(f"{key} must be a finite number")
        if float(cast(Any, value)) < 0 or (value == 0 and key != "ohmic_power_mw"):
            raise ValueError(f"{key} is outside its positive domain")
    if cast(float, request["minor_radius_m"]) >= cast(float, request["major_radius_m"]):
        raise ValueError("minor radius must be below major radius")
    if cast(float, request["coupling_fraction"]) > 1:
        raise ValueError("coupling_fraction must be at most one")
    count = 1
    for key in ("density_m3", "temperature_kev"):
        values = request[key]
        if not isinstance(values, list) or not 1 <= len(values) <= 256:
            raise ValueError(f"{key} must be a nonempty bounded list")
        for value in values:
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} requires finite positive numbers")
            if key == "temperature_kev" and not 0.2 <= value <= 100:
                raise ValueError("temperature must remain in the 0.2–100 keV fit domain")
        if len(set(values)) != len(values):
            raise ValueError(f"{key} contains duplicate coordinates")
        count *= len(values)
    if count > 256:
        raise ValueError("operating map exceeds 256 points")


def _point(
    request: Mapping[str, Any], density: float, temperature: float, local_rate: float
) -> dict[str, Any]:
    """Evaluate one uniform state using public upstream formula entry points."""
    unit = importlib.import_module("cfspopcon.unit_handling")
    u = unit.ureg
    reaction = importlib.import_module(
        "cfspopcon.formulas.fusion_power.fusion_data"
    ).DTFusionBoschHale()
    stored = importlib.import_module("cfspopcon.formulas.energy_confinement.plasma_stored_energy")
    confinement = importlib.import_module(
        "cfspopcon.formulas.energy_confinement.solve_for_input_power"
    )
    gain = importlib.import_module("cfspopcon.formulas.fusion_power.fusion_gain")
    volume = (
        2
        * math.pi**2
        * request["major_radius_m"]
        * request["minor_radius_m"] ** 2
        * request["elongation"]
    )
    n, t = density / u.m**3, temperature * u.keV
    energy = stored.calc_plasma_stored_energy(n, t, n, 0 / u.m**3, t, volume * u.m**3)
    tau, loss = confinement.solve_energy_confinement_scaling_for_input_power(
        confinement_time_scalar=request["h98"],
        plasma_current=request["plasma_current_ma"] * u.MA,
        magnetic_field_on_axis=request["magnetic_field_t"] * u.T,
        average_electron_density=n,
        major_radius=request["major_radius_m"] * u.m,
        areal_elongation=request["elongation"],
        separatrix_elongation=request["elongation"],
        inverse_aspect_ratio=request["minor_radius_m"] / request["major_radius_m"],
        average_ion_mass=2.5 * u.amu,
        triangularity_psi95=0.0,
        separatrix_triangularity=0.0,
        plasma_stored_energy=energy,
        q_star=3.0,
        energy_confinement_scaling="ITER98y2",
    )
    fusion = reaction.calc_power_density(t, 0.5) * n**2 * volume * u.m**3
    alpha = fusion / 5
    auxiliary = (loss - alpha - request["ohmic_power_mw"] * u.MW) / request["coupling_fraction"]
    upstream_rate = float(unit.magnitude_in_units(reaction.calc_rate_coefficient(t), u.m**3 / u.s))
    values = {
        "stored_energy_mj": float(unit.magnitude_in_units(energy, u.MJ)),
        "tau_e_s": float(unit.magnitude_in_units(tau, u.s)),
        "loss_power_mw": float(unit.magnitude_in_units(loss, u.MW)),
        "fusion_power_mw": float(unit.magnitude_in_units(fusion, u.MW)),
        "alpha_power_mw": float(unit.magnitude_in_units(alpha, u.MW)),
        "auxiliary_launched_mw": float(unit.magnitude_in_units(auxiliary, u.MW)),
        "greenwald_fraction": density
        / (request["plasma_current_ma"] / (math.pi * request["minor_radius_m"] ** 2) * 1e20),
        "reactivity_m3_s": upstream_rate,
        "reactivity_relative_difference": abs(local_rate - upstream_rate) / upstream_rate,
    }
    reasons = []
    if not all(math.isfinite(value) for value in values.values()):
        reasons.append("nonfinite_upstream_result")
    if values["auxiliary_launched_mw"] < 0:
        reasons.append("negative_auxiliary_requirement")
    if values["auxiliary_launched_mw"] > request["auxiliary_limit_mw"]:
        reasons.append("auxiliary_limit_exceeded")
    if values["greenwald_fraction"] > request["greenwald_limit"]:
        reasons.append("greenwald_limit_exceeded")
    if values["reactivity_relative_difference"] > 1e-12:
        reasons.append("reactivity_code_parity_failed")
    if values["auxiliary_launched_mw"] + request["ohmic_power_mw"] <= 1e-6:
        reasons.append("gain_denominator_at_or_below_upstream_floor")
    q = None
    if not reasons:
        q = float(
            unit.magnitude_in_units(
                gain.calc_fusion_gain(fusion, request["ohmic_power_mw"] * u.MW, auxiliary),
                u.dimensionless,
            )
        )
    return {
        "density_m3": density,
        "temperature_kev": temperature,
        "status": "outside_declared_limits" if reasons else "within_declared_limits",
        "reasons": reasons,
        "q_launched": q,
        "values": {key: value if math.isfinite(value) else None for key, value in values.items()},
    }


def run_operating_map(
    request: Mapping[str, object], *, reference_python: Path, reference_root: Path | None = None
) -> dict[str, Any]:
    """Calculate all requested points without promoting results to evidence.

    Parameters
    ----------
    request : Mapping[str, object]
        A validated uniform-DT request; see the committed reference input.
    reference_python : Path
        Python executable in the existing SCPN environment. The public local
        reactivity API runs in this separate process to isolate NumPy versions.
    reference_root : Path, optional
        SCPN checkout to compare; defaults to this checkout. The actual imported
        source hash is retained, including for a candidate under review.

    Returns
    -------
    dict[str, Any]
        Source-bound design-only results, including every refused point.

    Raises
    ------
    ValueError
        If inputs or installed source custody do not match the contract.
    ModuleNotFoundError
        If the optional upstream runtime has not been installed.
    """
    validate_request(request)
    upstream = importlib.import_module("cfspopcon")
    provenance = verify_upstream(Path(cast(str, upstream.__file__)).parent)
    provenance["version"] = importlib.metadata.version("cfspopcon")
    root = (
        reference_root.absolute()
        if reference_root is not None
        else Path(__file__).resolve().parents[1]
    )
    code = """import hashlib,json,sys
from pathlib import Path
import numpy
from scpn_fusion.core import uncertainty
print(json.dumps({"rates": [float(uncertainty.bosch_hale_reactivity(t)) for t in json.load(sys.stdin)], "source_sha256": hashlib.sha256(Path(uncertainty.__file__).read_bytes()).hexdigest(), "numpy_version": numpy.__version__, "python_version": sys.version.split()[0]}))
"""
    reference = subprocess.run(
        [str(reference_python.absolute()), "-c", code],
        input=json.dumps(request["temperature_kev"]),
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
        cwd=root,
        env=os.environ | {"PYTHONPATH": str(root / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    local = json.loads(reference.stdout)
    rates = local.pop("rates")
    rows = []
    for density in cast(list[float], request["density_m3"]):
        for temperature, rate in zip(
            cast(list[float], request["temperature_kev"]), rates, strict=True
        ):
            try:
                rows.append(_point(request, density, temperature, rate))
            except (ArithmeticError, ValueError) as error:
                rows.append(
                    {
                        "density_m3": density,
                        "temperature_kev": temperature,
                        "status": "calculation_failed",
                        "reasons": [type(error).__name__],
                        "q_launched": None,
                        "values": {},
                    }
                )
    canonical = json.dumps(request, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    return {
        "schema": RESULT_SCHEMA,
        "source": provenance,
        "request": dict(request),
        "request_sha256": hashlib.sha256(canonical).hexdigest(),
        "local_reference": local,
        "points": rows,
        "actionable": False,
        "evidence_claimed": False,
        "federated": False,
        "scope": "uniform DT; ITER98y2; no impurity radiation or L-H accessibility assessment; code parity only",
    }


def main() -> None:
    """Write a finite JSON operating map from an explicit JSON request path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-python", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    args = parser.parse_args()
    result = run_operating_map(
        json.loads(args.request.read_text()),
        reference_python=args.reference_python,
        reference_root=args.reference_root,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
