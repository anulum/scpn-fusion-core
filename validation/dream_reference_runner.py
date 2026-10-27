#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DREAM Reference Runner
"""Run a DREAM fluid runaway same-case and export reference time series.

Executes inside the dedicated DREAM pixi environment
(``external/dream-env``), with the DREAM python interface on ``PYTHONPATH``
(``external/DREAM/py``) and the compiled ``dreami`` binary resolved by
DREAM's own ``runiface`` (repo-relative). NOT importable in the main venv.

The scenario is a homogeneous cylindrical plasma with prescribed constant
electric field, density, and temperature — DREAM's fluid mode with the
Connor-Hastie Dreicer rate and the Rosenbluth-Putvinski fluid avalanche,
i.e. the SAME closed-form rate models implemented by
``scpn_fusion.core.runaway_electrons``. Collisions are completely screened
(the plasma is fully ionised deuterium) so the analytic formulas apply
without partial-screening corrections. The exported JSON carries full
provenance (DREAM git SHA, settings-file SHA-256, runtime metadata) so the
comparison lane in the main venv stays deterministic against the committed
artifact.

DREAM is open source (https://github.com/chalmersplasmatheory/DREAM, see
doi:10.1016/j.cpc.2021.108098); running it and redistributing derived
outputs with attribution is licence-compatible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DREAM_ROOT = REPO_ROOT / "external" / "DREAM"

# Scenario constants (fluid same-case; mirrored by the comparison gate)
E_FIELD_V_M = 6.0
N_E_M3 = 5.0e19
T_E_EV = 100.0
B0_T = 5.0
MINOR_RADIUS_M = 0.22
T_MAX_S = 1.0e-3
N_T = 100
N_RE_SEED_M3 = 1.0e10  # small seed so the avalanche term is exercised


def _dream_git_sha() -> str:
    """Return the checked-out DREAM commit SHA (provenance)."""
    return subprocess.run(
        ["git", "-C", str(DREAM_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    """Run the DREAM fluid same-case and write the reference artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(DREAM_ROOT / "py"))

    import DREAM.Settings.CollisionHandler as Collisions  # type: ignore[import-not-found]
    import DREAM.Settings.Equations.IonSpecies as Ions  # type: ignore[import-not-found]
    import DREAM.Settings.Equations.RunawayElectrons as Runaways  # type: ignore[import-not-found]
    import DREAM.Settings.Solver as Solver  # type: ignore[import-not-found]
    from DREAM import DREAMSettings, runiface

    ds = DREAMSettings()
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_COMPLETELY_SCREENED

    ds.eqsys.E_field.setPrescribedData(E_FIELD_V_M)
    ds.eqsys.T_cold.setPrescribedData(T_E_EV)
    ds.eqsys.n_i.addIon(name="D", Z=1, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=N_E_M3)

    # Fluid runaway generation with the same closed-form models as the
    # scpn_fusion.core.runaway_electrons lane.
    ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_CONNOR_HASTIE)
    ds.eqsys.n_re.setAvalanche(Runaways.AVALANCHE_MODE_FLUID)
    ds.eqsys.n_re.setInitialProfile(N_RE_SEED_M3)

    # Fluid-only: no kinetic grids.
    ds.hottailgrid.setEnabled(False)
    ds.runawaygrid.setEnabled(False)

    ds.radialgrid.setB0(B0_T)
    ds.radialgrid.setMinorRadius(MINOR_RADIUS_M)
    ds.radialgrid.setWallRadius(MINOR_RADIUS_M)
    ds.radialgrid.setNr(1)

    ds.solver.setType(Solver.LINEAR_IMPLICIT)
    ds.other.include("fluid")

    ds.timestep.setTmax(T_MAX_S)
    ds.timestep.setNt(N_T)

    with tempfile.TemporaryDirectory(prefix="dream_run_") as tmp:
        settings_path = Path(tmp) / "dream_settings.h5"
        output_path = Path(tmp) / "output.h5"
        ds.save(str(settings_path))
        settings_sha256 = hashlib.sha256(settings_path.read_bytes()).hexdigest()
        do = runiface(ds, str(output_path), quiet=True)

        time_s = [float(v) for v in do.grid.t[:]]
        n_re = [float(v) for v in do.eqsys.n_re[:].ravel()]
        other_fluid: dict[str, list[float]] = {}
        for name in ("gammaDreicer", "GammaAva", "runawayRate"):
            quantity = getattr(do.other.fluid, name, None)
            if quantity is not None:
                other_fluid[name] = [float(v) for v in quantity.data[:].ravel()]

    payload: dict[str, Any] = {
        "schema": "scpn-fusion-core.dream-reference-runaway.v1",
        "provenance": {
            "code": "DREAM",
            "code_url": "https://github.com/chalmersplasmatheory/DREAM",
            "paper_doi": "10.1016/j.cpc.2021.108098",
            "dream_git_sha": _dream_git_sha(),
            "settings_sha256": settings_sha256,
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "scenario": {
            "description": (
                "homogeneous cylindrical plasma, prescribed constant E/n/T, "
                "fluid Dreicer (Connor-Hastie) + fluid avalanche "
                "(Rosenbluth-Putvinski), completely screened collisions, "
                "fully ionised deuterium"
            ),
            "E_field_V_m": E_FIELD_V_M,
            "n_e_m3": N_E_M3,
            "T_e_eV": T_E_EV,
            "Z_eff": 1.0,
            "B0_T": B0_T,
            "minor_radius_m": MINOR_RADIUS_M,
            "t_max_s": T_MAX_S,
            "n_t": N_T,
            "n_re_seed_m3": N_RE_SEED_M3,
        },
        "series": {
            "time_s": time_s,
            "n_re_m3": n_re,
            "other_fluid": other_fluid,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
