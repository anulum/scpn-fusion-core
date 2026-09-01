# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Full Kinetic Radial DREAM Parity Deck
"""Generate the frozen full-kinetic DREAM parity settings.

This deck deliberately evolves the runaway distribution on all three physical
coordinates: radius, momentum and pitch.  It enables the collision, source,
radiation and radial-transport terms that a full-fidelity parity claim must
exercise.  The three named resolutions form one convergence family; they do
not select different physics.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np


DECK_SCHEMA: Final[str] = "scpn-fusion.dream-full-kinetic-radial-deck.v1"
DREAM_COMMIT: Final[str] = "ecdd5e146537c77602c9d7cc76b36100200e4b9a"


@dataclass(frozen=True)
class Resolution:
    """Numerical resolution for one member of the convergence family."""

    nr: int
    np: int
    nxi: int
    nt: int


RESOLUTIONS: Final[dict[str, Resolution]] = {
    "coarse": Resolution(nr=4, np=40, nxi=16, nt=8),
    "medium": Resolution(nr=6, np=60, nxi=24, nt=12),
    "fine": Resolution(nr=8, np=80, nxi=32, nt=16),
    "ultrafine": Resolution(nr=10, np=100, nxi=40, nt=20),
    "veryfine": Resolution(nr=12, np=120, nxi=48, nt=24),
    "superfine": Resolution(nr=14, np=140, nxi=56, nt=28),
}


@dataclass(frozen=True)
class PhysicalCase:
    """Frozen physical parameters shared by DREAM and the native runtime."""

    electric_field_v_per_m: float = 1.0
    cold_temperature_ev: float = 1.0e3
    free_electron_density_m3: float = 5.0e19
    seed_runaway_density_m3: float = 1.0e14
    argon_density_m3: float = 1.0e17
    argon_charge_state: int = 1
    magnetic_field_t: float = 5.0
    minor_radius_m: float = 0.22
    wall_radius_m: float = 0.22
    magnetic_perturbation: float = 1.0e-3
    p_min_mc: float = 2.0e-2
    p_max_mc: float = 8.0
    avalanche_cutoff_mc: float = 2.0e-2
    simulation_time_s: float = 2.0e-4


CASE: Final[PhysicalCase] = PhysicalCase()
REQUESTED_OTHER_QUANTITIES: Final[tuple[str, ...]] = (
    "fluid",
    "energy",
    "runaway/Ar",
    "runaway/Ap1",
    "runaway/Ap2",
    "runaway/Drr",
    "runaway/Dpp",
    "runaway/Dpx",
    "runaway/Dxp",
    "runaway/Dxx",
    "runaway/lnLambda_ee_f1",
    "runaway/lnLambda_ee_f2",
    "runaway/lnLambda_ei_f1",
    "runaway/lnLambda_ei_f2",
    "runaway/nu_D_f1",
    "runaway/nu_D_f2",
    "runaway/nu_s_f1",
    "runaway/nu_s_f2",
    "runaway/nu_par_f1",
    "runaway/nu_par_f2",
    "runaway/S_ava",
    "runaway/synchrotron_f1",
    "runaway/synchrotron_f2",
    "runaway/bremsstrahlung_f1",
    "lnLambda",
    "nu_s",
    "nu_D",
    "scalar",
    "transport",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_dream_api(dream_root: Path) -> tuple[Any, ...]:
    python_root = dream_root / "py"
    if not python_root.is_dir():
        raise FileNotFoundError(f"DREAM Python API not found at {python_root}")
    sys.path.insert(0, str(python_root))

    DreamSettings = importlib.import_module("DREAM.DREAMSettings").DREAMSettings
    Collisions = importlib.import_module("DREAM.Settings.CollisionHandler")
    Distribution = importlib.import_module("DREAM.Settings.Equations.DistributionFunction")
    Ions = importlib.import_module("DREAM.Settings.Equations.IonSpecies")
    Runaways = importlib.import_module("DREAM.Settings.Equations.RunawayElectrons")
    Solver = importlib.import_module("DREAM.Settings.Solver")
    Transport = importlib.import_module("DREAM.Settings.TransportSettings")

    return DreamSettings, Collisions, Distribution, Ions, Runaways, Solver, Transport


def build_settings(*, dream_root: Path, resolution: str, output: Path) -> Any:
    """Build one frozen DREAM settings object.

    Parameters
    ----------
    dream_root:
        Checkout of DREAM pinned to :data:`DREAM_COMMIT`.
    resolution:
        One of the named entries in :data:`RESOLUTIONS`.
    output:
        HDF5 output path written by DREAM when the settings are executed.
    """

    if resolution not in RESOLUTIONS:
        raise ValueError(f"unknown resolution {resolution!r}")
    grid = RESOLUTIONS[resolution]
    (
        DreamSettings,
        Collisions,
        Distribution,
        Ions,
        Runaways,
        Solver,
        Transport,
    ) = _load_dream_api(dream_root)

    settings = DreamSettings()
    case = CASE

    settings.collisions.collfreq_mode = Collisions.COLLFREQ_MODE_FULL
    settings.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
    settings.collisions.bremsstrahlung_mode = Collisions.BREMSSTRAHLUNG_MODE_STOPPING_POWER
    settings.collisions.lnlambda = Collisions.LNLAMBDA_ENERGY_DEPENDENT
    settings.collisions.pstar_mode = Collisions.PSTAR_MODE_COLLISIONLESS

    settings.eqsys.E_field.setPrescribedData(case.electric_field_v_per_m)
    settings.eqsys.T_cold.setPrescribedData(case.cold_temperature_ev)

    # Preserve the upstream 2kinetic free-electron density while adding a
    # trace partially ionized species that makes partial screening observable.
    deuterium_density = (
        case.free_electron_density_m3 - case.argon_charge_state * case.argon_density_m3
    )
    settings.eqsys.n_i.addIon(
        name="D",
        Z=1,
        iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED,
        n=deuterium_density,
    )
    settings.eqsys.n_i.addIon(
        name="Ar",
        Z=18,
        Z0=case.argon_charge_state,
        iontype=Ions.IONS_PRESCRIBED,
        n=case.argon_density_m3,
    )

    settings.hottailgrid.setEnabled(False)
    settings.runawaygrid.setNxi(grid.nxi)
    settings.runawaygrid.setNp(grid.np)
    settings.runawaygrid.setPmin(case.p_min_mc)
    settings.runawaygrid.setPmax(case.p_max_mc)

    settings.radialgrid.setB0(case.magnetic_field_t)
    settings.radialgrid.setMinorRadius(case.minor_radius_m)
    settings.radialgrid.setWallRadius(case.wall_radius_m)
    settings.radialgrid.setNr(grid.nr)

    radii = (np.arange(grid.nr, dtype=float) + 0.5) * case.minor_radius_m / grid.nr
    seed_profile = case.seed_runaway_density_m3 * (
        0.15 + 0.85 * np.exp(-((radii / (0.65 * case.minor_radius_m)) ** 2))
    )
    electric_profile = np.full(grid.nr, case.electric_field_v_per_m)
    settings.eqsys.n_re.setInitialProfile(seed_profile, radius=radii)
    settings.eqsys.f_re.setInitialAvalancheDistribution(
        E=electric_profile,
        r=radii,
    )
    settings.eqsys.n_re.setAvalanche(
        Runaways.AVALANCHE_MODE_KINETIC,
        pCutAvalanche=case.avalanche_cutoff_mc,
    )
    settings.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_DISABLED)
    settings.eqsys.f_re.setSynchrotronMode(Distribution.SYNCHROTRON_MODE_INCLUDE)
    settings.eqsys.f_re.setBoundaryCondition(Distribution.BC_F_0)
    settings.eqsys.f_re.setAdvectionInterpolationMethod(
        ad_int=Distribution.AD_INTERP_UPWIND,
        ad_jac=Distribution.AD_INTERP_JACOBIAN_FULL,
    )
    settings.eqsys.f_re.transport.setMagneticPerturbation(case.magnetic_perturbation)
    settings.eqsys.f_re.transport.setBoundaryCondition(Transport.BC_F_0)

    settings.solver.setType(Solver.NONLINEAR)
    settings.solver.setLinearSolver(Solver.LINEAR_SOLVER_LU)
    settings.solver.setMaxIterations(100)
    settings.solver.setVerbose(False)
    settings.solver.tolerance.set(reltol=2.0e-8)
    settings.solver.tolerance.set("f_re", reltol=2.0e-8, abstol=1.0e3)
    settings.solver.tolerance.set("n_re", reltol=2.0e-8, abstol=1.0e3)
    settings.solver.tolerance.set("j_re", reltol=2.0e-8, abstol=1.0e-6)

    # Request every quantity applicable to the enabled runaway grid. DREAM's
    # literal ``all`` also registers disabled hot-tail-grid quantities and
    # dereferences that absent grid at this pinned commit.
    settings.other.include(*REQUESTED_OTHER_QUANTITIES)
    settings.timestep.setTmax(case.simulation_time_s)
    settings.timestep.setNt(grid.nt)
    settings.output.setTiming(stdout=False, file=True)
    settings.output.setFilename(str(output.resolve()))
    return settings


def deck_manifest(*, resolution: str, settings_path: Path, output_path: Path) -> dict[str, object]:
    """Return the canonical, JSON-serializable deck manifest."""

    return {
        "schema": DECK_SCHEMA,
        "dream_commit": DREAM_COMMIT,
        "resolution_name": resolution,
        "resolution": asdict(RESOLUTIONS[resolution]),
        "case": asdict(CASE),
        "physics": {
            "coordinates": ["radius", "momentum", "pitch"],
            "avalanche": "kinetic Rosenbluth-Putvinski",
            "collision_frequency": "full partially screened",
            "bremsstrahlung": "stopping-power radiation reaction",
            "synchrotron": "kinetic radiation-reaction advection",
            "radial_transport": "kinetic Rechester-Rosenbluth diffusion",
            "radial_geometry": "upstream 2kinetic cylindrical geometry",
            "requested_other_quantities": list(REQUESTED_OTHER_QUANTITIES),
        },
        "settings_path": str(settings_path.resolve()),
        "output_path": str(output_path.resolve()),
    }


def main() -> int:
    """Generate one settings file and its sidecar manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", choices=tuple(RESOLUTIONS), required=True)
    parser.add_argument(
        "--dream-root",
        type=Path,
        default=_repo_root() / "data/external/full_fidelity_public_sources/repos/dream",
    )
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    settings = build_settings(
        dream_root=args.dream_root,
        resolution=args.resolution,
        output=args.output,
    )
    args.settings.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    settings.save(str(args.settings))
    args.manifest.write_text(
        json.dumps(
            deck_manifest(
                resolution=args.resolution,
                settings_path=args.settings,
                output_path=args.output,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
