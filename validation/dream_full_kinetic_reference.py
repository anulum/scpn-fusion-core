# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DREAM Full Kinetic Reference Adapter
"""Validate and adapt unprojected DREAM radius-momentum-pitch outputs."""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from scpn_fusion.core.runaway_kinetic_coefficients import (
    RunawayKineticCoefficients,
)
from scpn_fusion.core.runaway_kinetic_grid import FloatArray, RunawayKineticGrid
from scpn_fusion.core.runaway_kinetic_operator import (
    RunawayKineticGeometry,
    RunawayKineticOperator,
)


DREAM_COMMIT: Final[str] = "ecdd5e146537c77602c9d7cc76b36100200e4b9a"
ELECTRON_CHARGE_C: Final[float] = 1.60217662e-19
ELECTRON_MASS_KG: Final[float] = 9.10938356e-31
SPEED_OF_LIGHT_M_PER_S: Final[float] = 299792458.0
VACUUM_PERMITTIVITY_F_M: Final[float] = 8.85418782e-12

EXPECTED_FLAGS: Final[dict[str, int]] = {
    "settings/collisions/collfreq_mode": 2,
    "settings/collisions/collfreq_type": 3,
    "settings/collisions/bremsstrahlung_mode": 2,
    "settings/eqsys/n_re/avalanche": 4,
    "settings/eqsys/f_re/synchrotronmode": 2,
    "settings/eqsys/f_re/transport/type": 3,
}

REQUIRED_RUNAWAY_QUANTITIES: Final[tuple[str, ...]] = (
    "Ar",
    "Ap1",
    "Ap2",
    "Drr",
    "Dpp",
    "Dpx",
    "Dxp",
    "Dxx",
    "S_ava",
    "lnLambda_ee_f1",
    "lnLambda_ee_f2",
    "lnLambda_ei_f1",
    "lnLambda_ei_f2",
    "nu_D_f1",
    "nu_D_f2",
    "nu_s_f1",
    "nu_s_f2",
    "nu_par_f1",
    "nu_par_f2",
    "synchrotron_f1",
    "synchrotron_f2",
    "bremsstrahlung_f1",
)

REQUIRED_AUXILIARY_QUANTITIES: Final[tuple[str, ...]] = (
    "fluid/EDreic",
    "fluid/Eceff",
    "fluid/Ecfree",
    "fluid/Ectot",
    "fluid/GammaAva",
    "fluid/W_hot",
    "fluid/W_re",
    "fluid/Zeff",
    "fluid/conductivity",
    "fluid/gammaCompton",
    "fluid/gammaDreicer",
    "fluid/gammaTritium",
    "fluid/lnLambdaC",
    "fluid/lnLambdaT",
    "fluid/ni_negIonization",
    "fluid/ni_negRecombination",
    "fluid/ni_posIonization",
    "fluid/ni_posRecombination",
    "fluid/nusnuDatPStar",
    "fluid/pCrit",
    "fluid/pStar",
    "fluid/qR0",
    "fluid/runawayRate",
    "fluid/tIoniz",
    "fluid/tauEERel",
    "fluid/tauEETh",
    "scalar/E_mag",
    "scalar/L_i",
    "scalar/L_i_flux",
    "scalar/energyloss_f_re",
    "scalar/energyloss_T_cold",
    "scalar/l_i",
    "scalar/radialloss_f_re",
    "scalar/radialloss_n_re",
    "scalar/tIoniz",
)
ION_RATE_AUXILIARY_QUANTITIES: Final[frozenset[str]] = frozenset(
    {
        "fluid/ni_negIonization",
        "fluid/ni_negRecombination",
        "fluid/ni_posIonization",
        "fluid/ni_posRecombination",
    }
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a potentially large output without loading it twice."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _array(dataset: Any) -> FloatArray:
    return cast(FloatArray, np.asarray(dataset[()], dtype=np.float64))


def _text(dataset: Any) -> str:
    raw = np.asarray(dataset[()])
    return b"".join(raw.reshape(-1).tolist()).decode("utf-8")


@dataclass(frozen=True)
class DreamFullKineticOutput:
    """Authenticated full-dimensional DREAM state and operator history."""

    path: Path
    sha256: str
    commit: str
    times_s: FloatArray
    grid: RunawayKineticGrid
    geometry: RunawayKineticGeometry
    distribution: FloatArray
    density_m3: FloatArray
    current_density_a_m2: FloatArray
    total_electron_density_m3: FloatArray
    electric_field_v_m: FloatArray
    coefficients: dict[str, FloatArray]
    requested_quantities: tuple[str, ...]
    case_settings: dict[str, Any]
    auxiliary_diagnostics: dict[str, FloatArray]

    @classmethod
    def load(cls, path: Path) -> DreamFullKineticOutput:
        """Load an output and fail closed on any omitted axis or physics term."""

        h5py = importlib.import_module("h5py")
        resolved = path.resolve()
        with h5py.File(resolved, "r") as handle:
            commit = _text(handle["code/commit"])
            if commit != DREAM_COMMIT:
                raise ValueError(f"unexpected DREAM commit {commit!r}")
            for key, expected in EXPECTED_FLAGS.items():
                actual = int(_array(handle[key]).reshape(-1)[0])
                if actual != expected:
                    raise ValueError(f"{key}={actual}, expected {expected}")
            if not np.isinf(_array(handle["grid/R0"])[0]):
                raise ValueError("reference must retain upstream cylindrical geometry")

            runaway = handle["other/runaway"]
            missing = [name for name in REQUIRED_RUNAWAY_QUANTITIES if name not in runaway]
            if missing:
                raise ValueError(f"missing runaway diagnostics: {missing}")
            coefficients = {name: _array(runaway[name]) for name in REQUIRED_RUNAWAY_QUANTITIES}
            missing_auxiliary = [
                name for name in REQUIRED_AUXILIARY_QUANTITIES if f"other/{name}" not in handle
            ]
            if missing_auxiliary:
                raise ValueError(f"missing auxiliary diagnostics: {missing_auxiliary}")
            auxiliary: dict[str, FloatArray] = {}

            def collect_auxiliary(name: str, item: Any) -> None:
                if name.startswith("runaway/") or not hasattr(item, "shape"):
                    return
                auxiliary[name] = _array(item)

            handle["other"].visititems(collect_auxiliary)
            if not set(REQUIRED_AUXILIARY_QUANTITIES).issubset(auxiliary):
                raise ValueError("required auxiliary diagnostic collection is incomplete")
            # At this DREAM commit StoreEmpty() leaves the previous S_ava
            # buffer in place before SetVectorElements() adds the new source.
            # The saved series is therefore cumulative.  First differences
            # recover the instantaneous source used at each completed step.
            cumulative_avalanche = coefficients["S_ava"]
            coefficients["S_ava"] = np.concatenate(
                (
                    cumulative_avalanche[:1],
                    np.diff(cumulative_avalanche, axis=0),
                ),
                axis=0,
            )

            radius_faces = _array(handle["grid/r_f"])
            pitch_faces = _array(handle["grid/runaway/p2_f"])
            momentum_faces = _array(handle["grid/runaway/p1_f"])
            grid = RunawayKineticGrid(
                radius_faces_m=radius_faces,
                pitch_faces=pitch_faces,
                momentum_faces_mc=momentum_faces,
            )
            if grid.nr <= 1 or grid.nxi <= 1 or grid.np <= 1:
                raise ValueError("reference does not evolve all radius-momentum-pitch axes")

            dr = _array(handle["grid/dr"])
            dp = _array(handle["grid/runaway/dp1"])
            dxi = _array(handle["grid/runaway/dp2"])
            vprime = _array(handle["grid/runaway/Vprime"])
            vpvol = _array(handle["grid/VpVol"])
            vpvol_f = _array(handle["grid/VpVol_f"])
            momentum = grid.momentum_mc
            momentum_face = grid.momentum_faces_mc

            cell_measure = vprime * dr[:, None, None] * dxi[None, :, None] * dp[None, None, :]
            density_measure = vprime / vpvol[:, None, None] * dxi[None, :, None] * dp[None, None, :]
            radial_face_measure = (
                vpvol_f[:, None, None]
                * (2.0 * np.pi * momentum**2)[None, None, :]
                * dxi[None, :, None]
                * dp[None, None, :]
            )
            momentum_face_measure = (
                vpvol[:, None, None]
                * dr[:, None, None]
                * dxi[None, :, None]
                * (2.0 * np.pi * momentum_face**2)[None, None, :]
            )
            pitch_face_measure = (
                vpvol[:, None, None]
                * dr[:, None, None]
                * np.ones((1, grid.nxi + 1, 1), dtype=np.float64)
                * (2.0 * np.pi * momentum**2 * dp)[None, None, :]
            )
            geometry = RunawayKineticGeometry.checked(
                grid,
                cell_measure=cell_measure,
                density_cell_measure=density_measure,
                radial_face_measure=radial_face_measure,
                momentum_face_measure=momentum_face_measure,
                pitch_face_measure=pitch_face_measure,
            )

            times = _array(handle["grid/t"])
            distribution = _array(handle["eqsys/f_re"])
            density = _array(handle["eqsys/n_re"])
            current = _array(handle["eqsys/j_re"])
            total_density = _array(handle["eqsys/n_tot"])
            electric_field = _array(handle["eqsys/E_field"])
            requested = tuple(_text(handle["settings/other/include"]).split(";"))
            prescribed_ions = _array(handle["settings/eqsys/n_i/prescribed/x"])
            case_settings = {
                "magnetic_field_t": float(_array(handle["settings/radialgrid/B0"])[0]),
                "cold_temperature_ev": float(
                    _array(handle["settings/eqsys/T_cold/data/x"]).reshape(-1)[0]
                ),
                "magnetic_perturbation": float(
                    _array(handle["settings/eqsys/f_re/transport/dBB/x"]).reshape(-1)[0]
                ),
                "runaway_momentum_cutoff_mc": float(momentum_faces[0]),
                "ion_atomic_numbers": _array(handle["settings/eqsys/n_i/Z"]).tolist(),
                "prescribed_ion_charge_state_density_m3": prescribed_ions.reshape(-1).tolist(),
            }

        result = cls(
            path=resolved,
            sha256=sha256_file(resolved),
            commit=commit,
            times_s=times,
            grid=grid,
            geometry=geometry,
            distribution=distribution,
            density_m3=density,
            current_density_a_m2=current,
            total_electron_density_m3=total_density,
            electric_field_v_m=electric_field,
            coefficients=coefficients,
            requested_quantities=requested,
            case_settings=case_settings,
            auxiliary_diagnostics=auxiliary,
        )
        result.validate_complete()
        return result

    def validate_complete(self) -> None:
        """Validate shapes, finiteness and non-trivial required physics."""

        nt = self.times_s.size - 1
        if nt <= 0 or self.distribution.shape != (nt + 1, *self.grid.shape):
            raise ValueError("distribution history is incomplete")
        if self.density_m3.shape != (nt + 1, self.grid.nr):
            raise ValueError("runaway density history is incomplete")
        for name, values in (
            ("current_density_a_m2", self.current_density_a_m2),
            ("total_electron_density_m3", self.total_electron_density_m3),
            ("electric_field_v_m", self.electric_field_v_m),
        ):
            if values.shape != (nt + 1, self.grid.nr):
                raise ValueError(f"{name} history is incomplete")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains non-finite data")
        if not np.all(np.isfinite(self.density_m3)) or np.any(self.density_m3 < 0.0):
            raise ValueError("runaway density is non-finite or negative")
        if np.any(self.total_electron_density_m3 < 0.0):
            raise ValueError("total electron density is negative")
        if self.times_s[0] != 0.0 or np.any(np.diff(self.times_s) <= 0.0):
            raise ValueError("time grid is not a completed increasing trajectory")
        coefficient_shapes = {
            "Ar": (nt, self.grid.nr + 1, self.grid.nxi, self.grid.np),
            "Drr": (nt, self.grid.nr + 1, self.grid.nxi, self.grid.np),
            "S_ava": (nt, self.grid.nr, self.grid.nxi, self.grid.np),
        }
        coefficient_shapes.update(
            {
                name: (nt, self.grid.nr, self.grid.nxi, self.grid.np + 1)
                for name in (
                    "Ap1",
                    "Dpp",
                    "Dpx",
                    "lnLambda_ee_f1",
                    "lnLambda_ei_f1",
                    "nu_D_f1",
                    "nu_s_f1",
                    "nu_par_f1",
                    "synchrotron_f1",
                    "bremsstrahlung_f1",
                )
            }
        )
        coefficient_shapes.update(
            {
                name: (nt, self.grid.nr, self.grid.nxi + 1, self.grid.np)
                for name in (
                    "Ap2",
                    "Dxp",
                    "Dxx",
                    "lnLambda_ee_f2",
                    "lnLambda_ei_f2",
                    "nu_D_f2",
                    "nu_s_f2",
                    "nu_par_f2",
                    "synchrotron_f2",
                )
            }
        )
        for name, values in self.coefficients.items():
            coefficient_expected_shape = coefficient_shapes[name]
            if values.shape != coefficient_expected_shape:
                raise ValueError(
                    f"{name} has shape {values.shape}, expected {coefficient_expected_shape}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains non-finite data")
        for name, values in self.auxiliary_diagnostics.items():
            auxiliary_expected_shape: tuple[int, ...]
            if name in ION_RATE_AUXILIARY_QUANTITIES:
                auxiliary_expected_shape = (nt, 21, self.grid.nr)
            elif name.startswith("fluid/"):
                auxiliary_expected_shape = (nt, self.grid.nr)
            elif name.startswith("scalar/"):
                auxiliary_expected_shape = (nt, 1)
            else:
                raise ValueError(f"unexpected auxiliary diagnostic group for {name}")
            if values.shape != auxiliary_expected_shape:
                raise ValueError(
                    f"{name} has shape {values.shape}, expected {auxiliary_expected_shape}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains non-finite data")
        for name in (
            "Drr",
            "Dpp",
            "Dxx",
            "S_ava",
            "synchrotron_f1",
            "synchrotron_f2",
            "bremsstrahlung_f1",
        ):
            if not np.any(self.coefficients[name] != 0.0):
                raise ValueError(f"required physics diagnostic {name} is identically zero")
        if not np.all(np.isfinite(self.distribution)) or np.any(self.distribution < 0.0):
            raise ValueError("distribution is non-finite or negative")

    def density_moment(self) -> FloatArray:
        """Integrate the unprojected distribution with DREAM's own metric."""

        return cast(
            FloatArray,
            np.sum(
                self.distribution * self.geometry.density_cell_measure[None, :, :, :],
                axis=(2, 3),
            ),
        )

    def density_moment_relative_error(self) -> float:
        """Maximum relative finite-grid fraction of DREAM's total ``n_re``."""

        moment = self.density_moment()
        denominator = np.maximum(np.abs(self.density_m3), 1.0)
        return float(np.max(np.abs(moment - self.density_m3) / denominator))

    def current_moment(self) -> FloatArray:
        """Integrate the parallel-current moment with DREAM's phase-space metric."""

        momentum = self.grid.momentum_mc[None, :]
        pitch = self.grid.pitch[:, None]
        gamma = np.sqrt(1.0 + momentum * momentum)
        parallel_speed = SPEED_OF_LIGHT_M_PER_S * momentum * pitch / gamma
        return cast(
            FloatArray,
            ELECTRON_CHARGE_C
            * np.sum(
                self.distribution
                * self.geometry.density_cell_measure[None, :, :, :]
                * parallel_speed[None, None, :, :],
                axis=(2, 3),
            ),
        )

    def native_operator(self, step: int) -> RunawayKineticOperator:
        """Build the public native operator from one complete DREAM coefficient set."""

        nt = self.times_s.size - 1
        if step < 0 or step >= nt:
            raise IndexError(f"step must be in [0, {nt})")
        c = self.coefficients
        electric = self.electric_field_v_m[step + 1]
        prefactor = ELECTRON_CHARGE_C * electric / (ELECTRON_MASS_KG * SPEED_OF_LIGHT_M_PER_S)
        xi = self.grid.pitch
        xi_f = self.grid.pitch_faces
        momentum = self.grid.momentum_mc
        electric_p = np.broadcast_to(
            prefactor[:, None, None] * xi[None, :, None],
            (self.grid.nr, self.grid.nxi, self.grid.np + 1),
        ).copy()
        electric_xi = np.broadcast_to(
            prefactor[:, None, None] * (1.0 - xi_f**2)[None, :, None] / momentum[None, None, :],
            (self.grid.nr, self.grid.nxi + 1, self.grid.np),
        ).copy()
        synch_p = c["synchrotron_f1"][step]
        synch_xi = c["synchrotron_f2"][step]
        brems = c["bremsstrahlung_f1"][step]
        collision_p = c["Ap1"][step] - electric_p - synch_p - brems

        # The de-accumulated source at index ``step`` is evaluated with the
        # interval-end n_re state; dividing by that same state recovers the
        # time-independent Rosenbluth-Putvinski kernel to roundoff.
        n_re = self.density_m3[step + 1]
        n_tot = self.total_electron_density_m3[step + 1]
        source_denominator = np.maximum(
            n_re[:, None, None] * n_tot[:, None, None],
            np.finfo(np.float64).tiny,
        )
        avalanche_kernel = -c["S_ava"][step] / source_denominator
        momentum_cutoff_mc = float(self.grid.momentum_faces_mc[0])
        if momentum_cutoff_mc <= 0.0:
            raise ValueError(
                "DREAM total-density avalanche source requires a positive runaway momentum cutoff"
            )
        # DREAM ecdd5e1, AvalancheSourceRP::
        # EvaluateNormalizedTotalKnockOnNumber(pCutoff, infinity).
        epsmc = 4.0 * np.pi * VACUUM_PERMITTIVITY_F_M * ELECTRON_MASS_KG * SPEED_OF_LIGHT_M_PER_S
        avalanche_prefactor_m3_s = ELECTRON_CHARGE_C**4 / (epsmc**2 * SPEED_OF_LIGHT_M_PER_S)
        cutoff_squared = momentum_cutoff_mc**2
        cutoff_gamma = np.sqrt(1.0 + cutoff_squared)
        normalized_total_knock_on_m3_s = (
            2.0 * np.pi * avalanche_prefactor_m3_s * (cutoff_gamma + 1.0) / cutoff_squared
        )
        density_avalanche_rate_s_inv = normalized_total_knock_on_m3_s * n_tot
        coefficients = RunawayKineticCoefficients.checked(
            self.grid,
            radial_advection=c["Ar"][step],
            momentum_electric_advection=electric_p,
            momentum_collision_advection=collision_p,
            momentum_synchrotron_advection=synch_p,
            momentum_bremsstrahlung_advection=brems,
            pitch_electric_advection=electric_xi,
            pitch_synchrotron_advection=synch_xi,
            radial_diffusion=c["Drr"][step],
            momentum_diffusion=c["Dpp"][step],
            pitch_diffusion=c["Dxx"][step],
            momentum_pitch_diffusion=c["Dpx"][step],
            pitch_momentum_diffusion=c["Dxp"][step],
            avalanche_source_kernel=avalanche_kernel,
            total_electron_density_m3=n_tot,
            total_density_avalanche_rate_s_inv=density_avalanche_rate_s_inv,
            total_density_external_source_m3_s=np.zeros(self.grid.nr, dtype=np.float64),
            external_source=np.zeros(self.grid.shape, dtype=np.float64),
        )
        return RunawayKineticOperator(
            self.grid,
            coefficients,
            geometry=self.geometry,
        )

    def summary(self) -> dict[str, Any]:
        """Return a compact custody and completeness summary."""

        return {
            "artifact_filename": self.path.name,
            "sha256": self.sha256,
            "dream_commit": self.commit,
            "grid": {
                "nr": self.grid.nr,
                "nxi": self.grid.nxi,
                "np": self.grid.np,
                "nt": self.times_s.size - 1,
            },
            "final_time_s": float(self.times_s[-1]),
            "finite_grid_density_fraction_final": (
                self.density_moment()[-1] / self.density_m3[-1]
            ).tolist(),
            "finite_grid_vs_total_density_max_relative_gap": (self.density_moment_relative_error()),
            "minimum_distribution": float(np.min(self.distribution)),
            "runaway_density_growth_ratio": float(
                np.sum(self.density_m3[-1]) / np.sum(self.density_m3[0])
            ),
            "operator_nonzero": {
                name: bool(np.any(values != 0.0)) for name, values in self.coefficients.items()
            },
            "operator_shapes": {
                name: list(values.shape) for name, values in self.coefficients.items()
            },
            "requested_quantities": list(self.requested_quantities),
            "case_settings": self.case_settings,
            "auxiliary_diagnostics": {
                name: {
                    "shape": list(values.shape),
                    "nonzero": bool(np.any(values != 0.0)),
                }
                for name, values in self.auxiliary_diagnostics.items()
            },
        }


__all__ = ["DreamFullKineticOutput", "sha256_file"]
