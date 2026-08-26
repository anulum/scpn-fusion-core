# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TGLF External Gyrokinetic Solver
"""
TGLF (Trapped Gyro-Landau Fluid) external solver interface.

Generates current GACODE TGLF key-value decks, executes ``tglf`` via
subprocess, and parses growth-rate / flux output files. Missing or failed
external TGLF execution is a hard error so zero-flux placeholders cannot enter
production transport validation.

Reference: Staebler et al., Phys. Plasmas 14 (2007) 055909.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import subprocess
import tempfile
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core._tglf_interface_runtime import (
    _parse_gacode_tglf_output,
    _parse_gacode_tglf_spectrum,
    _resolve_tglf_command,
    render_tglf_input,
)
from scpn_fusion.core._tglf_interface_types import TGLFInputDeck, TGLFSpecies
from scpn_fusion.core.gk_interface import GKLocalParams, GKOutput, GKSolverBase, GKSpeciesFlux
from scpn_fusion.io.safe_loaders import checked_json_load

FloatArray: TypeAlias = NDArray[np.float64]

_DECK_METADATA_NAME = "scpn_fusion_tglf_deck.json"


def _deck_from_params(params: GKLocalParams) -> TGLFInputDeck:
    """Map the public GK contract to the shared current-GACODE deck."""
    if params.requires_nonlinear_solver:
        raise ValueError("TGLF is quasilinear and cannot satisfy a nonlinear GK request.")
    return TGLFInputDeck(
        rho=params.rho,
        s_hat=params.s_hat,
        q=params.q,
        alpha_mhd=params.alpha_MHD,
        R_LTi=params.R_L_Ti,
        R_LTe=params.R_L_Te,
        R_Lne=params.R_L_ne,
        R_Lni=params.R_L_ne,
        beta_e=params.beta_e,
        Z_eff=params.Z_eff,
        xnue=params.nu_star,
        T_e_keV=params.T_e_keV,
        T_i_keV=params.T_i_keV,
        n_e_19=params.n_e,
        R_major=params.R0,
        a_minor=params.a,
        B_toroidal=params.B0,
        kappa=params.kappa,
        delta=params.delta,
        use_bper=params.is_electromagnetic,
    )


def generate_tglf_input(params: GKLocalParams) -> str:
    """Render a current GACODE key-value input deck from local parameters."""
    return render_tglf_input(_deck_from_params(params))


def _load_deck_metadata(run_dir: Path) -> TGLFInputDeck:
    """Load the SCPN dimensional metadata required for physical scaling."""
    payload = checked_json_load(run_dir / _DECK_METADATA_NAME)
    if not isinstance(payload, dict):
        raise RuntimeError("TGLF deck metadata must be a JSON object.")
    try:
        raw_species = payload.get("species", ())
        if raw_species:
            if not isinstance(raw_species, list) or not all(
                isinstance(item, dict) for item in raw_species
            ):
                raise TypeError("species must be a list of objects")
            payload["species"] = tuple(TGLFSpecies(**item) for item in raw_species)
        return TGLFInputDeck(**payload)
    except TypeError as exc:
        raise RuntimeError(f"Invalid TGLF deck metadata: {exc}") from exc


def parse_tglf_output(run_dir: Path) -> GKOutput:
    """Parse real current-GACODE outputs with preserved dimensional metadata."""
    deck = _load_deck_metadata(run_dir)
    output = _parse_gacode_tglf_output(run_dir, deck)
    gamma, omega_r, k_y = _parse_gacode_tglf_spectrum(run_dir)
    dominant = _classify_dominant_mode(gamma, omega_r, k_y)
    return GKOutput(
        chi_i=output.chi_i,
        chi_e=output.chi_e,
        D_e=output.d_e,
        D_i=output.d_i,
        particle_flux_e_gb=output.particle_e,
        particle_flux_i_gb=output.particle_i,
        heat_flux_e_gb=output.q_e,
        heat_flux_i_gb=output.q_i,
        species_fluxes_gb=tuple(
            GKSpeciesFlux(
                species_index=item.species_index,
                name=item.name,
                charge_e=item.charge_e,
                particle_gb=item.particle_gb,
                energy_gb=item.energy_gb,
                momentum_gb=item.momentum_gb,
                exchange_gb=item.exchange_gb,
            )
            for item in output.species_fluxes
        ),
        gamma=gamma,
        omega_r=omega_r,
        k_y=k_y,
        dominant_mode=dominant,
        converged=True,
    )


def _classify_dominant_mode(
    gamma: FloatArray,
    omega_r: FloatArray,
    k_y: FloatArray | None = None,
) -> str:
    """Identify the dominant instability from sign and perpendicular scale.

    Electron-scale modes (``k_y rho_s >= 1``) are labelled ETG before using
    propagation direction to distinguish ion-scale ITG and TEM modes.
    """
    if len(gamma) == 0 or np.all(gamma <= 0):
        return "stable"
    idx = int(np.argmax(gamma))
    if k_y is not None and len(k_y) == len(gamma) and k_y[idx] >= 1.0:
        return "ETG"
    if omega_r[idx] < 0:
        return "ITG"  # ion diamagnetic direction
    return "TEM"  # electron diamagnetic direction


class TGLFSolver(GKSolverBase):
    """TGLF external solver via GACODE ``tglf`` binary.

    Parameters
    ----------
    binary : str
        Command name of the ``tglf`` executable, resolved exclusively via PATH.
    work_dir : Path or None
        Persistent working directory.  If None, uses a tempdir per call.
    """

    def __init__(
        self,
        binary: str = "tglf",
        work_dir: Path | None = None,
    ) -> None:
        self.binary = binary
        self.work_dir = work_dir

    def is_available(self) -> bool:
        """Return whether the configured TGLF executable is on ``PATH``."""
        try:
            _resolve_tglf_command(self.binary)
        except (FileNotFoundError, ValueError):
            return False
        return True

    def prepare_input(self, params: GKLocalParams) -> Path:
        """Create a TGLF run directory containing ``input.tglf``."""
        return self.prepare_deck(_deck_from_params(params))

    def prepare_deck(self, deck: TGLFInputDeck) -> Path:
        """Create a run directory from an explicit ordered-species deck."""
        base = self.work_dir or Path(tempfile.mkdtemp(prefix="tglf_"))
        base.mkdir(parents=True, exist_ok=True)
        input_file = base / "input.tglf"
        input_file.write_text(render_tglf_input(deck), encoding="utf-8")
        (base / _DECK_METADATA_NAME).write_text(
            json.dumps(asdict(deck), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return base

    def run(self, input_path: Path, *, timeout_s: float = 30.0) -> GKOutput:
        """Execute TGLF for a prepared input directory and parse transport output."""
        try:
            resolved = _resolve_tglf_command(self.binary)
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(f"TGLF command unavailable through PATH: {self.binary!r}.") from exc

        try:
            subprocess.run(
                [resolved, "-e", "."],
                cwd=str(input_path),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=True,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise RuntimeError(f"TGLF execution failed: {exc}") from exc

        return parse_tglf_output(input_path)
