# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TGLF Interface Runtime
"""Runtime subprocess helpers for the public TGLF interface."""

from __future__ import annotations

import logging
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import fields
from pathlib import Path
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core._tglf_interface_types import (
    TGLFInputDeck,
    TGLFOutput,
    TGLFParticleTransportIdentification,
    TGLFSpecies,
    TGLFSpeciesFlux,
    _TGLF_MAX_RETRIES_LIMIT,
    _TGLF_RETRY_BACKOFF_SECONDS,
)

logger = logging.getLogger(__name__)

_DEUTERIUM_MASS_KG = 2.0 * 1.67262192369e-27
_ELEMENTARY_CHARGE_C = 1.602176634e-19
_TGLF_MAX_SPECIES = 12
_SPECIES_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,63}$")


def _format_tglf_scientific(value: float) -> str:
    """Match current GACODE's compact exponent spelling."""
    return f"{value:.6e}".replace("e-0", "e-").replace("e+0", "e+")


def _normalize_tglf_timeout_seconds(timeout_s: float) -> float:
    timeout = float(timeout_s)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("timeout_s must be finite and > 0.")
    return timeout


def _normalize_tglf_max_retries(max_retries: int) -> int:
    if isinstance(max_retries, bool) or not isinstance(max_retries, int):
        raise ValueError(f"max_retries must be an integer in [0, {_TGLF_MAX_RETRIES_LIMIT}].")
    retries = int(max_retries)
    if retries < 0 or retries > _TGLF_MAX_RETRIES_LIMIT:
        raise ValueError(f"max_retries must be an integer in [0, {_TGLF_MAX_RETRIES_LIMIT}].")
    return retries


def _validate_tglf_command_name(command: str = "tglf") -> str:
    """Return a normalized command name after enforcing the PATH-only contract."""
    if not isinstance(command, str) or not command.strip():
        raise ValueError("TGLF command must be a non-empty PATH command name.")
    command = command.strip()
    if Path(command).name != command or "/" in command or "\\" in command:
        raise ValueError("TGLF must be resolved through PATH; filesystem paths are forbidden.")
    return command


def _resolve_tglf_command(command: str = "tglf") -> str:
    """Resolve a validated command name through ``PATH``."""
    command = _validate_tglf_command_name(command)
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(f"TGLF command not found on PATH: {command}")
    return resolved


def _validate_tglf_deck(deck: TGLFInputDeck) -> None:
    """Reject non-finite or physically invalid values before external execution."""
    for field_name, value in deck.__dict__.items():
        if field_name == "species":
            continue
        if field_name in {"use_bper", "use_bpar"}:
            if not isinstance(value, bool):
                raise ValueError(f"TGLF input {field_name!r} must be boolean.")
            continue
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError(f"TGLF input {field_name!r} must be numeric; got {value!r}.")
        if not np.isfinite(value):
            raise ValueError(f"TGLF input {field_name!r} must be finite; got {value!r}.")
    positive = {
        "q": deck.q,
        "kappa": deck.kappa,
        "T_e_keV": deck.T_e_keV,
        "T_i_keV": deck.T_i_keV,
        "n_e_19": deck.n_e_19,
        "R_major": deck.R_major,
        "a_minor": deck.a_minor,
        "B_toroidal": deck.B_toroidal,
    }
    for field_name, value in positive.items():
        if value <= 0.0:
            raise ValueError(f"TGLF input {field_name!r} must be > 0; got {value!r}.")
    if not 0.0 < deck.rho <= 1.0:
        raise ValueError(f"TGLF input 'rho' must be in (0, 1]; got {deck.rho!r}.")
    if deck.beta_e < 0.0 or deck.xnue < 0.0 or deck.Z_eff < 1.0:
        raise ValueError("TGLF beta_e/xnue must be non-negative and Z_eff must be >= 1.")
    species = deck.resolved_species()
    if not 2 <= len(species) <= _TGLF_MAX_SPECIES:
        raise ValueError(f"TGLF requires between 2 and {_TGLF_MAX_SPECIES} kinetic species.")
    if not all(isinstance(item, TGLFSpecies) for item in species):
        raise ValueError("Every explicit TGLF species must be a TGLFSpecies instance.")
    names: set[str] = set()
    for item in species:
        if (
            not isinstance(item.name, str)
            or _SPECIES_NAME_RE.fullmatch(item.name) is None
            or item.name in names
        ):
            raise ValueError("TGLF species names must be unique portable identifiers.")
        names.add(item.name)
        numeric = (
            item.mass_deuterium,
            item.charge_e,
            item.density_e_ratio,
            item.temperature_e_ratio,
            item.R_Ln,
            item.R_LT,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in numeric
        ) or not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"TGLF species {item.name!r} contains a non-finite value.")
        if (
            item.mass_deuterium <= 0.0
            or item.density_e_ratio <= 0.0
            or item.temperature_e_ratio <= 0.0
        ):
            raise ValueError(
                f"TGLF species {item.name!r} mass, density and temperature must be > 0."
            )
    if not math.isclose(species[0].charge_e, -1.0, rel_tol=0.0, abs_tol=1e-12) or any(
        item.charge_e <= 0.0 for item in species[1:]
    ):
        raise ValueError("TGLF species 1 must have Z=-1; all following species must be ions.")
    if not (
        math.isclose(species[0].density_e_ratio, 1.0, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(species[0].temperature_e_ratio, 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("TGLF electron AS_1 and TAUS_1 reference ratios must both equal 1.")
    electron_charge_density = abs(species[0].charge_e) * species[0].density_e_ratio
    ion_charge_density = sum(item.charge_e * item.density_e_ratio for item in species[1:])
    if not math.isclose(electron_charge_density, ion_charge_density, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("TGLF species densities and charges must satisfy quasineutrality.")


def render_tglf_input(deck: TGLFInputDeck) -> str:
    """Render a current GACODE ``input.tglf`` key-value deck.

    Public SCPN inputs use ``R/L`` gradients; GACODE TGLF consumes ``a/L``.
    Species order is explicit when ``deck.species`` is populated. An empty
    tuple resolves to the legacy electron/deuterium pair.
    """
    _validate_tglf_deck(deck)
    a_over_r = deck.a_minor / deck.R_major
    rmin_loc = deck.rho
    rmaj_loc = deck.R_major / deck.a_minor
    q_prime_loc = (
        deck.q_prime_loc if deck.q_prime_loc != 0.0 else deck.s_hat * (deck.q / rmin_loc) ** 2
    )
    p_prime_loc = (
        deck.p_prime_loc
        if deck.p_prime_loc != 0.0
        else -deck.alpha_mhd / (8.0 * math.pi * deck.q * rmaj_loc * rmin_loc)
    )
    species = deck.resolved_species()
    lines = [
        "# TGLF input deck generated by SCPN Fusion Core",
        f"# rho = {deck.rho:.6f}",
        "UNITS = GYRO",
        "USE_TRANSPORT_MODEL = .true.",
        "GEOMETRY_FLAG = 1",
        f"USE_BPER = {'.true.' if deck.use_bper else '.false.'}",
        f"USE_BPAR = {'.true.' if deck.use_bpar else '.false.'}",
        "SIGN_BT = 1.0",
        "SIGN_IT = 1.0",
        f"NS = {len(species)}",
        f"Q_LOC = {deck.q:.12g}",
        f"Q_PRIME_LOC = {q_prime_loc:.12g}",
        f"P_PRIME_LOC = {p_prime_loc:.12g}",
        f"S_KAPPA_LOC = {deck.s_kappa:.12g}",
        f"S_DELTA_LOC = {deck.s_delta:.12g}",
        f"KAPPA_LOC = {deck.kappa:.12g}",
        f"DELTA_LOC = {deck.delta:.12g}",
        f"XNUE = {deck.xnue:.12g}",
        f"BETAE = {deck.beta_e:.12g}",
        f"ZEFF = {deck.Z_eff:.12g}",
        f"RMAJ_LOC = {rmaj_loc:.12g}",
        f"RMIN_LOC = {rmin_loc:.12g}",
    ]
    for index, item in enumerate(species, start=1):
        lines.extend(
            [
                f"MASS_{index} = {_format_tglf_scientific(item.mass_deuterium)}",
                f"ZS_{index} = {item.charge_e:.1f}",
                f"RLNS_{index} = {item.R_Ln * a_over_r:.12g}",
                f"RLTS_{index} = {item.R_LT * a_over_r:.12g}",
                f"TAUS_{index} = {item.temperature_e_ratio:.12g}",
                f"AS_{index} = {item.density_e_ratio:.12g}",
            ]
        )
    return "\n".join(lines) + "\n"


def write_tglf_input_file(deck: TGLFInputDeck, output_dir: str | Path) -> Path:
    """Write a validated current-GACODE ``input.tglf`` file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "input.tglf"
    path.write_text(render_tglf_input(deck), encoding="utf-8")
    return path


def _gyro_bohm_diffusivity(deck: TGLFInputDeck) -> float:
    """Return the GACODE ``rho_s^2 c_s / a`` diffusivity scale in ``m^2/s``."""
    te_joule = deck.T_e_keV * 1.0e3 * _ELEMENTARY_CHARGE_C
    sound_speed = math.sqrt(te_joule / _DEUTERIUM_MASS_KG)
    rho_s = _DEUTERIUM_MASS_KG * sound_speed / (_ELEMENTARY_CHARGE_C * deck.B_toroidal)
    return rho_s * rho_s * sound_speed / deck.a_minor


def _effective_coefficient(flux_gb: float, gradient_a_over_l: float, chi_gb: float) -> float:
    """Convert signed gyro-Bohm flux to a gradient-normalised coefficient."""
    if abs(gradient_a_over_l) <= 1.0e-12:
        if abs(flux_gb) <= 1.0e-12:
            return 0.0
        raise RuntimeError("TGLF emitted finite flux for a zero normalising gradient.")
    return flux_gb * chi_gb / gradient_a_over_l


def _load_numeric_rows(path: Path, *, skiprows: int) -> NDArray[np.float64]:
    """Load a finite, non-empty numeric TGLF table with stable 2-D shape."""
    try:
        data = np.loadtxt(path, skiprows=skiprows, ndmin=2)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Could not parse TGLF output {path.name}: {exc}") from exc
    out = np.asarray(data, dtype=np.float64)
    if out.size == 0 or not bool(np.all(np.isfinite(out))):
        raise RuntimeError(f"TGLF output {path.name} is empty or non-finite.")
    return out


def _parse_gacode_tglf_output(output_dir: Path, deck: TGLFInputDeck) -> TGLFOutput:
    """Parse current GACODE flux and eigenvalue files for a completed run."""
    flux_path = output_dir / "out.tglf.gbflux"
    eigen_path = output_dir / "out.tglf.eigenvalue_spectrum"
    flux = _load_numeric_rows(flux_path, skiprows=0).reshape(-1)
    species = deck.resolved_species()
    expected_flux_values = 4 * len(species)
    if flux.size != expected_flux_values:
        raise RuntimeError(
            "TGLF out.tglf.gbflux must contain exactly "
            f"4 * NS = {expected_flux_values} values; got {flux.size}."
        )
    species_fluxes = tuple(
        TGLFSpeciesFlux(
            species_index=index,
            name=item.name,
            charge_e=item.charge_e,
            particle_gb=float(flux[index]),
            energy_gb=float(flux[len(species) + index]),
            momentum_gb=float(flux[2 * len(species) + index]),
            exchange_gb=float(flux[3 * len(species) + index]),
        )
        for index, item in enumerate(species)
    )
    electron = species_fluxes[0]
    main_ion_index = next(index for index, item in enumerate(species) if item.charge_e > 0.0)
    main_ion = species_fluxes[main_ion_index]
    particle_e = electron.particle_gb
    particle_i = main_ion.particle_gb
    q_e = electron.energy_gb
    q_i = main_ion.energy_gb

    eigen = _load_numeric_rows(eigen_path, skiprows=2)
    if eigen.shape[1] < 2 or eigen.shape[1] % 2 != 0:
        raise RuntimeError("TGLF eigenvalue spectrum must contain gamma/frequency pairs.")
    gamma_max = float(np.max(eigen[:, 0::2]))

    a_over_r = deck.a_minor / deck.R_major
    chi_gb = _gyro_bohm_diffusivity(deck)
    main_ion_density_temperature = (
        species[main_ion_index].density_e_ratio * species[main_ion_index].temperature_e_ratio
    )
    chi_i = (
        _effective_coefficient(
            q_i,
            species[main_ion_index].R_LT * a_over_r,
            chi_gb,
        )
        / main_ion_density_temperature
    )
    chi_e = _effective_coefficient(q_e, species[0].R_LT * a_over_r, chi_gb) / (
        species[0].density_e_ratio * species[0].temperature_e_ratio
    )
    d_e = (
        _effective_coefficient(particle_e, species[0].R_Ln * a_over_r, chi_gb)
        / species[0].density_e_ratio
    )
    d_i = (
        _effective_coefficient(particle_i, species[main_ion_index].R_Ln * a_over_r, chi_gb)
        / species[main_ion_index].density_e_ratio
    )
    return TGLFOutput(
        rho=deck.rho,
        chi_i=chi_i,
        chi_e=chi_e,
        gamma_max=gamma_max,
        q_i=q_i,
        q_e=q_e,
        particle_e=particle_e,
        particle_i=particle_i,
        d_e=d_e,
        d_i=d_i,
        species_fluxes=species_fluxes,
    )


def identify_tglf_particle_transport(
    decks: Sequence[TGLFInputDeck],
    outputs: Sequence[TGLFOutput],
    *,
    species_name: str,
) -> TGLFParticleTransportIdentification:
    """Identify diffusion and convective pinch from matched density-gradient runs.

    At least three runs are required so a residual can be reported. Every deck
    parameter and every non-target species field must match exactly; only the
    selected species ``R/L_n`` may vary.
    """
    if len(decks) != len(outputs) or len(decks) < 3:
        raise ValueError(
            "TGLF diffusion/pinch identification requires at least three matched runs."
        )
    reference_species = decks[0].resolved_species()
    matching_indices = [i for i, item in enumerate(reference_species) if item.name == species_name]
    if len(matching_indices) != 1:
        raise ValueError(
            f"TGLF species {species_name!r} is not uniquely present in the reference deck."
        )
    species_index = matching_indices[0]
    deck_field_names = tuple(item.name for item in fields(TGLFInputDeck) if item.name != "species")
    reference_scalars = tuple(getattr(decks[0], name) for name in deck_field_names)
    reference_species_signature = tuple(
        (
            item.name,
            item.mass_deuterium,
            item.charge_e,
            item.density_e_ratio,
            item.temperature_e_ratio,
            None if index == species_index else item.R_Ln,
            item.R_LT,
        )
        for index, item in enumerate(reference_species)
    )

    gradients: list[float] = []
    fluxes: list[float] = []
    for run_index, (deck, output) in enumerate(zip(decks, outputs, strict=True)):
        _validate_tglf_deck(deck)
        if tuple(getattr(deck, name) for name in deck_field_names) != reference_scalars:
            raise ValueError(f"TGLF paired-gradient deck {run_index} changes a non-species field.")
        run_species = deck.resolved_species()
        run_signature = tuple(
            (
                item.name,
                item.mass_deuterium,
                item.charge_e,
                item.density_e_ratio,
                item.temperature_e_ratio,
                None if index == species_index else item.R_Ln,
                item.R_LT,
            )
            for index, item in enumerate(run_species)
        )
        if run_signature != reference_species_signature:
            raise ValueError(
                f"TGLF paired-gradient deck {run_index} changes species order or state."
            )
        if len(output.species_fluxes) != len(run_species):
            raise ValueError(
                f"TGLF paired-gradient output {run_index} lacks canonical species fluxes."
            )
        flux = output.species_fluxes[species_index]
        if flux.name != species_name or flux.species_index != species_index:
            raise ValueError(
                f"TGLF paired-gradient output {run_index} changes species identity/order."
            )
        gradients.append(run_species[species_index].R_Ln * deck.a_minor / deck.R_major)
        fluxes.append(flux.particle_gb)

    gradient_array = np.asarray(gradients, dtype=np.float64)
    flux_array = np.asarray(fluxes, dtype=np.float64)
    density_e_ratio = reference_species[species_index].density_e_ratio
    normalized_flux_array = flux_array / density_e_ratio
    if np.unique(gradient_array).size < 3:
        raise ValueError("TGLF diffusion/pinch identification requires three distinct gradients.")
    design = np.column_stack((gradient_array, np.ones_like(gradient_array)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, normalized_flux_array, rcond=None)
    if rank != 2:
        raise ValueError("TGLF paired-gradient design matrix is rank deficient.")
    diffusion_gb, pinch_gb = (float(coefficients[0]), float(coefficients[1]))
    residual = normalized_flux_array - design @ coefficients
    chi_gb = _gyro_bohm_diffusivity(decks[0])
    return TGLFParticleTransportIdentification(
        species_index=species_index,
        species_name=species_name,
        density_e_ratio=density_e_ratio,
        gradients_a_over_l=tuple(float(value) for value in gradient_array),
        particle_fluxes_gb=tuple(float(value) for value in flux_array),
        normalized_particle_fluxes_gb=tuple(float(value) for value in normalized_flux_array),
        diffusion_gb=diffusion_gb,
        pinch_gb=pinch_gb,
        diffusion_m2_s=diffusion_gb * chi_gb,
        pinch_m_s=pinch_gb * chi_gb / decks[0].a_minor,
        residual_rms_gb_per_density=float(np.sqrt(np.mean(residual * residual))),
        residual_max_abs_gb_per_density=float(np.max(np.abs(residual))),
    )


def _parse_gacode_tglf_spectrum(
    output_dir: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return per-``ky`` dominant ``(gamma, frequency, ky)`` arrays."""
    eigen = _load_numeric_rows(output_dir / "out.tglf.eigenvalue_spectrum", skiprows=2)
    ky = _load_numeric_rows(output_dir / "out.tglf.ky_spectrum", skiprows=2).reshape(-1)
    if eigen.shape[1] < 2 or eigen.shape[1] % 2 != 0:
        raise RuntimeError("TGLF eigenvalue spectrum must contain gamma/frequency pairs.")
    if eigen.shape[0] != ky.size:
        raise RuntimeError("TGLF ky and eigenvalue spectra have inconsistent row counts.")
    gamma_modes = eigen[:, 0::2]
    frequency_modes = eigen[:, 1::2]
    dominant = np.argmax(gamma_modes, axis=1)
    rows = np.arange(eigen.shape[0])
    gamma = np.asarray(gamma_modes[rows, dominant], dtype=np.float64)
    frequency = np.asarray(frequency_modes[rows, dominant], dtype=np.float64)
    return gamma, frequency, np.asarray(ky, dtype=np.float64)


def run_tglf_binary(
    deck: TGLFInputDeck,
    *,
    tglf_command: str = "tglf",
    timeout_s: float = 120.0,
    work_dir: str | Path | None = None,
    max_retries: int = 2,
) -> TGLFOutput:
    """Execute PATH-resolved GACODE TGLF and parse its current output files."""
    timeout_s = _normalize_tglf_timeout_seconds(timeout_s)
    max_retries = _normalize_tglf_max_retries(max_retries)
    tglf_path = _resolve_tglf_command(tglf_command)
    _validate_tglf_deck(deck)

    cleanup = False
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="tglf_"))
        cleanup = True
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    last_exc: Exception | None = None
    try:
        for attempt in range(max_retries + 1):
            last_exc = None
            try:
                for stale in work_dir.glob("out.tglf.*"):
                    if stale.is_file():
                        stale.unlink()
                write_tglf_input_file(deck, work_dir)
                result = subprocess.run(
                    [tglf_path, "-e", "."],
                    cwd=str(work_dir),
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )

                if result.returncode != 0:
                    raise RuntimeError(
                        f"TGLF exited with code {result.returncode}: {result.stderr[:500]}"
                    )

                return _parse_gacode_tglf_output(work_dir, deck)
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                last_exc = exc
                if attempt < max_retries:
                    logger.warning("TGLF attempt %d failed: %s. Retrying...", attempt + 1, exc)
                    import time

                    time.sleep(_TGLF_RETRY_BACKOFF_SECONDS)
        assert last_exc is not None
        raise RuntimeError(f"TGLF execution failed after {max_retries + 1} attempts.") from last_exc
    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)


def _parse_tglf_run_output(
    path: Path,
    rho: float,
    species: Sequence[TGLFSpecies] | None = None,
) -> TGLFOutput:
    """Parse the human-readable GACODE run summary into signed raw fluxes."""
    species = tuple(species) if species is not None else TGLFInputDeck().resolved_species()
    parsed: dict[int, TGLFSpeciesFlux] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        tokens = line.split()
        label = tokens[0].lower() if tokens else ""
        if label == "elec":
            species_index = 0
        elif label.startswith("ion") and label[3:].isdigit():
            species_index = int(label[3:])
        else:
            continue
        if species_index >= len(species):
            continue
        try:
            if len(tokens) >= 9 and tokens[1].lower() == "particle":
                particle_flux = float(tokens[2])
                energy_flux = float(tokens[4])
                momentum_flux = float(tokens[6])
                exchange_flux = float(tokens[8])
            else:
                particle_flux = float(tokens[1])
                energy_flux = float(tokens[2])
                # Current out.tglf.run inserts Q_low between energy and momentum.
                momentum_index = 4 if len(tokens) >= 6 else 3
                exchange_index = 5 if len(tokens) >= 6 else 4
                momentum_flux = (
                    float(tokens[momentum_index]) if len(tokens) > momentum_index else 0.0
                )
                exchange_flux = (
                    float(tokens[exchange_index]) if len(tokens) > exchange_index else 0.0
                )
        except (IndexError, ValueError):
            continue
        if not all(
            math.isfinite(value)
            for value in (particle_flux, energy_flux, momentum_flux, exchange_flux)
        ):
            continue
        item = species[species_index]
        parsed[species_index] = TGLFSpeciesFlux(
            species_index=species_index,
            name=item.name,
            charge_e=item.charge_e,
            particle_gb=particle_flux,
            energy_gb=energy_flux,
            momentum_gb=momentum_flux,
            exchange_gb=exchange_flux,
        )
    species_fluxes = tuple(parsed[index] for index in sorted(parsed))
    electron = parsed.get(0)
    main_ion_index = next(index for index, item in enumerate(species) if item.charge_e > 0.0)
    main_ion = parsed.get(main_ion_index)
    return TGLFOutput(
        rho=rho,
        q_i=main_ion.energy_gb if main_ion is not None else 0.0,
        q_e=electron.energy_gb if electron is not None else 0.0,
        particle_e=electron.particle_gb if electron is not None else 0.0,
        particle_i=main_ion.particle_gb if main_ion is not None else 0.0,
        species_fluxes=species_fluxes,
    )
