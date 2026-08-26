# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TGLF Interface Types
"""Shared TGLF interface data containers and constants."""

from __future__ import annotations

from dataclasses import dataclass, field

from scpn_fusion._data_paths import data_root

TGLF_REF_DIR = data_root() / "validation" / "tglf_reference"
_TGLF_RETRY_BACKOFF_SECONDS = 1.0
_TGLF_MAX_RETRIES_LIMIT = 10
_TGLF_MAX_PARSED_VECTOR_LENGTH = 2048


@dataclass(frozen=True)
class TGLFSpecies:
    """One ordered kinetic species in current GACODE normalisation.

    ``mass_deuterium`` is ``MASS_s`` (mass relative to deuterium),
    ``density_e_ratio`` is ``AS_s`` (density relative to electron density),
    and ``temperature_e_ratio`` is ``TAUS_s`` (temperature relative to the
    electron temperature). Gradients remain in the public ``R/L`` convention.
    """

    name: str
    mass_deuterium: float
    charge_e: float
    density_e_ratio: float
    temperature_e_ratio: float
    R_Ln: float
    R_LT: float


@dataclass
class TGLFInputDeck:
    """All TGLF input parameters for a single flux surface."""

    rho: float = 0.5
    s_hat: float = 1.0
    q: float = 1.5
    q_prime_loc: float = 0.0
    alpha_mhd: float = 0.0
    p_prime_loc: float = 0.0
    kappa: float = 1.7
    delta: float = 0.3
    s_kappa: float = 0.0
    s_delta: float = 0.0
    R_LTi: float = 6.0
    R_LTe: float = 6.0
    R_Lne: float = 2.0
    R_Lni: float = 2.0
    beta_e: float = 0.01
    Z_eff: float = 1.5
    xnue: float = 0.0
    T_e_keV: float = 10.0
    T_i_keV: float = 10.0
    n_e_19: float = 8.0
    R_major: float = 6.2
    a_minor: float = 2.0
    B_toroidal: float = 5.3
    use_bper: bool = False
    use_bpar: bool = False
    species: tuple[TGLFSpecies, ...] = ()

    def resolved_species(self) -> tuple[TGLFSpecies, ...]:
        """Return the explicit ordered species or the legacy electron/D view."""
        if self.species:
            return tuple(self.species)
        return (
            TGLFSpecies(
                name="electron",
                mass_deuterium=2.723e-4,
                charge_e=-1.0,
                density_e_ratio=1.0,
                temperature_e_ratio=1.0,
                R_Ln=self.R_Lne,
                R_LT=self.R_LTe,
            ),
            TGLFSpecies(
                name="deuterium",
                mass_deuterium=1.0,
                charge_e=1.0,
                density_e_ratio=1.0,
                temperature_e_ratio=self.T_i_keV / self.T_e_keV,
                R_Ln=self.R_Lni,
                R_LT=self.R_LTi,
            ),
        )


@dataclass(frozen=True)
class TGLFSpeciesFlux:
    """The four signed GACODE flux blocks for one zero-based species index."""

    species_index: int
    name: str
    charge_e: float
    particle_gb: float
    energy_gb: float
    momentum_gb: float
    exchange_gb: float


@dataclass(frozen=True)
class TGLFParticleTransportIdentification:
    """Paired-gradient diffusion/pinch fit in explicit GACODE normalisation.

    GACODE particle flux contains the species density fraction ``AS_s``. The
    fitted relation is therefore ``Gamma_hat/AS_s = D_hat*(a/L_n) + V_hat``.
    Physical values use ``D = chi_GB*D_hat`` and ``V = chi_GB/a*V_hat``.
    """

    species_index: int
    species_name: str
    density_e_ratio: float
    gradients_a_over_l: tuple[float, ...]
    particle_fluxes_gb: tuple[float, ...]
    normalized_particle_fluxes_gb: tuple[float, ...]
    diffusion_gb: float
    pinch_gb: float
    diffusion_m2_s: float
    pinch_m_s: float
    residual_rms_gb_per_density: float
    residual_max_abs_gb_per_density: float


@dataclass
class TGLFOutput:
    """Parsed TGLF output for a single run.

    ``species_fluxes`` is the canonical ordered representation. The scalar
    electron/main-ion fields are a backwards-compatible convenience view.
    ``d_i`` and ``d_e`` are one-point signed effective coefficients, not an
    identified diffusion/pinch separation.
    """

    rho: float = 0.5
    chi_i: float = 0.0
    chi_e: float = 0.0
    gamma_max: float = 0.0
    q_i: float = 0.0
    q_e: float = 0.0
    particle_e: float = 0.0
    particle_i: float = 0.0
    d_e: float = 0.0
    d_i: float = 0.0
    species_fluxes: tuple[TGLFSpeciesFlux, ...] = ()


@dataclass
class TGLFComparisonResult:
    """Comparison between local transport and TGLF."""

    case_name: str = ""
    rho_points: list[float] = field(default_factory=list)
    our_chi_i: list[float] = field(default_factory=list)
    tglf_chi_i: list[float] = field(default_factory=list)
    our_chi_e: list[float] = field(default_factory=list)
    tglf_chi_e: list[float] = field(default_factory=list)
    rms_error_chi_i: float = 0.0
    rms_error_chi_e: float = 0.0
    correlation_chi_i: float = 0.0
    correlation_chi_e: float = 0.0
    max_rel_error_chi_i: float = 0.0
    max_rel_error_chi_e: float = 0.0


@dataclass
class TGLFProfileScanResult:
    """Interpolated transport profiles from a live TGLF radial scan."""

    rho_samples: list[float] = field(default_factory=list)
    chi_i_samples: list[float] = field(default_factory=list)
    chi_e_samples: list[float] = field(default_factory=list)
    gamma_samples: list[float] = field(default_factory=list)
    chi_i_profile: list[float] = field(default_factory=list)
    chi_e_profile: list[float] = field(default_factory=list)
    gamma_profile: list[float] = field(default_factory=list)


@dataclass
class TGLFReferenceCaseResult:
    """Reduced-closure comparison against a single TGLF reference regime."""

    case_name: str
    reference_mode: str
    predicted_mode: str
    mode_match: bool
    predicted_chi_i_gyrobohm: float
    predicted_chi_e_gyrobohm: float
    reference_chi_i_gyrobohm: float
    reference_chi_e_gyrobohm: float
    rel_error_chi_i: float
    rel_error_chi_e: float
