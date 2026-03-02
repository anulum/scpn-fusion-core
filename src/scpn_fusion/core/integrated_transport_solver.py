# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Integrated Transport Solver
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
from pathlib import Path
from typing import Any

try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel  # type: ignore[assignment]

from scpn_fusion.core.eped_pedestal import EpedPedestalModel
from scpn_fusion.core.integrated_transport_solver_runtime import (
    AdaptiveTimeController,
    TransportSolverRuntimeMixin,
)

_logger = logging.getLogger(__name__)

_rust_transport_available = False
_PyTransportSolver: Any = None
try:
    from scpn_fusion_rs import PyTransportSolver as _PyTransportSolver  # type: ignore[assignment,no-redef]
    _rust_transport_available = True
except ImportError as exc:
    _logger.debug("Rust transport bindings unavailable; using Python transport kernels: %s", exc)

_RUST_TRANSPORT_FALLBACK_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    FloatingPointError,
)
_EPED_FALLBACK_EXCEPTIONS = (
    ImportError,
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    FloatingPointError,
)
_RUST_CHANG_HINTON_DEFAULTS: dict[str, float] = {
    "R0": 6.2,
    "a": 2.0,
    "B0": 5.3,
    "A_ion": 2.0,
    "Z_eff": 1.5,
}
_RUST_CHANG_HINTON_PARAM_ATOL = 1e-12


class PhysicsError(RuntimeError):
    """Raised when a physics constraint is violated."""

_GYRO_BOHM_COEFF_PATH = (
    Path(__file__).resolve().parents[3]
    / "validation"
    / "reference_data"
    / "itpa"
    / "gyro_bohm_coefficients.json"
)

_GYRO_BOHM_DEFAULT = 0.1  # Compatibility default if JSON is unavailable


def _require_positive_finite_scalar(name: str, value: Any) -> float:
    """Validate finite-positive scalar inputs for transport kernels."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if (not np.isfinite(parsed)) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return parsed


def _coerce_matching_1d_profiles(**profiles: Any) -> dict[str, np.ndarray]:
    """Coerce profile-like inputs to matching 1-D float64 arrays."""
    out: dict[str, np.ndarray] = {}
    expected_shape: tuple[int, ...] | None = None
    for name, values in profiles.items():
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a 1-D array, got shape {arr.shape}")
        if arr.size == 0:
            raise ValueError(f"{name} must not be empty")
        if expected_shape is None:
            expected_shape = arr.shape
        elif arr.shape != expected_shape:
            raise ValueError(
                f"All profiles must have the same shape; {name} has {arr.shape}, "
                f"expected {expected_shape}"
            )
        out[name] = arr
    return out


def _rust_chang_hinton_params_match_defaults(
    *,
    R0: float,
    a: float,
    B0: float,
    A_ion: float,
    Z_eff: float,
) -> bool:
    """Return whether Chang-Hinton inputs match Rust default-parameter contract."""
    params = {
        "R0": float(R0),
        "a": float(a),
        "B0": float(B0),
        "A_ion": float(A_ion),
        "Z_eff": float(Z_eff),
    }
    for key, value in params.items():
        if not np.isclose(
            value,
            _RUST_CHANG_HINTON_DEFAULTS[key],
            rtol=0.0,
            atol=_RUST_CHANG_HINTON_PARAM_ATOL,
        ):
            return False
    return True


def _load_gyro_bohm_coefficient(
    path: Path | str | None = None,
) -> float:
    """Load the calibrated gyro-Bohm coefficient c_gB from JSON.

    Parameters
    ----------
    path : Path or str, optional
        Override path.  Defaults to the file shipped in
        ``validation/reference_data/itpa/gyro_bohm_coefficients.json``.

    Returns
    -------
    float
        The calibrated c_gB value, or 0.1 if the file is not found.
    """
    c_gB, _ = _load_gyro_bohm_coefficient_with_contract(path)
    return c_gB


def _load_gyro_bohm_coefficient_with_contract(
    path: Path | str | None = None,
) -> tuple[float, dict[str, Any]]:
    """Load calibrated c_gB and return value plus provenance contract."""
    p = Path(path) if path else _GYRO_BOHM_COEFF_PATH
    contract: dict[str, Any] = {
        "source": "json_file",
        "path": str(p),
        "fallback_used": False,
        "error": None,
    }
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        c_gB = float(data["c_gB"])
        if (not np.isfinite(c_gB)) or c_gB <= 0.0:
            raise ValueError(f"Invalid c_gB={c_gB!r}")
        _logger.debug("Loaded c_gB = %.6f from %s", c_gB, p)
        return c_gB, contract
    except (FileNotFoundError, KeyError, json.JSONDecodeError, TypeError, ValueError) as exc:
        _logger.warning(
            "Could not load c_gB from %s (%s), using default %.4f",
            p, exc, _GYRO_BOHM_DEFAULT,
        )
        contract["source"] = "default_fallback"
        contract["fallback_used"] = True
        contract["error"] = f"{exc.__class__.__name__}:{exc}"
        return _GYRO_BOHM_DEFAULT, contract


_gyro_bohm_cache: float | None = None
_gyro_bohm_cache_contract: dict[str, Any] | None = None


def _load_gyro_bohm_coefficient_cached() -> float:
    """Return the default-path c_gB value, reading the file at most once."""
    value, _ = _load_gyro_bohm_coefficient_cached_with_contract()
    return value


def _load_gyro_bohm_coefficient_cached_with_contract() -> tuple[float, dict[str, Any]]:
    """Cached c_gB loader with provenance contract."""
    global _gyro_bohm_cache  # noqa: PLW0603
    global _gyro_bohm_cache_contract  # noqa: PLW0603
    if _gyro_bohm_cache is None or _gyro_bohm_cache_contract is None:
        _gyro_bohm_cache, _gyro_bohm_cache_contract = _load_gyro_bohm_coefficient_with_contract()
    return float(_gyro_bohm_cache), dict(_gyro_bohm_cache_contract)


def chang_hinton_chi_profile(rho, T_i, n_e_19, q, R0, a, B0, A_ion=2.0, Z_eff=1.5):
    """
    Chang-Hinton (1982) neoclassical ion thermal diffusivity profile [m²/s].

    Parameters
    ----------
    rho : array  — normalised radius [0,1]
    T_i : array  — ion temperature [keV]
    n_e_19 : array  — electron density [10^19 m^-3]
    q : array  — safety factor profile
    R0 : float  — major radius [m]
    a : float  — minor radius [m]
    B0 : float  — toroidal field [T]
    A_ion : float  — ion mass number (default 2 = deuterium)
    Z_eff : float  — effective charge

    Returns
    -------
    chi_nc : array  — neoclassical chi_i [m²/s]
    """
    prof = _coerce_matching_1d_profiles(rho=rho, T_i=T_i, n_e_19=n_e_19, q=q)
    rho_arr = np.clip(
        np.nan_to_num(prof["rho"], nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )
    T_i_arr = np.maximum(
        np.nan_to_num(prof["T_i"], nan=0.01, posinf=1e3, neginf=0.01),
        0.01,
    )
    n_e_arr = np.maximum(
        np.nan_to_num(prof["n_e_19"], nan=0.1, posinf=1e3, neginf=0.1),
        0.1,
    )
    q_arr = np.maximum(
        np.nan_to_num(prof["q"], nan=1.0, posinf=10.0, neginf=0.1),
        0.1,
    )

    r0 = _require_positive_finite_scalar("R0", R0)
    a_minor = _require_positive_finite_scalar("a", a)
    b0 = _require_positive_finite_scalar("B0", B0)
    a_ion = _require_positive_finite_scalar("A_ion", A_ion)
    z_eff = _require_positive_finite_scalar("Z_eff", Z_eff)

    # Rust fast-path (parameters must match NeoclassicalParams::default())
    # The Rust PyTransportSolver.chang_hinton_chi_profile() uses
    # NeoclassicalParams::default() (R0=6.2, a=2.0, B0=5.3, A_ion=2.0,
    # Z_eff=1.5) with only q_profile overridden. Delegate when the
    # caller's parameters match these defaults.
    use_rust_fast_path = _rust_transport_available and _rust_chang_hinton_params_match_defaults(
        R0=r0,
        a=a_minor,
        B0=b0,
        A_ion=a_ion,
        Z_eff=z_eff,
    )
    if use_rust_fast_path:
        try:
            solver = _PyTransportSolver()
            chi_rust = np.asarray(
                solver.chang_hinton_chi_profile(rho_arr, T_i_arr, n_e_arr, q_arr),
                dtype=np.float64,
            )
            _logger.debug("chang_hinton_chi_profile: Rust fast-path (%d pts)", len(rho_arr))
            return chi_rust
        except _RUST_TRANSPORT_FALLBACK_EXCEPTIONS:
            _logger.debug("chang_hinton_chi_profile: Rust fast-path failed, falling back to Python")
    elif _rust_transport_available:
        _logger.debug(
            "chang_hinton_chi_profile: Rust fast-path skipped for non-default geometry/charge "
            "(R0=%.6g, a=%.6g, B0=%.6g, A_ion=%.6g, Z_eff=%.6g)",
            r0,
            a_minor,
            b0,
            a_ion,
            z_eff,
        )

    # Vectorized Python fallback
    e_charge = 1.602176634e-19
    eps0 = 8.854187812e-12
    m_p = 1.672621924e-27
    m_i = a_ion * m_p

    chi_nc = np.full_like(rho_arr, 0.01)

    # Mask: valid points only
    valid = (rho_arr > 0.0) & (T_i_arr > 0.0) & (n_e_arr > 0.0) & (q_arr > 0.0)
    epsilon = rho_arr * a_minor / r0
    valid &= epsilon >= 1e-6

    if not np.any(valid):
        return chi_nc

    eps_v = epsilon[valid]
    T_J = T_i_arr[valid] * 1.602176634e-16
    v_ti = np.sqrt(2.0 * T_J / m_i)
    rho_i = m_i * v_ti / (e_charge * b0)

    n_e = n_e_arr[valid] * 1e19
    ln_lambda = 17.0
    nu_ii = (n_e * z_eff ** 2 * e_charge ** 4 * ln_lambda
             / (12.0 * np.pi ** 1.5 * eps0 ** 2 * m_i ** 0.5 * T_J ** 1.5))

    eps32 = eps_v ** 1.5
    nu_star = np.maximum(nu_ii * q_arr[valid] * r0 / (eps32 * v_ti), 0.0)

    chi_val = (0.66 * (1.0 + 1.54 * eps_v) * q_arr[valid] ** 2
               * rho_i ** 2 * nu_ii
               / (eps32 * (1.0 + 0.74 * nu_star ** (2.0 / 3.0))))

    chi_val = np.where(np.isfinite(chi_val), np.maximum(chi_val, 0.01), 0.01)
    chi_nc[valid] = chi_val

    return chi_nc


def calculate_sauter_bootstrap_current_full(rho, Te, Ti, ne, q, R0, a, B0, Z_eff=1.5):
    """Full Sauter bootstrap current model (Sauter et al., Phys. Plasmas 6, 1999).

    Parameters
    ----------
    rho : array — normalised radius [0,1]
    Te : array — electron temperature [keV]
    Ti : array — ion temperature [keV]
    ne : array — electron density [10^19 m^-3]
    q : array — safety factor profile
    R0 : float — major radius [m]
    a : float — minor radius [m]
    B0 : float — toroidal field [T]
    Z_eff : float — effective charge

    Returns
    -------
    j_bs : array — bootstrap current density [A/m^2]
    """
    prof = _coerce_matching_1d_profiles(rho=rho, Te=Te, Ti=Ti, ne=ne, q=q)
    rho_arr = np.clip(
        np.nan_to_num(prof["rho"], nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )
    te_arr = np.maximum(
        np.nan_to_num(prof["Te"], nan=0.01, posinf=1e3, neginf=0.01),
        0.01,
    )
    ti_arr = np.maximum(
        np.nan_to_num(prof["Ti"], nan=0.01, posinf=1e3, neginf=0.01),
        0.01,
    )
    ne_arr = np.maximum(
        np.nan_to_num(prof["ne"], nan=0.1, posinf=1e3, neginf=0.1),
        0.1,
    )
    q_arr = np.maximum(
        np.nan_to_num(prof["q"], nan=1.0, posinf=10.0, neginf=0.1),
        0.1,
    )

    r0 = _require_positive_finite_scalar("R0", R0)
    a_minor = _require_positive_finite_scalar("a", a)
    b0 = _require_positive_finite_scalar("B0", B0)
    z_eff = _require_positive_finite_scalar("Z_eff", Z_eff)

    if _rust_transport_available:  # Rust fast-path
        try:
            solver = _PyTransportSolver()
            eps_arr = np.clip(rho_arr * a_minor / r0, 1e-6, 0.999999)
            j_rust = np.asarray(
                solver.sauter_bootstrap_profile(
                    rho_arr, te_arr, ti_arr, ne_arr, q_arr, eps_arr, b0,
                ),
                dtype=np.float64,
            )
            _logger.debug("sauter_bootstrap: Rust fast-path (%d pts)", len(rho_arr))
            return j_rust
        except _RUST_TRANSPORT_FALLBACK_EXCEPTIONS:
            _logger.debug("sauter_bootstrap: Rust fast-path failed, falling back to Python")

    # Vectorized Python fallback
    n = len(rho_arr)
    j_bs = np.zeros(n)
    e_charge = 1.602176634e-19
    m_e = 9.10938370e-31
    eps0 = 8.854187812e-12

    # Interior points only (i=1..n-2) — boundary derivatives undefined
    sl = slice(1, n - 1)
    eps = np.clip(rho_arr[sl] * a_minor / r0, 1e-6, 0.999999)
    valid = (te_arr[sl] > 0) & (ne_arr[sl] > 0) & (q_arr[sl] > 0)

    if not np.any(valid):
        return j_bs

    eps_v = eps[valid]
    te_v = te_arr[sl][valid]
    ti_v = ti_arr[sl][valid]
    ne_v = ne_arr[sl][valid]
    q_v = q_arr[sl][valid]

    # Trapped fraction (Sauter)
    sqrt_trap = np.sqrt(np.maximum(1.0 - eps_v ** 2, 1e-12))
    f_t = 1.0 - (1.0 - eps_v) ** 2 / (sqrt_trap * (1.0 + 1.46 * np.sqrt(eps_v)))
    f_t = np.clip(f_t, 0.0, 1.0)

    T_e_J = te_v * 1e3 * e_charge
    v_te = np.sqrt(2.0 * T_e_J / m_e)
    n_e = ne_v * 1e19
    ln_lambda = 17.0
    nu_ei = n_e * z_eff * e_charge ** 4 * ln_lambda / (
        12.0 * np.pi ** 1.5 * eps0 ** 2 * m_e ** 0.5 * T_e_J ** 1.5
    )
    nu_ei = np.where(np.isfinite(nu_ei) & (nu_ei >= 0.0), nu_ei, 0.0)

    nu_star_e = np.where(
        v_te > 0,
        nu_ei * q_v * r0 / (eps_v ** 1.5 * v_te),
        1e6,
    )
    nu_star_e = np.where(np.isfinite(nu_star_e) & (nu_star_e >= 0.0), nu_star_e, 1e6)

    alpha_31 = 1.0 / (1.0 + 0.36 / z_eff)
    L31 = f_t * alpha_31 / (1.0 + alpha_31 * np.sqrt(nu_star_e) +
           0.25 * nu_star_e * (1.0 - f_t) ** 2)
    L32 = f_t * (0.05 + 0.62 * z_eff) / (z_eff * (1.0 + 0.44 * z_eff))
    L32 /= (1.0 + 0.22 * np.sqrt(nu_star_e) + 0.19 * nu_star_e * (1.0 - f_t))
    L34 = L31 * ti_v / np.maximum(te_v, 0.01)

    # Central-difference gradients at interior points
    idx_valid = np.where(valid)[0]  # indices into the interior array
    i_full = idx_valid + 1  # indices into the full array
    dr = (rho_arr[i_full + 1] - rho_arr[i_full - 1]) * a_minor
    dr_ok = np.abs(dr) >= 1e-12

    dn_dr = np.where(dr_ok, (ne_arr[i_full + 1] - ne_arr[i_full - 1]) * 1e19 / np.where(dr_ok, dr, 1.0), 0.0)
    dTe_dr = np.where(dr_ok, (te_arr[i_full + 1] - te_arr[i_full - 1]) * 1e3 * e_charge / np.where(dr_ok, dr, 1.0), 0.0)
    dTi_dr = np.where(dr_ok, (ti_arr[i_full + 1] - ti_arr[i_full - 1]) * 1e3 * e_charge / np.where(dr_ok, dr, 1.0), 0.0)

    B_pol = b0 * eps_v / np.maximum(q_v, 0.1)
    B_ok = B_pol >= 1e-10

    # Temperature floor: 10 eV
    _T_FLOOR_J = 10.0 * e_charge
    p_e = n_e * T_e_J
    j_val = -(p_e / np.where(B_ok, B_pol, 1.0)) * (
        L31 * dn_dr / np.maximum(n_e, 1e10) +
        L32 * dTe_dr / np.maximum(T_e_J, _T_FLOOR_J) +
        L34 * dTi_dr / np.maximum(ti_v * 1e3 * e_charge, _T_FLOOR_J)
    )

    j_val = np.where(dr_ok & B_ok & np.isfinite(j_val), j_val, 0.0)
    j_bs[i_full] = j_val

    j_bs = np.nan_to_num(j_bs, nan=0.0, posinf=0.0, neginf=0.0)
    j_bs[0] = 0.0
    j_bs[-1] = 0.0
    return j_bs


class TransportSolver(TransportSolverRuntimeMixin, FusionKernel):
    """
    1.5D Integrated Transport Code.
    Solves Heat and Particle diffusion equations on flux surfaces,
    coupled self-consistently with the 2D Grad-Shafranov equilibrium.

    When ``multi_ion=True``, the solver evolves separate D/T fuel densities,
    He-ash transport with pumping (configurable ``tau_He``), independent
    electron temperature Te, coronal-equilibrium tungsten radiation
    (Pütterich et al. 2010), and per-cell Bremsstrahlung.
    """
    def __init__(self, config_path: str | Path, *, multi_ion: bool = False) -> None:
        super().__init__(config_path)
        self.external_profile_mode = True # Tell Kernel to respect our calculated profiles
        self.nr = 50 # Radial grid points (normalized radius rho)
        self.rho = np.linspace(0, 1, self.nr)
        self.drho = 1.0 / (self.nr - 1)

        self.multi_ion: bool = multi_ion

        # PROFILES (Evolving state variables)
        # Te = Electron Temp (keV), Ti = Ion Temp (keV), ne = Density (10^19 m-3)
        self.Te = 1.0 * (1 - self.rho**2) # Initial guess
        self.Ti = 1.0 * (1 - self.rho**2)
        self.ne = 5.0 * (1 - self.rho**2)**0.5

        # Transport Coefficients (Anomalous Transport Models)
        self.chi_e = np.ones(self.nr) # Electron diffusivity
        self.chi_i = np.ones(self.nr) # Ion diffusivity
        self.D_n = np.ones(self.nr)   # Particle diffusivity

        # Impurity Profile (Tungsten density)
        self.n_impurity = np.zeros(self.nr)
        
        # Boundary Condition (Pedestal Top)
        self.T_edge_keV = 0.08
        self.pedestal_model = None
        if self.cfg.get("physics", {}).get("pedestal_mode") == "eped":
            self.pedestal_model = EpedPedestalModel(
                R0=(self.cfg["dimensions"]["R_min"] + self.cfg["dimensions"]["R_max"]) / 2.0,
                a=(self.cfg["dimensions"]["R_max"] - self.cfg["dimensions"]["R_min"]) / 2.0,
                B0=5.3, # Should pull from coils/config but defaulting for now
                Ip_MA=self.cfg.get("physics", {}).get("plasma_current_target", 5.0),
            )

        # Neoclassical transport configuration (None = constant chi_base=0.5)
        self.neoclassical_params: dict[str, Any] | None = None

        # Energy conservation diagnostic (updated each evolve_profiles call)
        self._last_conservation_error: float = 0.0

        # Multi-ion species densities [10^19 m^-3]
        if self.multi_ion:
            self.n_D = 0.5 * self.ne.copy()       # Deuterium
            self.n_T = 0.5 * self.ne.copy()       # Tritium
            self.n_He = np.zeros(self.nr)          # He-4 ash
        else:
            self.n_D = None  # type: ignore[assignment]
            self.n_T = None  # type: ignore[assignment]
            self.n_He = None  # type: ignore[assignment]

        # He-ash pumping time (default 5 * tau_E, ITER design baseline)
        self.tau_He_factor: float = 5.0

        # Particle diffusivity for species transport
        self.D_species: float = 0.3  # m^2/s (typical for ITER)

        # Z_eff tracking (updated every evolve step in multi-ion mode)
        self._Z_eff: float = 1.5

        # Auxiliary-heating source model parameters
        self.aux_heating_profile_width: float = 0.1
        self.aux_heating_electron_fraction: float = 0.5

        # Last-step auxiliary-heating power-balance telemetry
        self._last_aux_heating_balance: dict[str, float] = {
            "target_total_MW": 0.0,
            "target_ion_MW": 0.0,
            "target_electron_MW": 0.0,
            "reconstructed_ion_MW": 0.0,
            "reconstructed_electron_MW": 0.0,
            "reconstructed_total_MW": 0.0,
        }
        self._last_gyro_bohm_contract: dict[str, Any] = {
            "used": False,
            "source": "uninitialized",
            "path": str(_GYRO_BOHM_COEFF_PATH),
            "c_gB": float(_GYRO_BOHM_DEFAULT),
            "fallback_used": False,
            "error": None,
        }
        self._last_pedestal_contract: dict[str, Any] = {
            "used": False,
            "in_domain": True,
            "extrapolation_score": 0.0,
            "extrapolation_penalty": 1.0,
            "domain_violations": [],
            "fallback_used": False,
        }
        self._last_pedestal_bc_contract: dict[str, Any] = {
            "used": False,
            "updated": False,
            "fallback_used": False,
            "in_domain": None,
            "extrapolation_penalty": None,
            "n_ped_1e19": None,
            "t_edge_keV_before": float(self.T_edge_keV),
            "t_edge_keV_after": float(self.T_edge_keV),
            "error": None,
        }

        # Numerical hardening telemetry (non-finite replacements per step)
        self._last_numerical_recovery_count: int = 0
        self._last_numerical_recovery_breakdown: dict[str, int] = {}
        self._last_numerical_recovery_limit: int | None = None

        # Optional hard cap for non-finite/clamped recovery operations per step.
        # None means telemetry-only mode (legacy behavior).
        raw_recovery_cap = self.cfg.get("solver", {}).get("max_numerical_recoveries_per_step")
        if raw_recovery_cap is None:
            self.max_numerical_recoveries_per_step: int | None = None
        else:
            if isinstance(raw_recovery_cap, bool) or int(raw_recovery_cap) < 0:
                raise ValueError(
                    "solver.max_numerical_recoveries_per_step must be a non-negative integer."
                )
            self.max_numerical_recoveries_per_step = int(raw_recovery_cap)

    def set_numerical_recovery_limit(self, max_recoveries: int | None) -> None:
        """Set optional per-step numerical-recovery cap.

        Parameters
        ----------
        max_recoveries : int or None
            Maximum allowed recoveries per evolve step.  ``None`` disables
            budget enforcement and keeps telemetry-only behavior.
        """
        if max_recoveries is None:
            self.max_numerical_recoveries_per_step = None
            return
        if isinstance(max_recoveries, bool) or int(max_recoveries) < 0:
            raise ValueError("max_recoveries must be a non-negative integer or None.")
        self.max_numerical_recoveries_per_step = int(max_recoveries)

    def _record_recovery(self, label: str, count: int) -> None:
        """Track recoveries by category for per-step diagnostics."""
        if count <= 0:
            return
        self._last_numerical_recovery_breakdown[label] = (
            self._last_numerical_recovery_breakdown.get(label, 0) + int(count)
        )

    def _resolve_recovery_limit(self, override: int | None) -> int | None:
        """Resolve and validate an optional per-step recovery cap override."""
        if override is None:
            return self.max_numerical_recoveries_per_step
        if isinstance(override, bool) or int(override) < 0:
            raise ValueError("max_numerical_recoveries must be a non-negative integer or None.")
        return int(override)

    def _enforce_recovery_budget(
        self,
        *,
        enforce_numerical_recovery: bool,
        max_numerical_recoveries: int | None,
    ) -> None:
        """Fail fast when recovery volume exceeds configured hardening budget."""
        limit = self._resolve_recovery_limit(max_numerical_recoveries)
        self._last_numerical_recovery_limit = limit
        if not enforce_numerical_recovery or limit is None:
            return
        if self._last_numerical_recovery_count <= limit:
            return

        details = ", ".join(
            f"{name}={count}" for name, count in sorted(self._last_numerical_recovery_breakdown.items())
        ) or "no breakdown"
        raise PhysicsError(
            "Numerical recovery budget exceeded: "
            f"{self._last_numerical_recovery_count} > {limit}. "
            f"Breakdown: {details}"
        )

    def set_neoclassical(self, R0: float, a: float, B0: float, A_ion: float = 2.0, Z_eff: float = 1.5, q0: float = 1.0, q_edge: float = 3.0) -> None:
        """Configure Chang-Hinton neoclassical transport model.

        When set, update_transport_model uses the Chang-Hinton formula instead
        of the constant chi_base = 0.5.
        """
        r0 = _require_positive_finite_scalar("R0", R0)
        a_minor = _require_positive_finite_scalar("a", a)
        b0 = _require_positive_finite_scalar("B0", B0)
        a_ion = _require_positive_finite_scalar("A_ion", A_ion)
        z_eff = _require_positive_finite_scalar("Z_eff", Z_eff)
        q0_f = _require_positive_finite_scalar("q0", q0)
        q_edge_f = _require_positive_finite_scalar("q_edge", q_edge)

        q_profile = q0_f + (q_edge_f - q0_f) * self.rho**2
        if (not np.all(np.isfinite(q_profile))) or np.any(q_profile <= 0.0):
            raise ValueError("Generated q_profile contains invalid values")
        self.neoclassical_params = {
            'R0': r0, 'a': a_minor, 'B0': b0,
            'A_ion': a_ion, 'Z_eff': z_eff,
            'q_profile': q_profile,
        }

    def chang_hinton_chi_profile(self) -> np.ndarray:
        """Backward-compatible Chang-Hinton profile helper.

        Older parity tests call this no-arg method on a partially-initialized
        transport object. Keep the method as a thin adapter over the module
        function so those tests remain stable.
        """
        rho = np.asarray(getattr(self, "rho"), dtype=np.float64)

        t_i_raw = getattr(self, "t_i", None)
        if t_i_raw is None:
            t_i_raw = getattr(self, "Ti")
        t_i = np.asarray(t_i_raw, dtype=np.float64)

        n_e_raw = getattr(self, "n_e", None)
        if n_e_raw is None:
            n_e_raw = getattr(self, "ne")
        n_e = np.asarray(n_e_raw, dtype=np.float64)
        q_profile = np.asarray(
            getattr(self, "q_profile", np.linspace(1.0, 3.0, len(rho))),
            dtype=np.float64,
        )

        params = getattr(self, "neoclassical_params", None)
        if not isinstance(params, dict):
            params = {}
        R0 = float(params.get("R0", 6.2))
        a = float(params.get("a", 2.0))
        B0 = float(params.get("B0", 5.3))
        A_ion = float(params.get("A_ion", 2.0))
        Z_eff = float(params.get("Z_eff", 1.5))

        if q_profile.shape != rho.shape:
            q_profile = np.linspace(1.0, 3.0, len(rho), dtype=np.float64)

        return chang_hinton_chi_profile(
            rho, t_i, n_e, q_profile, R0, a, B0, A_ion=A_ion, Z_eff=Z_eff
        )

    def inject_impurities(self, flux_from_wall_per_sec, dt):
        """
        Models impurity influx from PWI erosion.
        Simple diffusion model: Source at edge, diffuses inward.
        """
        # Source at edge (last grid point)
        # Flux is total particles. Volume of edge shell approx 20 m3.
        # Delta_n = Flux * dt / Vol_edge
        # Scaling factor adjusted for simulation stability
        d_n_edge = (flux_from_wall_per_sec * dt) / 20.0 * 1e-18 
        
        # Add to edge
        self.n_impurity[-1] += d_n_edge
        
        # Diffuse inward (Explicit step)
        D_imp = 1.0 # m2/s
        new_imp = self.n_impurity.copy()
        
        grad = np.gradient(self.n_impurity, self.drho)
        flux = -D_imp * grad
        div = np.gradient(flux, self.drho) / (self.rho + 1e-6)
        
        new_imp += (-div) * dt
        
        # Boundary
        new_imp[0] = new_imp[1] # Axis symmetry
        
        self.n_impurity = np.maximum(0, new_imp)

    def _evolve_impurity(self, dt: float) -> None:
        """Autonomous impurity evolution with edge source and diffusion.

        Models a small constant edge influx (sputtering) and diffuses
        inward.  Updates ``self.n_impurity`` in place and recomputes
        ``self._Z_eff`` for non-multi-ion feedback.
        """
        # Edge source: small constant sputtering influx [10^19 m^-3 / s]
        edge_source_rate = 0.01  # modest influx
        self.n_impurity[-1] += edge_source_rate * dt

        # Diffuse inward (explicit step, same scheme as inject_impurities)
        D_imp = 1.0  # m^2/s
        grad = np.gradient(self.n_impurity, self.drho)
        flux = -D_imp * grad
        div = np.gradient(flux, self.drho) / (self.rho + 1e-6)
        self.n_impurity += (-div) * dt

        # Boundary conditions
        self.n_impurity[0] = self.n_impurity[1]  # axis symmetry
        self.n_impurity = np.maximum(0.0, self.n_impurity)

        # Update Z_eff (non-multi-ion path: light impurity model)
        # Assume carbon-like impurity with Z_imp ~ 6
        Z_imp = 6.0
        ne_safe = np.maximum(self.ne, 0.1) * 1e19
        n_imp_m3 = self.n_impurity * 1e19
        # Z_eff = (n_e * 1 + n_imp * Z^2) / (n_e + n_imp * Z)  (reduced-order closure)
        sum_nZ2 = ne_safe + n_imp_m3 * Z_imp ** 2
        sum_nZ = ne_safe + n_imp_m3 * Z_imp
        self._Z_eff = float(np.clip(np.mean(sum_nZ2 / np.maximum(sum_nZ, 1e10)), 1.0, 10.0))

    def calculate_bootstrap_current_simple(self, R0, B_pol):
        """
        Calculates the neoclassical bootstrap current density [A/m2]
        using a calibrated-heuristic Sauter model.
        J_bs = - (R/B_pol) * [ L31 * dP/dpsi + L32 * n*dTe/dpsi + L34 * n*dTi/dpsi ]
        """
        a = (self.cfg["dimensions"]["R_max"] - self.cfg["dimensions"]["R_min"]) / 2.0
        r = self.rho * a
        epsilon = r / R0
        
        # Trapped fraction (approximate)
        f_trapped = 1.46 * np.sqrt(epsilon)
        
        # Gradients (SI units: J/m3 per m)
        e_charge = 1.602e-16 # keV to J
        n_e = self.ne * 1e19
        dn_dr = np.gradient(n_e, self.drho * a)
        dTe_dr = np.gradient(self.Te * e_charge, self.drho * a)
        dTi_dr = np.gradient(self.Ti * e_charge, self.drho * a)

        # Effective B_pol for toroidal geometry
        B_pol = np.maximum(B_pol, 0.1) 

        # Sauter coefficients approximations (L-mode, low collisionality)
        L31 = f_trapped / (1.0 + 0.3 * np.sqrt(epsilon))
        L32 = 0.5 * L31
        zeff_eff = float(np.clip(getattr(self, "_Z_eff", 1.5), 1.0, 5.0))
        L34 = -0.1 * L31 * (1.0 + 0.08 * (zeff_eff - 1.0))  # bounded temperature-gradient coupling

        # J_bs ~ -(1/B_pol) * [ L31 * (Te+Ti) * dn/dr + L32 * n * dTe/dr + L34 * n * dTi/dr ]
        J_bs = -(1.0 / B_pol) * (
            L31 * (self.Te + self.Ti) * e_charge * dn_dr +
            L32 * n_e * dTe_dr +
            L34 * n_e * dTi_dr
        )

        # Scale to match global empirical observations (calibration factor)
        J_bs *= 1.4 

        # Ensure it's zero at axis and edge
        J_bs[0] = 0
        J_bs[-1] = 0

        return J_bs

    def calculate_bootstrap_current(self, R0, B_pol):
        """Calculate bootstrap current. Uses full Sauter if neoclassical params set."""
        if hasattr(self, 'neoclassical_params') and self.neoclassical_params is not None:
            return calculate_sauter_bootstrap_current_full(
                self.rho, self.Te, self.Ti, self.ne,
                self.neoclassical_params.get('q_profile', np.linspace(1, 4, len(self.rho))),
                R0, self.neoclassical_params.get('a', 2.0),
                self.neoclassical_params.get('B0', 5.3),
                self.neoclassical_params.get('Z_eff', 1.5),
            )
        return self.calculate_bootstrap_current_simple(R0, B_pol)

    def _gyro_bohm_chi(self) -> np.ndarray:
        """Gyro-Bohm anomalous transport diffusivity [m^2/s].

        chi_gB = c_gB * rho_s^2 * c_s / (a * q * R)

        where rho_s = sqrt(T_i m_i) / (e B), c_s = sqrt(T_e / m_i).

        The calibration coefficient c_gB is loaded from
        ``validation/reference_data/itpa/gyro_bohm_coefficients.json``
        if available (calibrated against the ITPA H-mode confinement
        database by ``tools/calibrate_gyro_bohm.py``).  Falls back to
        the value in ``neoclassical_params['c_gB']`` if explicitly set,
        or to the module-level default (0.1) otherwise.
        """
        if self.neoclassical_params is None:
            self._last_gyro_bohm_contract = {
                "used": False,
                "source": "neoclassical_disabled",
                "path": str(_GYRO_BOHM_COEFF_PATH),
                "c_gB": float(_GYRO_BOHM_DEFAULT),
                "fallback_used": True,
                "error": "neoclassical_params_missing",
            }
            return np.full_like(self.rho, 0.5)

        p = self.neoclassical_params
        R0 = p['R0']
        a = p['a']
        B0 = p['B0']
        A_ion = p.get('A_ion', 2.0)
        q = p['q_profile']

        # Load c_gB: explicit param > cached JSON file > default
        if 'c_gB' in p:
            try:
                c_gB = float(p['c_gB'])
                if (not np.isfinite(c_gB)) or c_gB <= 0.0:
                    raise ValueError(f"Invalid c_gB={p['c_gB']!r}")
                self._last_gyro_bohm_contract = {
                    "used": True,
                    "source": "neoclassical_params",
                    "path": None,
                    "c_gB": float(c_gB),
                    "fallback_used": False,
                    "error": None,
                }
            except (TypeError, ValueError) as exc:
                c_gB, fallback_contract = _load_gyro_bohm_coefficient_cached_with_contract()
                self._last_gyro_bohm_contract = {
                    "used": True,
                    "source": "neoclassical_params_invalid_fallback",
                    "path": fallback_contract.get("path"),
                    "c_gB": float(c_gB),
                    "fallback_used": True,
                    "error": f"{exc.__class__.__name__}:{exc}",
                    "requested_c_gB": p.get("c_gB"),
                    "fallback_source": fallback_contract.get("source"),
                }
        else:
            c_gB, loader_contract = _load_gyro_bohm_coefficient_cached_with_contract()
            self._last_gyro_bohm_contract = {
                "used": True,
                "source": loader_contract.get("source", "json_file"),
                "path": loader_contract.get("path"),
                "c_gB": float(c_gB),
                "fallback_used": bool(loader_contract.get("fallback_used", False)),
                "error": loader_contract.get("error"),
            }

        e_charge = 1.602176634e-19
        m_i = A_ion * 1.672621924e-27

        # Vectorised gyro-Bohm chi (replaces per-point Python for-loop)
        Ti_keV = np.maximum(self.Ti, 0.01)
        Te_keV = np.maximum(self.Te, 0.01)
        qi = np.maximum(q, 0.5)

        T_i_J = Ti_keV * 1e3 * e_charge
        T_e_J = Te_keV * 1e3 * e_charge

        rho_s = np.sqrt(T_i_J * m_i) / (e_charge * B0)
        c_s = np.sqrt(T_e_J / m_i)

        denom = np.maximum(a * qi * R0, 1e-6)
        chi_gB = c_gB * rho_s**2 * c_s / denom
        chi_gB = np.where(np.isfinite(chi_gB), np.maximum(chi_gB, 0.01), 0.01)

        return chi_gB

    def update_transport_model(self, P_aux):
        """
        Gyro-Bohm + neoclassical transport model with EPED-like pedestal.

        When neoclassical params are set, uses:
        - Chang-Hinton neoclassical chi as additive floor
        - Gyro-Bohm anomalous transport (calibrated c_gB)
        - EPED-like pedestal model for H-mode boundary condition

        Falls back to constant chi_base=0.5 when neoclassical is not configured.
        """
        grad_T = np.gradient(self.Ti, self.drho)
        threshold = 2.0

        # Base Level: neoclassical + gyro-Bohm, or constant fallback
        if self.neoclassical_params is not None:
            p = self.neoclassical_params
            chi_nc = chang_hinton_chi_profile(
                self.rho, self.Ti, self.ne,
                p['q_profile'], p['R0'], p['a'], p['B0'],
                p['A_ion'], p['Z_eff']
            )
            chi_gB = self._gyro_bohm_chi()
            chi_base = chi_nc + chi_gB
        else:
            chi_base = np.full_like(self.rho, 0.5)

        # Turbulent Level (critical gradient excess)
        chi_turb = 5.0 * np.maximum(0, -grad_T - threshold)

        # H-Mode detection and EPED-like pedestal model
        is_H_mode = P_aux > 30.0  # MW
        self._last_pedestal_contract = {
            "used": False,
            "in_domain": True,
            "extrapolation_score": 0.0,
            "extrapolation_penalty": 1.0,
            "domain_violations": [],
            "fallback_used": False,
        }

        if is_H_mode and self.neoclassical_params is not None:
            try:
                from scpn_fusion.core.eped_pedestal import EpedPedestalModel

                p = self.neoclassical_params
                # Reuse cached pedestal model if parameters match; avoids
                # re-instantiation every H-mode transport step.
                if self.pedestal_model is None or getattr(
                    self.pedestal_model, '_neo_params_hash', None
                ) != id(self.neoclassical_params):
                    eped = EpedPedestalModel(
                        R0=p['R0'], a=p['a'], B0=p['B0'],
                        Ip_MA=p.get('Ip_MA', 15.0),
                        kappa=p.get('kappa', 1.7),
                        A_ion=p.get('A_ion', 2.0),
                        Z_eff=p.get('Z_eff', 1.5),
                    )
                    eped._neo_params_hash = id(self.neoclassical_params)  # type: ignore[attr-defined]
                    self.pedestal_model = eped
                else:
                    eped = self.pedestal_model
                # Use current edge density for pedestal prediction
                n_ped = max(float(self.ne[-5]), 1.0)
                ped = eped.predict(n_ped)
                self._last_pedestal_contract = {
                    "used": True,
                    "in_domain": bool(ped.in_domain),
                    "extrapolation_score": float(ped.extrapolation_score),
                    "extrapolation_penalty": float(ped.extrapolation_penalty),
                    "domain_violations": list(ped.domain_violations),
                    "fallback_used": False,
                    "n_ped_1e19": float(n_ped),
                }

                # Apply pedestal: suppress transport inside pedestal region
                ped_start = 1.0 - ped.Delta_ped
                edge_mask = self.rho > ped_start
                chi_turb[edge_mask] *= 0.05  # Strong transport barrier

                # Set pedestal boundary conditions on profiles
                ped_idx = np.searchsorted(self.rho, ped_start)
                if ped_idx < len(self.Te):
                    self.Te[ped_idx:] = np.minimum(
                        self.Te[ped_idx:],
                        ped.T_ped_keV * np.linspace(1.0, 0.1, len(self.Te[ped_idx:]))
                    )
                    self.Ti[ped_idx:] = np.minimum(
                        self.Ti[ped_idx:],
                        ped.T_ped_keV * np.linspace(1.0, 0.1, len(self.Ti[ped_idx:]))
                    )
            except _EPED_FALLBACK_EXCEPTIONS as exc:
                # Fallback: simple edge suppression
                edge_mask = self.rho > 0.9
                chi_turb[edge_mask] *= 0.1
                self._last_pedestal_contract = {
                    "used": False,
                    "in_domain": False,
                    "extrapolation_score": 0.0,
                    "extrapolation_penalty": 1.0,
                    "domain_violations": [f"eped_failure:{exc}"],
                    "fallback_used": True,
                }
        elif is_H_mode:
            # No neoclassical params — simple suppression
            edge_mask = self.rho > 0.9
            chi_turb[edge_mask] *= 0.1
            self._last_pedestal_contract = {
                "used": False,
                "in_domain": False,
                "extrapolation_score": 0.0,
                "extrapolation_penalty": 1.0,
                "domain_violations": ["neoclassical_params_missing"],
                "fallback_used": True,
            }

        self.chi_e = chi_base + chi_turb
        self.chi_i = chi_base + chi_turb
        self.D_n = 0.1 * self.chi_e

IntegratedTransportSolver = TransportSolver
