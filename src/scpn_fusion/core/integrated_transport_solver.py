# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Integrated Transport Solver
"""Top-level integrated transport solver orchestration and factory helpers.

This module wires together transport-model and runtime mixins into the public
``IntegratedTransportSolver`` interface. It also exposes conservative fallback
paths (Python implementations when Rust bindings are unavailable) and preserves
legacy compatibility symbols required by downstream callers.

Docstring scope in this module is intentionally high-level because most detailed
numerical contracts are documented in the extracted model/runtime mixins.
"""

from __future__ import annotations

import json
import logging
import numpy as np
import numpy.typing as npt
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from scpn_fusion.exceptions import FusionCoreError as _FusionCoreError
from scpn_fusion.io.safe_loaders import checked_json_load

# `_rust_compat` resolves FusionKernel through the canonical multi-backend
# dispatcher (Rust when available, NumPy otherwise) and always imports cleanly.
from scpn_fusion.core._rust_compat import FusionKernel

from scpn_fusion.core.integrated_transport_solver_adaptive import (
    AdaptiveTimeController,  # re-export for backward compatibility
)
from scpn_fusion.core.integrated_transport_solver_contracts import (
    coerce_matching_1d_profiles as _coerce_matching_1d_profiles,
    require_positive_finite_scalar as _require_positive_finite_scalar,
    rust_chang_hinton_params_match_defaults as _rust_chang_hinton_params_match_defaults,
)
from scpn_fusion.core._integrated_transport_solver_init import (
    TransportSolverInitializationMixin,
)
from scpn_fusion.core.integrated_transport_solver_model import (
    TransportSolverModelMixin,
)
from scpn_fusion.core.integrated_transport_solver_runtime import (
    TransportSolverRuntimeMixin,
)

_logger = logging.getLogger(__name__)

__all__ = [
    "AdaptiveTimeController",
    "PhysicsError",
    "TransportSolver",
    "IntegratedTransportSolver",
    "_load_gyro_bohm_coefficient",
    "chang_hinton_chi_profile",
    "calculate_sauter_bootstrap_current_full",
]

_rust_transport_available = False
_PyTransportSolver: Any = None
try:
    from scpn_fusion.core._multi_compat import dispatch_rust_symbol

    _PyTransportSolver = dispatch_rust_symbol("PyTransportSolver")
    _rust_transport_available = True
except (ImportError, AttributeError, TypeError) as exc:
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
FloatArray = npt.NDArray[np.float64]


class PhysicsError(RuntimeError, _FusionCoreError):
    """Raised when a physics constraint is violated."""


_GYRO_BOHM_COEFF_PATH = (
    Path(__file__).resolve().parents[3]
    / "validation"
    / "reference_data"
    / "itpa"
    / "gyro_bohm_coefficients.json"
)

_GYRO_BOHM_DEFAULT = 0.1  # Compatibility default if JSON is unavailable


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
        data = checked_json_load(p)
        c_gB = float(data["c_gB"])
        if (not np.isfinite(c_gB)) or c_gB <= 0.0:
            raise ValueError(f"Invalid c_gB={c_gB!r}")
        _logger.debug("Loaded c_gB = %.6f from %s", c_gB, p)
        return c_gB, contract
    except (FileNotFoundError, KeyError, json.JSONDecodeError, TypeError, ValueError) as exc:
        _logger.warning(
            "Could not load c_gB from %s (%s), using default %.4f",
            p,
            exc,
            _GYRO_BOHM_DEFAULT,
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
    """Return the cached c_gB coefficient with its provenance contract."""
    global _gyro_bohm_cache  # noqa: PLW0603
    global _gyro_bohm_cache_contract  # noqa: PLW0603
    if _gyro_bohm_cache is None or _gyro_bohm_cache_contract is None:
        _gyro_bohm_cache, _gyro_bohm_cache_contract = _load_gyro_bohm_coefficient_with_contract()
    return float(_gyro_bohm_cache), dict(_gyro_bohm_cache_contract)


def chang_hinton_chi_profile(
    rho: npt.ArrayLike,
    T_i: npt.ArrayLike,
    n_e_19: npt.ArrayLike,
    q: npt.ArrayLike,
    R0: float,
    a: float,
    B0: float,
    A_ion: float = 2.0,
    Z_eff: float = 1.5,
) -> FloatArray:
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
    # NRL Plasma Formulary: ln_Lambda = 17.7 + ln(T_keV/10) - 0.5*ln(n_e/1e20)
    T_keV = np.maximum(T_i_arr[valid], 0.01)
    ln_lambda = np.clip(
        17.7 + np.log(T_keV / 10.0) - 0.5 * np.log(np.maximum(n_e, 1e10) / 1e20), 10.0, 25.0
    )
    nu_ii = (
        n_e
        * z_eff**2
        * e_charge**4
        * ln_lambda
        / (12.0 * np.pi**1.5 * eps0**2 * m_i**0.5 * T_J**1.5)
    )

    eps32 = eps_v**1.5
    nu_star = np.maximum(nu_ii * q_arr[valid] * r0 / (eps32 * v_ti), 0.0)

    chi_val = (
        0.66
        * (1.0 + 1.54 * eps_v)
        * q_arr[valid] ** 2
        * rho_i**2
        * nu_ii
        / (eps32 * (1.0 + 0.74 * nu_star ** (2.0 / 3.0)))
    )

    chi_val = np.where(np.isfinite(chi_val), np.maximum(chi_val, 0.01), 0.01)
    chi_nc[valid] = chi_val

    return chi_nc


def calculate_sauter_bootstrap_current_full(
    rho: npt.ArrayLike,
    Te: npt.ArrayLike,
    Ti: npt.ArrayLike,
    ne: npt.ArrayLike,
    q: npt.ArrayLike,
    R0: float,
    a: float,
    B0: float,
    Z_eff: float = 1.5,
) -> FloatArray:
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
                    rho_arr,
                    te_arr,
                    ti_arr,
                    ne_arr,
                    q_arr,
                    eps_arr,
                    b0,
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
    sqrt_trap = np.sqrt(np.maximum(1.0 - eps_v**2, 1e-12))
    f_t = 1.0 - (1.0 - eps_v) ** 2 / (sqrt_trap * (1.0 + 1.46 * np.sqrt(eps_v)))
    f_t = np.clip(f_t, 0.0, 1.0)

    T_e_J = te_v * 1e3 * e_charge
    v_te = np.sqrt(2.0 * T_e_J / m_e)
    n_e = ne_v * 1e19
    ln_lambda = np.clip(
        17.7 + np.log(np.maximum(te_v, 0.01) / 10.0) - 0.5 * np.log(np.maximum(n_e, 1e10) / 1e20),
        10.0,
        25.0,
    )
    nu_ei = (
        n_e
        * z_eff
        * e_charge**4
        * ln_lambda
        / (12.0 * np.pi**1.5 * eps0**2 * m_e**0.5 * T_e_J**1.5)
    )
    nu_ei = np.where(np.isfinite(nu_ei) & (nu_ei >= 0.0), nu_ei, 0.0)

    nu_star_e = np.where(
        v_te > 0,
        nu_ei * q_v * r0 / (eps_v**1.5 * v_te),
        1e6,
    )
    nu_star_e = np.where(np.isfinite(nu_star_e) & (nu_star_e >= 0.0), nu_star_e, 1e6)

    alpha_31 = 1.0 / (1.0 + 0.36 / z_eff)
    L31 = (
        f_t * alpha_31 / (1.0 + alpha_31 * np.sqrt(nu_star_e) + 0.25 * nu_star_e * (1.0 - f_t) ** 2)
    )
    L32 = f_t * (0.05 + 0.62 * z_eff) / (z_eff * (1.0 + 0.44 * z_eff))
    L32 /= 1.0 + 0.22 * np.sqrt(nu_star_e) + 0.19 * nu_star_e * (1.0 - f_t)
    L34 = L31 * ti_v / np.maximum(te_v, 0.01)

    # Central-difference gradients at interior points
    idx_valid = np.where(valid)[0]  # indices into the interior array
    i_full = idx_valid + 1  # indices into the full array
    dr = (rho_arr[i_full + 1] - rho_arr[i_full - 1]) * a_minor
    dr_ok = np.abs(dr) >= 1e-12

    dn_dr = np.where(
        dr_ok, (ne_arr[i_full + 1] - ne_arr[i_full - 1]) * 1e19 / np.where(dr_ok, dr, 1.0), 0.0
    )
    dTe_dr = np.where(
        dr_ok,
        (te_arr[i_full + 1] - te_arr[i_full - 1]) * 1e3 * e_charge / np.where(dr_ok, dr, 1.0),
        0.0,
    )
    dTi_dr = np.where(
        dr_ok,
        (ti_arr[i_full + 1] - ti_arr[i_full - 1]) * 1e3 * e_charge / np.where(dr_ok, dr, 1.0),
        0.0,
    )

    B_pol = b0 * eps_v / np.maximum(q_v, 0.1)
    B_ok = B_pol >= 1e-10

    # Temperature floor: 10 eV
    _T_FLOOR_J = 10.0 * e_charge
    p_e = n_e * T_e_J
    j_val = -(p_e / np.where(B_ok, B_pol, 1.0)) * (
        L31 * dn_dr / np.maximum(n_e, 1e10)
        + L32 * dTe_dr / np.maximum(T_e_J, _T_FLOOR_J)
        + L34 * dTi_dr / np.maximum(ti_v * 1e3 * e_charge, _T_FLOOR_J)
    )

    j_val = np.where(dr_ok & B_ok & np.isfinite(j_val), j_val, 0.0)
    j_bs[i_full] = j_val

    j_bs[:] = np.nan_to_num(j_bs, nan=0.0, posinf=0.0, neginf=0.0)
    j_bs[0] = 0.0
    j_bs[-1] = 0.0
    return j_bs


class TransportSolver(
    TransportSolverInitializationMixin,
    TransportSolverModelMixin,
    TransportSolverRuntimeMixin,
    FusionKernel,
):
    """1.5D Integrated Transport Code.

    Solves Heat and Particle diffusion equations on flux surfaces,
    coupled self-consistently with the 2D Grad-Shafranov equilibrium.

    When ``multi_ion=True``, the solver evolves separate D/T fuel densities,
    He-ash transport with pumping (configurable ``tau_He``), independent
    electron temperature Te, coronal-equilibrium tungsten radiation
    (Pütterich et al. 2010), and per-cell Bremsstrahlung.
    """

    def __init__(self, config_path: str | Path, *, multi_ion: bool = False) -> None:
        """Build a transport solver with deterministic defaults and requested mode."""
        fusion_kernel_init = cast(Callable[[Any, str | Path], None], FusionKernel.__init__)
        fusion_kernel_init(self, config_path)
        self._initialize_transport_solver_state(multi_ion=multi_ion)

    # Model/configuration methods are provided by TransportSolverModelMixin.


IntegratedTransportSolver = TransportSolver
