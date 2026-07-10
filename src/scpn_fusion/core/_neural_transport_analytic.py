# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Analytic reduced gyrokinetic closure for neural transport fallback."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._neural_transport_types import FloatArray, TransportFluxes, TransportInputs
from .neural_transport_math import _compute_nustar


# Critical gradient thresholds (Dimits shift included)
_CRIT_ITG = 4.0
_CRIT_TEM = 5.0
_CRIT_ETG = 12.0
_CHI_GB = 1.0

# Transport stiffness exponent. Physical range 1.5-4.0 (Dimits PoP 2000,
# Citrin NF 2015); values outside [1.0, 6.0] are non-physical.
_STIFFNESS = 2.0
_STIFFNESS_MIN = 1.0
_STIFFNESS_MAX = 6.0
_TRANSPORT_FLOOR = 1e-6


def _gyro_bohm_diffusivity(inp: TransportInputs) -> float:
    """Estimate local gyro-Bohm diffusivity from reduced geometry inputs."""
    e_charge = 1.602176634e-19
    m_i = 2.0 * 1.672621924e-27
    te_kev = max(float(inp.te_kev), 0.01)
    b_t = max(float(inp.b_tesla), 0.1)
    r_major = max(float(inp.r_major_m), 0.1)

    te_j = te_kev * 1e3 * e_charge
    cs = np.sqrt(te_j / m_i)
    rho_s = np.sqrt(m_i * te_j) / (e_charge * b_t)
    chi_gb = rho_s**2 * cs / r_major
    if not np.isfinite(chi_gb):
        return _TRANSPORT_FLOOR
    return float(max(chi_gb, _TRANSPORT_FLOOR))


def _dominant_channel(
    *,
    chi_i_itg: float,
    chi_e_itg: float,
    chi_e_tem: float,
    chi_e_etg: float,
) -> str:
    """Return the dominant reduced transport channel."""
    strengths = {
        "ITG": float(chi_i_itg + chi_e_itg),
        "TEM": float(chi_e_tem),
        "ETG": float(chi_e_etg),
    }
    name, value = max(strengths.items(), key=lambda item: item[1])
    return name if value > 0.0 else "stable"


def critical_gradient_model(
    inp: TransportInputs,
    *,
    stiffness: float = _STIFFNESS,
) -> TransportFluxes:
    """Reduced multichannel gyrokinetic closure used as analytic fallback."""
    if not (_STIFFNESS_MIN <= stiffness <= _STIFFNESS_MAX):
        raise ValueError(
            f"stiffness={stiffness} outside physical range [{_STIFFNESS_MIN}, {_STIFFNESS_MAX}]"
        )
    eps = float(np.clip(inp.rho * inp.a_minor_m / max(inp.r_major_m, 1e-6), 0.0, 0.8))
    trapped_frac = float(np.clip(1.46 * np.sqrt(max(eps, 0.0)), 0.0, 1.0))
    nustar = float(
        _compute_nustar(
            inp.te_kev,
            inp.ne_19,
            inp.q,
            inp.rho,
            inp.r_major_m,
            inp.a_minor_m,
            inp.z_eff,
        )
    )
    chi_gb = _gyro_bohm_diffusivity(inp)
    shear_supp = 1.0 / (1.0 + 0.35 * max(inp.s_hat, 0.0) ** 2)
    beta_supp = 1.0 / (1.0 + max(inp.beta_e, 0.0) / 0.03)
    electron_ratio = float(np.clip(inp.te_kev / max(inp.ti_kev, 0.05), 0.5, 4.0))

    crit_itg = _CRIT_ITG + 0.4 * max(inp.s_hat, 0.0) + 8.0 * max(inp.beta_e, 0.0)
    density_excess = max(inp.grad_ne - 2.5, 0.0)
    crit_tem = max(
        2.5,
        5.0 + 1.1 * eps + 0.12 * min(max(nustar, 0.0), 10.0) - 0.35 * density_excess,
    )
    crit_etg = 10.5 + 1.0 * eps + 0.3 * max(inp.s_hat, 0.0) + 0.2 * max(nustar, 0.0)

    excess_itg = max(0.0, inp.grad_ti - crit_itg)
    excess_tem = max(0.0, inp.grad_te - crit_tem)
    excess_etg = max(0.0, inp.grad_te - crit_etg)

    chi_i_itg = chi_gb * excess_itg**stiffness * shear_supp * beta_supp
    chi_e_itg = 0.35 * chi_i_itg

    collisional_tem = 1.0 / (1.0 + 0.8 * max(nustar, 0.0))
    density_drive = 0.15 + 0.35 * density_excess
    chi_e_tem = (
        chi_gb * excess_tem**stiffness * trapped_frac * collisional_tem * beta_supp * density_drive
    )

    collisional_etg = 1.0 / (1.0 + 1.5 * max(nustar, 0.0))
    etg_shear = 1.0 / (1.0 + 0.2 * max(inp.s_hat, 0.0) ** 2)
    electron_gradient_split = 1.0 + 0.18 * max(inp.grad_te - inp.grad_ti, 0.0)
    chi_e_etg = (
        0.85
        * chi_gb
        * excess_etg ** (0.9 * stiffness)
        * collisional_etg
        * etg_shear
        * electron_ratio
        * electron_gradient_split
    )

    chi_i = max(chi_i_itg, 0.0)
    chi_e = max(chi_e_itg + chi_e_tem + chi_e_etg, 0.0)
    d_e = chi_e * (0.1 + 0.5 * np.sqrt(max(eps, 0.0)))
    channel = _dominant_channel(
        chi_i_itg=chi_i_itg,
        chi_e_itg=chi_e_itg,
        chi_e_tem=chi_e_tem,
        chi_e_etg=chi_e_etg,
    )

    return TransportFluxes(
        chi_e=chi_e,
        chi_i=chi_i,
        d_e=d_e,
        channel=channel,
        chi_e_itg=max(chi_e_itg, 0.0),
        chi_e_tem=max(chi_e_tem, 0.0),
        chi_e_etg=max(chi_e_etg, 0.0),
        chi_i_itg=max(chi_i_itg, 0.0),
    )


def reduced_gyrokinetic_profile_model(
    rho: FloatArray,
    te: FloatArray,
    ti: FloatArray,
    ne: FloatArray,
    q_profile: FloatArray,
    s_hat_profile: FloatArray,
    *,
    r_major: float = 6.2,
    a_minor: float = 2.0,
    b_toroidal: float = 5.3,
) -> tuple[FloatArray, FloatArray, FloatArray, dict[str, Any]]:
    """Evaluate the reduced ITG/TEM/ETG closure across a radial profile."""
    rho = np.asarray(rho, dtype=np.float64)
    te = np.asarray(te, dtype=np.float64)
    ti = np.asarray(ti, dtype=np.float64)
    ne = np.asarray(ne, dtype=np.float64)
    q_profile = np.asarray(q_profile, dtype=np.float64)
    s_hat_profile = np.asarray(s_hat_profile, dtype=np.float64)

    if any(arr.ndim != 1 for arr in (rho, te, ti, ne, q_profile, s_hat_profile)):
        raise ValueError("rho/te/ti/ne/q_profile/s_hat_profile must all be 1D arrays.")
    n = int(rho.size)
    if n < 3:
        raise ValueError("profile arrays must contain at least 3 points.")
    if any(arr.size != n for arr in (te, ti, ne, q_profile, s_hat_profile)):
        raise ValueError("profile arrays must all have identical length.")
    if not np.all(np.isfinite(rho)):
        raise ValueError("rho must contain finite values.")
    if not np.all(np.isfinite(te)):
        raise ValueError("te must contain finite values.")
    if not np.all(np.isfinite(ti)):
        raise ValueError("ti must contain finite values.")
    if not np.all(np.isfinite(ne)):
        raise ValueError("ne must contain finite values.")
    if not np.all(np.isfinite(q_profile)):
        raise ValueError("q_profile must contain finite values.")
    if not np.all(np.isfinite(s_hat_profile)):
        raise ValueError("s_hat_profile must contain finite values.")
    if not np.all(np.diff(rho) > 0.0):
        raise ValueError("rho must be strictly increasing.")
    if rho[0] < 0.0 or rho[-1] > 1.2:
        raise ValueError("rho must satisfy 0 <= rho <= 1.2.")

    r_major = float(r_major)
    a_minor = float(a_minor)
    b_toroidal = float(b_toroidal)
    if (not np.isfinite(r_major)) or r_major <= 0.0:
        raise ValueError("r_major must be finite and > 0.")
    if (not np.isfinite(a_minor)) or a_minor <= 0.0:
        raise ValueError("a_minor must be finite and > 0.")
    if (not np.isfinite(b_toroidal)) or b_toroidal <= 0.0:
        raise ValueError("b_toroidal must be finite and > 0.")

    def norm_grad(x: FloatArray) -> FloatArray:
        dx = np.gradient(x, rho)
        safe_x = np.maximum(np.abs(x), 1e-6)
        return np.asarray(-r_major * dx / safe_x, dtype=np.float64)

    grad_te_raw = norm_grad(te)
    grad_ti_raw = norm_grad(ti)
    grad_ne_raw = norm_grad(ne)
    grad_te = np.clip(grad_te_raw, 0.0, 50.0)
    grad_ti = np.clip(grad_ti_raw, 0.0, 50.0)
    grad_ne = np.clip(grad_ne_raw, -10.0, 30.0)
    beta_e = 4.03e-3 * ne * te

    fluxes = [
        critical_gradient_model(
            TransportInputs(
                rho=float(rho[i]),
                te_kev=float(te[i]),
                ti_kev=float(ti[i]),
                ne_19=float(ne[i]),
                grad_te=float(grad_te[i]),
                grad_ti=float(grad_ti[i]),
                grad_ne=float(grad_ne[i]),
                q=float(q_profile[i]),
                s_hat=float(s_hat_profile[i]),
                beta_e=float(beta_e[i]),
                r_major_m=r_major,
                a_minor_m=a_minor,
                b_tesla=b_toroidal,
            )
        )
        for i in range(n)
    ]

    chi_e = np.array([f.chi_e for f in fluxes], dtype=np.float64)
    chi_i = np.array([f.chi_i for f in fluxes], dtype=np.float64)
    d_e = np.array([f.d_e for f in fluxes], dtype=np.float64)
    dominant_channels = [f.channel for f in fluxes]

    channel_energy = {
        "ITG": float(np.sum([f.chi_i_itg + f.chi_e_itg for f in fluxes])),
        "TEM": float(np.sum([f.chi_e_tem for f in fluxes])),
        "ETG": float(np.sum([f.chi_e_etg for f in fluxes])),
    }
    dominant_channel = max(channel_energy.items(), key=lambda item: item[1])[0]
    if channel_energy[dominant_channel] <= 0.0:
        dominant_channel = "stable"

    metadata: dict[str, Any] = {
        "model": "reduced_multichannel_analytic",
        "dominant_channel": dominant_channel,
        "channel_counts": {
            name: int(sum(ch == name for ch in dominant_channels))
            for name in ("ITG", "TEM", "ETG", "stable")
        },
        "channel_energy": channel_energy,
        "gradient_clip_counts": {
            "grad_te": int(np.count_nonzero((grad_te_raw < 0.0) | (grad_te_raw > 50.0))),
            "grad_ti": int(np.count_nonzero((grad_ti_raw < 0.0) | (grad_ti_raw > 50.0))),
            "grad_ne": int(np.count_nonzero((grad_ne_raw < -10.0) | (grad_ne_raw > 30.0))),
        },
        "profile_contract": {
            "n_points": int(n),
            "rho_min": float(rho[0]),
            "rho_max": float(rho[-1]),
            "r_major": r_major,
            "a_minor": a_minor,
            "b_toroidal": b_toroidal,
        },
        "edge_etg_fraction": (
            float(
                np.mean(
                    [
                        1.0 if ch == "ETG" else 0.0
                        for ch, r in zip(dominant_channels, rho)
                        if r >= 0.8
                    ]
                )
            )
            if np.any(rho >= 0.8)
            else 0.0
        ),
    }
    return chi_e, chi_i, d_e, metadata


__all__ = [
    "_CHI_GB",
    "_CRIT_ETG",
    "_CRIT_ITG",
    "_CRIT_TEM",
    "_STIFFNESS",
    "_STIFFNESS_MAX",
    "_STIFFNESS_MIN",
    "_TRANSPORT_FLOOR",
    "_dominant_channel",
    "_gyro_bohm_diffusivity",
    "critical_gradient_model",
    "reduced_gyrokinetic_profile_model",
]
