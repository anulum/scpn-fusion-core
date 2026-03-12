# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Full-Chain Uncertainty Propagation
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Full-chain Monte Carlo uncertainty propagation:
equilibrium → transport → fusion power → gain → normalised beta.

Extends the base IPB98 UQ in ``uncertainty.py`` by additionally perturbing
gyro-Bohm transport coefficients, EPED pedestal height, and equilibrium
boundary shape.

References
----------
- ITER Physics Basis, Nucl. Fusion 39 (1999) 2175
- Verdoolaege et al., Nucl. Fusion 61 (2021) 076006
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from scpn_fusion.core.uncertainty import (
    IPB98_CENTRAL,
    IPB98_SIGMA,
    PlasmaScenario,
    ipb98_tau_e,
    fusion_power_from_tau,
    _validate_n_samples,
    _validate_seed,
    _validate_scenario,
)


@dataclass
class EquilibriumUncertainty:
    """Uncertainty contributions from equilibrium reconstruction.

    Captures how boundary perturbations propagate into psi-field and
    magnetic-axis location uncertainty.
    """

    psi_nrmse_mean: float = 0.0  # Mean normalised RMSE of psi reconstruction
    psi_nrmse_sigma: float = 0.01  # 1-sigma spread from boundary perturbation
    R_axis_sigma: float = 0.02  # Magnetic axis R location uncertainty (m)
    Z_axis_sigma: float = 0.01  # Magnetic axis Z location uncertainty (m)


@dataclass
class TransportUncertainty:
    """Uncertainty contributions from transport model coefficients.

    Captures the dominant sources of uncertainty in the gyro-Bohm diffusivity
    model and the EPED pedestal prediction.
    """

    chi_gB_factor_sigma: float = 0.3  # Gyro-Bohm coefficient fractional uncertainty (30%)
    pedestal_height_sigma: float = 0.2  # EPED pedestal height fractional uncertainty (20%)


@dataclass
class FullChainUQResult:
    """Extended uncertainty-quantified prediction covering the full
    equilibrium -> transport -> fusion power chain.

    All ``*_bands`` fields are length-3 arrays: [5th, 50th, 95th] percentiles.
    """

    tau_E: float
    P_fusion: float
    Q: float

    tau_E_sigma: float
    P_fusion_sigma: float
    Q_sigma: float

    psi_nrmse_bands: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tau_E_bands: np.ndarray = field(default_factory=lambda: np.zeros(3))
    P_fusion_bands: np.ndarray = field(default_factory=lambda: np.zeros(3))
    Q_bands: np.ndarray = field(default_factory=lambda: np.zeros(3))
    beta_N_bands: np.ndarray = field(default_factory=lambda: np.zeros(3))

    tau_E_percentiles: np.ndarray = field(default_factory=lambda: np.zeros(5))
    P_fusion_percentiles: np.ndarray = field(default_factory=lambda: np.zeros(5))
    Q_percentiles: np.ndarray = field(default_factory=lambda: np.zeros(5))

    n_samples: int = 0


def _build_ipb98_covariance() -> np.ndarray:
    """Build covariance matrix for correlated IPB98 coefficient sampling."""
    keys = [
        "C",
        "alpha_I",
        "alpha_B",
        "alpha_P",
        "alpha_n",
        "alpha_R",
        "alpha_A",
        "alpha_kappa",
        "alpha_M",
    ]
    sigmas = np.array([IPB98_SIGMA[k] for k in keys], dtype=np.float64)
    cov = np.diag(sigmas**2)

    # Known physical correlations from global scaling regressions.
    idx_c = 0
    idx_r = 5
    corr_cr = -0.7
    cov[idx_c, idx_r] = corr_cr * sigmas[idx_c] * sigmas[idx_r]
    cov[idx_r, idx_c] = cov[idx_c, idx_r]

    idx_i = 1
    idx_b = 2
    corr_ib = 0.4
    cov[idx_i, idx_b] = corr_ib * sigmas[idx_i] * sigmas[idx_b]
    cov[idx_b, idx_i] = cov[idx_i, idx_b]

    return cov


def quantify_full_chain(
    scenario: PlasmaScenario,
    n_samples: int = 5000,
    seed: Optional[int] = None,
    chi_gB_sigma: float = 0.3,
    pedestal_sigma: float = 0.2,
    boundary_sigma: float = 0.02,
) -> FullChainUQResult:
    """
    Full-chain Monte Carlo uncertainty propagation:
    equilibrium -> transport -> fusion power -> gain.

    Extends :func:`quantify_uncertainty` by additionally perturbing the
    gyro-Bohm transport coefficient, EPED pedestal height, and equilibrium
    boundary shape, then computing normalised beta as an extra observable.

    Parameters
    ----------
    scenario : PlasmaScenario
        Nominal plasma parameters (held at central values; perturbations are
        multiplicative).
    n_samples : int
        Number of Monte Carlo draws (default 5000).
    seed : int, optional
        Random seed for reproducibility.
    chi_gB_sigma : float
        Log-normal sigma for gyro-Bohm coefficient perturbation (default 0.3).
    pedestal_sigma : float
        Gaussian sigma (fractional) for pedestal height perturbation (default 0.2).
    boundary_sigma : float
        Gaussian sigma (fractional) for major-radius boundary perturbation
        (default 0.02, i.e. 2%).

    Returns
    -------
    FullChainUQResult
        Bands at [5%, 50%, 95%] for psi_nrmse, tau_E, P_fusion, Q, beta_N.
    """
    scenario = _validate_scenario(scenario)
    n_samples = _validate_n_samples(n_samples)
    seed = _validate_seed(seed)

    def _validate_sigma(name: str, value: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite and >= 0") from exc
        if not np.isfinite(parsed) or parsed < 0.0:
            raise ValueError(f"{name} must be finite and >= 0")
        return parsed

    chi_gB_sigma = _validate_sigma("chi_gB_sigma", chi_gB_sigma)
    pedestal_sigma = _validate_sigma("pedestal_sigma", pedestal_sigma)
    boundary_sigma = _validate_sigma("boundary_sigma", boundary_sigma)

    rng = np.random.default_rng(seed)

    tau_samples = np.zeros(n_samples)
    pfus_samples = np.zeros(n_samples)
    q_samples = np.zeros(n_samples)
    beta_n_samples = np.zeros(n_samples)
    psi_nrmse_samples = np.zeros(n_samples)

    keys = [
        "C",
        "alpha_I",
        "alpha_B",
        "alpha_P",
        "alpha_n",
        "alpha_R",
        "alpha_A",
        "alpha_kappa",
        "alpha_M",
    ]
    means = np.array([IPB98_CENTRAL[k] for k in keys], dtype=np.float64)
    cov = _build_ipb98_covariance()
    try:
        ipb_samples = rng.multivariate_normal(means, cov, size=n_samples)
    except np.linalg.LinAlgError:
        cov = cov + np.eye(len(keys), dtype=np.float64) * 1e-12
        ipb_samples = rng.multivariate_normal(means, cov, size=n_samples)

    for i in range(n_samples):
        params = {keys[j]: float(ipb_samples[i, j]) for j in range(len(keys))}
        params["C"] = max(params["C"], 1e-4)
        params["alpha_P"] = min(params["alpha_P"], -0.1)

        chi_factor = rng.lognormal(0.0, chi_gB_sigma)

        ped_factor = rng.normal(1.0, pedestal_sigma)
        ped_factor = max(ped_factor, 0.1)

        R_pert = scenario.R * (1.0 + rng.normal(0.0, boundary_sigma))
        R_pert = max(R_pert, 0.5)

        psi_nrmse = abs(R_pert - scenario.R) / scenario.R
        psi_nrmse_samples[i] = psi_nrmse

        pert_scenario = PlasmaScenario(
            I_p=scenario.I_p,
            B_t=scenario.B_t,
            P_heat=scenario.P_heat,
            n_e=scenario.n_e,
            R=R_pert,
            A=scenario.A,
            kappa=scenario.kappa,
            M=scenario.M,
        )
        tau = ipb98_tau_e(pert_scenario, params)
        tau = tau * ped_factor / chi_factor
        tau = max(tau, 1e-6)

        pfus = fusion_power_from_tau(pert_scenario, tau)
        pfus = max(pfus, 0.0)

        q = pfus / scenario.P_heat if scenario.P_heat > 0 else 0.0
        if not np.isfinite(q):
            q = 0.0

        # beta_N = beta_t(%) / I_N where I_N = I_p / (a * B_t)
        a_pert = R_pert / scenario.A
        V_approx = 2.0 * np.pi**2 * R_pert * a_pert**2 * scenario.kappa
        p_avg = scenario.P_heat * tau * 1e6 / V_approx  # Pa
        mu0 = 4.0 * np.pi * 1e-7
        beta_t = 2.0 * mu0 * p_avg / (scenario.B_t**2)
        beta_t_pct = beta_t * 100.0
        I_N = scenario.I_p / (a_pert * scenario.B_t)
        beta_N = beta_t_pct / I_N if I_N > 1e-6 else 0.0

        tau_samples[i] = tau
        pfus_samples[i] = pfus
        q_samples[i] = q
        beta_n_samples[i] = beta_N

    band_pcts = [5, 50, 95]
    full_pcts = [5, 25, 50, 75, 95]

    return FullChainUQResult(
        tau_E=float(np.median(tau_samples)),
        P_fusion=float(np.median(pfus_samples)),
        Q=float(np.median(q_samples)),
        tau_E_sigma=float(np.std(tau_samples)),
        P_fusion_sigma=float(np.std(pfus_samples)),
        Q_sigma=float(np.std(q_samples)),
        psi_nrmse_bands=np.asarray(
            np.percentile(psi_nrmse_samples, band_pcts),
            dtype=float,
        ),
        tau_E_bands=np.asarray(np.percentile(tau_samples, band_pcts), dtype=float),
        P_fusion_bands=np.asarray(
            np.percentile(pfus_samples, band_pcts),
            dtype=float,
        ),
        Q_bands=np.asarray(np.percentile(q_samples, band_pcts), dtype=float),
        beta_N_bands=np.asarray(np.percentile(beta_n_samples, band_pcts), dtype=float),
        tau_E_percentiles=np.asarray(np.percentile(tau_samples, full_pcts), dtype=float),
        P_fusion_percentiles=np.asarray(
            np.percentile(pfus_samples, full_pcts),
            dtype=float,
        ),
        Q_percentiles=np.asarray(np.percentile(q_samples, full_pcts), dtype=float),
        n_samples=n_samples,
    )


def summarize_uq(result: FullChainUQResult) -> dict:
    """
    Convert a FullChainUQResult to a plain dict suitable for ``json.dumps()``.

    All numpy arrays are converted to Python lists; floats are rounded
    to 6 significant figures.
    """

    def _round(x: float, sig: int = 6) -> float:
        return float(f"{x:.{sig}g}")

    def _arr(a: np.ndarray) -> list:
        return [_round(float(v)) for v in a]

    return {
        "central": {
            "tau_E_s": _round(result.tau_E),
            "P_fusion_MW": _round(result.P_fusion),
            "Q": _round(result.Q),
        },
        "sigma": {
            "tau_E_s": _round(result.tau_E_sigma),
            "P_fusion_MW": _round(result.P_fusion_sigma),
            "Q": _round(result.Q_sigma),
        },
        "bands_5_50_95": {
            "psi_nrmse": _arr(result.psi_nrmse_bands),
            "tau_E_s": _arr(result.tau_E_bands),
            "P_fusion_MW": _arr(result.P_fusion_bands),
            "Q": _arr(result.Q_bands),
            "beta_N": _arr(result.beta_N_bands),
        },
        "n_samples": result.n_samples,
    }
