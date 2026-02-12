# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Bayesian Uncertainty Quantification
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Bayesian uncertainty quantification for fusion performance predictions.

Provides error bars on confinement time, fusion power, and Q-factor by
Monte Carlo sampling over scaling-law parameter uncertainties. Uses the
IPB98(y,2) H-mode confinement scaling as the baseline model.

References
----------
- ITER Physics Basis, Nucl. Fusion 39 (1999) 2175
- Verdoolaege et al., Nucl. Fusion 61 (2021) 076006
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# IPB98(y,2) scaling: τ_E = C · I_p^α_I · B^α_B · P^α_P · n^α_n · R^α_R · A^α_A · κ^α_κ · M^α_M
# Central values from ITER Physics Basis
IPB98_CENTRAL = {
    'C':     0.0562,
    'alpha_I':  0.93,
    'alpha_B':  0.15,
    'alpha_P': -0.69,
    'alpha_n':  0.41,
    'alpha_R':  1.97,
    'alpha_A': -0.58,
    'alpha_kappa': 0.78,
    'alpha_M':  0.19,
}

# 1-sigma uncertainties (from Verdoolaege 2021 Bayesian regression)
IPB98_SIGMA = {
    'C':     0.008,
    'alpha_I':  0.04,
    'alpha_B':  0.05,
    'alpha_P':  0.03,
    'alpha_n':  0.04,
    'alpha_R':  0.08,
    'alpha_A':  0.05,
    'alpha_kappa': 0.06,
    'alpha_M':  0.05,
}


@dataclass
class PlasmaScenario:
    """Input plasma parameters for a confinement prediction."""
    I_p: float        # Plasma current (MA)
    B_t: float        # Toroidal field (T)
    P_heat: float     # Total heating power (MW)
    n_e: float        # Line-average electron density (10^19 m^-3)
    R: float          # Major radius (m)
    A: float          # Aspect ratio R/a
    kappa: float      # Elongation
    M: float = 2.5    # Effective ion mass (AMU, 2.5 for D-T)


@dataclass
class UQResult:
    """Uncertainty-quantified prediction result."""
    # Central estimates
    tau_E: float          # Confinement time (s)
    P_fusion: float       # Fusion power (MW)
    Q: float              # Fusion gain Q = P_fus / P_heat

    # Uncertainties (1-sigma)
    tau_E_sigma: float
    P_fusion_sigma: float
    Q_sigma: float

    # Percentiles [5%, 25%, 50%, 75%, 95%]
    tau_E_percentiles: np.ndarray = field(default_factory=lambda: np.zeros(5))
    P_fusion_percentiles: np.ndarray = field(default_factory=lambda: np.zeros(5))
    Q_percentiles: np.ndarray = field(default_factory=lambda: np.zeros(5))

    # Raw samples (for custom analysis)
    n_samples: int = 0


def ipb98_tau_e(scenario: PlasmaScenario,
                params: Optional[dict] = None) -> float:
    """
    Compute IPB98(y,2) confinement time for given plasma parameters.

    Parameters
    ----------
    scenario : PlasmaScenario
        Plasma parameters.
    params : dict, optional
        Scaling law coefficients. Defaults to IPB98 central values.

    Returns
    -------
    float — confinement time in seconds.
    """
    p = params or IPB98_CENTRAL
    return (
        p['C']
        * scenario.I_p ** p['alpha_I']
        * scenario.B_t ** p['alpha_B']
        * scenario.P_heat ** p['alpha_P']
        * scenario.n_e ** p['alpha_n']
        * scenario.R ** p['alpha_R']
        * scenario.A ** p['alpha_A']
        * scenario.kappa ** p['alpha_kappa']
        * scenario.M ** p['alpha_M']
    )


def fusion_power_from_tau(scenario: PlasmaScenario, tau_E: float) -> float:
    """
    Estimate fusion power from confinement time using simplified power balance.

    P_fus ≈ 5 · n_e^2 · <σv> · V · E_fus / (4 · tau_E_loss_factor)

    For a rough estimate we use the empirical relation:
    P_fus ≈ (n_e * tau_E * T_i)^2 scaling, simplified to:
    P_fus ≈ C_fus · (n_e · 1e19)^2 · tau_E^2 · R^3 / A^2 · kappa

    The constant C_fus is calibrated to ITER Q=10 scenario.
    """
    # Simplified fusion power model calibrated to ITER:
    # ITER: n=10.1e19, tau=3.7s, R=6.2, A=3.1, kappa=1.7 → P_fus=500 MW
    C_fus = 500.0 / (10.1**2 * 3.7**2 * 6.2**3 / 3.1**2 * 1.7)
    n19 = scenario.n_e  # already in 10^19 m^-3
    V_factor = scenario.R**3 / scenario.A**2 * scenario.kappa
    return C_fus * n19**2 * tau_E**2 * V_factor


def quantify_uncertainty(scenario: PlasmaScenario,
                         n_samples: int = 10000,
                         seed: Optional[int] = None) -> UQResult:
    """
    Monte Carlo uncertainty quantification for fusion performance.

    Samples scaling-law coefficients from their Gaussian posteriors and
    propagates through the confinement and fusion power models.

    Parameters
    ----------
    scenario : PlasmaScenario
        Plasma parameters (held fixed).
    n_samples : int
        Number of Monte Carlo samples (default 10,000).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    UQResult — central estimates + error bars + percentiles.
    """
    rng = np.random.default_rng(seed)

    tau_samples = np.zeros(n_samples)
    pfus_samples = np.zeros(n_samples)
    q_samples = np.zeros(n_samples)

    for i in range(n_samples):
        # Sample scaling law parameters
        params = {}
        for key in IPB98_CENTRAL:
            params[key] = rng.normal(IPB98_CENTRAL[key], IPB98_SIGMA[key])

        # Ensure physical constraints
        params['C'] = max(params['C'], 1e-4)
        params['alpha_P'] = min(params['alpha_P'], -0.1)  # must be negative

        tau = ipb98_tau_e(scenario, params)
        tau = max(tau, 1e-6)  # floor

        pfus = fusion_power_from_tau(scenario, tau)
        pfus = max(pfus, 0.0)

        q = pfus / scenario.P_heat if scenario.P_heat > 0 else 0.0

        tau_samples[i] = tau
        pfus_samples[i] = pfus
        q_samples[i] = q

    pcts = [5, 25, 50, 75, 95]

    return UQResult(
        tau_E=float(np.median(tau_samples)),
        P_fusion=float(np.median(pfus_samples)),
        Q=float(np.median(q_samples)),
        tau_E_sigma=float(np.std(tau_samples)),
        P_fusion_sigma=float(np.std(pfus_samples)),
        Q_sigma=float(np.std(q_samples)),
        tau_E_percentiles=np.percentile(tau_samples, pcts),
        P_fusion_percentiles=np.percentile(pfus_samples, pcts),
        Q_percentiles=np.percentile(q_samples, pcts),
        n_samples=n_samples,
    )
