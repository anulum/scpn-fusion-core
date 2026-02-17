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


def _validate_n_samples(n_samples: int) -> int:
    """Validate Monte Carlo sample count and normalise to int."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise ValueError("n_samples must be an integer >= 1")
    parsed = int(n_samples)
    if parsed < 1:
        raise ValueError("n_samples must be an integer >= 1")
    return parsed


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


@dataclass
class EquilibriumUncertainty:
    """Uncertainty contributions from equilibrium reconstruction.

    Captures how boundary perturbations propagate into psi-field and
    magnetic-axis location uncertainty.
    """
    psi_nrmse_mean: float = 0.0     # Mean normalised RMSE of psi reconstruction
    psi_nrmse_sigma: float = 0.01   # 1-sigma spread from boundary perturbation
    R_axis_sigma: float = 0.02      # Magnetic axis R location uncertainty (m)
    Z_axis_sigma: float = 0.01      # Magnetic axis Z location uncertainty (m)


@dataclass
class TransportUncertainty:
    """Uncertainty contributions from transport model coefficients.

    Captures the dominant sources of uncertainty in the gyro-Bohm diffusivity
    model and the EPED pedestal prediction.
    """
    chi_gB_factor_sigma: float = 0.3   # Gyro-Bohm coefficient fractional uncertainty (30%)
    pedestal_height_sigma: float = 0.2  # EPED pedestal height fractional uncertainty (20%)


@dataclass
class FullChainUQResult:
    """Extended uncertainty-quantified prediction covering the full
    equilibrium -> transport -> fusion power chain.

    All ``*_bands`` fields are length-3 arrays: [5th, 50th, 95th] percentiles.
    """
    # Central estimates (medians)
    tau_E: float
    P_fusion: float
    Q: float

    # 1-sigma spreads
    tau_E_sigma: float
    P_fusion_sigma: float
    Q_sigma: float

    # Percentile bands [5%, 50%, 95%]
    psi_nrmse_bands: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tau_E_bands: np.ndarray = field(default_factory=lambda: np.zeros(3))
    P_fusion_bands: np.ndarray = field(default_factory=lambda: np.zeros(3))
    Q_bands: np.ndarray = field(default_factory=lambda: np.zeros(3))
    beta_N_bands: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Legacy-compatible percentiles [5, 25, 50, 75, 95]
    tau_E_percentiles: np.ndarray = field(default_factory=lambda: np.zeros(5))
    P_fusion_percentiles: np.ndarray = field(default_factory=lambda: np.zeros(5))
    Q_percentiles: np.ndarray = field(default_factory=lambda: np.zeros(5))

    n_samples: int = 0


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
    n_samples = _validate_n_samples(n_samples)

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

    a_nominal = scenario.R / scenario.A  # minor radius (m)

    for i in range(n_samples):
        # --- (a) Perturb IPB98 scaling-law coefficients ---
        params = {}
        for key in IPB98_CENTRAL:
            params[key] = rng.normal(IPB98_CENTRAL[key], IPB98_SIGMA[key])
        params['C'] = max(params['C'], 1e-4)
        params['alpha_P'] = min(params['alpha_P'], -0.1)

        # --- (b) Perturb gyro-Bohm transport coefficient ---
        chi_factor = rng.lognormal(0.0, chi_gB_sigma)

        # --- (c) Perturb pedestal height ---
        ped_factor = rng.normal(1.0, pedestal_sigma)
        ped_factor = max(ped_factor, 0.1)  # floor to 10% of nominal

        # --- (d) Perturb equilibrium boundary (major radius) ---
        R_pert = scenario.R * (1.0 + rng.normal(0.0, boundary_sigma))
        R_pert = max(R_pert, 0.5)  # physical floor

        # --- (i) psi NRMSE from boundary perturbation ---
        # Boundary displacement maps roughly linearly to psi reconstruction error
        psi_nrmse = abs(R_pert - scenario.R) / scenario.R
        psi_nrmse_samples[i] = psi_nrmse

        # --- (e) Compute tau_E with perturbed chi applied multiplicatively ---
        # Use perturbed R for the scaling law
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
        # chi_factor > 1 means stronger transport → shorter confinement;
        # pedestal height factor > 1 means better pedestal → longer confinement
        tau = tau * ped_factor / chi_factor
        tau = max(tau, 1e-6)

        # --- (f) Compute P_fusion from perturbed tau_E ---
        pfus = fusion_power_from_tau(pert_scenario, tau)
        pfus = max(pfus, 0.0)

        # --- (g) Compute Q ---
        q = pfus / scenario.P_heat if scenario.P_heat > 0 else 0.0

        # --- (h) Compute normalised beta ---
        # beta_t = (n_e * 1e19 * k_B * T_i) / (B^2 / 2mu0)
        # For a rough estimate, T_i ~ tau_E * P_heat / (n_e * V) gives
        # beta_t ~ C * n_e * tau_E * P_heat / (B^2 * V)
        # Then beta_N = beta_t(%) / (I_p / (a * B_t))
        a_pert = R_pert / scenario.A
        V_approx = 2.0 * np.pi**2 * R_pert * a_pert**2 * scenario.kappa
        # Volume-average pressure proxy: P_heat * tau_E / V  (MW·s / m^3 = MJ/m^3)
        # beta_t = 2 mu0 * <p> / B^2;  mu0 = 4pi*1e-7;  1 MJ/m^3 = 1e6 Pa
        p_avg = scenario.P_heat * tau * 1e6 / V_approx  # Pa
        mu0 = 4.0 * np.pi * 1e-7
        beta_t = 2.0 * mu0 * p_avg / (scenario.B_t**2)  # dimensionless
        beta_t_pct = beta_t * 100.0
        I_N = scenario.I_p / (a_pert * scenario.B_t)  # normalised current (MA / (m·T))
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
            np.percentile(psi_nrmse_samples, band_pcts), dtype=float,
        ),
        tau_E_bands=np.asarray(np.percentile(tau_samples, band_pcts), dtype=float),
        P_fusion_bands=np.asarray(
            np.percentile(pfus_samples, band_pcts), dtype=float,
        ),
        Q_bands=np.asarray(np.percentile(q_samples, band_pcts), dtype=float),
        beta_N_bands=np.asarray(np.percentile(beta_n_samples, band_pcts), dtype=float),
        tau_E_percentiles=np.asarray(np.percentile(tau_samples, full_pcts), dtype=float),
        P_fusion_percentiles=np.asarray(
            np.percentile(pfus_samples, full_pcts), dtype=float,
        ),
        Q_percentiles=np.asarray(np.percentile(q_samples, full_pcts), dtype=float),
        n_samples=n_samples,
    )


def summarize_uq(result: FullChainUQResult) -> dict:
    """
    Pretty-print a FullChainUQResult as a plain dict suitable for
    ``json.dumps()``.

    All numpy arrays are converted to Python lists; floats are rounded
    to 6 significant figures.

    Parameters
    ----------
    result : FullChainUQResult
        Output of :func:`quantify_full_chain`.

    Returns
    -------
    dict — JSON-serialisable summary.
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
    n_samples = _validate_n_samples(n_samples)
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
        tau_E_percentiles=np.asarray(np.percentile(tau_samples, pcts), dtype=float),
        P_fusion_percentiles=np.asarray(
            np.percentile(pfus_samples, pcts), dtype=float,
        ),
        Q_percentiles=np.asarray(np.percentile(q_samples, pcts), dtype=float),
        n_samples=n_samples,
    )
