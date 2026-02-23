# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Bayesian Uncertainty Quantification
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

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


_SAFE_LOG_MAX = 700.0
_SAFE_LOG_MIN = -745.0


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


def _validate_seed(seed: Optional[int]) -> Optional[int]:
    """Validate optional RNG seed."""
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer >= 0 or None")
    parsed = int(seed)
    if parsed < 0:
        raise ValueError("seed must be an integer >= 0 or None")
    return parsed


def _require_positive_finite(name: str, value: float) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return parsed


def _validate_scenario(scenario: PlasmaScenario) -> PlasmaScenario:
    """Validate that all scenario parameters are finite and physically positive."""
    scenario.I_p = _require_positive_finite("scenario.I_p", scenario.I_p)
    scenario.B_t = _require_positive_finite("scenario.B_t", scenario.B_t)
    scenario.P_heat = _require_positive_finite("scenario.P_heat", scenario.P_heat)
    scenario.n_e = _require_positive_finite("scenario.n_e", scenario.n_e)
    scenario.R = _require_positive_finite("scenario.R", scenario.R)
    scenario.A = _require_positive_finite("scenario.A", scenario.A)
    scenario.kappa = _require_positive_finite("scenario.kappa", scenario.kappa)
    scenario.M = _require_positive_finite("scenario.M", scenario.M)
    return scenario


def _safe_exp_from_log(log_value: float, *, name: str) -> float:
    """Exponentiate bounded log value; raise when outside stable float range."""
    if not np.isfinite(log_value):
        raise ValueError(f"{name} became non-finite in log space")
    if log_value > _SAFE_LOG_MAX or log_value < _SAFE_LOG_MIN:
        raise ValueError(
            f"{name} outside numerically stable range (log={log_value:.2f})"
        )
    out = float(np.exp(log_value))
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} is non-finite after exponentiation")
    return out


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
    scenario = _validate_scenario(scenario)
    p = params or IPB98_CENTRAL
    required = (
        "C",
        "alpha_I",
        "alpha_B",
        "alpha_P",
        "alpha_n",
        "alpha_R",
        "alpha_A",
        "alpha_kappa",
        "alpha_M",
    )
    for key in required:
        if key not in p:
            raise ValueError(f"params missing required key '{key}'")
        if not np.isfinite(float(p[key])):
            raise ValueError(f"params.{key} must be finite")
    c = float(p["C"])
    if c <= 0.0:
        raise ValueError("params.C must be finite and > 0")

    log_tau = (
        np.log(c)
        + float(p['alpha_I']) * np.log(scenario.I_p)
        + float(p['alpha_B']) * np.log(scenario.B_t)
        + float(p['alpha_P']) * np.log(scenario.P_heat)
        + float(p['alpha_n']) * np.log(scenario.n_e)
        + float(p['alpha_R']) * np.log(scenario.R)
        + float(p['alpha_A']) * np.log(scenario.A)
        + float(p['alpha_kappa']) * np.log(scenario.kappa)
        + float(p['alpha_M']) * np.log(scenario.M)
    )
    return _safe_exp_from_log(float(log_tau), name="ipb98_tau_e")


def _dt_reactivity(Ti_keV: float) -> float:
    """D-T fusion reactivity <sigma v> [m^3/s] using Gamow peak approximation.

    Valid for Ti in [1, 100] keV.
    Reference: Huba, NRL Plasma Formulary (2019); Bosch & Hale, NF 32 (1992) 611.
    """
    T = float(max(Ti_keV, 1.0))
    # Gamow peak form: <sv> = A * T^{-2/3} * exp(-B * T^{-1/3})
    # A = 6.68e-12 cm^3/s, B = 19.94 keV^{1/3}  (D-T)
    # Convert cm^3/s -> m^3/s: factor 1e-6
    A_m3 = 6.68e-18  # m^3/s
    B = 19.94
    sv = A_m3 * T ** (-2.0 / 3.0) * np.exp(-B * T ** (-1.0 / 3.0))
    return float(sv)


def fusion_power_from_tau(scenario: PlasmaScenario, tau_E: float) -> float:
    """
    Estimate fusion power from cross-section integrated thermal reactivity.
    P_fus = n_D * n_T * <sigma v> * V * E_fusion

    Includes alpha self-heating iteration: in a burning plasma, 20% of
    fusion energy (3.5 MeV alphas out of 17.6 MeV total) heats the
    plasma, raising Ti and thus reactivity.  Three fixed-point iterations
    converge for ITER-class Q~10 scenarios.
    """
    scenario = _validate_scenario(scenario)
    tau_E = _require_positive_finite("tau_E", tau_E)

    # Volume and Density (SI)
    a = scenario.R / scenario.A
    V = 2.0 * np.pi**2 * scenario.R * a**2 * scenario.kappa
    ne = scenario.n_e * 1e19  # m^-3

    k_B_keV = 1.602176634e-16  # J per keV
    E_fus_J = 17.6e6 * 1.602176634e-19  # J per D-T reaction
    f_alpha = 3.5 / 17.6  # fraction of fusion energy to alphas

    def _pfus_at_Ptotal(P_tot: float) -> float:
        W_MJ = P_tot * tau_E
        Ti = (W_MJ * 1e6) / (3.0 * ne * k_B_keV * V)
        Ti = float(np.clip(Ti, 0.5, 100.0))
        sv = _dt_reactivity(Ti)
        return (0.25 * ne**2 * sv * V * E_fus_J) / 1e6

    # Step 1: fusion power from external heating only
    pfus_0 = _pfus_at_Ptotal(scenario.P_heat)

    # Step 2: single alpha-heating correction (one Picard step)
    pfus_mw = _pfus_at_Ptotal(scenario.P_heat + f_alpha * pfus_0)

    return float(pfus_mw)


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

    a_nominal = scenario.R / scenario.A  # minor radius (m)

    for i in range(n_samples):
        # (a) Perturb IPB98 scaling-law coefficients
        params = {}
        for key in IPB98_CENTRAL:
            params[key] = rng.normal(IPB98_CENTRAL[key], IPB98_SIGMA[key])
        params['C'] = max(params['C'], 1e-4)
        params['alpha_P'] = min(params['alpha_P'], -0.1)

        # (b) Perturb gyro-Bohm transport coefficient
        chi_factor = rng.lognormal(0.0, chi_gB_sigma)

        # (c) Perturb pedestal height
        ped_factor = rng.normal(1.0, pedestal_sigma)
        ped_factor = max(ped_factor, 0.1)  # floor to 10% of nominal

        # (d) Perturb equilibrium boundary (major radius)
        R_pert = scenario.R * (1.0 + rng.normal(0.0, boundary_sigma))
        R_pert = max(R_pert, 0.5)  # physical floor

        # (i) psi NRMSE from boundary perturbation
        # Boundary displacement maps roughly linearly to psi reconstruction error
        psi_nrmse = abs(R_pert - scenario.R) / scenario.R
        psi_nrmse_samples[i] = psi_nrmse

        # (e) Compute tau_E with perturbed chi
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

        # (f) Compute P_fusion from perturbed tau_E
        pfus = fusion_power_from_tau(pert_scenario, tau)
        pfus = max(pfus, 0.0)

        # (g) Compute Q
        q = pfus / scenario.P_heat if scenario.P_heat > 0 else 0.0
        if not np.isfinite(q):
            q = 0.0

        # (h) Compute normalised beta
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
    scenario = _validate_scenario(scenario)
    n_samples = _validate_n_samples(n_samples)
    seed = _validate_seed(seed)
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
        if not np.isfinite(q):
            q = 0.0

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
