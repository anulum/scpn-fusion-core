// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! Nonlinear plasma source term for the Grad-Shafranov equation.
//!
//! Port of fusion_kernel.py `update_plasma_source_nonlinear()` (lines 118-171).
//! Computes J_phi using:
//!   J_phi = R · p'(ψ_norm) + (1/(μ₀R)) · FF'(ψ_norm)
//! where ψ_norm = (Ψ - Ψ_axis) / (Ψ_boundary - Ψ_axis).

use fusion_types::error::{FusionError, FusionResult};
use fusion_types::state::Grid2D;
use ndarray::Array2;

/// Default pressure/current mixing ratio (0=pure poloidal, 1=pure pressure).
/// Python line 158: `beta_mix = 0.5`
const DEFAULT_BETA_MIX: f64 = 0.5;

/// Minimum denominator for flux normalization (avoid div-by-zero).
/// Python line 130: `if abs(denom) < 1e-9: denom = 1e-9`
const MIN_FLUX_DENOMINATOR: f64 = 1e-9;

/// Minimum current integral threshold for renormalization.
/// Python line 165: `if abs(I_current) > 1e-9:`
const MIN_CURRENT_INTEGRAL: f64 = 1e-9;

/// Explicit GEQDSK profile-source convention transforms accepted by the native solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeqdskSourceConvention {
    Canonical,
    Negated,
    ScaledByTwoPi,
    ScaledByMinusTwoPi,
    ScaledByInvTwoPi,
    ScaledByMinusInvTwoPi,
    TimesFluxSpan,
    OverFluxSpan,
    NegatedTimesFluxSpan,
    NegatedOverFluxSpan,
    NotEvaluated,
}

impl GeqdskSourceConvention {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Canonical => "canonical",
            Self::Negated => "negated",
            Self::ScaledByTwoPi => "scaled_by_2pi",
            Self::ScaledByMinusTwoPi => "scaled_by_minus_2pi",
            Self::ScaledByInvTwoPi => "scaled_by_inv_2pi",
            Self::ScaledByMinusInvTwoPi => "scaled_by_minus_inv_2pi",
            Self::TimesFluxSpan => "times_flux_span",
            Self::OverFluxSpan => "over_flux_span",
            Self::NegatedTimesFluxSpan => "negated_times_flux_span",
            Self::NegatedOverFluxSpan => "negated_over_flux_span",
            Self::NotEvaluated => "not_evaluated",
        }
    }
}

impl TryFrom<&str> for GeqdskSourceConvention {
    type Error = FusionError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "canonical" => Ok(Self::Canonical),
            "negated" => Ok(Self::Negated),
            "scaled_by_2pi" => Ok(Self::ScaledByTwoPi),
            "scaled_by_minus_2pi" => Ok(Self::ScaledByMinusTwoPi),
            "scaled_by_inv_2pi" => Ok(Self::ScaledByInvTwoPi),
            "scaled_by_minus_inv_2pi" => Ok(Self::ScaledByMinusInvTwoPi),
            "times_flux_span" => Ok(Self::TimesFluxSpan),
            "over_flux_span" => Ok(Self::OverFluxSpan),
            "negated_times_flux_span" => Ok(Self::NegatedTimesFluxSpan),
            "negated_over_flux_span" => Ok(Self::NegatedOverFluxSpan),
            "not_evaluated" => Ok(Self::NotEvaluated),
            other => Err(FusionError::ConfigError(format!(
                "unknown GEQDSK source convention '{other}'"
            ))),
        }
    }
}

/// Result of ranking named GEQDSK source-convention transforms against an operator source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeqdskSourceConventionAdapter {
    pub convention: GeqdskSourceConvention,
    pub residual_l2: f64,
    pub pass: bool,
}

/// Residual-ranked executable GEQDSK source-convention candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeqdskSourceConventionCandidate {
    pub convention: GeqdskSourceConvention,
    pub residual_l2: f64,
}

/// GEQDSK profile-source components assembled on an R-Z grid.
#[derive(Debug, Clone)]
pub struct GeqdskProfileSourceComponents {
    pub pressure_source: Array2<f64>,
    pub ffprime_source: Array2<f64>,
    pub total_source: Array2<f64>,
    pub plasma_mask: Array2<bool>,
    pub plasma_mask_fraction: f64,
    pub pressure_source_norm: f64,
    pub ffprime_source_norm: f64,
    pub total_source_norm: f64,
}

/// mTanh profile parameters used by inverse reconstruction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfileParams {
    pub ped_top: f64,
    pub ped_width: f64,
    pub ped_height: f64,
    pub core_alpha: f64,
}

impl Default for ProfileParams {
    fn default() -> Self {
        Self {
            ped_top: 0.9,
            ped_width: 0.08,
            ped_height: 1.0,
            core_alpha: 0.2,
        }
    }
}

/// Context passed to profile-driven source update.
#[derive(Debug, Clone, Copy)]
pub struct SourceProfileContext<'a> {
    pub psi: &'a Array2<f64>,
    pub grid: &'a Grid2D,
    pub psi_axis: f64,
    pub psi_boundary: f64,
    pub mu0: f64,
    pub i_target: f64,
}

fn validate_source_inputs(
    psi: &Array2<f64>,
    grid: &Grid2D,
    psi_axis: f64,
    psi_boundary: f64,
    mu0: f64,
    i_target: f64,
) -> FusionResult<()> {
    if grid.nz == 0 || grid.nr == 0 {
        return Err(FusionError::ConfigError(
            "source update grid requires nz,nr >= 1".to_string(),
        ));
    }
    if !grid.dr.is_finite()
        || !grid.dz.is_finite()
        || grid.dr.abs() <= f64::EPSILON
        || grid.dz.abs() <= f64::EPSILON
    {
        return Err(FusionError::ConfigError(format!(
            "source update requires finite non-zero grid spacing, got dr={} dz={}",
            grid.dr, grid.dz
        )));
    }
    if psi.nrows() != grid.nz || psi.ncols() != grid.nr {
        return Err(FusionError::ConfigError(format!(
            "source update psi shape mismatch: expected ({}, {}), got ({}, {})",
            grid.nz,
            grid.nr,
            psi.nrows(),
            psi.ncols()
        )));
    }
    if grid.rr.nrows() != grid.nz || grid.rr.ncols() != grid.nr {
        return Err(FusionError::ConfigError(format!(
            "source update grid.rr shape mismatch: expected ({}, {}), got ({}, {})",
            grid.nz,
            grid.nr,
            grid.rr.nrows(),
            grid.rr.ncols()
        )));
    }
    if psi.iter().any(|v| !v.is_finite()) || grid.rr.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::ConfigError(
            "source update inputs must be finite".to_string(),
        ));
    }
    if !psi_axis.is_finite() || !psi_boundary.is_finite() {
        return Err(FusionError::ConfigError(
            "source update psi_axis/psi_boundary must be finite".to_string(),
        ));
    }
    if !mu0.is_finite() || mu0 <= 0.0 {
        return Err(FusionError::ConfigError(format!(
            "source update requires finite mu0 > 0, got {mu0}"
        )));
    }
    if !i_target.is_finite() {
        return Err(FusionError::ConfigError(
            "source update target current must be finite".to_string(),
        ));
    }
    let denom = psi_boundary - psi_axis;
    if !denom.is_finite() || denom.abs() < MIN_FLUX_DENOMINATOR {
        return Err(FusionError::ConfigError(format!(
            "source update flux denominator must satisfy |psi_boundary-psi_axis| >= {MIN_FLUX_DENOMINATOR}, got {}",
            denom
        )));
    }
    Ok(())
}

fn validate_profile_params(params: &ProfileParams, label: &str) -> FusionResult<()> {
    if !params.ped_top.is_finite() || params.ped_top <= 0.0 {
        return Err(FusionError::ConfigError(format!(
            "{label}.ped_top must be finite and > 0, got {}",
            params.ped_top
        )));
    }
    if !params.ped_width.is_finite() || params.ped_width <= 0.0 {
        return Err(FusionError::ConfigError(format!(
            "{label}.ped_width must be finite and > 0, got {}",
            params.ped_width
        )));
    }
    if !params.ped_height.is_finite() || !params.core_alpha.is_finite() {
        return Err(FusionError::ConfigError(format!(
            "{label}.ped_height/core_alpha must be finite"
        )));
    }
    Ok(())
}

/// Return the explicit multiplier for a named GEQDSK source convention.
pub fn geqdsk_source_convention_multiplier(
    convention: GeqdskSourceConvention,
    flux_span: f64,
) -> FusionResult<f64> {
    match convention {
        GeqdskSourceConvention::Canonical => Ok(1.0),
        GeqdskSourceConvention::Negated => Ok(-1.0),
        GeqdskSourceConvention::ScaledByTwoPi => Ok(2.0 * std::f64::consts::PI),
        GeqdskSourceConvention::ScaledByMinusTwoPi => Ok(-2.0 * std::f64::consts::PI),
        GeqdskSourceConvention::ScaledByInvTwoPi => Ok(1.0 / (2.0 * std::f64::consts::PI)),
        GeqdskSourceConvention::ScaledByMinusInvTwoPi => Ok(-1.0 / (2.0 * std::f64::consts::PI)),
        GeqdskSourceConvention::TimesFluxSpan => checked_flux_span_multiplier(flux_span, false),
        GeqdskSourceConvention::OverFluxSpan => {
            checked_inverse_flux_span_multiplier(flux_span, false)
        }
        GeqdskSourceConvention::NegatedTimesFluxSpan => {
            checked_flux_span_multiplier(flux_span, true)
        }
        GeqdskSourceConvention::NegatedOverFluxSpan => {
            checked_inverse_flux_span_multiplier(flux_span, true)
        }
        GeqdskSourceConvention::NotEvaluated => Err(FusionError::ConfigError(
            "not_evaluated is not an executable GEQDSK source convention".to_string(),
        )),
    }
}

fn checked_flux_span_multiplier(flux_span: f64, negated: bool) -> FusionResult<f64> {
    if !flux_span.is_finite() || flux_span.abs() < 1.0e-15 {
        return Err(FusionError::ConfigError(
            "flux-span source conventions require a finite non-zero flux span".to_string(),
        ));
    }
    Ok(if negated { -flux_span } else { flux_span })
}

fn checked_inverse_flux_span_multiplier(flux_span: f64, negated: bool) -> FusionResult<f64> {
    if !flux_span.is_finite() || flux_span.abs() < 1.0e-15 {
        return Err(FusionError::ConfigError(
            "flux-span source conventions require a finite non-zero flux span".to_string(),
        ));
    }
    Ok(if negated {
        -1.0 / flux_span
    } else {
        1.0 / flux_span
    })
}

/// Apply an explicit, documented GEQDSK source convention transform.
pub fn apply_geqdsk_source_convention(
    source: &Array2<f64>,
    convention: GeqdskSourceConvention,
    flux_span: f64,
) -> FusionResult<Array2<f64>> {
    if source.iter().any(|value| !value.is_finite()) {
        return Err(FusionError::ConfigError(
            "GEQDSK source convention input must contain only finite values".to_string(),
        ));
    }
    let multiplier = geqdsk_source_convention_multiplier(convention, flux_span)?;
    let transformed = source.mapv(|value| value * multiplier);
    if transformed.iter().any(|value| !value.is_finite()) {
        return Err(FusionError::ConfigError(format!(
            "GEQDSK source convention {} produced non-finite values",
            convention.as_str()
        )));
    }
    Ok(transformed)
}

fn validate_flux_profile_inputs(psi_norm: &Array2<f64>, profile: &[f64]) -> FusionResult<()> {
    if profile.len() < 3 {
        return Err(FusionError::ConfigError(
            "GEQDSK flux profile must contain at least three samples".to_string(),
        ));
    }
    if psi_norm.iter().any(|value| !value.is_finite())
        || profile.iter().any(|value| !value.is_finite())
    {
        return Err(FusionError::ConfigError(
            "GEQDSK psi_norm and profile samples must contain only finite values".to_string(),
        ));
    }
    Ok(())
}

fn linear_flux_profile_value(psi_norm: f64, profile: &[f64]) -> f64 {
    let x = psi_norm.clamp(0.0, 1.0);
    let scale = (profile.len() - 1) as f64;
    let lower = (x * scale).floor().min(scale - 1.0) as usize;
    let upper = lower + 1;
    let x0 = lower as f64 / scale;
    let x1 = upper as f64 / scale;
    let t = if x1 > x0 { (x - x0) / (x1 - x0) } else { 0.0 };
    profile[lower] * (1.0 - t) + profile[upper] * t
}

fn quadratic_flux_profile_value(psi_norm: f64, profile: &[f64]) -> f64 {
    let x = psi_norm.clamp(0.0, 1.0);
    let n = profile.len();
    let scale = (n - 1) as f64;
    let mut idx = (x * scale).floor() as isize;
    idx = idx.clamp(1, (n - 2) as isize);
    let i = idx as usize;

    let x0 = (i - 1) as f64 / scale;
    let x1 = i as f64 / scale;
    let x2 = (i + 1) as f64 / scale;
    let y0 = profile[i - 1];
    let y1 = profile[i];
    let y2 = profile[i + 1];

    let term0 = y0 * (x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2));
    let term1 = y1 * (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2));
    let term2 = y2 * (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1));
    term0 + term1 + term2
}

/// Interpolate a GEQDSK p'/FF' flux profile with local quadratic stencils.
///
/// Profile samples are assumed to live on a uniform normalized-flux grid
/// `0 <= psi_N <= 1`, matching the GEQDSK profile-array convention. Values are
/// clipped to the profile domain before interpolation.
pub fn interpolate_flux_profile_second_order(
    psi_norm: &Array2<f64>,
    profile: &[f64],
) -> FusionResult<Array2<f64>> {
    validate_flux_profile_inputs(psi_norm, profile)?;
    Ok(psi_norm.mapv(|psi| quadratic_flux_profile_value(psi, profile)))
}

/// Interpolate a flux profile while preserving the masked weighted integral.
///
/// The quadratic interpolation gives second-order local profile accuracy.  The
/// final scalar correction preserves the same masked weighted integral as the
/// linear GEQDSK profile contract, preventing hidden net-current drift when the
/// source is assembled from p' and FF' profiles.
pub fn interpolate_flux_profile_current_conserving(
    psi_norm: &Array2<f64>,
    profile: &[f64],
    weights: &Array2<f64>,
    mask: &Array2<bool>,
) -> FusionResult<Array2<f64>> {
    validate_flux_profile_inputs(psi_norm, profile)?;
    if weights.dim() != psi_norm.dim() {
        return Err(FusionError::ConfigError(format!(
            "GEQDSK profile weights shape mismatch: expected {:?}, got {:?}",
            psi_norm.dim(),
            weights.dim()
        )));
    }
    if mask.dim() != psi_norm.dim() {
        return Err(FusionError::ConfigError(format!(
            "GEQDSK profile mask shape mismatch: expected {:?}, got {:?}",
            psi_norm.dim(),
            mask.dim()
        )));
    }
    if weights.iter().any(|value| !value.is_finite()) {
        return Err(FusionError::ConfigError(
            "GEQDSK profile weights must contain only finite values".to_string(),
        ));
    }
    if weights
        .iter()
        .zip(mask.iter())
        .any(|(weight, include)| *include && *weight < 0.0)
    {
        return Err(FusionError::ConfigError(
            "GEQDSK profile weights must be non-negative on the masked domain".to_string(),
        ));
    }

    let quadratic = interpolate_flux_profile_second_order(psi_norm, profile)?;
    if !mask.iter().any(|include| *include) {
        return Ok(quadratic);
    }

    let mut target_integral = 0.0;
    let mut observed_integral = 0.0;
    for ((psi, weight), include) in psi_norm.iter().zip(weights.iter()).zip(mask.iter()) {
        if *include {
            target_integral += linear_flux_profile_value(*psi, profile) * weight;
            observed_integral += quadratic_flux_profile_value(*psi, profile) * weight;
        }
    }
    let scale = target_integral.abs().max(1.0);
    if target_integral.abs() <= 1.0e-15 && observed_integral.abs() <= 1.0e-15 {
        return Ok(quadratic);
    }
    if observed_integral.abs() <= 1.0e-15 * scale {
        return Err(FusionError::ConfigError(
            "GEQDSK quadratic profile interpolation has zero weighted integral".to_string(),
        ));
    }
    Ok(quadratic.mapv(|value| value * target_integral / observed_integral))
}

fn interior_norm(values: &Array2<f64>) -> f64 {
    if values.nrows() <= 2 || values.ncols() <= 2 {
        return 0.0;
    }
    let mut sum_sq = 0.0;
    for iz in 1..values.nrows() - 1 {
        for ir in 1..values.ncols() - 1 {
            let value = values[[iz, ir]];
            sum_sq += value * value;
        }
    }
    sum_sq.sqrt()
}

fn interior_mask_fraction(mask: &Array2<bool>) -> f64 {
    if mask.nrows() <= 2 || mask.ncols() <= 2 {
        return 0.0;
    }
    let mut count = 0usize;
    let mut total = 0usize;
    for iz in 1..mask.nrows() - 1 {
        for ir in 1..mask.ncols() - 1 {
            total += 1;
            if mask[[iz, ir]] {
                count += 1;
            }
        }
    }
    count as f64 / total as f64
}

/// Assemble GEQDSK-derived Grad-Shafranov source components on an R-Z grid.
///
/// The profile arrays are interpolated through the same second-order,
/// current-conserving flux-profile contract used by Python. Boundary rows and
/// columns are excluded from the physical source domain before the pressure and
/// FFprime terms are assembled.
pub fn compute_geqdsk_profile_source_components(
    psi_norm: &Array2<f64>,
    rr: &Array2<f64>,
    pprime: &[f64],
    ffprime: &[f64],
    mu0: f64,
) -> FusionResult<GeqdskProfileSourceComponents> {
    if psi_norm.dim() != rr.dim() {
        return Err(FusionError::ConfigError(format!(
            "GEQDSK source R grid shape mismatch: psi {:?}, rr {:?}",
            psi_norm.dim(),
            rr.dim()
        )));
    }
    if psi_norm.nrows() < 3 || psi_norm.ncols() < 3 {
        return Err(FusionError::ConfigError(
            "GEQDSK source grid must be at least 3x3".to_string(),
        ));
    }
    if !mu0.is_finite() || mu0 <= 0.0 {
        return Err(FusionError::ConfigError(format!(
            "GEQDSK source requires finite mu0 > 0, got {mu0}"
        )));
    }
    if rr.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Err(FusionError::ConfigError(
            "GEQDSK source R grid must contain finite positive radii".to_string(),
        ));
    }
    validate_flux_profile_inputs(psi_norm, pprime)?;
    validate_flux_profile_inputs(psi_norm, ffprime)?;
    if pprime.len() != ffprime.len() {
        return Err(FusionError::ConfigError(format!(
            "GEQDSK pprime/ffprime lengths must match: got {} and {}",
            pprime.len(),
            ffprime.len()
        )));
    }

    let psi_clipped = psi_norm.mapv(|value| value.clamp(0.0, 1.0));
    let plasma_mask = psi_norm.mapv(|value| (0.0..1.0).contains(&value));
    let mut source_mask = plasma_mask.clone();
    let nrows = source_mask.nrows();
    let ncols = source_mask.ncols();
    for ir in 0..ncols {
        source_mask[[0, ir]] = false;
        source_mask[[nrows - 1, ir]] = false;
    }
    for iz in 0..nrows {
        source_mask[[iz, 0]] = false;
        source_mask[[iz, ncols - 1]] = false;
    }

    let rr_safe = rr.mapv(|value| value.max(1.0e-12));
    let inv_rr = rr_safe.mapv(|value| 1.0 / value);
    let pprime_2d =
        interpolate_flux_profile_current_conserving(&psi_clipped, pprime, &rr_safe, &source_mask)?;
    let ffprime_2d =
        interpolate_flux_profile_current_conserving(&psi_clipped, ffprime, &inv_rr, &source_mask)?;

    let mut pressure_source = Array2::zeros(psi_norm.dim());
    let mut ffprime_source = Array2::zeros(psi_norm.dim());
    let mut total_source = Array2::zeros(psi_norm.dim());
    for iz in 0..psi_norm.nrows() {
        for ir in 0..psi_norm.ncols() {
            if source_mask[[iz, ir]] {
                let pressure = -(mu0 * rr_safe[[iz, ir]].powi(2) * pprime_2d[[iz, ir]]);
                let ff = -ffprime_2d[[iz, ir]];
                pressure_source[[iz, ir]] = pressure;
                ffprime_source[[iz, ir]] = ff;
                total_source[[iz, ir]] = pressure + ff;
            }
        }
    }
    if pressure_source.iter().any(|value| !value.is_finite())
        || ffprime_source.iter().any(|value| !value.is_finite())
        || total_source.iter().any(|value| !value.is_finite())
    {
        return Err(FusionError::ConfigError(
            "GEQDSK source components produced non-finite values".to_string(),
        ));
    }

    Ok(GeqdskProfileSourceComponents {
        pressure_source_norm: interior_norm(&pressure_source),
        ffprime_source_norm: interior_norm(&ffprime_source),
        total_source_norm: interior_norm(&total_source),
        plasma_mask_fraction: interior_mask_fraction(&plasma_mask),
        pressure_source,
        ffprime_source,
        total_source,
        plasma_mask,
    })
}

fn source_relative_l2(
    operator_source: &Array2<f64>,
    candidate_source: &Array2<f64>,
) -> FusionResult<f64> {
    if operator_source.dim() != candidate_source.dim() {
        return Err(FusionError::ConfigError(format!(
            "source convention shape mismatch: operator {:?}, candidate {:?}",
            operator_source.dim(),
            candidate_source.dim()
        )));
    }
    if operator_source.iter().any(|value| !value.is_finite())
        || candidate_source.iter().any(|value| !value.is_finite())
    {
        return Err(FusionError::ConfigError(
            "source convention arrays must contain only finite values".to_string(),
        ));
    }
    let operator_norm = operator_source
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if !operator_norm.is_finite() || operator_norm <= 1.0e-15 {
        return Err(FusionError::ConfigError(
            "source convention operator norm must be finite and non-zero".to_string(),
        ));
    }
    let residual = operator_source
        .iter()
        .zip(candidate_source.iter())
        .map(|(operator, candidate)| {
            let diff = operator - candidate;
            diff * diff
        })
        .sum::<f64>()
        .sqrt()
        / operator_norm;
    if !residual.is_finite() {
        return Err(FusionError::ConfigError(
            "source convention residual became non-finite".to_string(),
        ));
    }
    Ok(residual)
}

fn executable_geqdsk_source_conventions(flux_span: f64) -> Vec<GeqdskSourceConvention> {
    let mut conventions = vec![
        GeqdskSourceConvention::Canonical,
        GeqdskSourceConvention::Negated,
        GeqdskSourceConvention::ScaledByTwoPi,
        GeqdskSourceConvention::ScaledByMinusTwoPi,
        GeqdskSourceConvention::ScaledByInvTwoPi,
        GeqdskSourceConvention::ScaledByMinusInvTwoPi,
    ];
    if flux_span.is_finite() && flux_span.abs() >= 1.0e-15 {
        conventions.extend([
            GeqdskSourceConvention::TimesFluxSpan,
            GeqdskSourceConvention::OverFluxSpan,
            GeqdskSourceConvention::NegatedTimesFluxSpan,
            GeqdskSourceConvention::NegatedOverFluxSpan,
        ]);
    }
    conventions
}

/// Rank executable named GEQDSK source-convention transforms by relative L2 residual.
///
/// This mirrors the Python benchmark adapter's audit surface: only documented
/// named transforms are ranked, flux-span transforms are omitted unless the
/// physical flux span is finite and non-zero, and fitted scales are not
/// represented as executable candidates.
pub fn rank_geqdsk_source_convention_candidates(
    operator_source: &Array2<f64>,
    profile_source: &Array2<f64>,
    flux_span: f64,
) -> FusionResult<Vec<GeqdskSourceConventionCandidate>> {
    let mut candidates = Vec::new();
    for convention in executable_geqdsk_source_conventions(flux_span) {
        let candidate = apply_geqdsk_source_convention(profile_source, convention, flux_span)?;
        let residual_l2 = source_relative_l2(operator_source, &candidate)?;
        candidates.push(GeqdskSourceConventionCandidate {
            convention,
            residual_l2,
        });
    }
    candidates.sort_by(|left, right| {
        left.residual_l2
            .partial_cmp(&right.residual_l2)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(candidates)
}

/// Select the best named GEQDSK source convention without accepting fitted scales.
pub fn select_geqdsk_source_convention_adapter(
    operator_source: &Array2<f64>,
    profile_source: &Array2<f64>,
    flux_span: f64,
    residual_threshold: f64,
) -> FusionResult<GeqdskSourceConventionAdapter> {
    if !residual_threshold.is_finite() || residual_threshold <= 0.0 {
        return Err(FusionError::ConfigError(
            "source convention residual threshold must be finite and positive".to_string(),
        ));
    }
    let Some(best) =
        rank_geqdsk_source_convention_candidates(operator_source, profile_source, flux_span)?
            .into_iter()
            .next()
    else {
        return Ok(GeqdskSourceConventionAdapter {
            convention: GeqdskSourceConvention::NotEvaluated,
            residual_l2: f64::INFINITY,
            pass: false,
        });
    };
    Ok(GeqdskSourceConventionAdapter {
        convention: best.convention,
        residual_l2: best.residual_l2,
        pass: best.residual_l2 <= residual_threshold,
    })
}

/// mTanh profile:
/// f(psi_n) = 0.5 * h * (1 + tanh(y)) + alpha * core(psi_n)
/// y = (ped_top - psi_n) / ped_width
/// core(psi_n) = max(0, 1 - (psi_n/ped_top)^2)
pub fn mtanh_profile(psi_norm: f64, params: &ProfileParams) -> f64 {
    let w = params.ped_width.abs().max(1e-8);
    let ped_top = params.ped_top.abs().max(1e-8);
    let y = (params.ped_top - psi_norm) / w;
    let tanh_y = y.tanh();
    let core = (1.0 - (psi_norm / ped_top).powi(2)).max(0.0);
    0.5 * params.ped_height * (1.0 + tanh_y) + params.core_alpha * core
}

/// Analytical derivatives of mTanh profile with respect to:
/// [ped_height, ped_top, ped_width, core_alpha].
pub fn mtanh_profile_derivatives(psi_norm: f64, params: &ProfileParams) -> [f64; 4] {
    let w = params.ped_width.abs().max(1e-8);
    let ped_top = params.ped_top.abs().max(1e-8);
    let y = (params.ped_top - psi_norm) / w;
    let tanh_y = y.tanh();
    let sech2 = 1.0 - tanh_y * tanh_y;
    let core = (1.0 - (psi_norm / ped_top).powi(2)).max(0.0);

    let d_core_d_ped_top = if psi_norm.abs() < ped_top {
        2.0 * psi_norm.powi(2) / ped_top.powi(3)
    } else {
        0.0
    };

    let d_ped_height = 0.5 * (1.0 + tanh_y);
    let d_ped_top = 0.5 * params.ped_height * sech2 / w + params.core_alpha * d_core_d_ped_top;
    let d_ped_width = -0.5 * params.ped_height * sech2 * y / w;
    let d_core_alpha = core;

    [d_ped_height, d_ped_top, d_ped_width, d_core_alpha]
}

/// Update the toroidal current density J_phi using the full Grad-Shafranov source term.
///
/// Algorithm:
/// 1. Normalize flux: ψ_norm = (Ψ - Ψ_axis) / (Ψ_boundary - Ψ_axis)
/// 2. Define profile shape: f(ψ_norm) = (1 - ψ_norm) inside plasma (0 ≤ ψ_norm < 1)
/// 3. Pressure term: J_p = R · f(ψ_norm)
/// 4. Current term: J_f = (1/(μ₀R)) · f(ψ_norm)
/// 5. Mix: J_raw = β_mix · J_p + (1 - β_mix) · J_f
/// 6. Renormalize: scale J_raw so that ∫J_phi dR dZ = I_target
///
/// Returns the updated J_phi `[nz, nr]`.
pub fn update_plasma_source_nonlinear(
    psi: &Array2<f64>,
    grid: &Grid2D,
    psi_axis: f64,
    psi_boundary: f64,
    mu0: f64,
    i_target: f64,
) -> FusionResult<Array2<f64>> {
    validate_source_inputs(psi, grid, psi_axis, psi_boundary, mu0, i_target)?;
    let nz = grid.nz;
    let nr = grid.nr;

    // Normalize flux
    let denom = psi_boundary - psi_axis;

    let mut j_phi = Array2::zeros((nz, nr));

    for iz in 0..nz {
        for ir in 0..nr {
            let psi_norm = (psi[[iz, ir]] - psi_axis) / denom;
            if !psi_norm.is_finite() {
                return Err(FusionError::ConfigError(format!(
                    "source update produced non-finite psi_norm at ({iz}, {ir})"
                )));
            }

            // Only inside plasma (0 ≤ ψ_norm < 1)
            if (0.0..1.0).contains(&psi_norm) {
                let profile = 1.0 - psi_norm;

                let r = grid.rr[[iz, ir]];
                if r <= 0.0 {
                    return Err(FusionError::ConfigError(format!(
                        "source update requires R > 0 inside plasma at ({iz}, {ir}), got {r}"
                    )));
                }

                // Pressure-driven current (dominates at large R)
                let j_p = r * profile;

                // Poloidal field current (dominates at small R)
                let j_f = (1.0 / (mu0 * r)) * profile;
                if !j_p.is_finite() || !j_f.is_finite() {
                    return Err(FusionError::ConfigError(format!(
                        "source update produced non-finite current components at ({iz}, {ir})"
                    )));
                }

                // Mix
                j_phi[[iz, ir]] = DEFAULT_BETA_MIX * j_p + (1.0 - DEFAULT_BETA_MIX) * j_f;
            }
        }
    }

    // Renormalize to match target current
    let i_current: f64 = j_phi.iter().sum::<f64>() * grid.dr * grid.dz;
    if !i_current.is_finite() {
        return Err(FusionError::ConfigError(
            "source update current integral became non-finite".to_string(),
        ));
    }

    if i_current.abs() > MIN_CURRENT_INTEGRAL {
        let scale = i_target / i_current;
        if !scale.is_finite() {
            return Err(FusionError::ConfigError(
                "source update renormalization scale became non-finite".to_string(),
            ));
        }
        j_phi.mapv_inplace(|v| v * scale);
    } else {
        j_phi.fill(0.0);
    }

    if j_phi.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::ConfigError(
            "source update output contains non-finite values".to_string(),
        ));
    }
    Ok(j_phi)
}

/// Update toroidal current density using externally provided profile parameters.
///
/// This is the kernel-facing hook used by inverse reconstruction when profile
/// parameters are being estimated from measurements.
pub fn update_plasma_source_with_profiles(
    ctx: SourceProfileContext<'_>,
    params_p: &ProfileParams,
    params_ff: &ProfileParams,
) -> FusionResult<Array2<f64>> {
    let SourceProfileContext {
        psi,
        grid,
        psi_axis,
        psi_boundary,
        mu0,
        i_target,
    } = ctx;
    validate_source_inputs(psi, grid, psi_axis, psi_boundary, mu0, i_target)?;
    validate_profile_params(params_p, "params_p")?;
    validate_profile_params(params_ff, "params_ff")?;
    let nz = grid.nz;
    let nr = grid.nr;

    let denom = psi_boundary - psi_axis;

    let mut j_phi = Array2::zeros((nz, nr));
    for iz in 0..nz {
        for ir in 0..nr {
            let psi_norm = (psi[[iz, ir]] - psi_axis) / denom;
            if !psi_norm.is_finite() {
                return Err(FusionError::ConfigError(format!(
                    "profile source update produced non-finite psi_norm at ({iz}, {ir})"
                )));
            }
            if (0.0..1.0).contains(&psi_norm) {
                let r = grid.rr[[iz, ir]];
                if r <= 0.0 {
                    return Err(FusionError::ConfigError(format!(
                        "profile source update requires R > 0 inside plasma at ({iz}, {ir}), got {r}"
                    )));
                }
                let p_profile = mtanh_profile(psi_norm, params_p);
                let ff_profile = mtanh_profile(psi_norm, params_ff);
                if !p_profile.is_finite() || !ff_profile.is_finite() {
                    return Err(FusionError::ConfigError(format!(
                        "profile source update produced non-finite profile values at ({iz}, {ir})"
                    )));
                }

                let j_p = r * p_profile;
                let j_f = (1.0 / (mu0 * r)) * ff_profile;
                if !j_p.is_finite() || !j_f.is_finite() {
                    return Err(FusionError::ConfigError(format!(
                        "profile source update produced non-finite current components at ({iz}, {ir})"
                    )));
                }
                j_phi[[iz, ir]] = DEFAULT_BETA_MIX * j_p + (1.0 - DEFAULT_BETA_MIX) * j_f;
            }
        }
    }

    let i_current: f64 = j_phi.iter().sum::<f64>() * grid.dr * grid.dz;
    if !i_current.is_finite() {
        return Err(FusionError::ConfigError(
            "profile source update current integral became non-finite".to_string(),
        ));
    }
    if i_current.abs() > MIN_CURRENT_INTEGRAL {
        let scale = i_target / i_current;
        if !scale.is_finite() {
            return Err(FusionError::ConfigError(
                "profile source update renormalization scale became non-finite".to_string(),
            ));
        }
        j_phi.mapv_inplace(|v| v * scale);
    } else {
        j_phi.fill(0.0);
    }
    if j_phi.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::ConfigError(
            "profile source update output contains non-finite values".to_string(),
        ));
    }
    Ok(j_phi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_zero_outside_plasma() {
        let grid = Grid2D::new(16, 16, 1.0, 9.0, -5.0, 5.0);
        // Ψ everywhere = 0, axis = 1.0, boundary = 0.0
        // ψ_norm = (0 - 1) / (0 - 1) = 1.0 → outside plasma
        let psi = Array2::zeros((16, 16));
        let j = update_plasma_source_nonlinear(&psi, &grid, 1.0, 0.0, 1.0, 1e6)
            .expect("valid source-update inputs");

        // Everything should be zero (all ψ_norm = 1.0, exactly at boundary)
        let max_j = j.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
        assert!(max_j < 1e-15, "Should be zero outside plasma: {max_j}");
    }

    #[test]
    fn test_source_renormalization() {
        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        // Create Gaussian-like Ψ peaked at center
        let psi = Array2::from_shape_fn((33, 33), |(iz, ir)| {
            let r = grid.rr[[iz, ir]];
            let z = grid.zz[[iz, ir]];
            (-(((r - 5.0).powi(2) + z.powi(2)) / 4.0)).exp()
        });

        let psi_axis = 1.0; // peak
        let psi_boundary = 0.0; // edge
        let i_target = 15e6; // 15 MA

        let j = update_plasma_source_nonlinear(&psi, &grid, psi_axis, psi_boundary, 1.0, i_target)
            .expect("valid source-update inputs");

        // Check integral matches target
        let i_actual: f64 = j.iter().sum::<f64>() * grid.dr * grid.dz;
        let rel_error = ((i_actual - i_target) / i_target).abs();
        assert!(
            rel_error < 1e-10,
            "Current integral {i_actual} should match target {i_target}"
        );
    }

    #[test]
    fn test_mtanh_derivatives_match_finite_difference() {
        let params = ProfileParams {
            ped_top: 0.92,
            ped_width: 0.07,
            ped_height: 1.2,
            core_alpha: 0.3,
        };
        let psi = 0.35;
        let analytic = mtanh_profile_derivatives(psi, &params);
        let eps = 1e-6;

        let mut p = params;
        p.ped_height += eps;
        let fd_h = (mtanh_profile(psi, &p) - mtanh_profile(psi, &params)) / eps;

        p = params;
        p.ped_top += eps;
        let fd_top = (mtanh_profile(psi, &p) - mtanh_profile(psi, &params)) / eps;

        p = params;
        p.ped_width += eps;
        let fd_w = (mtanh_profile(psi, &p) - mtanh_profile(psi, &params)) / eps;

        p = params;
        p.core_alpha += eps;
        let fd_a = (mtanh_profile(psi, &p) - mtanh_profile(psi, &params)) / eps;

        let fd = [fd_h, fd_top, fd_w, fd_a];
        for i in 0..4 {
            let denom = fd[i].abs().max(1e-8);
            let rel = (analytic[i] - fd[i]).abs() / denom;
            assert!(
                rel < 1e-3,
                "Derivative mismatch at index {i}: analytic={}, fd={}, rel={}",
                analytic[i],
                fd[i],
                rel
            );
        }
    }

    #[test]
    fn test_source_with_profiles_finite() {
        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        let psi = Array2::from_shape_fn((33, 33), |(iz, ir)| {
            let r = grid.rr[[iz, ir]];
            let z = grid.zz[[iz, ir]];
            (-(((r - 5.0).powi(2) + z.powi(2)) / 4.0)).exp()
        });

        let params_p = ProfileParams {
            ped_top: 0.9,
            ped_width: 0.08,
            ped_height: 1.1,
            core_alpha: 0.25,
        };
        let params_ff = ProfileParams {
            ped_top: 0.85,
            ped_width: 0.06,
            ped_height: 0.95,
            core_alpha: 0.1,
        };

        let j = update_plasma_source_with_profiles(
            SourceProfileContext {
                psi: &psi,
                grid: &grid,
                psi_axis: 1.0,
                psi_boundary: 0.0,
                mu0: 1.0,
                i_target: 15e6,
            },
            &params_p,
            &params_ff,
        )
        .expect("valid profile-source-update inputs");
        assert!(
            j.iter().all(|v| v.is_finite()),
            "Profile source contains non-finite values"
        );
        let i_actual: f64 = j.iter().sum::<f64>() * grid.dr * grid.dz;
        let rel_error = ((i_actual - 15e6) / 15e6).abs();
        assert!(rel_error < 1e-10, "Current mismatch after renormalization");
    }

    #[test]
    fn test_geqdsk_source_convention_adapter_accepts_named_2pi_contract() {
        let profile_source =
            Array2::from_shape_fn((5, 5), |(iz, ir)| 0.25 + iz as f64 + 0.5 * ir as f64);
        let operator_source = profile_source.mapv(|value| value * 2.0 * std::f64::consts::PI);

        let adapter =
            select_geqdsk_source_convention_adapter(&operator_source, &profile_source, 0.4, 0.15)
                .expect("finite source arrays should rank source conventions");

        assert_eq!(adapter.convention, GeqdskSourceConvention::ScaledByTwoPi);
        assert!(adapter.pass);
        assert!(adapter.residual_l2 < 1.0e-14);
    }

    #[test]
    fn test_flux_profile_second_order_is_exact_for_quadratic_profile() {
        let profile: Vec<f64> = (0..5)
            .map(|idx| {
                let x = idx as f64 / 4.0;
                1.0 - 0.5 * x + 2.0 * x * x
            })
            .collect();
        let psi_norm = Array2::from_shape_vec((2, 3), vec![0.0, 0.125, 0.375, 0.5, 0.875, 1.0])
            .expect("valid shape");

        let interpolated = interpolate_flux_profile_second_order(&psi_norm, &profile)
            .expect("valid quadratic interpolation");

        for (psi, value) in psi_norm.iter().zip(interpolated.iter()) {
            let expected = 1.0 - 0.5 * psi + 2.0 * psi * psi;
            assert!((value - expected).abs() < 1.0e-14);
        }
    }

    #[test]
    fn test_current_conserving_flux_profile_preserves_weighted_linear_integral() {
        let profile = vec![0.25, 0.7, 1.4, 2.0, 2.8];
        let psi_norm = Array2::from_shape_vec((2, 3), vec![0.05, 0.21, 0.43, 0.66, 0.82, 0.98])
            .expect("valid shape");
        let weights = Array2::from_shape_vec((2, 3), vec![1.0, 1.5, 2.0, 0.5, 1.25, 0.75])
            .expect("valid shape");
        let mask = Array2::from_shape_vec((2, 3), vec![true, true, false, true, true, true])
            .expect("valid shape");

        let interpolated =
            interpolate_flux_profile_current_conserving(&psi_norm, &profile, &weights, &mask)
                .expect("valid current-conserving interpolation");
        let target_integral: f64 = psi_norm
            .iter()
            .zip(weights.iter())
            .zip(mask.iter())
            .filter_map(|((psi, weight), include)| {
                include.then_some(linear_flux_profile_value(*psi, &profile) * weight)
            })
            .sum();
        let observed_integral: f64 = interpolated
            .iter()
            .zip(weights.iter())
            .zip(mask.iter())
            .filter_map(|((value, weight), include)| include.then_some(value * weight))
            .sum();

        assert!((observed_integral - target_integral).abs() < 1.0e-12);
    }

    #[test]
    fn test_current_conserving_flux_profile_rejects_invalid_contracts() {
        let psi_norm = Array2::from_elem((2, 2), 0.5);
        let weights = Array2::from_elem((2, 2), 1.0);
        let mask = Array2::from_elem((2, 2), true);

        assert!(interpolate_flux_profile_current_conserving(
            &psi_norm,
            &[1.0, 2.0],
            &weights,
            &mask,
        )
        .is_err());

        let bad_weights = Array2::from_elem((2, 2), -1.0);
        assert!(interpolate_flux_profile_current_conserving(
            &psi_norm,
            &[0.0, 1.0, 2.0],
            &bad_weights,
            &mask,
        )
        .is_err());

        let wrong_shape = Array2::from_elem((1, 4), true);
        assert!(interpolate_flux_profile_current_conserving(
            &psi_norm,
            &[0.0, 1.0, 2.0],
            &weights,
            &wrong_shape,
        )
        .is_err());
    }

    #[test]
    fn test_geqdsk_profile_source_components_mask_boundary_and_report_norms() {
        let psi_norm = Array2::from_shape_fn((5, 5), |(iz, ir)| {
            if iz == 0 || ir == 0 || iz == 4 || ir == 4 {
                1.0
            } else {
                0.2 + 0.05 * iz as f64 + 0.04 * ir as f64
            }
        });
        let rr = Array2::from_shape_fn((5, 5), |(_iz, ir)| 1.0 + 0.25 * ir as f64);
        let pprime = vec![0.2, 0.4, 0.7, 1.0, 1.4];
        let ffprime = vec![0.1, 0.05, -0.05, -0.1, -0.2];

        let components =
            compute_geqdsk_profile_source_components(&psi_norm, &rr, &pprime, &ffprime, 1.0)
                .expect("valid GEQDSK source components");

        for idx in 0..5 {
            assert_eq!(components.total_source[[0, idx]], 0.0);
            assert_eq!(components.total_source[[4, idx]], 0.0);
            assert_eq!(components.total_source[[idx, 0]], 0.0);
            assert_eq!(components.total_source[[idx, 4]], 0.0);
        }
        assert!(components.plasma_mask_fraction > 0.0);
        assert!(components.pressure_source_norm > 0.0);
        assert!(components.ffprime_source_norm > 0.0);
        assert!(components.total_source_norm > 0.0);
    }

    #[test]
    fn test_geqdsk_profile_source_components_preserve_total_source_identity() {
        let psi_norm = Array2::from_shape_fn((4, 4), |(iz, ir)| 0.1 + 0.1 * (iz + ir) as f64);
        let rr = Array2::from_shape_fn((4, 4), |(_iz, ir)| 1.2 + 0.1 * ir as f64);
        let pprime = vec![0.5, 0.75, 1.0, 1.25];
        let ffprime = vec![0.25, 0.1, -0.1, -0.25];

        let components =
            compute_geqdsk_profile_source_components(&psi_norm, &rr, &pprime, &ffprime, 1.0)
                .expect("valid GEQDSK source components");

        for ((pressure, ffprime), total) in components
            .pressure_source
            .iter()
            .zip(components.ffprime_source.iter())
            .zip(components.total_source.iter())
        {
            assert!((*pressure + *ffprime - *total).abs() < 1.0e-14);
        }
    }

    #[test]
    fn test_geqdsk_profile_source_components_reject_invalid_inputs() {
        let psi_norm = Array2::from_elem((3, 3), 0.5);
        let rr = Array2::from_elem((3, 3), 1.0);
        assert!(compute_geqdsk_profile_source_components(
            &psi_norm,
            &rr,
            &[1.0, 2.0],
            &[1.0, 2.0],
            1.0,
        )
        .is_err());

        let bad_rr = Array2::from_elem((3, 3), 0.0);
        assert!(compute_geqdsk_profile_source_components(
            &psi_norm,
            &bad_rr,
            &[0.0, 1.0, 2.0],
            &[0.0, 1.0, 2.0],
            1.0,
        )
        .is_err());

        assert!(compute_geqdsk_profile_source_components(
            &psi_norm,
            &rr,
            &[0.0, 1.0, 2.0],
            &[0.0, 1.0, 2.0, 3.0],
            1.0,
        )
        .is_err());
    }

    #[test]
    fn test_geqdsk_source_convention_adapter_rejects_unclassified_scale() {
        let profile_source = Array2::from_shape_fn((5, 5), |(iz, ir)| 1.0 + iz as f64 + ir as f64);
        let operator_source = profile_source.mapv(|value| value * 3.0);

        let adapter =
            select_geqdsk_source_convention_adapter(&operator_source, &profile_source, 0.4, 0.15)
                .expect("finite source arrays should rank source conventions");

        assert_ne!(adapter.convention, GeqdskSourceConvention::NotEvaluated);
        assert!(!adapter.pass);
        assert!(adapter.residual_l2 > 0.15);
    }

    #[test]
    fn test_geqdsk_source_convention_rankings_expose_named_candidates_only() {
        let profile_source =
            Array2::from_shape_fn((4, 4), |(iz, ir)| 0.5 + iz as f64 + 0.25 * ir as f64);
        let operator_source = profile_source.mapv(|value| value / 0.5);

        let ranked =
            rank_geqdsk_source_convention_candidates(&operator_source, &profile_source, 0.5)
                .expect("finite source arrays should rank executable source conventions");

        assert_eq!(ranked[0].convention, GeqdskSourceConvention::OverFluxSpan);
        assert!(ranked[0].residual_l2 < 1.0e-14);
        assert_eq!(ranked.len(), 10);
        assert!(!ranked
            .iter()
            .any(|candidate| candidate.convention == GeqdskSourceConvention::NotEvaluated));

        let degenerate_flux_span =
            rank_geqdsk_source_convention_candidates(&operator_source, &profile_source, 0.0)
                .expect("degenerate flux span should still rank finite non-flux conventions");
        assert_eq!(degenerate_flux_span.len(), 6);
        assert!(!degenerate_flux_span.iter().any(|candidate| {
            matches!(
                candidate.convention,
                GeqdskSourceConvention::TimesFluxSpan
                    | GeqdskSourceConvention::OverFluxSpan
                    | GeqdskSourceConvention::NegatedTimesFluxSpan
                    | GeqdskSourceConvention::NegatedOverFluxSpan
            )
        }));
    }

    #[test]
    fn test_geqdsk_source_convention_transform_supports_flux_span_and_rejects_degenerate_span() {
        let source = Array2::from_elem((3, 3), 2.0);
        let transformed =
            apply_geqdsk_source_convention(&source, GeqdskSourceConvention::OverFluxSpan, 0.5)
                .expect("non-degenerate flux span should be accepted");

        assert!(transformed
            .iter()
            .all(|value| (*value - 4.0).abs() < 1.0e-15));
        assert!(
            apply_geqdsk_source_convention(&source, GeqdskSourceConvention::OverFluxSpan, 0.0,)
                .is_err()
        );
    }

    #[test]
    fn test_geqdsk_source_convention_names_round_trip_strictly() {
        let conventions = [
            GeqdskSourceConvention::Canonical,
            GeqdskSourceConvention::Negated,
            GeqdskSourceConvention::ScaledByTwoPi,
            GeqdskSourceConvention::ScaledByMinusTwoPi,
            GeqdskSourceConvention::ScaledByInvTwoPi,
            GeqdskSourceConvention::ScaledByMinusInvTwoPi,
            GeqdskSourceConvention::TimesFluxSpan,
            GeqdskSourceConvention::OverFluxSpan,
            GeqdskSourceConvention::NegatedTimesFluxSpan,
            GeqdskSourceConvention::NegatedOverFluxSpan,
            GeqdskSourceConvention::NotEvaluated,
        ];

        for convention in conventions {
            let parsed = GeqdskSourceConvention::try_from(convention.as_str())
                .expect("documented convention labels should parse");
            assert_eq!(parsed, convention);
        }

        let err = GeqdskSourceConvention::try_from("least_squares_fit")
            .expect_err("fitted conventions are not executable native adapters");
        assert!(matches!(err, FusionError::ConfigError(_)));
    }

    #[test]
    fn test_source_rejects_invalid_runtime_inputs() {
        let mut grid = Grid2D::new(16, 16, 1.0, 9.0, -5.0, 5.0);
        let psi = Array2::zeros((16, 16));

        let err = update_plasma_source_nonlinear(&psi, &grid, 1.0, 1.0, 1.0, 1.0)
            .expect_err("degenerate flux normalization must fail");
        assert!(matches!(err, FusionError::ConfigError(_)));

        grid.rr[[3, 3]] = 0.0;
        let psi_inside = Array2::from_elem((16, 16), 0.5);
        let err = update_plasma_source_nonlinear(&psi_inside, &grid, 1.0, 0.0, 1.0, 1.0)
            .expect_err("non-positive radius inside plasma must fail");
        assert!(matches!(err, FusionError::ConfigError(_)));

        let params_bad = ProfileParams {
            ped_top: 0.9,
            ped_width: 0.0,
            ped_height: 1.0,
            core_alpha: 0.2,
        };
        let err = update_plasma_source_with_profiles(
            SourceProfileContext {
                psi: &Array2::from_elem((16, 16), 0.5),
                grid: &Grid2D::new(16, 16, 1.0, 9.0, -5.0, 5.0),
                psi_axis: 1.0,
                psi_boundary: 0.0,
                mu0: 1.0,
                i_target: 1.0,
            },
            &params_bad,
            &ProfileParams::default(),
        )
        .expect_err("invalid profile params must fail");
        assert!(matches!(err, FusionError::ConfigError(_)));
    }
}
