// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Native Rust Polyglot GS Solver
//! Dependency-light native Grad-Shafranov reference solver.
//!
//! This crate parses the governed `[grad_shafranov]` case format, solves the
//! fixed-boundary equation with Picard/Jacobi iteration, and exposes matching
//! flux-operator and toroidal-current diagnostics for polyglot validation.
#![deny(missing_docs)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

mod case;

use case::validate_case;
pub use case::{load_case, parse_case};

#[derive(Clone, Debug)]
/// Inputs for one fixed-boundary Grad-Shafranov solve.
///
/// The domain is a uniformly sampled rectangular R-Z grid. All coefficients
/// are validated by [`parse_case`] and [`solve_grad_shafranov`] before use.
pub struct GradShafranovCase {
    /// Minimum major radius in metres; must be positive.
    pub r_min: f64,
    /// Maximum major radius in metres; must exceed [`Self::r_min`].
    pub r_max: f64,
    /// Minimum vertical coordinate in metres.
    pub z_min: f64,
    /// Maximum vertical coordinate in metres; must exceed [`Self::z_min`].
    pub z_max: f64,
    /// Number of uniformly spaced radial grid points; must be at least three.
    pub nr: usize,
    /// Number of uniformly spaced vertical grid points; must be at least three.
    pub nz: usize,
    /// Target total toroidal plasma current in amperes.
    pub ip_target: f64,
    /// Vacuum permeability used by the source and current-density relations.
    pub mu0: f64,
    /// Number of outer nonlinear Picard iterations; must be non-zero.
    pub n_picard: usize,
    /// Number of inner Jacobi sweeps per Picard iteration; must be non-zero.
    pub n_jacobi: usize,
    /// Picard update fraction in the interval `(0, 1]`.
    pub alpha: f64,
    /// Jacobi relaxation fraction in the interval `(0, 2)`.
    pub omega_j: f64,
    /// Pressure-versus-poloidal-current source mixture in `[0, 1]`.
    pub beta_mix: f64,
}

#[derive(Clone, Debug)]
/// Native grid and poloidal-flux result from [`solve_grad_shafranov`].
pub struct GradShafranovResult {
    /// Uniform radial coordinates, ordered from `r_min` through `r_max`.
    pub r: Vec<f64>,
    /// Uniform vertical coordinates, ordered from `z_min` through `z_max`.
    pub z: Vec<f64>,
    /// Row-major flux matrix indexed as `psi[z_index][r_index]`.
    pub psi: Vec<Vec<f64>>,
}

/// Solves one validated fixed-boundary Grad-Shafranov case.
///
/// The returned flux matrix has shape `(case.nz, case.nr)`, is indexed as
/// `[z][r]`, and has zero Dirichlet values on every boundary.
///
/// # Errors
///
/// Returns an error string when any case invariant accepted by [`parse_case`]
/// is violated, including when a caller constructs the public case directly.
pub fn solve_grad_shafranov(case: &GradShafranovCase) -> Result<GradShafranovResult, String> {
    validate_case(case)?;

    let r = linspace(case.r_min, case.r_max, case.nr);
    let z = linspace(case.z_min, case.z_max, case.nz);
    let dr = (case.r_max - case.r_min) / ((case.nr - 1) as f64);
    let dz = (case.z_max - case.z_min) / ((case.nz - 1) as f64);
    if !((dr * dr).is_finite() && (dz * dz).is_finite() && dr * dr > 0.0 && dz * dz > 0.0) {
        return Err("solver spacing arithmetic became non-finite or degenerate".to_owned());
    }
    let r_center = 0.5 * (case.r_min + case.r_max);
    if !r_center.is_finite() {
        return Err("initial Gaussian centre became non-finite".to_owned());
    }

    let mut psi = vec![vec![0.0; case.nr]; case.nz];
    for row in &mut psi {
        for (ir, value) in row.iter_mut().enumerate() {
            let exponent = -((r[ir] - r_center).powi(2)) / 0.5;
            if !exponent.is_finite() {
                return Err("initial Gaussian exponent became non-finite".to_owned());
            }
            *value = exponent.exp() * 0.01;
            if !value.is_finite() {
                return Err("initial flux became non-finite".to_owned());
            }
        }
    }
    enforce_boundary(&mut psi);

    for _ in 0..case.n_picard {
        let source = compute_source(case, &r, &psi, dr, dz)?;
        let mut elliptic = psi.clone();
        for _ in 0..case.n_jacobi {
            elliptic = jacobi_step(case, &r, &elliptic, &source)?;
        }
        for iz in 0..case.nz {
            for ir in 0..case.nr {
                psi[iz][ir] = (1.0 - case.alpha) * psi[iz][ir] + case.alpha * elliptic[iz][ir];
                if !psi[iz][ir].is_finite() {
                    return Err("Picard flux became non-finite".to_owned());
                }
            }
        }
        enforce_boundary(&mut psi);
    }

    Ok(GradShafranovResult { r, z, psi })
}

fn linspace(min: f64, max: f64, count: usize) -> Vec<f64> {
    let step = (max - min) / ((count - 1) as f64);
    (0..count).map(|idx| min + step * (idx as f64)).collect()
}

/// Build the current profile without hiding non-finite arithmetic in clamps/reductions.
fn compute_source(
    case: &GradShafranovCase,
    r: &[f64],
    psi: &[Vec<f64>],
    dr: f64,
    dz: f64,
) -> Result<Vec<Vec<f64>>, String> {
    let mut psi_axis = f64::NEG_INFINITY;
    for row in psi.iter().take(case.nz - 1).skip(1) {
        for value in row.iter().take(case.nr - 1).skip(1) {
            psi_axis = psi_axis.max(*value);
        }
    }

    let mut denominator = -psi_axis;
    if denominator.abs() < 1.0e-9 {
        denominator = if denominator.is_sign_negative() {
            -1.0e-9
        } else {
            1.0e-9
        };
    }

    let mut raw = vec![vec![0.0; case.nr]; case.nz];
    let mut current = 0.0;
    for iz in 0..case.nz {
        for (ir, radius) in r.iter().enumerate() {
            let psi_norm = (psi[iz][ir] - psi_axis) / denominator;
            if !psi_norm.is_finite() {
                return Err("normalised flux became non-finite".to_owned());
            }
            let psi_norm = psi_norm.clamp(0.0, 1.0);
            let profile = if (0.0..1.0).contains(&psi_norm) {
                1.0 - psi_norm
            } else {
                0.0
            };
            let jp = radius * profile;
            let denominator = case.mu0 * radius.max(1.0e-6);
            if !denominator.is_finite() || denominator == 0.0 {
                return Err("source denominator became non-finite or zero".to_owned());
            }
            let jf = profile / denominator;
            raw[iz][ir] = case.beta_mix * jp + (1.0 - case.beta_mix) * jf;
            if !(jp.is_finite() && jf.is_finite() && raw[iz][ir].is_finite()) {
                return Err("current profile became non-finite".to_owned());
            }
            current += raw[iz][ir] * dr * dz;
            if !current.is_finite() {
                return Err("profile current became non-finite".to_owned());
            }
        }
    }

    let scale = case.ip_target / current.abs().max(1.0e-9);
    if !scale.is_finite() {
        return Err("current scaling became non-finite".to_owned());
    }
    let mut source = vec![vec![0.0; case.nr]; case.nz];
    for iz in 0..case.nz {
        for (ir, radius) in r.iter().enumerate() {
            let scaled_current = raw[iz][ir] * scale;
            if !scaled_current.is_finite() {
                return Err("scaled current became non-finite".to_owned());
            }
            source[iz][ir] = -case.mu0 * radius * scaled_current;
            if !source[iz][ir].is_finite() {
                return Err("GS source became non-finite".to_owned());
            }
        }
    }
    Ok(source)
}

/// Apply one checked relaxation sweep and propagate numerical failure.
fn jacobi_step(
    case: &GradShafranovCase,
    r: &[f64],
    psi: &[Vec<f64>],
    source: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, String> {
    let dr = (case.r_max - case.r_min) / ((case.nr - 1) as f64);
    let dz = (case.z_max - case.z_min) / ((case.nz - 1) as f64);
    let dr2 = dr * dr;
    let dz2 = dz * dz;
    let a_ns = 1.0 / dz2;
    let a_c = 2.0 / dr2 + 2.0 / dz2;
    if !(a_ns.is_finite() && a_c.is_finite() && a_ns > 0.0 && a_c > 0.0) {
        return Err("Jacobi coefficient arithmetic failed".to_owned());
    }

    let mut out = psi.to_vec();
    for iz in 1..case.nz - 1 {
        for (ir, radius) in r.iter().enumerate().take(case.nr - 1).skip(1) {
            let ae = 1.0 / dr2 - 1.0 / (2.0 * radius * dr);
            let aw = 1.0 / dr2 + 1.0 / (2.0 * radius * dr);
            if !((2.0 * radius * dr).is_finite()
                && 2.0 * radius * dr > 0.0
                && ae.is_finite()
                && aw.is_finite())
            {
                return Err("radial Jacobi coefficient arithmetic failed".to_owned());
            }
            let update = (ae * psi[iz][ir + 1]
                + aw * psi[iz][ir - 1]
                + a_ns * (psi[iz - 1][ir] + psi[iz + 1][ir])
                - source[iz][ir])
                / a_c;
            out[iz][ir] = (1.0 - case.omega_j) * psi[iz][ir] + case.omega_j * update;
            if !(update.is_finite() && out[iz][ir].is_finite()) {
                return Err("Jacobi flux became non-finite".to_owned());
            }
        }
    }
    enforce_boundary(&mut out);
    Ok(out)
}

fn enforce_boundary(psi: &mut [Vec<f64>]) {
    if psi.is_empty() || psi[0].is_empty() {
        return;
    }
    let nz = psi.len();
    let nr = psi[0].len();
    psi[0].fill(0.0);
    psi[nz - 1].fill(0.0);
    for row in psi {
        row[0] = 0.0;
        row[nr - 1] = 0.0;
    }
}

fn validate_flux_matrix(case: &GradShafranovCase, psi: &[Vec<f64>]) -> Result<(), String> {
    validate_case(case)?;
    if psi.len() != case.nz {
        return Err("psi row count must match Grad-Shafranov case grid".to_string());
    }
    for row in psi {
        if row.len() != case.nr {
            return Err("psi column count must match Grad-Shafranov case grid".to_string());
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err("psi must contain only finite values".to_string());
        }
    }
    Ok(())
}

/// Evaluates the cylindrical Grad-Shafranov operator Δ\*ψ on the native grid.
///
/// The central-difference stencil is evaluated on interior cells; boundary
/// values in the returned `(nz, nr)` matrix are zero.
///
/// # Errors
///
/// Returns an error string when the case is invalid or `psi` is non-finite or
/// does not have the exact row-major `(nz, nr)` shape.
pub fn grad_shafranov_delta_star(
    case: &GradShafranovCase,
    psi: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, String> {
    validate_flux_matrix(case, psi)?;
    let r = linspace(case.r_min, case.r_max, case.nr);
    let dr = (case.r_max - case.r_min) / ((case.nr - 1) as f64);
    let dz = (case.z_max - case.z_min) / ((case.nz - 1) as f64);
    let dr2 = dr * dr;
    let dz2 = dz * dz;
    let mut delta_star = vec![vec![0.0; case.nr]; case.nz];

    for iz in 1..(case.nz - 1) {
        for (ir, radius) in r.iter().enumerate().take(case.nr - 1).skip(1) {
            let d2_dr2 = (psi[iz][ir + 1] - 2.0 * psi[iz][ir] + psi[iz][ir - 1]) / dr2;
            let d_dr_over_r = (psi[iz][ir + 1] - psi[iz][ir - 1]) / (2.0 * dr * radius);
            let d2_dz2 = (psi[iz + 1][ir] - 2.0 * psi[iz][ir] + psi[iz - 1][ir]) / dz2;
            delta_star[iz][ir] = d2_dr2 - d_dr_over_r + d2_dz2;
        }
    }
    Ok(delta_star)
}

/// Returns `J_phi` implied by `Δ*ψ = -mu0 R J_phi`.
///
/// The returned matrix matches the input shape and has zero boundary cells
/// because [`grad_shafranov_delta_star`] uses an interior-only stencil.
///
/// # Errors
///
/// Returns an error string when the case or flux matrix is invalid.
pub fn toroidal_current_density_from_flux(
    case: &GradShafranovCase,
    psi: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, String> {
    validate_flux_matrix(case, psi)?;
    let r = linspace(case.r_min, case.r_max, case.nr);
    let delta_star = grad_shafranov_delta_star(case, psi)?;
    let mut current_density = vec![vec![0.0; case.nr]; case.nz];
    for iz in 1..(case.nz - 1) {
        for (ir, radius) in r.iter().enumerate().take(case.nr - 1).skip(1) {
            current_density[iz][ir] = -delta_star[iz][ir] / (case.mu0 * radius);
        }
    }
    Ok(current_density)
}

/// Integrates implied `J_phi` over the interior of the native R-Z grid.
///
/// Interior cells receive uniform `dr * dz` weights and boundary cells are
/// excluded.
///
/// # Errors
///
/// Returns an error string when the case or flux matrix is invalid, or when
/// the integrated value is non-finite.
pub fn total_toroidal_current_from_flux(
    case: &GradShafranovCase,
    psi: &[Vec<f64>],
) -> Result<f64, String> {
    let current_density = toroidal_current_density_from_flux(case, psi)?;
    let dr = (case.r_max - case.r_min) / ((case.nr - 1) as f64);
    let dz = (case.z_max - case.z_min) / ((case.nz - 1) as f64);
    let mut total = 0.0;
    for row in current_density.iter().take(case.nz - 1).skip(1) {
        for value in row.iter().take(case.nr - 1).skip(1) {
            total += value * dr * dz;
        }
    }
    if !total.is_finite() {
        return Err("integrated toroidal current became non-finite".to_string());
    }
    Ok(total)
}

/// Integrates implied `J_phi` with full-domain trapezoidal weights.
///
/// Boundary rows and columns receive half weights in their respective axes.
/// The current-density operator itself still yields zero boundary cells.
///
/// # Errors
///
/// Returns an error string when the case or flux matrix is invalid, or when
/// the integrated value is non-finite.
pub fn total_toroidal_current_from_flux_trapezoidal(
    case: &GradShafranovCase,
    psi: &[Vec<f64>],
) -> Result<f64, String> {
    let current_density = toroidal_current_density_from_flux(case, psi)?;
    let dr = (case.r_max - case.r_min) / ((case.nr - 1) as f64);
    let dz = (case.z_max - case.z_min) / ((case.nz - 1) as f64);
    let mut total = 0.0;
    for (iz, row) in current_density.iter().enumerate().take(case.nz) {
        let z_weight = if iz == 0 || iz + 1 == case.nz {
            0.5
        } else {
            1.0
        };
        for (ir, value) in row.iter().enumerate().take(case.nr) {
            let r_weight = if ir == 0 || ir + 1 == case.nr {
                0.5
            } else {
                1.0
            };
            total += value * z_weight * r_weight * dr * dz;
        }
    }
    if !total.is_finite() {
        return Err("trapezoidal integrated toroidal current became non-finite".to_string());
    }
    Ok(total)
}

/// Integrate J_phi implied by a flux grid over an explicit R-Z domain mask.
///
/// The mask must match the flux matrix shape. Boundary cells may be present in
/// the mask, but contribute zero because the Delta* stencil is interior-only.
///
/// # Errors
///
/// Returns an error string when the case, flux matrix, or mask shape is
/// invalid; when the mask selects no cells; or when integration is non-finite.
pub fn total_toroidal_current_from_flux_masked(
    case: &GradShafranovCase,
    psi: &[Vec<f64>],
    domain_mask: &[Vec<bool>],
) -> Result<f64, String> {
    validate_flux_matrix(case, psi)?;
    if domain_mask.len() != case.nz || domain_mask.iter().any(|row| row.len() != case.nr) {
        return Err(format!(
            "toroidal current mask shape must match case shape ({}, {})",
            case.nz, case.nr
        ));
    }
    if !domain_mask.iter().flatten().any(|value| *value) {
        return Err("toroidal current mask must include at least one cell".to_string());
    }

    let current_density = toroidal_current_density_from_flux(case, psi)?;
    let dr = (case.r_max - case.r_min) / ((case.nr - 1) as f64);
    let dz = (case.z_max - case.z_min) / ((case.nz - 1) as f64);
    let mut total = 0.0;
    for iz in 0..case.nz {
        for ir in 0..case.nr {
            if domain_mask[iz][ir] {
                total += current_density[iz][ir] * dr * dz;
            }
        }
    }
    if !total.is_finite() {
        return Err("masked integrated toroidal current became non-finite".to_string());
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_case() -> GradShafranovCase {
        GradShafranovCase {
            r_min: 1.0,
            r_max: 3.0,
            z_min: -1.2,
            z_max: 1.2,
            nr: 17,
            nz: 17,
            ip_target: 1.0e6,
            mu0: 4.0e-7 * std::f64::consts::PI,
            n_picard: 8,
            n_jacobi: 16,
            alpha: 0.1,
            omega_j: 2.0 / 3.0,
            beta_mix: 0.5,
        }
    }

    #[test]
    fn solve_preserves_boundary_and_nontrivial_interior() {
        let case = reference_case();
        let result = solve_grad_shafranov(&case).expect("reference case should solve");

        assert_eq!(result.psi.len(), case.nz);
        assert_eq!(result.psi[0].len(), case.nr);
        assert!(result.psi.iter().flatten().all(|value| value.is_finite()));
        assert!(result.psi[0].iter().all(|value| value.abs() < 1.0e-14));
        assert!(result.psi[case.nz - 1]
            .iter()
            .all(|value| value.abs() < 1.0e-14));
        assert!(result.psi.iter().all(|row| row[0].abs() < 1.0e-14));
        assert!(result
            .psi
            .iter()
            .all(|row| row[case.nr - 1].abs() < 1.0e-14));
        let interior_max = result.psi[1..case.nz - 1]
            .iter()
            .flat_map(|row| &row[1..case.nr - 1])
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(interior_max > 1.0e-6);
    }

    #[test]
    fn parse_case_rejects_missing_required_field() {
        let text = "\
[grad_shafranov]
R_min = 1.0
R_max = 3.0
Z_min = -1.2
Z_max = 1.2
NR = 17
NZ = 17
Ip_target = 1.0e6
mu0 = 1.2566370614359173e-6
n_picard = 8
n_jacobi = 16
alpha = 0.1
omega_j = 0.6666666666666666
";
        let err = parse_case(text).expect_err("missing beta_mix must fail closed");
        assert!(err.contains("missing required Grad-Shafranov case field: beta_mix"));
    }

    #[test]
    fn operator_current_closure_matches_z_quadratic_manufactured_solution() {
        let mut case = reference_case();
        case.z_min = -1.0;
        case.z_max = 1.0;
        case.nr = 17;
        case.nz = 19;
        let z = linspace(case.z_min, case.z_max, case.nz);
        let coeff = -0.25_f64;
        let psi: Vec<Vec<f64>> = z
            .iter()
            .map(|z_value| vec![coeff * z_value * z_value; case.nr])
            .collect();

        let delta_star = grad_shafranov_delta_star(&case, &psi).unwrap();
        let current_density = toroidal_current_density_from_flux(&case, &psi).unwrap();
        let total_current = total_toroidal_current_from_flux(&case, &psi).unwrap();
        let trapezoidal_current =
            total_toroidal_current_from_flux_trapezoidal(&case, &psi).unwrap();
        let dr = (case.r_max - case.r_min) / ((case.nr - 1) as f64);
        let dz = (case.z_max - case.z_min) / ((case.nz - 1) as f64);

        let mut expected_total = 0.0;
        for iz in 1..(case.nz - 1) {
            for ir in 1..(case.nr - 1) {
                let r = case.r_min + (ir as f64) * dr;
                let expected_j = -2.0 * coeff / (case.mu0 * r);
                assert!((delta_star[iz][ir] - 2.0 * coeff).abs() < 1.0e-12);
                assert!((current_density[iz][ir] - expected_j).abs() < 1.0e-6);
                expected_total += expected_j * dr * dz;
            }
        }
        assert!(((total_current - expected_total) / expected_total).abs() < 1.0e-12);
        assert!(((trapezoidal_current - expected_total) / expected_total).abs() < 1.0e-12);

        let mask: Vec<Vec<bool>> = (0..case.nz)
            .map(|iz| {
                (0..case.nr)
                    .map(|ir| iz > 2 && iz < case.nz - 3 && ir > 3 && ir < case.nr - 4)
                    .collect()
            })
            .collect();
        let masked_current = total_toroidal_current_from_flux_masked(&case, &psi, &mask).unwrap();
        let mut expected_masked_total = 0.0;
        for row in mask.iter().take(case.nz - 1).skip(1) {
            for (ir, in_domain) in row.iter().enumerate().take(case.nr - 1).skip(1) {
                if *in_domain {
                    let r = case.r_min + (ir as f64) * dr;
                    expected_masked_total += -2.0 * coeff / (case.mu0 * r) * dr * dz;
                }
            }
        }
        assert!(((masked_current - expected_masked_total) / expected_masked_total).abs() < 1.0e-12);
        assert!(masked_current.abs() < total_current.abs());
        assert!(total_toroidal_current_from_flux_masked(&case, &psi, &[]).is_err());

        let radial_coeff = 0.03125_f64;
        let vertical_coeff = -0.125_f64;
        let r = linspace(case.r_min, case.r_max, case.nr);
        let psi_radial: Vec<Vec<f64>> = z
            .iter()
            .map(|z_value| {
                r.iter()
                    .map(|r_value| {
                        radial_coeff * r_value.powi(4) + vertical_coeff * z_value.powi(2)
                    })
                    .collect()
            })
            .collect();
        let delta_star_radial = grad_shafranov_delta_star(&case, &psi_radial).unwrap();
        let current_density_radial =
            toroidal_current_density_from_flux(&case, &psi_radial).unwrap();

        for iz in 1..(case.nz - 1) {
            for ir in 1..(case.nr - 1) {
                let r_value = r[ir];
                let expected_delta = 8.0 * radial_coeff * r_value * r_value + 2.0 * vertical_coeff
                    - 2.0 * radial_coeff * dr * dr;
                let expected_j = -expected_delta / (case.mu0 * r_value);
                assert!((delta_star_radial[iz][ir] - expected_delta).abs() < 1.0e-12);
                assert!((current_density_radial[iz][ir] - expected_j).abs() < 1.0e-6);
            }
        }

        let mixed_coeff = 0.05_f64;
        let psi_mixed: Vec<Vec<f64>> = z
            .iter()
            .map(|z_value| {
                r.iter()
                    .map(|r_value| {
                        mixed_coeff * r_value.powi(2) * z_value.powi(2)
                            + vertical_coeff * z_value.powi(2)
                    })
                    .collect()
            })
            .collect();
        let delta_star_mixed = grad_shafranov_delta_star(&case, &psi_mixed).unwrap();
        let current_density_mixed = toroidal_current_density_from_flux(&case, &psi_mixed).unwrap();

        for iz in 1..(case.nz - 1) {
            for ir in 1..(case.nr - 1) {
                let r_value = r[ir];
                let expected_delta = 2.0 * mixed_coeff * r_value * r_value + 2.0 * vertical_coeff;
                let expected_j = -expected_delta / (case.mu0 * r_value);
                assert!((delta_star_mixed[iz][ir] - expected_delta).abs() < 1.0e-12);
                assert!((current_density_mixed[iz][ir] - expected_j).abs() < 1.0e-6);
            }
        }
    }
}
