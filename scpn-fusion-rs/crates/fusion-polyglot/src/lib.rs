// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Native Rust Polyglot GS Solver

use std::collections::HashMap;
use std::fs;
use std::path::Path;

const REQUIRED_FIELDS: [&str; 13] = [
    "R_min",
    "R_max",
    "Z_min",
    "Z_max",
    "NR",
    "NZ",
    "Ip_target",
    "mu0",
    "n_picard",
    "n_jacobi",
    "alpha",
    "omega_j",
    "beta_mix",
];

#[derive(Clone, Debug)]
pub struct GradShafranovCase {
    pub r_min: f64,
    pub r_max: f64,
    pub z_min: f64,
    pub z_max: f64,
    pub nr: usize,
    pub nz: usize,
    pub ip_target: f64,
    pub mu0: f64,
    pub n_picard: usize,
    pub n_jacobi: usize,
    pub alpha: f64,
    pub omega_j: f64,
    pub beta_mix: f64,
}

#[derive(Clone, Debug)]
pub struct GradShafranovResult {
    pub r: Vec<f64>,
    pub z: Vec<f64>,
    pub psi: Vec<Vec<f64>>,
}

pub fn load_case(path: &Path) -> Result<GradShafranovCase, String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read Grad-Shafranov case: {err}"))?;
    parse_case(&text)
}

pub fn parse_case(text: &str) -> Result<GradShafranovCase, String> {
    let mut values: HashMap<String, String> = HashMap::new();
    let mut in_section = false;

    for raw_line in text.lines() {
        let line = raw_line
            .split_once('#')
            .map_or(raw_line, |(before, _)| before)
            .trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            in_section = line == "[grad_shafranov]";
            continue;
        }
        if !in_section {
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("invalid Grad-Shafranov case line: {line}"))?;
        values.insert(key.trim().to_string(), value.trim().to_string());
    }

    for field in REQUIRED_FIELDS {
        if !values.contains_key(field) {
            return Err(format!(
                "missing required Grad-Shafranov case field: {field}"
            ));
        }
    }

    let case = GradShafranovCase {
        r_min: parse_f64(&values, "R_min")?,
        r_max: parse_f64(&values, "R_max")?,
        z_min: parse_f64(&values, "Z_min")?,
        z_max: parse_f64(&values, "Z_max")?,
        nr: parse_usize(&values, "NR")?,
        nz: parse_usize(&values, "NZ")?,
        ip_target: parse_f64(&values, "Ip_target")?,
        mu0: parse_f64(&values, "mu0")?,
        n_picard: parse_usize(&values, "n_picard")?,
        n_jacobi: parse_usize(&values, "n_jacobi")?,
        alpha: parse_f64(&values, "alpha")?,
        omega_j: parse_f64(&values, "omega_j")?,
        beta_mix: parse_f64(&values, "beta_mix")?,
    };
    validate_case(&case)?;
    Ok(case)
}

pub fn solve_grad_shafranov(case: &GradShafranovCase) -> Result<GradShafranovResult, String> {
    validate_case(case)?;

    let r = linspace(case.r_min, case.r_max, case.nr);
    let z = linspace(case.z_min, case.z_max, case.nz);
    let dr = (case.r_max - case.r_min) / ((case.nr - 1) as f64);
    let dz = (case.z_max - case.z_min) / ((case.nz - 1) as f64);
    let r_center = 0.5 * (case.r_min + case.r_max);

    let mut psi = vec![vec![0.0; case.nr]; case.nz];
    for row in &mut psi {
        for (ir, value) in row.iter_mut().enumerate() {
            *value = (-((r[ir] - r_center).powi(2)) / 0.5).exp() * 0.01;
        }
    }
    enforce_boundary(&mut psi);

    for _ in 0..case.n_picard {
        let source = compute_source(case, &r, &psi, dr, dz);
        let mut elliptic = psi.clone();
        for _ in 0..case.n_jacobi {
            elliptic = jacobi_step(case, &r, &elliptic, &source);
        }
        for iz in 0..case.nz {
            for ir in 0..case.nr {
                psi[iz][ir] = (1.0 - case.alpha) * psi[iz][ir] + case.alpha * elliptic[iz][ir];
            }
        }
        enforce_boundary(&mut psi);
    }

    Ok(GradShafranovResult { r, z, psi })
}

fn parse_f64(values: &HashMap<String, String>, field: &str) -> Result<f64, String> {
    values[field]
        .parse::<f64>()
        .map_err(|err| format!("invalid Grad-Shafranov field {field}: {err}"))
}

fn parse_usize(values: &HashMap<String, String>, field: &str) -> Result<usize, String> {
    values[field]
        .parse::<usize>()
        .map_err(|err| format!("invalid Grad-Shafranov field {field}: {err}"))
}

fn validate_case(case: &GradShafranovCase) -> Result<(), String> {
    if !(case.r_min.is_finite()
        && case.r_max.is_finite()
        && case.z_min.is_finite()
        && case.z_max.is_finite()
        && case.ip_target.is_finite()
        && case.mu0.is_finite()
        && case.alpha.is_finite()
        && case.omega_j.is_finite()
        && case.beta_mix.is_finite())
    {
        return Err("Grad-Shafranov case contains non-finite scalar".to_string());
    }
    if case.nr < 3 || case.nz < 3 {
        return Err("Grad-Shafranov grid must have at least 3 x 3 points".to_string());
    }
    if case.r_min <= 0.0 || case.r_max <= case.r_min {
        return Err(
            "Grad-Shafranov major-radius bounds must be positive and increasing".to_string(),
        );
    }
    if case.z_max <= case.z_min {
        return Err("Grad-Shafranov vertical bounds must be increasing".to_string());
    }
    if case.mu0 <= 0.0 || case.n_picard == 0 || case.n_jacobi == 0 {
        return Err(
            "Grad-Shafranov physical constants and iteration counts must be positive".to_string(),
        );
    }
    if !(0.0..=1.0).contains(&case.alpha) || !(0.0..=1.0).contains(&case.beta_mix) {
        return Err("Grad-Shafranov mixing coefficients must be bounded in [0, 1]".to_string());
    }
    if !(0.0..=1.0).contains(&case.omega_j) {
        return Err("Grad-Shafranov Jacobi relaxation must be bounded in [0, 1]".to_string());
    }
    Ok(())
}

fn linspace(min: f64, max: f64, count: usize) -> Vec<f64> {
    let step = (max - min) / ((count - 1) as f64);
    (0..count).map(|idx| min + step * (idx as f64)).collect()
}

fn compute_source(
    case: &GradShafranovCase,
    r: &[f64],
    psi: &[Vec<f64>],
    dr: f64,
    dz: f64,
) -> Vec<Vec<f64>> {
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
            let psi_norm = ((psi[iz][ir] - psi_axis) / denominator).clamp(0.0, 1.0);
            let profile = if (0.0..1.0).contains(&psi_norm) {
                1.0 - psi_norm
            } else {
                0.0
            };
            let jp = radius * profile;
            let jf = profile / (case.mu0 * radius.max(1.0e-6));
            raw[iz][ir] = case.beta_mix * jp + (1.0 - case.beta_mix) * jf;
            current += raw[iz][ir] * dr * dz;
        }
    }

    let scale = case.ip_target / current.abs().max(1.0e-9);
    let mut source = vec![vec![0.0; case.nr]; case.nz];
    for iz in 0..case.nz {
        for (ir, radius) in r.iter().enumerate() {
            source[iz][ir] = -case.mu0 * radius * raw[iz][ir] * scale;
        }
    }
    source
}

fn jacobi_step(
    case: &GradShafranovCase,
    r: &[f64],
    psi: &[Vec<f64>],
    source: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let dr = (case.r_max - case.r_min) / ((case.nr - 1) as f64);
    let dz = (case.z_max - case.z_min) / ((case.nz - 1) as f64);
    let dr2 = dr * dr;
    let dz2 = dz * dz;
    let a_ns = 1.0 / dz2;
    let a_c = 2.0 / dr2 + 2.0 / dz2;

    let mut out = psi.to_vec();
    for iz in 1..case.nz - 1 {
        for (ir, radius) in r.iter().enumerate().take(case.nr - 1).skip(1) {
            let ae = 1.0 / dr2 - 1.0 / (2.0 * radius * dr);
            let aw = 1.0 / dr2 + 1.0 / (2.0 * radius * dr);
            let update = (ae * psi[iz][ir + 1]
                + aw * psi[iz][ir - 1]
                + a_ns * (psi[iz - 1][ir] + psi[iz + 1][ir])
                - source[iz][ir])
                / a_c;
            out[iz][ir] = (1.0 - case.omega_j) * psi[iz][ir] + case.omega_j * update;
        }
    }
    enforce_boundary(&mut out);
    out
}

fn enforce_boundary(psi: &mut [Vec<f64>]) {
    if psi.is_empty() || psi[0].is_empty() {
        return;
    }
    let nz = psi.len();
    let nr = psi[0].len();
    for value in &mut psi[0] {
        *value = 0.0;
    }
    for value in &mut psi[nz - 1] {
        *value = 0.0;
    }
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

/// Evaluate the cylindrical Grad-Shafranov operator Delta*psi on the native grid.
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

/// Return J_phi implied by Delta*psi = -mu0 R J_phi.
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

/// Integrate J_phi implied by a flux grid over the native R-Z grid.
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

/// Integrate J_phi implied by a flux grid with full-domain trapezoidal weights.
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
