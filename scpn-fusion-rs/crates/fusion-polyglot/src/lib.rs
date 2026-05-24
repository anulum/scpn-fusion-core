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
}
