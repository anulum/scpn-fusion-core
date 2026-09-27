// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Rust typed physical case admission
//! Conforming TOML parsing and checked physical case validation.

use std::{fs, path::Path};

use super::GradShafranovCase;

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

/// Load a UTF-8 TOML 1.0 case with the same admission rules as [`parse_case`].
///
/// # Errors
///
/// Refuses unreadable/non-UTF-8 input, malformed TOML or an invalid physical case.
pub fn load_case(path: &Path) -> Result<GradShafranovCase, String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read Grad-Shafranov case: {err}"))?;
    parse_case(&text)
}

/// Parse the exact thirteen-field `grad_shafranov` tree from conforming TOML 1.0.
///
/// Equivalent quoted, dotted and inline forms are accepted. All fields are
/// mandatory; duplicate definitions, unknown entries and implicit numeric
/// coercions are refused. R/Z use metres, current uses amperes and mu0 uses H/m.
///
/// # Errors
///
/// Returns a typed error result for syntax, numeric kinds/exactness, finite
/// physical domains, count/work caps and unrepresentable mesh geometry.
pub fn parse_case(text: &str) -> Result<GradShafranovCase, String> {
    let root = text
        .parse::<toml::Table>()
        .map_err(|err| format!("invalid case TOML: {err}"))?;
    if root.len() != 1 {
        return Err("expected only the grad_shafranov table".to_owned());
    }
    let fields = root
        .get("grad_shafranov")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| "expected grad_shafranov table".to_owned())?;
    for key in REQUIRED_FIELDS {
        if !fields.contains_key(key) {
            return Err(format!("missing required Grad-Shafranov case field: {key}"));
        }
    }
    if fields.len() != REQUIRED_FIELDS.len() {
        return Err("unknown Grad-Shafranov case field".to_owned());
    }
    let case = GradShafranovCase {
        r_min: parse_real(fields, "R_min")?,
        r_max: parse_real(fields, "R_max")?,
        z_min: parse_real(fields, "Z_min")?,
        z_max: parse_real(fields, "Z_max")?,
        nr: parse_count(fields, "NR")?,
        nz: parse_count(fields, "NZ")?,
        ip_target: parse_real(fields, "Ip_target")?,
        mu0: parse_real(fields, "mu0")?,
        n_picard: parse_count(fields, "n_picard")?,
        n_jacobi: parse_count(fields, "n_jacobi")?,
        alpha: parse_real(fields, "alpha")?,
        omega_j: parse_real(fields, "omega_j")?,
        beta_mix: parse_real(fields, "beta_mix")?,
    };
    validate_case(&case)?;
    Ok(case)
}

/// Convert only an actual TOML float or mathematically exact binary64 integer.
fn parse_real(fields: &toml::Table, key: &str) -> Result<f64, String> {
    let out = match &fields[key] {
        toml::Value::Float(value) => *value,
        toml::Value::Integer(value) => {
            let converted = *value as f64;
            // i128 preserves the rounded 2^63 endpoint instead of a saturating
            // i64 roundtrip that would incorrectly admit i64::MAX.
            if converted as i128 != i128::from(*value) {
                return Err(format!(
                    "{key}: integer is not exactly representable in binary64"
                ));
            }
            converted
        }
        _ => return Err(format!("{key}: expected TOML float or exact integer")),
    };
    if !out.is_finite() {
        return Err(format!("{key}: scalar must be finite"));
    }
    Ok(out)
}

/// Validate the TOML integer kind and mathematical value before native narrowing.
fn parse_count(fields: &toml::Table, key: &str) -> Result<usize, String> {
    let integer = fields[key]
        .as_integer()
        .ok_or_else(|| format!("{key}: expected TOML integer"))?;
    if integer < 1 {
        return Err(format!("{key}: count must be positive"));
    }
    usize::try_from(integer).map_err(|err| format!("{key}: {err}"))
}

/// Admit the same domain and bounded workload for file and direct solver cases.
pub(super) fn validate_case(case: &GradShafranovCase) -> Result<(), String> {
    if [
        case.r_min,
        case.r_max,
        case.z_min,
        case.z_max,
        case.ip_target,
        case.mu0,
        case.alpha,
        case.omega_j,
        case.beta_mix,
    ]
    .iter()
    .any(|v| !v.is_finite())
    {
        return Err("Grad-Shafranov case contains non-finite scalar".to_owned());
    }
    if !(3..=1025).contains(&case.nr) || !(3..=1025).contains(&case.nz) {
        return Err("case grid counts must be in [3,1025]".to_owned());
    }
    if !(1..=10000).contains(&case.n_picard) || !(1..=10000).contains(&case.n_jacobi) {
        return Err("case iteration counts must be in [1,10000]".to_owned());
    }
    let mut work = 1usize;
    for count in [case.nr, case.nz, case.n_picard, case.n_jacobi] {
        if count > 100000000 / work {
            return Err("case exceeds 100000000 point iterations".to_owned());
        }
        work *= count;
    }
    if !(case.r_min > 0.0 && case.r_max > case.r_min && case.z_max > case.z_min) {
        return Err("case requires positive ordered R and ordered Z".to_owned());
    }
    if case.mu0 <= 0.0 {
        return Err("mu0 must be positive".to_owned());
    }
    if !(case.alpha > 0.0
        && case.alpha <= 1.0
        && case.omega_j > 0.0
        && case.omega_j < 2.0
        && (0.0..=1.0).contains(&case.beta_mix))
    {
        return Err("invalid relaxation or profile scalar".to_owned());
    }
    for (start, stop, count) in [
        (case.r_min, case.r_max, case.nr),
        (case.z_min, case.z_max, case.nz),
    ] {
        let step = (stop - start) / ((count - 1) as f64);
        if !step.is_finite() || step <= 0.0 {
            return Err("grid spacing must be finite and positive".to_owned());
        }
        if stop <= start + step * ((count - 2) as f64) {
            return Err("inclusive grid endpoint has collapsed adjacent nodes".to_owned());
        }
        let mut previous = start;
        for index in 1..count {
            let current = start + step * (index as f64);
            if !current.is_finite() || current <= previous {
                return Err("grid axis is non-finite or has collapsed adjacent nodes".to_owned());
            }
            previous = current;
        }
    }
    Ok(())
}
