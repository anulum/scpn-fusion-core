// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — VMEC Interface
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Lightweight VMEC-compatible boundary-state wrapper.
//!
//! This module intentionally does not implement a full 3D force-balance solve.
//! Instead it provides a deterministic interoperability lane for exchanging
//! reduced Fourier boundary states with external VMEC-class workflows.

use fusion_types::error::{FusionError, FusionResult};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VmecFourierMode {
    pub m: i32,
    pub n: i32,
    pub r_cos: f64,
    pub r_sin: f64,
    pub z_cos: f64,
    pub z_sin: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VmecBoundaryState {
    pub r_axis: f64,
    pub z_axis: f64,
    pub a_minor: f64,
    pub kappa: f64,
    pub triangularity: f64,
    pub nfp: usize,
    pub modes: Vec<VmecFourierMode>,
}

impl VmecBoundaryState {
    pub fn validate(&self) -> FusionResult<()> {
        if !self.r_axis.is_finite()
            || !self.z_axis.is_finite()
            || !self.a_minor.is_finite()
            || !self.kappa.is_finite()
            || !self.triangularity.is_finite()
        {
            return Err(FusionError::PhysicsViolation(
                "VMEC boundary contains non-finite scalar".to_string(),
            ));
        }
        if self.a_minor <= 0.0 {
            return Err(FusionError::PhysicsViolation(
                "VMEC boundary requires a_minor > 0".to_string(),
            ));
        }
        if self.kappa <= 0.0 {
            return Err(FusionError::PhysicsViolation(
                "VMEC boundary requires kappa > 0".to_string(),
            ));
        }
        if self.nfp < 1 {
            return Err(FusionError::PhysicsViolation(
                "VMEC boundary requires nfp >= 1".to_string(),
            ));
        }
        let mut seen_modes: HashSet<(i32, i32)> = HashSet::with_capacity(self.modes.len());
        for (idx, mode) in self.modes.iter().enumerate() {
            if mode.m < 0 {
                return Err(FusionError::PhysicsViolation(format!(
                    "VMEC boundary mode[{idx}] requires m >= 0, got {}",
                    mode.m
                )));
            }
            if !mode.r_cos.is_finite()
                || !mode.r_sin.is_finite()
                || !mode.z_cos.is_finite()
                || !mode.z_sin.is_finite()
            {
                return Err(FusionError::PhysicsViolation(format!(
                    "VMEC boundary mode[{idx}] contains non-finite coefficients"
                )));
            }
            let key = (mode.m, mode.n);
            if !seen_modes.insert(key) {
                return Err(FusionError::PhysicsViolation(format!(
                    "VMEC boundary contains duplicate mode (m={}, n={})",
                    mode.m, mode.n
                )));
            }
        }
        Ok(())
    }
}

pub fn export_vmec_like_text(state: &VmecBoundaryState) -> FusionResult<String> {
    state.validate()?;
    let mut out = String::new();
    out.push_str("format=vmec_like_v1\n");
    out.push_str(&format!("r_axis={:.16e}\n", state.r_axis));
    out.push_str(&format!("z_axis={:.16e}\n", state.z_axis));
    out.push_str(&format!("a_minor={:.16e}\n", state.a_minor));
    out.push_str(&format!("kappa={:.16e}\n", state.kappa));
    out.push_str(&format!("triangularity={:.16e}\n", state.triangularity));
    out.push_str(&format!("nfp={}\n", state.nfp));
    for mode in &state.modes {
        out.push_str(&format!(
            "mode,{},{},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            mode.m, mode.n, mode.r_cos, mode.r_sin, mode.z_cos, mode.z_sin
        ));
    }
    Ok(out)
}

fn parse_float(key: &str, text: &str) -> FusionResult<f64> {
    let val = text.parse::<f64>().map_err(|e| {
        FusionError::PhysicsViolation(format!("Failed to parse VMEC key '{key}' as float: {e}"))
    })?;
    if !val.is_finite() {
        return Err(FusionError::PhysicsViolation(format!(
            "VMEC key '{key}' must be finite, got {val}"
        )));
    }
    Ok(val)
}

fn parse_int<T>(key: &str, text: &str) -> FusionResult<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    text.parse::<T>().map_err(|e| {
        FusionError::PhysicsViolation(format!("Failed to parse VMEC key '{key}' as integer: {e}"))
    })
}

pub fn import_vmec_like_text(text: &str) -> FusionResult<VmecBoundaryState> {
    let mut format_seen = false;
    let mut r_axis: Option<f64> = None;
    let mut z_axis: Option<f64> = None;
    let mut a_minor: Option<f64> = None;
    let mut kappa: Option<f64> = None;
    let mut triangularity: Option<f64> = None;
    let mut nfp: Option<usize> = None;
    let mut modes: Vec<VmecFourierMode> = Vec::new();

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(format_name) = line.strip_prefix("format=") {
            if format_seen {
                return Err(FusionError::PhysicsViolation(
                    "Duplicate VMEC key: format".to_string(),
                ));
            }
            if format_name.trim() != "vmec_like_v1" {
                return Err(FusionError::PhysicsViolation(format!(
                    "Unsupported VMEC format: {}",
                    format_name.trim()
                )));
            }
            format_seen = true;
            continue;
        }
        if let Some(rest) = line.strip_prefix("mode,") {
            let cols: Vec<&str> = rest.split(',').map(|v| v.trim()).collect();
            if cols.len() != 6 {
                return Err(FusionError::PhysicsViolation(
                    "VMEC mode line must contain exactly 6 columns".to_string(),
                ));
            }
            modes.push(VmecFourierMode {
                m: parse_int("mode.m", cols[0])?,
                n: parse_int("mode.n", cols[1])?,
                r_cos: parse_float("mode.r_cos", cols[2])?,
                r_sin: parse_float("mode.r_sin", cols[3])?,
                z_cos: parse_float("mode.z_cos", cols[4])?,
                z_sin: parse_float("mode.z_sin", cols[5])?,
            });
            continue;
        }
        let (key, value) = line.split_once('=').ok_or_else(|| {
            FusionError::PhysicsViolation(format!("Invalid VMEC line (missing '='): {line}"))
        })?;
        let key = key.trim();
        let value = value.trim();
        match key {
            "r_axis" => {
                if r_axis.is_some() {
                    return Err(FusionError::PhysicsViolation(
                        "Duplicate VMEC key: r_axis".to_string(),
                    ));
                }
                r_axis = Some(parse_float(key, value)?);
            }
            "z_axis" => {
                if z_axis.is_some() {
                    return Err(FusionError::PhysicsViolation(
                        "Duplicate VMEC key: z_axis".to_string(),
                    ));
                }
                z_axis = Some(parse_float(key, value)?);
            }
            "a_minor" => {
                if a_minor.is_some() {
                    return Err(FusionError::PhysicsViolation(
                        "Duplicate VMEC key: a_minor".to_string(),
                    ));
                }
                a_minor = Some(parse_float(key, value)?);
            }
            "kappa" => {
                if kappa.is_some() {
                    return Err(FusionError::PhysicsViolation(
                        "Duplicate VMEC key: kappa".to_string(),
                    ));
                }
                kappa = Some(parse_float(key, value)?);
            }
            "triangularity" => {
                if triangularity.is_some() {
                    return Err(FusionError::PhysicsViolation(
                        "Duplicate VMEC key: triangularity".to_string(),
                    ));
                }
                triangularity = Some(parse_float(key, value)?);
            }
            "nfp" => {
                if nfp.is_some() {
                    return Err(FusionError::PhysicsViolation(
                        "Duplicate VMEC key: nfp".to_string(),
                    ));
                }
                nfp = Some(parse_int(key, value)?);
            }
            other => {
                return Err(FusionError::PhysicsViolation(format!(
                    "Unknown VMEC key: {other}"
                )));
            }
        }
    }

    let state = VmecBoundaryState {
        r_axis: r_axis
            .ok_or_else(|| FusionError::PhysicsViolation("Missing VMEC key: r_axis".to_string()))?,
        z_axis: z_axis
            .ok_or_else(|| FusionError::PhysicsViolation("Missing VMEC key: z_axis".to_string()))?,
        a_minor: a_minor.ok_or_else(|| {
            FusionError::PhysicsViolation("Missing VMEC key: a_minor".to_string())
        })?,
        kappa: kappa
            .ok_or_else(|| FusionError::PhysicsViolation("Missing VMEC key: kappa".to_string()))?,
        triangularity: triangularity.ok_or_else(|| {
            FusionError::PhysicsViolation("Missing VMEC key: triangularity".to_string())
        })?,
        nfp: nfp
            .ok_or_else(|| FusionError::PhysicsViolation("Missing VMEC key: nfp".to_string()))?,
        modes,
    };
    state.validate()?;
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_state() -> VmecBoundaryState {
        VmecBoundaryState {
            r_axis: 6.2,
            z_axis: 0.0,
            a_minor: 2.0,
            kappa: 1.7,
            triangularity: 0.33,
            nfp: 1,
            modes: vec![
                VmecFourierMode {
                    m: 1,
                    n: 1,
                    r_cos: 0.03,
                    r_sin: -0.01,
                    z_cos: 0.0,
                    z_sin: 0.02,
                },
                VmecFourierMode {
                    m: 2,
                    n: 1,
                    r_cos: 0.015,
                    r_sin: 0.0,
                    z_cos: 0.0,
                    z_sin: 0.008,
                },
            ],
        }
    }

    #[test]
    fn test_vmec_text_roundtrip() {
        let state = sample_state();
        let text = export_vmec_like_text(&state).expect("export must succeed");
        let parsed = import_vmec_like_text(&text).expect("import must succeed");
        assert_eq!(parsed.nfp, state.nfp);
        assert_eq!(parsed.modes.len(), state.modes.len());
        assert!((parsed.r_axis - state.r_axis).abs() < 1e-12);
        assert!((parsed.kappa - state.kappa).abs() < 1e-12);
    }

    #[test]
    fn test_vmec_missing_key_errors() {
        let text = "format=vmec_like_v1\nr_axis=6.2\n";
        let err = import_vmec_like_text(text).expect_err("missing keys should error");
        match err {
            FusionError::PhysicsViolation(msg) => assert!(msg.contains("Missing VMEC key")),
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_vmec_invalid_minor_radius_errors() {
        let text = "\
format=vmec_like_v1
r_axis=6.2
z_axis=0.0
a_minor=0.0
kappa=1.7
triangularity=0.2
nfp=1
";
        let err = import_vmec_like_text(text).expect_err("a_minor=0 must fail");
        match err {
            FusionError::PhysicsViolation(msg) => assert!(msg.contains("a_minor")),
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_vmec_rejects_invalid_modes_and_duplicates() {
        let mut state = sample_state();
        state.modes[0].r_cos = f64::NAN;
        let err = export_vmec_like_text(&state).expect_err("non-finite mode coeff must fail");
        match err {
            FusionError::PhysicsViolation(msg) => assert!(msg.contains("mode")),
            other => panic!("Unexpected error: {other:?}"),
        }

        let dup_text = "\
format=vmec_like_v1
r_axis=6.2
r_axis=6.3
z_axis=0.0
a_minor=2.0
kappa=1.7
triangularity=0.2
nfp=1
";
        let err = import_vmec_like_text(dup_text).expect_err("duplicate keys must fail");
        match err {
            FusionError::PhysicsViolation(msg) => assert!(msg.contains("Duplicate VMEC key")),
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_vmec_rejects_duplicate_modes_and_bad_format() {
        let mut state = sample_state();
        state.modes.push(VmecFourierMode {
            m: state.modes[0].m,
            n: state.modes[0].n,
            r_cos: 0.001,
            r_sin: 0.0,
            z_cos: 0.0,
            z_sin: 0.001,
        });
        let err = export_vmec_like_text(&state).expect_err("duplicate mode index must fail");
        match err {
            FusionError::PhysicsViolation(msg) => assert!(msg.contains("duplicate mode")),
            other => panic!("Unexpected error: {other:?}"),
        }

        let bad_format = "\
format=vmec_like_v2
r_axis=6.2
z_axis=0.0
a_minor=2.0
kappa=1.7
triangularity=0.2
nfp=1
";
        let err = import_vmec_like_text(bad_format).expect_err("unsupported format must fail");
        match err {
            FusionError::PhysicsViolation(msg) => assert!(msg.contains("Unsupported VMEC format")),
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_vmec_rejects_malformed_mode_and_duplicate_format() {
        let malformed_mode = "\
format=vmec_like_v1
r_axis=6.2
z_axis=0.0
a_minor=2.0
kappa=1.7
triangularity=0.2
nfp=1
mode,1,1,0.1,0.0,0.2
";
        let err =
            import_vmec_like_text(malformed_mode).expect_err("mode with wrong arity must fail");
        match err {
            FusionError::PhysicsViolation(msg) => assert!(msg.contains("exactly 6 columns")),
            other => panic!("Unexpected error: {other:?}"),
        }

        let duplicate_format = "\
format=vmec_like_v1
format=vmec_like_v1
r_axis=6.2
z_axis=0.0
a_minor=2.0
kappa=1.7
triangularity=0.2
nfp=1
";
        let err =
            import_vmec_like_text(duplicate_format).expect_err("duplicate format key must fail");
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("Duplicate VMEC key: format"))
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }
}
