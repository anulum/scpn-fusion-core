// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Rust public physical case contract tests
//! Real public parser and direct solver admission regressions.

use fusion_polyglot::{parse_case, solve_grad_shafranov};
use std::{fs, path::Path};

const DOCUMENT: &str = include_str!("../../../../validation/polyglot/gs_picard_reference.toml");
const FIELDS: [&str; 13] = [
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

/// The same corpus proves exact physical mapping and public load/direct solve outcomes.
#[test]
fn shared_physical_case_corpus() {
    let folder =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../validation/polyglot/case_contract");
    let manifest = fs::read_to_string(folder.join("manifest.toml"))
        .unwrap()
        .parse::<toml::Table>()
        .unwrap();
    assert_eq!(
        manifest["schema"].as_str(),
        Some("scpn.physical-case-corpus.v1")
    );
    for fixture in manifest["case"].as_array().unwrap() {
        let name = fixture["name"].as_str().unwrap();
        let result = fusion_polyglot::load_case(&folder.join(fixture["path"].as_str().unwrap()));
        if !fixture["valid"].as_bool().unwrap() {
            assert!(result.is_err(), "{name}");
            continue;
        }
        let c = result.unwrap_or_else(|err| panic!("{name}: {err}"));
        let e = fixture["expected"].as_table().unwrap();
        for (key, actual) in [
            ("R_min", c.r_min),
            ("R_max", c.r_max),
            ("Z_min", c.z_min),
            ("Z_max", c.z_max),
            ("Ip_target", c.ip_target),
            ("mu0", c.mu0),
            ("alpha", c.alpha),
            ("omega_j", c.omega_j),
            ("beta_mix", c.beta_mix),
        ] {
            assert_eq!(actual, e[key].as_float().unwrap(), "{name}:{key}");
        }
        for (key, actual) in [
            ("NR", c.nr),
            ("NZ", c.nz),
            ("n_picard", c.n_picard),
            ("n_jacobi", c.n_jacobi),
        ] {
            assert_eq!(
                actual,
                e[key].as_integer().unwrap() as usize,
                "{name}:{key}"
            );
        }
        if fixture["solve"].as_bool().unwrap() {
            let output = solve_grad_shafranov(&c).unwrap();
            assert_eq!(output.psi.len(), c.nz);
            assert_eq!(output.psi[0].len(), c.nr);
            assert!(output.psi.iter().flatten().all(|value| value.is_finite()));
        }
    }
}

/// Replace one physical field while preserving the remainder of the reference bytes.
fn replace_value(document: &str, field: &str, value: &str) -> String {
    document
        .lines()
        .map(|line| {
            if line.starts_with(&format!("{field} = ")) {
                format!("{field} = {value}")
            } else {
                line.to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Conforming representations deliver identical thirteen-field physical mappings.
#[test]
fn typed_tree_spellings_match_physical_reference() {
    let assignments: Vec<_> = DOCUMENT
        .lines()
        .filter(|line| line.contains(" = "))
        .collect();
    let dotted = assignments
        .iter()
        .map(|line| format!("grad_shafranov.{line}\n"))
        .collect::<String>();
    let inline = format!("grad_shafranov = {{ {} }}", assignments.join(", "));
    let quoted = DOCUMENT.replace("[grad_shafranov]", "[\"grad_shafranov\"]");
    let radix = replace_value(&replace_value(DOCUMENT, "NR", "0x11"), "NZ", "0b10001");
    for document in [DOCUMENT.to_owned(), dotted, inline, quoted, radix] {
        let c = parse_case(&document).expect("conforming tree must be accepted");
        assert_eq!((c.r_min, c.r_max, c.z_min, c.z_max), (1.0, 3.0, -1.2, 1.2));
        assert_eq!((c.nr, c.nz, c.n_picard, c.n_jacobi), (17, 17, 8, 16));
        assert_eq!(
            (c.ip_target, c.mu0, c.alpha, c.omega_j, c.beta_mix),
            (1e6, 1.2566370614359173e-6, 0.1, 0.6666666666666666, 0.5)
        );
    }
    for value in ["-1_000_000", "0", "9007199254740994"] {
        assert!(parse_case(&replace_value(DOCUMENT, "Ip_target", value)).is_ok());
    }
}

/// Every required field rejects nonnumeric delivered kinds and absent values.
#[test]
fn invalid_tree_and_field_kinds_are_refused() {
    for field in FIELDS {
        for value in ["true", "\"1#literal\"", "[1]", "{x=1}", "1979-05-27"] {
            assert!(
                parse_case(&replace_value(DOCUMENT, field, value)).is_err(),
                "{field}={value}"
            );
        }
        let missing = DOCUMENT
            .lines()
            .filter(|line| !line.starts_with(&format!("{field} = ")))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(parse_case(&missing).is_err(), "missing {field}");
    }
    for field in ["NR", "NZ", "n_picard", "n_jacobi"] {
        for value in ["17.0", "-1", "0", "9223372036854775807"] {
            assert!(parse_case(&replace_value(DOCUMENT, field, value)).is_err());
        }
    }
    for field in [
        "R_min",
        "R_max",
        "Z_min",
        "Z_max",
        "Ip_target",
        "mu0",
        "alpha",
        "omega_j",
        "beta_mix",
    ] {
        for value in ["nan", "+inf", "-inf", "9007199254740993"] {
            assert!(
                parse_case(&replace_value(DOCUMENT, field, value)).is_err(),
                "{field}={value}"
            );
        }
    }
    for invalid in [
        format!("{DOCUMENT}\nNR = 17"),
        format!("{DOCUMENT}\n[grad_shafranov]"),
        format!("{DOCUMENT}\nextra = 1"),
        format!("extra = 1\n{DOCUMENT}"),
        format!("{DOCUMENT}\n[other]\nx=1"),
        format!("{DOCUMENT}\nbroken line"),
        replace_value(DOCUMENT, "NR", "017"),
        replace_value(DOCUMENT, "Ip_target", "9223372036854775808"),
    ] {
        assert!(parse_case(&invalid).is_err(), "invalid document admitted");
    }
}

/// Direct public solves refuse overflow and cap violations before allocation.
#[test]
fn direct_case_limits_and_geometry_are_enforced() {
    let original = parse_case(DOCUMENT).unwrap();
    for name in [
        "grid_cap",
        "iteration_cap",
        "product",
        "count_overflow",
        "alpha_zero",
        "omega_upper",
        "spacing_overflow",
        "spacing_underflow",
        "collapsed_axis",
    ] {
        let mut c = original.clone();
        match name {
            "grid_cap" => c.nr = 1026,
            "iteration_cap" => c.n_jacobi = 10001,
            "product" => {
                c.nr = 10;
                c.nz = 10;
                c.n_picard = 10000;
                c.n_jacobi = 101;
            }
            "count_overflow" => c.nr = usize::MAX,
            "alpha_zero" => c.alpha = 0.0,
            "omega_upper" => c.omega_j = 2.0,
            "spacing_overflow" => {
                c.z_min = -f64::MAX;
                c.z_max = f64::MAX;
            }
            "spacing_underflow" => {
                c.z_min = 0.0;
                c.z_max = f64::from_bits(1);
            }
            "collapsed_axis" => {
                c.r_min = 1.0;
                c.r_max = f64::from_bits(1.0_f64.to_bits() + 1);
            }
            _ => unreachable!("exhaustive fixed test cases"),
        }
        assert!(solve_grad_shafranov(&c).is_err(), "{name}");
    }
    let mut exact = replace_value(&replace_value(DOCUMENT, "NR", "10"), "NZ", "10");
    exact = replace_value(
        &replace_value(&exact, "n_picard", "10000"),
        "n_jacobi",
        "100",
    );
    assert!(parse_case(&exact).is_ok());
    assert!(parse_case(&replace_value(&exact, "n_jacobi", "101")).is_err());
    assert!(parse_case(&replace_value(&exact, "n_picard", "9999")).is_ok());
    assert!(parse_case(&replace_value(DOCUMENT, "omega_j", "1.5")).is_ok());
}

/// Accepted finite endpoints cannot turn arithmetic failure into successful flux.
#[test]
fn accepted_case_numerical_failure_returns_error() {
    let text = replace_value(&replace_value(DOCUMENT, "R_min", "1e200"), "R_max", "2e200");
    let c = parse_case(&text).expect("finite ordered mesh must pass admission");
    assert!(solve_grad_shafranov(&c).is_err());
}

/// Admitted inputs must refuse overflow before exponential or physical rescaling hides it.
#[test]
fn hidden_intermediate_overflow_is_refused() {
    for seed in [Some("1e155"), Some("3e154"), None] {
        let mut document =
            replace_value(&replace_value(DOCUMENT, "n_picard", "1"), "n_jacobi", "1");
        let diagnostic = if let Some(radius) = seed {
            document = replace_value(&replace_value(&document, "R_min", "1e154"), "R_max", radius);
            "initial Gaussian"
        } else {
            for (field, value) in [
                ("R_min", "1.0"),
                ("R_max", "1.01"),
                ("Z_min", "-0.01"),
                ("Z_max", "0.01"),
                ("Ip_target", "1e308"),
            ] {
                document = replace_value(&document, field, value);
            }
            "scaled current"
        };
        let case = parse_case(&document).expect("finite physical inputs must pass admission");
        let error =
            solve_grad_shafranov(&case).expect_err("overflow must not yield successful flux");
        assert!(
            error.contains(diagnostic),
            "expected {diagnostic}, got {error}"
        );
    }
}
