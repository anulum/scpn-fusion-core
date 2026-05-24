// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Native Rust Polyglot GS Solver

use std::env;
use std::io::{self, Write};
use std::path::Path;
use std::process::ExitCode;

use fusion_polyglot::{load_case, solve_grad_shafranov};

fn main() -> ExitCode {
    let mut args = env::args();
    let program = args.next().unwrap_or_else(|| "gs_picard_csv".to_string());
    let Some(case_path) = args.next() else {
        eprintln!("usage: {program} <grad-shafranov-case.toml>");
        return ExitCode::from(2);
    };
    if args.next().is_some() {
        eprintln!("usage: {program} <grad-shafranov-case.toml>");
        return ExitCode::from(2);
    }

    let case = match load_case(Path::new(&case_path)) {
        Ok(case) => case,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let result = match solve_grad_shafranov(&case) {
        Ok(result) => result,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = write_csv(&result.psi) {
        eprintln!("failed to write Grad-Shafranov CSV: {err}");
        return ExitCode::from(1);
    }

    ExitCode::SUCCESS
}

fn write_csv(psi: &[Vec<f64>]) -> io::Result<()> {
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    for row in psi {
        for (idx, value) in row.iter().enumerate() {
            if idx > 0 {
                write!(handle, ",")?;
            }
            write!(handle, "{value:.17}")?;
        }
        writeln!(handle)?;
    }
    Ok(())
}
