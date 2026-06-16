// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Rust MAST SNN Validation CLI

use fusion_ml::mast_snn::{
    default_policy_grid, evaluate_mast_snn_npz_panel, load_independent_label_manifest,
    sweep_mast_snn_npz_panel, sweep_mast_snn_npz_panel_with_independent_labels, MastSnnConfig,
};
use std::path::PathBuf;

fn main() {
    if let Err(err) = run() {
        eprintln!("mast_snn_validation: {err}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut cache_dir = PathBuf::from("data/mast_cache");
    let mut out = PathBuf::from("../validation/reports/mast_snn_rust_validation.json");
    let mut train_shots = Vec::new();
    let mut val_shots = Vec::new();
    let mut negative_shots = Vec::new();
    let mut label_manifest: Option<PathBuf> = None;
    let mut config = MastSnnConfig::default();
    let mut sweep = false;

    let mut idx = 1usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--cache-dir" => {
                idx += 1;
                cache_dir = value_path(&args, idx, "--cache-dir")?;
            }
            "--out" => {
                idx += 1;
                out = value_path(&args, idx, "--out")?;
            }
            "--train-shots" => {
                idx += 1;
                while idx < args.len() && !args[idx].starts_with("--") {
                    train_shots.push(args[idx].parse::<i32>()?);
                    idx += 1;
                }
                continue;
            }
            "--val-shots" => {
                idx += 1;
                while idx < args.len() && !args[idx].starts_with("--") {
                    val_shots.push(args[idx].parse::<i32>()?);
                    idx += 1;
                }
                continue;
            }
            "--negative-shots" => {
                idx += 1;
                while idx < args.len() && !args[idx].starts_with("--") {
                    negative_shots.push(args[idx].parse::<i32>()?);
                    idx += 1;
                }
                continue;
            }
            "--label-manifest" => {
                idx += 1;
                label_manifest = Some(value_path(&args, idx, "--label-manifest")?);
            }
            "--sweep" => {
                sweep = true;
            }
            "--epochs" => {
                idx += 1;
                config.epochs = value(&args, idx, "--epochs")?.parse::<usize>()?;
            }
            "--seed" => {
                idx += 1;
                config.seed = value(&args, idx, "--seed")?.parse::<u64>()?;
            }
            "--min-train-shots" => {
                idx += 1;
                config.min_train_shots =
                    value(&args, idx, "--min-train-shots")?.parse::<usize>()?;
            }
            "--min-validation-shots" => {
                idx += 1;
                config.min_validation_shots =
                    value(&args, idx, "--min-validation-shots")?.parse::<usize>()?;
            }
            "--n-neurons" => {
                idx += 1;
                config.n_neurons = value(&args, idx, "--n-neurons")?.parse::<usize>()?;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            unknown => {
                return Err(format!("unknown argument: {unknown}").into());
            }
        }
        idx += 1;
    }

    if train_shots.is_empty() {
        return Err("--train-shots requires at least one shot".into());
    }

    let independent_labels = label_manifest
        .as_deref()
        .map(load_independent_label_manifest)
        .transpose()?;
    if let Some(labels) = &independent_labels {
        val_shots = labels.disruptive_ids();
        negative_shots = labels.non_disruptive.clone();
    }

    if val_shots.is_empty() {
        return Err("--val-shots or --label-manifest requires at least one disruptive shot".into());
    }

    if sweep {
        let policies = default_policy_grid();
        let report = match &independent_labels {
            Some(labels) => sweep_mast_snn_npz_panel_with_independent_labels(
                &cache_dir,
                &train_shots,
                labels,
                config,
                &policies,
            )?,
            None => sweep_mast_snn_npz_panel(
                &cache_dir,
                &train_shots,
                &val_shots,
                &negative_shots,
                config,
                &policies,
            )?,
        };
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&out, serde_json::to_string_pretty(&report)? + "\n")?;
        println!("wrote={}", out.display());
        println!(
            "train_available={} disruptive_available={} negative_available={} best_policy={:?} recall={:?} fpr={:?}",
            report.train_available_count,
            report.disruptive_available_count,
            report.negative_available_count,
            report.best_policy_index,
            report.recall,
            report.false_positive_rate
        );
        return Ok(());
    }

    let report = evaluate_mast_snn_npz_panel(&cache_dir, &train_shots, &val_shots, config)?;
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out, serde_json::to_string_pretty(&report)? + "\n")?;
    println!("wrote={}", out.display());
    println!(
        "train_available={} validation_available={} detected={}",
        report.train_available_count,
        report.validation_available_count,
        report.detected_validation_count
    );
    Ok(())
}

fn value<'a>(args: &'a [String], idx: usize, flag: &str) -> Result<&'a str, String> {
    args.get(idx)
        .map(String::as_str)
        .filter(|value| !value.starts_with("--"))
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn value_path(args: &[String], idx: usize, flag: &str) -> Result<PathBuf, String> {
    value(args, idx, flag).map(PathBuf::from)
}

fn print_help() {
    println!(
        "Usage: mast_snn_validation --cache-dir DIR --train-shots ... (--val-shots ... | --label-manifest PATH) [--negative-shots ...] [--sweep] [--out PATH]"
    );
}
