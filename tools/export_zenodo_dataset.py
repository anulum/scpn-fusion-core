#!/usr/bin/env python3
"""Export benchmark data for Zenodo dataset deposit.

Bundles all benchmark reports, plots, and campaign data into a single
directory ready for upload to Zenodo as a Dataset.

Usage:
    python tools/export_zenodo_dataset.py [--output-dir /path/to/output]

Default output: ../Paper_SNN_Tokamak_Control/zenodo_dataset/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT.parent.parent / "01_MANUSCRIPTS" / "SCPN_PAPERS" / "Paper_SNN_Tokamak_Control" / "zenodo_dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Zenodo dataset bundle")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help="Output directory for the dataset bundle",
    )
    args = parser.parse_args()
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"Exporting Zenodo dataset to: {out}")

    # ── 1. Benchmark reports (JSON + Markdown) ──────────────────────
    reports_src = ROOT / "validation" / "reports"
    reports_dst = out / "benchmark_reports"
    reports_dst.mkdir(exist_ok=True)

    n_reports = 0
    if reports_src.is_dir():
        for f in sorted(reports_src.iterdir()):
            if f.suffix in (".json", ".md"):
                shutil.copy2(f, reports_dst / f.name)
                n_reports += 1
    print(f"  Copied {n_reports} report files -> benchmark_reports/")

    # ── 2. Plot figures ─────────────────────────────────────────────
    figures_dst = out / "figures"
    figures_dst.mkdir(exist_ok=True)

    plot_files = [
        "controller_latency_comparison.png",
        "fno_suppression.png",
        "snn_trajectory.png",
    ]
    assets_dir = ROOT / "docs" / "assets"
    n_figs = 0
    for name in plot_files:
        src = assets_dir / name
        if src.exists():
            shutil.copy2(src, figures_dst / name)
            n_figs += 1
        else:
            print(f"  WARNING: {src} not found — skipping")
    print(f"  Copied {n_figs} figures -> figures/")

    # ── 3. RESULTS.md (the summary table) ───────────────────────────
    results_src = ROOT / "RESULTS.md"
    if results_src.exists():
        shutil.copy2(results_src, out / "RESULTS.md")
        print("  Copied RESULTS.md")

    # ── 4. VERSION ──────────────────────────────────────────────────
    ver_src = ROOT / "src" / "scpn_fusion" / "VERSION"
    if ver_src.exists():
        shutil.copy2(ver_src, out / "VERSION")
        version = ver_src.read_text().strip()
    else:
        version = "unknown"

    # ── 5. Generate dataset README ──────────────────────────────────
    readme = out / "README.md"
    readme.write_text(_build_readme(version), encoding="utf-8")
    print("  Generated README.md")

    # ── 6. Generate dataset metadata JSON ───────────────────────────
    meta = out / "dataset_metadata.json"
    meta.write_text(json.dumps(_build_metadata(version), indent=2, ensure_ascii=False), encoding="utf-8")
    print("  Generated dataset_metadata.json")

    # ── Summary ─────────────────────────────────────────────────────
    total_size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    n_files = sum(1 for f in out.rglob("*") if f.is_file())
    print(f"\nDone. {n_files} files, {total_size / 1024:.0f} KB total")
    print(f"Upload the contents of {out} to Zenodo as type 'Dataset'.")


def _build_readme(version: str) -> str:
    return f"""\
# SCPN Fusion Core v{version} — Benchmark Dataset

This dataset accompanies the paper:

> **Neuromorphic Spiking Neural Network Control for Tokamak Plasma Stabilisation:
> A Comparative Benchmark Study**
>
> Miroslav Šotek, Anulum Research (2026)

## Contents

| Directory / File | Description |
|-----------------|-------------|
| `benchmark_reports/` | Raw JSON + Markdown reports from all 11 benchmark modules |
| `figures/` | Publication-quality PNG plots (3 figures) |
| `RESULTS.md` | Auto-generated summary table with all metrics |
| `VERSION` | Software version that produced this data |
| `dataset_metadata.json` | Zenodo deposit metadata |
| `README.md` | This file |

## Benchmark Reports

### Controller Stress-Test Campaign
- `stress_test_campaign.json` — 400-episode campaign (100 per controller × 4 controllers)
  - PID, H∞, NMPC-JAX, Nengo-SNN
  - Per-controller: latency percentiles, mean reward, disruption rate

### Physics Benchmarks
- `task6_heating_neutronics_realism.json` — Q scan, ECRH, TBR
- `task4_quasi_3d_modeling.json` — 3D force balance equilibrium
- `task5_disruption_mitigation_integration.json` — Disruption metrics, halo/RE currents
- `scpn_end_to_end_latency.json` — HIL control-loop latency
- `transport_power_balance_benchmark.json` — Energy transport
- `multi_ion_transport_conservation_benchmark.json` — Multi-species transport
- `eped_domain_contract_benchmark.json` — Pedestal stability
- `transport_uncertainty_envelope_benchmark.json` — Transport uncertainty
- `task2_pretrained_surrogates_benchmark.json` — Neural surrogates (MLP, FNO)
- `gneu_01_benchmark.json` — Neural equilibrium solver
- `gneu_03_fueling.json` — Fueling and density control

## Figures

1. **`controller_latency_comparison.png`** — Bar chart: P50/P95/P99 latency per controller
2. **`fno_suppression.png`** — FNO turbulence surrogate: field heatmap + energy timeseries
3. **`snn_trajectory.png`** — SNN controller: R-axis position tracking vs target

## Reproducibility

To regenerate this dataset from source:

```bash
pip install scpn-fusion-core[all]
python validation/collect_results.py      # ~5 hours, generates RESULTS.md
python tools/generate_benchmark_plots.py  # ~2 minutes, generates PNGs
python tools/export_zenodo_dataset.py     # bundles everything
```

## Environment

- CPU: Intel Core i7-10700K
- RAM: 31.8 GB
- OS: Windows 11 (10.0.26200)
- Python: 3.12.5
- NumPy: 1.26.4
- Nengo: 4.1.0

## License

Data: CC-BY-4.0
Software: GNU AGPL v3.0

## DOI

Software: https://doi.org/10.5281/zenodo.18731981
"""


def _build_metadata(version: str) -> dict:
    return {
        "metadata": {
            "title": f"Benchmark Dataset for SCPN Fusion Core v{version} — Neuromorphic SNN Tokamak Control",
            "upload_type": "dataset",
            "publication_date": "2026-02-22",
            "creators": [
                {
                    "name": "Sotek, Miroslav",
                    "affiliation": "Anulum Research",
                    "orcid": "0009-0009-3560-0851",
                }
            ],
            "description": (
                "<p>Benchmark dataset for the paper: <em>Neuromorphic Spiking Neural Network "
                "Control for Tokamak Plasma Stabilisation: A Comparative Benchmark Study</em>.</p>"
                "<p>Contains raw JSON reports from an 11-module benchmark suite, a 400-episode "
                "controller stress-test campaign (PID, H&infin;, NMPC-JAX, Nengo-SNN), "
                "hardware-in-the-loop latency measurements, physics validation metrics "
                "(Q=15, TBR=1.141, ECRH 99%), and publication-quality figures.</p>"
                "<p>All data generated deterministically from SCPN Fusion Core v"
                + version
                + " with fixed random seeds.</p>"
            ),
            "access_right": "open",
            "license": "cc-by-4.0",
            "keywords": [
                "tokamak",
                "plasma control",
                "spiking neural networks",
                "benchmark dataset",
                "fusion energy",
                "neuromorphic computing",
                "controller comparison",
                "ITER",
            ],
            "related_identifiers": [
                {
                    "identifier": "10.5281/zenodo.18731981",
                    "relation": "isSupplementTo",
                    "resource_type": "software",
                    "scheme": "doi",
                }
            ],
            "version": version,
            "language": "eng",
            "notes": (
                "Generated on Intel Core i7-10700K, 31.8 GB RAM, Windows 11. "
                "Python 3.12.5, NumPy 1.26.4, Nengo 4.1.0. "
                "Wall-clock for full benchmark suite: ~17,700 s."
            ),
        }
    }


if __name__ == "__main__":
    main()
