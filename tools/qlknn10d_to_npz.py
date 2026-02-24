# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — QLKNN-10D Data Pipeline
# © 1998–2026 Miroslav Sotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
"""
Convert QLKNN-10D Zenodo HDF5 data to training-ready .npz files.

Input contract (from QLKNN-10D HDF5):
  10 input columns: Ati, Ate, An, q, smag, x, Ti_Te, Zeff, alpha, Machtor
  3 flux outputs:   efi_GB, efe_GB, pfe_GB  (gyro-Bohm normalised)

Output contract (for NeuralTransportModel):
  X: shape (N, 10), dtype float64
     [rho, Te, Ti, ne, R/LTe, R/LTi, R/Lne, q, s_hat, beta_e]
  Y: shape (N, 3),  dtype float64
     [chi_e, chi_i, D_e]  in m^2/s
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "qlknn10d"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "qlknn10d_processed"

# QLKNN-10D column names (expected in HDF5)
QLKNN_INPUT_COLS = ["Ati", "Ate", "An", "q", "smag", "x", "Ti_Te", "Zeff", "alpha", "Machtor"]
QLKNN_FLUX_COLS = ["efi_GB", "efe_GB", "pfe_GB"]

# Physical sampling ranges for augmentation
TE_RANGE = (1.0, 25.0)   # keV
NE_RANGE = (1.0, 15.0)   # 10^19 m^-3
BT_REF = 5.3              # T
R_REF = 6.2               # m
MI_KG = 3.344e-27          # deuterium
E_CHARGE = 1.602e-19


def _gyrobohm_chi(te_kev: np.ndarray) -> np.ndarray:
    te_j = te_kev * 1e3 * E_CHARGE
    cs = np.sqrt(te_j / MI_KG)
    rho_s = np.sqrt(MI_KG * te_j) / (E_CHARGE * BT_REF)
    return rho_s**2 * cs / R_REF


def _detect_format(input_dir: Path) -> tuple[str, list[Path]]:
    h5_files = sorted(input_dir.glob("*.h5")) + sorted(input_dir.glob("*.hdf5"))
    if h5_files: return "hdf5", h5_files
    nc_files = sorted(input_dir.glob("*.nc"))
    if nc_files: return "netcdf", nc_files
    raise FileNotFoundError(f"No HDF5 or NetCDF files found in {input_dir}.")


def _load_chunk_hdf5(path: Path, start: int, count: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import h5py
    import pandas as pd
    with h5py.File(path, "r") as f:
        if "input" in f and "output" in f:
            df_in = pd.read_hdf(path, key="input", start=start, stop=start+count)
            inputs = np.zeros((len(df_in), len(QLKNN_INPUT_COLS)), dtype=np.float64)
            for i, col in enumerate(QLKNN_INPUT_COLS):
                if col in df_in.columns:
                    inputs[:, i] = df_in[col].values
                elif col == "alpha" and "alpha_MHD" in df_in.columns:
                    inputs[:, i] = df_in["alpha_MHD"].values
            
            fluxes = np.zeros((len(df_in), len(QLKNN_FLUX_COLS)), dtype=np.float64)
            for i, col in enumerate(QLKNN_FLUX_COLS):
                out_key = f"output/{col}"
                try:
                    df_out = pd.read_hdf(path, key=out_key, start=start, stop=start+len(df_in))
                    fluxes[:, i] = df_out[col].values if col in df_out.columns else df_out.iloc[:, 0].values
                except:
                    with h5py.File(path, "r") as f_loc:
                        if out_key in f_loc: fluxes[:, i] = f_loc[out_key][start:start+len(df_in)]
            return inputs, fluxes, list(df_in.columns)
    return np.zeros((0, 10)), np.zeros((0, 3)), []


def _get_total_rows_hdf5(path: Path) -> int:
    import h5py
    with h5py.File(path, "r") as f:
        if "input" in f and "block0_values" in f["input"]:
            return f["input"]["block0_values"].shape[0]
    return 0


def _classify_regime(ati: np.ndarray, ate: np.ndarray) -> np.ndarray:
    regime = np.zeros(len(ati), dtype=np.int32)
    regime[ati > 4.0] = 1
    regime[(ate > 5.0) & (ati <= 4.0)] = 2
    return regime


def process(input_dir: Path, output_dir: Path, max_samples: int = 5_000_000, seed: int = 42, regime_balanced: bool = False, gb_normalized: bool = True) -> None:
    rng = np.random.default_rng(seed)
    fmt, files = _detect_format(input_dir)
    total_rows = sum(_get_total_rows_hdf5(f) for f in files)
    subsample_rate = min(1.0, max_samples / max(total_rows, 1))
    
    CHUNK_SIZE = 500_000
    all_inputs, all_fluxes, loaded = [], [], 0
    t0 = time.monotonic()

    for fpath in files:
        n_rows = _get_total_rows_hdf5(fpath)
        offset = 0
        while offset < n_rows and loaded < max_samples:
            chunk_n = min(CHUNK_SIZE, n_rows - offset)
            inputs, fluxes, cols = _load_chunk_hdf5(fpath, offset, chunk_n)
            offset += chunk_n
            
            # Filter Zeff=1.0 for determinism in 10D MLP
            if "Zeff" in cols:
                idx = QLKNN_INPUT_COLS.index("Zeff")
                mask = (np.abs(inputs[:, idx] - 1.0) < 1e-3)
                inputs, fluxes = inputs[mask], fluxes[mask]

            if subsample_rate < 1.0:
                mask = rng.random(len(inputs)) < subsample_rate
                inputs, fluxes = inputs[mask], fluxes[mask]

            valid = np.all(np.isfinite(inputs), axis=1) & np.all(np.isfinite(fluxes), axis=1)
            inputs, fluxes = inputs[valid], fluxes[valid]
            # Clip negative fluxes to 0 (stable regime) instead of dropping.
            # Keeping these samples teaches the model the ITG/TEM onset threshold.
            fluxes = np.clip(fluxes, 0.0, None)

            if len(inputs) > 0:
                all_inputs.append(inputs); all_fluxes.append(fluxes); loaded += len(inputs)
                print(f"\r  Loaded {loaded:,} samples", end="")

    raw_inputs = np.vstack(all_inputs)
    raw_fluxes = np.vstack(all_fluxes)
    
    # Shuffle entire loaded set before slicing to max_samples
    # to avoid bias from ordered HDF5 grid
    N_loaded = len(raw_inputs)
    perm_load = rng.permutation(N_loaded)
    raw_inputs = raw_inputs[perm_load][:max_samples]
    raw_fluxes = raw_fluxes[perm_load][:max_samples]
    
    N = len(raw_inputs)

    ati, ate, an, q, smag, x, ti_te = raw_inputs[:, 0], raw_inputs[:, 1], raw_inputs[:, 2], raw_inputs[:, 3], raw_inputs[:, 4], raw_inputs[:, 5], raw_inputs[:, 6]
    alpha = raw_inputs[:, 8]

    te_kev = np.exp(rng.uniform(np.log(TE_RANGE[0]), np.log(TE_RANGE[1]), size=N))
    ti_kev = te_kev * np.clip(ti_te, 0.5, 3.0)
    ne_19 = rng.uniform(*NE_RANGE, size=N)

    X = np.column_stack([x, te_kev, ti_kev, ne_19, ate, ati, an, q, smag, 4.03e-3 * ne_19 * te_kev])

    if gb_normalized:
        # Save raw GB-normalized fluxes — no chi_gb multiplication.
        # Column mapping: efi_GB → chi_i, efe_GB → chi_e, pfe_GB → D_e
        # Non-negative efi/efe already enforced by the valid filter above.
        Y = np.column_stack([raw_fluxes[:, 1], raw_fluxes[:, 0], np.abs(raw_fluxes[:, 2])])
    else:
        chi_gb = _gyrobohm_chi(te_kev)
        chi_i, chi_e = raw_fluxes[:, 0] * chi_gb, raw_fluxes[:, 1] * chi_gb
        d_e = np.abs(raw_fluxes[:, 2]) * chi_gb
        Y = np.column_stack([np.clip(chi_e, 0, 500), np.clip(chi_i, 0, 500), np.clip(d_e, 0, 200)])

    regimes = _classify_regime(ati, ate)
    perm = rng.permutation(N)
    X, Y, regimes = X[perm], Y[perm], regimes[perm]
    
    n_train, n_val = int(N * 0.9), int(N * 0.05)
    output_dir.mkdir(parents=True, exist_ok=True)
    _meta = {"gb_normalized": np.array(1 if gb_normalized else 0)}
    np.savez(output_dir / "train.npz", X=X[:n_train], Y=Y[:n_train], regimes=regimes[:n_train], **_meta)
    np.savez(output_dir / "val.npz", X=X[n_train:n_train+n_val], Y=Y[n_train:n_train+n_val], regimes=regimes[n_train:n_train+n_val], **_meta)
    np.savez(output_dir / "test.npz", X=X[n_train+n_val:], Y=Y[n_train+n_val:], regimes=regimes[n_train+n_val:], **_meta)
    mode_str = "GB-normalized" if gb_normalized else "physical (m^2/s)"
    print(f"\nDone. Saved {N} samples ({mode_str}).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=500000)
    parser.add_argument("--no-gb-normalized", dest="gb_normalized", action="store_false",
                        help="Save physical fluxes (m^2/s) instead of GB-normalized")
    parser.set_defaults(gb_normalized=True)
    args = parser.parse_args()
    process(DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, args.max_samples, gb_normalized=args.gb_normalized)

if __name__ == "__main__": main()
