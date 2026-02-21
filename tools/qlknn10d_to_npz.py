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

The QLKNN-10D uses normalised gradient-length-scale inputs.  Since
our NeuralTransportModel expects physical Te/Ti/ne alongside gradients,
we augment each QLKNN row by sampling Te/ne from tokamak-relevant
distributions and converting gyro-Bohm fluxes to physical units.

Usage
-----
    python tools/qlknn10d_to_npz.py
    python tools/qlknn10d_to_npz.py --max-samples 1000000
    python tools/qlknn10d_to_npz.py --check
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
BT_REF = 5.3              # T (reference toroidal field)
R_REF = 6.2               # m (reference major radius, ITER-like)
MI_KG = 3.344e-27          # deuterium ion mass [kg]
E_CHARGE = 1.602e-19       # elementary charge [C]


def _gyrobohm_chi(te_kev: np.ndarray) -> np.ndarray:
    """Compute gyro-Bohm chi normalisation: rho_s^2 * cs / R.

    chi_GB = (Te / e*B)^2 * sqrt(Te*e / m_i) / R
    Simplified: chi_GB ~ Te^{5/2} / (B^2 * R * sqrt(m_i) * e^{3/2})

    For typical tokamak parameters this is ~1-10 m^2/s.
    """
    te_j = te_kev * 1e3 * E_CHARGE  # keV -> J
    cs = np.sqrt(te_j / MI_KG)       # ion sound speed [m/s]
    rho_s = np.sqrt(MI_KG * te_j) / (E_CHARGE * BT_REF)  # Larmor radius [m]
    return rho_s**2 * cs / R_REF      # [m^2/s]


def _detect_format(input_dir: Path) -> tuple[str, list[Path]]:
    """Detect the data format in the input directory.

    Returns (format_type, file_list) where format_type is one of:
    'hdf5', 'csv', 'netcdf'.
    """
    h5_files = sorted(input_dir.glob("*.h5")) + sorted(input_dir.glob("*.hdf5"))
    if h5_files:
        return "hdf5", h5_files

    csv_files = sorted(input_dir.glob("*.csv"))
    if csv_files:
        return "csv", csv_files

    nc_files = sorted(input_dir.glob("*.nc"))
    if nc_files:
        return "netcdf", nc_files

    # Check for .pkl or .pickle (pandas)
    pkl_files = sorted(input_dir.glob("*.pkl")) + sorted(input_dir.glob("*.pickle"))
    if pkl_files:
        return "pickle", pkl_files

    raise FileNotFoundError(
        f"No HDF5, CSV, NetCDF, or pickle files found in {input_dir}.\n"
        "Run tools/download_qlknn10d.py first."
    )


def _load_chunk_hdf5(path: Path, start: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Load a chunk of rows from an HDF5 file."""
    import h5py
    with h5py.File(path, "r") as f:
        # Auto-detect dataset structure
        # QLKNN-10D may store data as separate datasets per column
        # or as a single large table
        keys = list(f.keys())

        if all(col in f for col in QLKNN_INPUT_COLS + QLKNN_FLUX_COLS):
            # Column-per-dataset layout
            end = min(start + count, len(f[QLKNN_INPUT_COLS[0]]))
            inputs = np.column_stack([
                np.array(f[col][start:end], dtype=np.float64)
                for col in QLKNN_INPUT_COLS
            ])
            fluxes = np.column_stack([
                np.array(f[col][start:end], dtype=np.float64)
                for col in QLKNN_FLUX_COLS
            ])
            return inputs, fluxes

        # Try 'data' or 'table' dataset
        for dname in ("data", "table", "dataset"):
            if dname in f:
                ds = f[dname]
                end = min(start + count, ds.shape[0])
                chunk = ds[start:end]
                if hasattr(chunk, "dtype") and chunk.dtype.names:
                    # Structured array
                    inputs = np.column_stack([
                        np.array(chunk[col], dtype=np.float64)
                        for col in QLKNN_INPUT_COLS
                    ])
                    fluxes = np.column_stack([
                        np.array(chunk[col], dtype=np.float64)
                        for col in QLKNN_FLUX_COLS
                    ])
                    return inputs, fluxes

        raise KeyError(
            f"Could not find expected columns in {path}.\n"
            f"Available keys: {keys}\n"
            f"Expected input columns: {QLKNN_INPUT_COLS}\n"
            f"Expected flux columns: {QLKNN_FLUX_COLS}"
        )


def _get_total_rows_hdf5(path: Path) -> int:
    """Get total number of rows in an HDF5 file."""
    import h5py
    with h5py.File(path, "r") as f:
        if QLKNN_INPUT_COLS[0] in f:
            return len(f[QLKNN_INPUT_COLS[0]])
        for dname in ("data", "table", "dataset"):
            if dname in f:
                return f[dname].shape[0]
    return 0


def _classify_regime(ati: np.ndarray, ate: np.ndarray) -> np.ndarray:
    """Classify turbulence regime from gradient ratios.

    Returns array of 0=stable, 1=ITG, 2=TEM.
    """
    regime = np.zeros(len(ati), dtype=np.int32)
    regime[ati > 4.0] = 1  # ITG dominated
    regime[(ate > 5.0) & (ati <= 4.0)] = 2  # TEM dominated
    return regime


def process(
    input_dir: Path,
    output_dir: Path,
    max_samples: int = 5_000_000,
    seed: int = 42,
    regime_balanced: bool = False,
) -> None:
    """Convert QLKNN-10D data to training .npz files."""
    rng = np.random.default_rng(seed)

    fmt, files = _detect_format(input_dir)
    print(f"Detected format: {fmt}, {len(files)} file(s)")

    if fmt != "hdf5":
        raise NotImplementedError(
            f"Format '{fmt}' not yet supported. "
            "Currently only HDF5 is implemented."
        )

    # Count total rows
    total_rows = sum(_get_total_rows_hdf5(f) for f in files)
    print(f"Total rows across all files: {total_rows:,}")

    # Determine subsampling rate
    subsample_rate = min(1.0, max_samples / max(total_rows, 1))
    print(f"Subsample rate: {subsample_rate:.4f} (target {max_samples:,} samples)")

    # Chunked loading with subsampling
    CHUNK_SIZE = 500_000
    all_inputs = []
    all_fluxes = []
    loaded = 0
    t0 = time.monotonic()

    for fpath in files:
        n_rows = _get_total_rows_hdf5(fpath)
        offset = 0
        while offset < n_rows and loaded < max_samples:
            chunk_n = min(CHUNK_SIZE, n_rows - offset)
            inputs, fluxes = _load_chunk_hdf5(fpath, offset, chunk_n)
            offset += chunk_n

            # Random subsample
            if subsample_rate < 1.0:
                mask = rng.random(len(inputs)) < subsample_rate
                inputs = inputs[mask]
                fluxes = fluxes[mask]

            if len(inputs) == 0:
                continue

            # Filter: remove NaN, Inf, negative fluxes
            valid = np.all(np.isfinite(inputs), axis=1) & np.all(np.isfinite(fluxes), axis=1)
            # Allow negative fluxes in pfe_GB (particle flux can be inward)
            # but require non-negative energy fluxes
            valid &= (fluxes[:, 0] >= 0)  # efi_GB >= 0
            valid &= (fluxes[:, 1] >= 0)  # efe_GB >= 0
            inputs = inputs[valid]
            fluxes = fluxes[valid]

            all_inputs.append(inputs)
            all_fluxes.append(fluxes)
            loaded += len(inputs)

            elapsed = time.monotonic() - t0
            print(
                f"\r  Loaded {loaded:,} samples "
                f"({elapsed:.0f}s, {loaded/max(elapsed, 0.01):.0f} rows/s)",
                end="", flush=True,
            )

    print()

    if loaded == 0:
        print("ERROR: No valid samples loaded. Check data format.")
        sys.exit(1)

    raw_inputs = np.vstack(all_inputs)[:max_samples]
    raw_fluxes = np.vstack(all_fluxes)[:max_samples]
    N = len(raw_inputs)
    print(f"Loaded {N:,} valid samples")

    # ── Map QLKNN columns to our 10D convention ──────────────────────
    # QLKNN order: [Ati, Ate, An, q, smag, x, Ti_Te, Zeff, alpha, Machtor]
    ati = raw_inputs[:, 0]
    ate = raw_inputs[:, 1]
    an = raw_inputs[:, 2]
    q = raw_inputs[:, 3]
    smag = raw_inputs[:, 4]
    x = raw_inputs[:, 5]       # rho
    ti_te = raw_inputs[:, 6]
    # Zeff = raw_inputs[:, 7]  # not used in our 10D
    alpha = raw_inputs[:, 8]   # alpha_MHD ~ beta_e
    # Machtor = raw_inputs[:, 9]  # not used

    # ── Augment with physical Te/ne for gyro-Bohm conversion ────────
    te_kev = rng.uniform(*TE_RANGE, size=N)
    ti_kev = te_kev * np.clip(ti_te, 0.5, 3.0)  # Ti = Te * Ti_Te ratio
    ne_19 = rng.uniform(*NE_RANGE, size=N)

    # Our 10D: [rho, Te, Ti, ne, R/LTe, R/LTi, R/Lne, q, s_hat, beta_e]
    X = np.column_stack([
        x,          # rho
        te_kev,     # Te [keV]
        ti_kev,     # Ti [keV]
        ne_19,      # ne [10^19 m^-3]
        ate,        # R/LTe
        ati,        # R/LTi
        an,         # R/Lne
        q,          # safety factor
        smag,       # magnetic shear
        alpha,      # beta_e (alpha_MHD proxy)
    ])

    # ── Convert gyro-Bohm fluxes to physical units ──────────────────
    chi_gb = _gyrobohm_chi(te_kev)  # [m^2/s]

    efi_gb = raw_fluxes[:, 0]  # ion energy flux (GB)
    efe_gb = raw_fluxes[:, 1]  # electron energy flux (GB)
    pfe_gb = raw_fluxes[:, 2]  # particle flux (GB)

    chi_i = efi_gb * chi_gb
    chi_e = efe_gb * chi_gb
    d_e = np.abs(pfe_gb) * chi_gb  # particle diffusivity magnitude

    # Sanity cap: clip unreasonably large values
    chi_i = np.clip(chi_i, 0, 500.0)
    chi_e = np.clip(chi_e, 0, 500.0)
    d_e = np.clip(d_e, 0, 200.0)

    Y = np.column_stack([chi_e, chi_i, d_e])

    # ── Regime classification ────────────────────────────────────────
    regimes = _classify_regime(ati, ate)
    n_stable = np.sum(regimes == 0)
    n_itg = np.sum(regimes == 1)
    n_tem = np.sum(regimes == 2)
    print(f"Regime distribution: ITG={n_itg:,} ({n_itg/N*100:.1f}%), "
          f"TEM={n_tem:,} ({n_tem/N*100:.1f}%), "
          f"Stable={n_stable:,} ({n_stable/N*100:.1f}%)")

    # ── Stratified split ─────────────────────────────────────────────
    if regime_balanced:
        min_regime = min(n_stable, n_itg, n_tem)
        idx_stable = rng.choice(np.where(regimes == 0)[0], min(min_regime, n_stable), replace=False)
        idx_itg = rng.choice(np.where(regimes == 1)[0], min(min_regime, n_itg), replace=False)
        idx_tem = rng.choice(np.where(regimes == 2)[0], min(min_regime, n_tem), replace=False)
        indices = np.concatenate([idx_stable, idx_itg, idx_tem])
        rng.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        regimes = regimes[indices]
        N = len(X)
        print(f"After regime balancing: {N:,} samples")

    # Shuffle all data
    perm = rng.permutation(N)
    X = X[perm]
    Y = Y[perm]
    regimes = regimes[perm]

    # Split: 90/5/5
    n_train = int(N * 0.90)
    n_val = int(N * 0.05)

    X_train, Y_train, R_train = X[:n_train], Y[:n_train], regimes[:n_train]
    X_val, Y_val, R_val = X[n_train:n_train+n_val], Y[n_train:n_train+n_val], regimes[n_train:n_train+n_val]
    X_test, Y_test, R_test = X[n_train+n_val:], Y[n_train+n_val:], regimes[n_train+n_val:]

    print(f"Split: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}")

    # ── Compute normalization stats on train set ONLY ────────────────
    input_mean = np.mean(X_train, axis=0)
    input_std = np.std(X_train, axis=0)
    output_mean = np.mean(Y_train, axis=0)
    output_std = np.std(Y_train, axis=0)

    # ── Save ─────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(output_dir / "train.npz", X=X_train, Y=Y_train, regimes=R_train)
    np.savez(output_dir / "val.npz", X=X_val, Y=Y_val, regimes=R_val)
    np.savez(output_dir / "test.npz", X=X_test, Y=Y_test, regimes=R_test)

    metadata = {
        "source": "QLKNN-10D (Zenodo DOI 10.5281/zenodo.3497066)",
        "citation": "van de Plassche et al., Phys. Plasmas 27, 022310 (2020)",
        "processed": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "total_samples": N,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "regime_balanced": regime_balanced,
        "regime_counts": {"stable": int(n_stable), "ITG": int(n_itg), "TEM": int(n_tem)},
        "input_columns": ["rho", "Te_keV", "Ti_keV", "ne_19", "R_LTe", "R_LTi", "R_Lne", "q", "s_hat", "beta_e"],
        "output_columns": ["chi_e", "chi_i", "D_e"],
        "input_mean": input_mean.tolist(),
        "input_std": input_std.tolist(),
        "output_mean": output_mean.tolist(),
        "output_std": output_std.tolist(),
        "gyrobohm_params": {"B_T": BT_REF, "R_major": R_REF, "m_i_kg": MI_KG},
        "physical_ranges": {
            "Te_keV": list(TE_RANGE),
            "ne_19": list(NE_RANGE),
        },
        "sanity_checks": {
            "chi_i_mean": float(np.mean(chi_i)),
            "chi_i_median": float(np.median(chi_i)),
            "chi_e_mean": float(np.mean(chi_e)),
            "chi_e_median": float(np.median(chi_e)),
            "d_e_mean": float(np.mean(d_e)),
        },
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"\nSaved to {output_dir}/")
    print(f"  train.npz: {X_train.shape}")
    print(f"  val.npz:   {X_val.shape}")
    print(f"  test.npz:  {X_test.shape}")
    print(f"\nSanity checks:")
    print(f"  chi_i mean: {np.mean(chi_i):.2f} m^2/s (expect ~1-10 for ITER)")
    print(f"  chi_e mean: {np.mean(chi_e):.2f} m^2/s")
    print(f"  D_e mean:   {np.mean(d_e):.2f} m^2/s")


def check(output_dir: Path) -> bool:
    """Verify processed data integrity."""
    if not output_dir.exists():
        print(f"Directory {output_dir} does not exist. Run without --check first.")
        return False

    ok = True
    for name in ("train", "val", "test"):
        path = output_dir / f"{name}.npz"
        if not path.exists():
            print(f"  MISSING: {path}")
            ok = False
            continue
        data = np.load(path)
        X, Y = data["X"], data["Y"]
        if X.shape[1] != 10:
            print(f"  WRONG SHAPE: {path} X has {X.shape[1]} columns (expected 10)")
            ok = False
        if Y.shape[1] != 3:
            print(f"  WRONG SHAPE: {path} Y has {Y.shape[1]} columns (expected 3)")
            ok = False
        if np.any(~np.isfinite(X)):
            print(f"  NaN/Inf in X: {path}")
            ok = False
        if np.any(~np.isfinite(Y)):
            print(f"  NaN/Inf in Y: {path}")
            ok = False
        print(f"  OK: {name}.npz — X{X.shape}, Y{Y.shape}")

    meta_path = output_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"  Source: {meta.get('source', 'unknown')}")
        print(f"  Processed: {meta.get('processed', 'unknown')}")
    else:
        print(f"  MISSING: metadata.json")
        ok = False

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert QLKNN-10D HDF5 data to training .npz files."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regime-balanced", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        ok = check(args.output_dir)
        sys.exit(0 if ok else 1)
    else:
        process(args.input_dir, args.output_dir, args.max_samples, args.seed, args.regime_balanced)


if __name__ == "__main__":
    main()
