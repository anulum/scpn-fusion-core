# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tokamak Synthetic Shot Utilities
"""Synthetic shot database helpers extracted from tokamak_archive monolith."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


#: Machine parameter specifications for synthetic shot generation.
_MACHINE_SPECS: dict[str, dict[str, Any]] = {
    "ITER": {
        "n_shots": 15,
        "Ip_MA": (15.0, 15.0),
        "BT_T": (5.3, 5.3),
        "ne_1e19": (10.0, 12.0),
        "Te_keV": (8.0, 25.0),
        "Ti_keV_ratio": (0.85, 1.0),
        "q95": (2.8, 3.5),
        "beta_N": (1.8, 2.5),
    },
    "SPARC": {
        "n_shots": 15,
        "Ip_MA": (8.7, 8.7),
        "BT_T": (12.2, 12.2),
        "ne_1e19": (25.0, 40.0),
        "Te_keV": (10.0, 22.0),
        "Ti_keV_ratio": (0.80, 0.95),
        "q95": (3.0, 4.0),
        "beta_N": (1.5, 2.8),
    },
    "DIII-D": {
        "n_shots": 20,
        "Ip_MA": (1.0, 2.0),
        "BT_T": (2.1, 2.1),
        "ne_1e19": (3.0, 8.0),
        "Te_keV": (2.0, 8.0),
        "Ti_keV_ratio": (0.75, 1.05),
        "q95": (3.0, 6.0),
        "beta_N": (1.0, 3.0),
    },
    "EAST": {
        "n_shots": 10,
        "Ip_MA": (0.4, 1.0),
        "BT_T": (2.5, 2.5),
        "ne_1e19": (2.0, 5.0),
        "Te_keV": (1.0, 5.0),
        "Ti_keV_ratio": (0.70, 0.95),
        "q95": (4.0, 7.0),
        "beta_N": (0.8, 2.2),
    },
}


def _parabolic_profile(
    rho: NDArray[np.float64],
    peak: float,
    edge_fraction: float = 0.1,
    alpha: float = 0.5,
) -> NDArray[np.float64]:
    """Parabolic profile: peak * ((1 - rho^2)^alpha * (1 - edge_fraction) + edge_fraction)."""
    return np.asarray(
        peak * ((1.0 - rho**2) ** alpha * (1.0 - edge_fraction) + edge_fraction),
        dtype=np.float64,
    )


def generate_synthetic_shot_database(
    *,
    n_shots: int = 60,
    output_dir: Path,
    seed: int = 20260218,
) -> list[dict[str, Any]]:
    """Generate synthetic-realistic tokamak shot profiles for ITER/SPARC/DIII-D/EAST."""
    rng = np.random.default_rng(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_total = sum(s["n_shots"] for s in _MACHINE_SPECS.values())
    scale = max(1.0, n_shots / spec_total)
    machine_counts: dict[str, int] = {
        m: max(int(round(s["n_shots"] * scale)), s["n_shots"]) for m, s in _MACHINE_SPECS.items()
    }

    rho = np.linspace(0.0, 1.0, 1000, dtype=np.float64)
    time_s = np.linspace(0.0, 10.0, 1000, dtype=np.float64)

    catalogue: list[dict[str, Any]] = []
    shot_counter = 0

    for machine, spec in _MACHINE_SPECS.items():
        count = machine_counts[machine]
        for _ in range(count):
            shot_counter += 1
            is_disruption = bool(rng.random() < 0.20)

            ip = float(rng.uniform(*spec["Ip_MA"]))
            bt = float(rng.uniform(*spec["BT_T"]))
            ne0 = float(rng.uniform(*spec["ne_1e19"]))
            te0 = float(rng.uniform(*spec["Te_keV"]))
            ti_ratio = float(rng.uniform(*spec["Ti_keV_ratio"]))
            ti0 = te0 * ti_ratio
            q95_val = float(rng.uniform(*spec["q95"]))
            beta_n = float(rng.uniform(*spec["beta_N"]))

            if is_disruption:
                beta_n = float(np.clip(beta_n * 1.4, spec["beta_N"][0], spec["beta_N"][1] * 1.5))
                q95_val = float(np.clip(q95_val * 0.75, 1.5, spec["q95"][1]))

            ne_profile = _parabolic_profile(rho, ne0, edge_fraction=0.12, alpha=0.5)
            te_profile = _parabolic_profile(rho, te0, edge_fraction=0.08, alpha=0.6)
            ti_profile = _parabolic_profile(rho, ti0, edge_fraction=0.08, alpha=0.55)

            ip_trace = np.full(1000, ip, dtype=np.float64)
            bt_trace = np.full(1000, bt, dtype=np.float64)
            q95_trace = np.full(1000, q95_val, dtype=np.float64)
            beta_n_trace = np.full(1000, beta_n, dtype=np.float64)

            ip_trace += rng.normal(0.0, 0.01 * ip, size=1000)
            q95_trace += rng.normal(0.0, 0.02 * q95_val, size=1000)
            beta_n_trace += rng.normal(0.0, 0.02 * beta_n, size=1000)

            shot_id = f"{machine.replace('-', '')}_{shot_counter:04d}"
            fname = f"shot_{shot_id}.npz"
            np.savez_compressed(
                str(out_dir / fname),
                time_s=time_s,
                Ip_MA=ip_trace,
                BT_T=bt_trace,
                ne_1e19=ne_profile,
                Te_keV=te_profile,
                Ti_keV=ti_profile,
                q95=q95_trace,
                beta_N=beta_n_trace,
                disruption_label=np.bool_(is_disruption),
                machine=np.array(machine, dtype="U16"),
            )
            catalogue.append(
                {
                    "file": fname,
                    "machine": machine,
                    "shot_id": shot_id,
                    "disruption_label": is_disruption,
                    "Ip_MA": round(ip, 4),
                    "BT_T": round(bt, 4),
                    "ne0_1e19": round(ne0, 4),
                    "Te0_keV": round(te0, 4),
                    "q95": round(q95_val, 4),
                    "beta_N": round(beta_n, 4),
                }
            )

    return catalogue


def list_synthetic_shots(
    *,
    synthetic_dir: Path,
) -> list[str]:
    """Return sorted names of synthetic shot NPZ files (without extension)."""
    d = Path(synthetic_dir)
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.npz"))


def load_synthetic_shot(
    shot_name_or_path: str | Path,
    *,
    synthetic_dir: Path,
) -> dict[str, Any]:
    """Load one synthetic shot NPZ file."""
    p = Path(shot_name_or_path)
    if not p.suffix:
        p = Path(synthetic_dir) / f"{p.name}.npz"
    if p.suffix.lower() != ".npz":
        raise ValueError(f"Synthetic shot file must be .npz: {p}")
    if not p.exists():
        raise FileNotFoundError(f"Synthetic shot file not found: {p}")

    with np.load(str(p), allow_pickle=False) as raw:
        required_array_keys = {
            "time_s",
            "Ip_MA",
            "BT_T",
            "ne_1e19",
            "Te_keV",
            "Ti_keV",
            "q95",
            "beta_N",
        }
        required_scalar_keys = {"disruption_label", "machine"}
        all_required = required_array_keys | required_scalar_keys
        present = set(raw.files)
        missing = all_required - present
        if missing:
            raise ValueError(f"NPZ file {p.name} missing keys: {sorted(missing)}")

        result: dict[str, Any] = {}
        for k in required_array_keys:
            result[k] = np.asarray(raw[k], dtype=np.float64)
        result["disruption_label"] = bool(np.asarray(raw["disruption_label"]).reshape(()).item())
        result["machine"] = str(np.asarray(raw["machine"]).reshape(()).item())
        return result
