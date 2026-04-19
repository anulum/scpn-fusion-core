# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tokamak Archive Profile Utilities
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.eqdsk import read_geqdsk


_DATA_ROOT_ENV = "SCPN_DATA_DIR"


def default_reference_data_root() -> Path:
    """Return the base directory for reference data assets."""
    env = os.environ.get(_DATA_ROOT_ENV, "").strip()
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[3] / "validation" / "reference_data"


def default_diiid_dir() -> Path:
    return default_reference_data_root() / "diiid"


def default_disruption_dir() -> Path:
    return default_diiid_dir() / "disruption_shots"


def default_itpa_csv() -> Path:
    return default_reference_data_root() / "itpa" / "hmode_confinement.csv"


def default_synthetic_dir() -> Path:
    return default_reference_data_root() / "synthetic_shots"


@dataclass(frozen=True)
class TokamakProfile:
    machine: str
    shot: int
    time_ms: float
    beta_n: float
    q95: float
    tau_e_ms: float
    psi_contour: tuple[float, ...]
    sensor_trace: tuple[float, ...]
    toroidal_n1_amp: float
    toroidal_n2_amp: float
    toroidal_n3_amp: float
    disruption: bool


def _profile_key(profile: TokamakProfile) -> tuple[str, int, int]:
    return (
        profile.machine,
        int(profile.shot),
        int(round(float(profile.time_ms))),
    )


def _normalize_machine(machine: str) -> str:
    text = machine.strip().lower()
    if text in {"diii-d", "diiid"}:
        return "DIII-D"
    if text in {"alcator c-mod", "c-mod", "cmod"}:
        return "C-Mod"
    raise ValueError(f"Unsupported machine: {machine!r}")


def _coerce_int(name: str, value: Any, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    out = int(value)
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return out


def _coerce_finite(name: str, value: Any, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError(f"{name} must be a finite number.")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return out


def _resample_1d(vec: NDArray[np.float64], points: int) -> NDArray[np.float64]:
    if vec.ndim != 1 or vec.size < 2:
        raise ValueError("Expected a 1D array with at least 2 values.")
    x = np.linspace(0.0, 1.0, vec.size, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, int(points), dtype=np.float64)
    return np.interp(x_new, x, vec).astype(np.float64)


def _build_sensor_trace(
    psi_contour: NDArray[np.float64],
    points: int,
    seed: int,
) -> NDArray[np.float64]:
    rng = np.random.default_rng(int(seed))
    grad = np.gradient(psi_contour)
    base = np.concatenate((psi_contour, grad), axis=0)
    base = _resample_1d(base, max(points, 16))
    t = np.linspace(0.0, 1.0, base.size, dtype=np.float64)
    mod = 0.08 * np.sin(2.0 * np.pi * 4.0 * t + 0.3)
    noise = rng.normal(0.0, 0.01, size=base.size)
    out = np.clip(base + mod + noise, 0.0, 2.5)
    return np.asarray(out, dtype=np.float64)


def _profile_from_geqdsk(
    path: Path,
    *,
    shot: int,
    time_ms: float,
    contour_points: int,
    sensor_points: int,
) -> TokamakProfile:
    eq = read_geqdsk(path)
    psi_mid = np.asarray(eq.psirz[eq.nh // 2, :], dtype=np.float64)
    denom = float(eq.sibry - eq.simag)
    if abs(denom) < 1e-12:
        raise ValueError(f"Degenerate psi normalization in {path.name}.")
    psi_norm = np.clip((psi_mid - float(eq.simag)) / denom, 0.0, 1.0)
    psi_contour = _resample_1d(psi_norm, contour_points)

    q_profile = np.asarray(eq.qpsi, dtype=np.float64)
    q95 = 4.5
    if q_profile.size > 0 and np.all(np.isfinite(q_profile)):
        idx95 = min(q_profile.size - 1, int(0.95 * (q_profile.size - 1)))
        q95 = float(abs(q_profile[idx95]))

    ip_ma = abs(float(eq.current)) / 1.0e6
    name = path.stem.lower()
    is_hmode = "hmode" in name
    beta_n = float(np.clip(0.9 + 0.48 * ip_ma + (0.18 if is_hmode else 0.0), 0.6, 4.2))
    tau_e_ms = float(np.clip(68.0 + 18.0 * ip_ma + (9.0 if is_hmode else 0.0), 15.0, 380.0))
    disruption = bool(("snowflake" in name) or ("negdelta" in name) or ("lmode" in name))

    sensor_trace = _build_sensor_trace(psi_contour, points=sensor_points, seed=shot)
    return TokamakProfile(
        machine="DIII-D",
        shot=int(shot),
        time_ms=float(time_ms),
        beta_n=beta_n,
        q95=q95,
        tau_e_ms=tau_e_ms,
        psi_contour=tuple(float(v) for v in psi_contour),
        sensor_trace=tuple(float(v) for v in sensor_trace),
        toroidal_n1_amp=float(np.clip(0.09 + 0.06 * beta_n, 0.05, 0.45)),
        toroidal_n2_amp=float(np.clip(0.05 + 0.03 * beta_n, 0.02, 0.30)),
        toroidal_n3_amp=float(np.clip(0.02 + 0.02 * beta_n, 0.01, 0.22)),
        disruption=disruption,
    )


def _stable_shot_from_text(text: str) -> int:
    acc = 0
    for ch in text:
        acc = (acc * 131 + ord(ch)) % 900_000
    return 100_000 + acc


def _synthetic_cmod_psi_contour(kappa: float, delta: float, points: int) -> NDArray[np.float64]:
    theta = np.linspace(0.0, 2.0 * np.pi, int(points), endpoint=False, dtype=np.float64)
    r = 0.55 + 0.20 * np.cos(theta + 0.35 * delta) + 0.08 * np.sin(2.0 * theta)
    z = 0.45 * kappa * np.sin(theta)
    psi = np.sqrt((r - np.mean(r)) ** 2 + (z - np.mean(z)) ** 2)
    psi_norm = (psi - np.min(psi)) / (np.max(psi) - np.min(psi) + 1e-12)
    return np.asarray(psi_norm, dtype=np.float64)


__all__ = [
    "TokamakProfile",
    "default_reference_data_root",
    "default_diiid_dir",
    "default_disruption_dir",
    "default_itpa_csv",
    "default_synthetic_dir",
    "_profile_key",
    "_normalize_machine",
    "_coerce_int",
    "_coerce_finite",
    "_resample_1d",
    "_build_sensor_trace",
    "_profile_from_geqdsk",
    "_stable_shot_from_text",
    "_synthetic_cmod_psi_contour",
]
