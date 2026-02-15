# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Tokamak Archive Feed Adapters
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Empirical tokamak profile loaders with optional live MDSplus fallback."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scpn_fusion.core.eqdsk import read_geqdsk


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIIID_DIR = REPO_ROOT / "validation" / "reference_data" / "diiid"
DEFAULT_ITPA_CSV = REPO_ROOT / "validation" / "reference_data" / "itpa" / "hmode_confinement.csv"

DEFAULT_MDSPLUS_NODE_MAP: dict[str, str] = {
    "time_ms": "\\time_ms",
    "beta_n": "\\betan",
    "q95": "\\q95",
    "tau_e_ms": "\\taue_ms",
    "psi_contour": "\\psi_contour",
    "sensor_trace": "\\sensor_trace",
    "toroidal_n1_amp": "\\toroidal_n1_amp",
    "toroidal_n2_amp": "\\toroidal_n2_amp",
    "toroidal_n3_amp": "\\toroidal_n3_amp",
    "disruption": "\\disruption_flag",
}


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


def _resample_1d(vec: np.ndarray, points: int) -> np.ndarray:
    if vec.ndim != 1 or vec.size < 2:
        raise ValueError("Expected a 1D array with at least 2 values.")
    x = np.linspace(0.0, 1.0, vec.size, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, int(points), dtype=np.float64)
    return np.interp(x_new, x, vec).astype(np.float64)


def _build_sensor_trace(psi_contour: np.ndarray, points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    grad = np.gradient(psi_contour)
    base = np.concatenate((psi_contour, grad), axis=0)
    base = _resample_1d(base, max(points, 16))
    t = np.linspace(0.0, 1.0, base.size, dtype=np.float64)
    mod = 0.08 * np.sin(2.0 * np.pi * 4.0 * t + 0.3)
    noise = rng.normal(0.0, 0.01, size=base.size)
    out = np.clip(base + mod + noise, 0.0, 2.5)
    return out.astype(np.float64)


def _profile_from_geqdsk(path: Path, *, shot: int, time_ms: float, contour_points: int, sensor_points: int) -> TokamakProfile:
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


def load_diiid_reference_profiles(
    *,
    reference_dir: Path | None = None,
    contour_points: int = 64,
    sensor_points: int = 96,
) -> list[TokamakProfile]:
    ref_dir = reference_dir if reference_dir is not None else DEFAULT_DIIID_DIR
    files = sorted(Path(ref_dir).glob("*.geqdsk"))
    profiles: list[TokamakProfile] = []
    for idx, path in enumerate(files):
        shot = 163000 + idx
        time_ms = 850.0 + 65.0 * idx
        profiles.append(
            _profile_from_geqdsk(
                path,
                shot=shot,
                time_ms=time_ms,
                contour_points=contour_points,
                sensor_points=sensor_points,
            )
        )
    if not profiles:
        raise ValueError(f"No DIII-D reference files found in {ref_dir}.")
    return profiles


def _stable_shot_from_text(text: str) -> int:
    acc = 0
    for ch in text:
        acc = (acc * 131 + ord(ch)) % 900_000
    return 100_000 + acc


def _synthetic_cmod_psi_contour(kappa: float, delta: float, points: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, int(points), endpoint=False, dtype=np.float64)
    r = 0.55 + 0.20 * np.cos(theta + 0.35 * delta) + 0.08 * np.sin(2.0 * theta)
    z = 0.45 * kappa * np.sin(theta)
    psi = np.sqrt((r - np.mean(r)) ** 2 + (z - np.mean(z)) ** 2)
    psi_norm = (psi - np.min(psi)) / (np.max(psi) - np.min(psi) + 1e-12)
    return psi_norm.astype(np.float64)


def load_cmod_reference_profiles(
    *,
    itpa_csv_path: Path | None = None,
    contour_points: int = 64,
    sensor_points: int = 96,
) -> list[TokamakProfile]:
    csv_path = itpa_csv_path if itpa_csv_path is not None else DEFAULT_ITPA_CSV
    rows: list[TokamakProfile] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            machine = str(row.get("machine", "")).strip().lower()
            if machine not in {"c-mod", "alcator c-mod", "cmod"}:
                continue

            raw_shot = str(row.get("shot", "")).strip()
            try:
                shot = int(raw_shot)
            except Exception:
                shot = _stable_shot_from_text(raw_shot or "cmod")

            ip_ma = _coerce_finite("Ip_MA", float(row["Ip_MA"]), minimum=0.0)
            bt_t = _coerce_finite("BT_T", float(row["BT_T"]), minimum=0.0)
            tau_e_ms = _coerce_finite("tau_E_s", float(row["tau_E_s"]), minimum=0.0) * 1e3
            h98 = _coerce_finite("H98y2", float(row["H98y2"]), minimum=0.0)
            kappa = _coerce_finite("kappa", float(row["kappa"]), minimum=1.0)
            delta = _coerce_finite("delta", float(row["delta"]))

            beta_n = float(np.clip(0.85 + 0.42 * ip_ma + 0.01 * bt_t, 0.6, 4.5))
            q95 = float(np.clip(3.2 + 0.45 * ip_ma + 0.15 * (2.0 - h98), 2.5, 7.0))
            disruption = bool(h98 < 0.92)
            time_ms = float(640.0 + 120.0 * (len(rows) + 1))

            psi_contour = _synthetic_cmod_psi_contour(kappa, delta, contour_points)
            sensor_trace = _build_sensor_trace(psi_contour, points=sensor_points, seed=shot)

            rows.append(
                TokamakProfile(
                    machine="C-Mod",
                    shot=shot,
                    time_ms=time_ms,
                    beta_n=beta_n,
                    q95=q95,
                    tau_e_ms=float(tau_e_ms),
                    psi_contour=tuple(float(v) for v in psi_contour),
                    sensor_trace=tuple(float(v) for v in sensor_trace),
                    toroidal_n1_amp=float(np.clip(0.07 + 0.05 * beta_n, 0.04, 0.40)),
                    toroidal_n2_amp=float(np.clip(0.04 + 0.03 * beta_n, 0.02, 0.26)),
                    toroidal_n3_amp=float(np.clip(0.02 + 0.015 * beta_n, 0.01, 0.20)),
                    disruption=disruption,
                )
            )
    if not rows:
        raise ValueError(f"No C-Mod rows found in {csv_path}.")
    return rows


def _to_scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Empty MDSplus node payload.")
    flat = arr.reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        raise ValueError("No finite values in MDSplus payload.")
    return float(finite[-1])


def _to_trace(value: Any, points: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.repeat(arr.reshape(1), 8)
    elif arr.ndim > 1:
        arr = arr.reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        raise ValueError("Trace payload must contain at least 2 finite values.")
    return _resample_1d(arr, points)


def fetch_mdsplus_profiles(
    *,
    machine: str,
    host: str,
    tree: str,
    shots: list[int],
    node_map: Mapping[str, str] | None = None,
    contour_points: int = 64,
    sensor_points: int = 96,
    allow_partial: bool = True,
) -> list[TokamakProfile]:
    if not shots:
        raise ValueError("shots must be non-empty.")
    try:
        import MDSplus  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("MDSplus is not available in this environment.") from exc

    normalized_machine = _normalize_machine(machine)
    nodes = dict(DEFAULT_MDSPLUS_NODE_MAP)
    if node_map is not None:
        nodes.update(dict(node_map))

    out: list[TokamakProfile] = []
    conn = MDSplus.Connection(str(host))  # pragma: no cover - live integration path
    for shot in shots:
        try:
            shot_i = _coerce_int("shot", shot, minimum=0)
            conn.openTree(str(tree), shot_i)

            def _fetch(node_key: str) -> Any:
                node = nodes[node_key]
                payload = conn.get(node)
                return payload.data() if hasattr(payload, "data") else payload

            psi = _to_trace(_fetch("psi_contour"), contour_points)
            trace = _to_trace(_fetch("sensor_trace"), sensor_points)
            beta_n = _coerce_finite("beta_n", _to_scalar(_fetch("beta_n")), minimum=0.0)
            q95 = _coerce_finite("q95", _to_scalar(_fetch("q95")), minimum=0.0)
            tau_ms = _coerce_finite("tau_e_ms", _to_scalar(_fetch("tau_e_ms")), minimum=0.0)
            time_ms = _coerce_finite("time_ms", _to_scalar(_fetch("time_ms")), minimum=0.0)
            n1 = _coerce_finite("toroidal_n1_amp", _to_scalar(_fetch("toroidal_n1_amp")), minimum=0.0)
            n2 = _coerce_finite("toroidal_n2_amp", _to_scalar(_fetch("toroidal_n2_amp")), minimum=0.0)
            n3 = _coerce_finite("toroidal_n3_amp", _to_scalar(_fetch("toroidal_n3_amp")), minimum=0.0)
            disruption = bool(round(_to_scalar(_fetch("disruption"))))

            out.append(
                TokamakProfile(
                    machine=normalized_machine,
                    shot=shot_i,
                    time_ms=time_ms,
                    beta_n=beta_n,
                    q95=q95,
                    tau_e_ms=tau_ms,
                    psi_contour=tuple(float(v) for v in psi),
                    sensor_trace=tuple(float(v) for v in trace),
                    toroidal_n1_amp=n1,
                    toroidal_n2_amp=n2,
                    toroidal_n3_amp=n3,
                    disruption=disruption,
                )
            )
        except Exception:
            if not allow_partial:
                raise
            continue
    return out


def load_machine_profiles(
    *,
    machine: str,
    prefer_live: bool = False,
    host: str | None = None,
    tree: str | None = None,
    shots: list[int] | None = None,
    node_map: Mapping[str, str] | None = None,
    contour_points: int = 64,
    sensor_points: int = 96,
) -> tuple[list[TokamakProfile], dict[str, Any]]:
    normalized_machine = _normalize_machine(machine)
    if normalized_machine == "DIII-D":
        ref = load_diiid_reference_profiles(
            contour_points=contour_points,
            sensor_points=sensor_points,
        )
    else:
        ref = load_cmod_reference_profiles(
            contour_points=contour_points,
            sensor_points=sensor_points,
        )

    live_attempted = bool(prefer_live)
    live_error: str | None = None
    live_profiles: list[TokamakProfile] = []
    if prefer_live:
        if not host or not tree:
            live_error = "Missing host/tree for live MDSplus fetch."
        else:
            live_shots = shots if shots is not None else [p.shot for p in ref[:6]]
            try:
                live_profiles = fetch_mdsplus_profiles(
                    machine=normalized_machine,
                    host=host,
                    tree=tree,
                    shots=live_shots,
                    node_map=node_map,
                    contour_points=contour_points,
                    sensor_points=sensor_points,
                    allow_partial=True,
                )
            except Exception as exc:
                live_error = str(exc)

    if live_profiles:
        by_key: dict[tuple[int, int], TokamakProfile] = {
            (p.shot, int(round(p.time_ms))): p for p in ref
        }
        for p in live_profiles:
            by_key[(p.shot, int(round(p.time_ms)))] = p
        merged = list(by_key.values())
        source = "live+reference"
    else:
        merged = ref
        source = "reference"

    return merged, {
        "machine": normalized_machine,
        "source": source,
        "reference_count": len(ref),
        "live_count": len(live_profiles),
        "live_attempted": live_attempted,
        "live_error": live_error,
    }
