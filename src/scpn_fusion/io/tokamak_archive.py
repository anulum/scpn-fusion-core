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
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.eqdsk import read_geqdsk


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIIID_DIR = REPO_ROOT / "validation" / "reference_data" / "diiid"
DEFAULT_DISRUPTION_DIR = DEFAULT_DIIID_DIR / "disruption_shots"
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


def _build_sensor_trace(psi_contour: NDArray[np.float64], points: int, seed: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(int(seed))
    grad = np.gradient(psi_contour)
    base = np.concatenate((psi_contour, grad), axis=0)
    base = _resample_1d(base, max(points, 16))
    t = np.linspace(0.0, 1.0, base.size, dtype=np.float64)
    mod = 0.08 * np.sin(2.0 * np.pi * 4.0 * t + 0.3)
    noise = rng.normal(0.0, 0.01, size=base.size)
    out = np.clip(base + mod + noise, 0.0, 2.5)
    return np.asarray(out, dtype=np.float64)


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


def _synthetic_cmod_psi_contour(kappa: float, delta: float, points: int) -> NDArray[np.float64]:
    theta = np.linspace(0.0, 2.0 * np.pi, int(points), endpoint=False, dtype=np.float64)
    r = 0.55 + 0.20 * np.cos(theta + 0.35 * delta) + 0.08 * np.sin(2.0 * theta)
    z = 0.45 * kappa * np.sin(theta)
    psi = np.sqrt((r - np.mean(r)) ** 2 + (z - np.mean(z)) ** 2)
    psi_norm = (psi - np.min(psi)) / (np.max(psi) - np.min(psi) + 1e-12)
    return np.asarray(psi_norm, dtype=np.float64)


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


def _to_trace(value: Any, points: int) -> NDArray[np.float64]:
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
        import MDSplus
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


def poll_mdsplus_feed(
    *,
    machine: str,
    host: str,
    tree: str,
    shots: list[int],
    polls: int = 3,
    poll_interval_ms: int = 100,
    node_map: Mapping[str, str] | None = None,
    contour_points: int = 64,
    sensor_points: int = 96,
    allow_partial: bool = True,
    fallback_to_reference: bool = True,
) -> tuple[list[TokamakProfile], dict[str, Any]]:
    """
    Poll live MDSplus feed snapshots with deterministic merge + fallback metadata.

    Notes:
    - This function performs immediate poll iterations without sleeping to keep
      CI/runtime deterministic. `poll_interval_ms` is recorded as intent.
    - If all live polls fail or are empty and `fallback_to_reference=True`,
      reference data is returned so validation lanes remain runnable.
    """
    polls_i = _coerce_int("polls", polls, minimum=1)
    interval_i = _coerce_int("poll_interval_ms", poll_interval_ms, minimum=1)
    normalized_machine = _normalize_machine(machine)
    merged: dict[tuple[str, int, int], TokamakProfile] = {}
    poll_records: list[dict[str, Any]] = []

    for poll_idx in range(polls_i):
        rec: dict[str, Any] = {
            "poll_index": int(poll_idx),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "live_count": 0,
            "error": None,
        }
        try:
            live = fetch_mdsplus_profiles(
                machine=normalized_machine,
                host=host,
                tree=tree,
                shots=shots,
                node_map=node_map,
                contour_points=contour_points,
                sensor_points=sensor_points,
                allow_partial=allow_partial,
            )
            rec["live_count"] = int(len(live))
            for row in live:
                merged[_profile_key(row)] = row
        except Exception as exc:
            rec["error"] = str(exc)
        poll_records.append(rec)

    fallback_meta: dict[str, Any] | None = None
    if not merged and fallback_to_reference:
        fallback_rows, fallback_meta = load_machine_profiles(
            machine=normalized_machine,
            prefer_live=False,
            contour_points=contour_points,
            sensor_points=sensor_points,
        )
        for row in fallback_rows:
            merged[_profile_key(row)] = row

    rows = list(merged.values())
    rows.sort(key=lambda r: (r.machine, int(r.shot), float(r.time_ms)))
    total_live = int(sum(int(r["live_count"]) for r in poll_records))
    source = "live_stream" if total_live > 0 else ("reference_fallback" if fallback_meta is not None else "empty")
    return rows, {
        "machine": normalized_machine,
        "host": str(host),
        "tree": str(tree),
        "shots_requested": [int(s) for s in shots],
        "polls": int(polls_i),
        "poll_interval_ms": int(interval_i),
        "live_total_profiles": total_live,
        "poll_records": poll_records,
        "fallback_meta": fallback_meta,
        "source": source,
    }


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


# ---------------------------------------------------------------------------
# Disruption shot NPZ loaders (synthetic DIII-D reference data)
# ---------------------------------------------------------------------------

def list_disruption_shots(
    *,
    disruption_dir: Path | None = None,
) -> list[str]:
    """
    Return the names of all available disruption shot NPZ files.

    Each name corresponds to a file in ``disruption_dir`` (default:
    ``validation/reference_data/diiid/disruption_shots/``).  The returned
    names are *without* the ``.npz`` extension and sorted alphabetically.

    Example::

        >>> names = list_disruption_shots()
        >>> names
        ['shot_154406_hybrid', 'shot_155916_locked_mode', ...]
    """
    d = disruption_dir if disruption_dir is not None else DEFAULT_DISRUPTION_DIR
    d = Path(d)
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.npz"))


def load_disruption_shot(
    shot_name_or_path: str | Path,
    *,
    disruption_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Load a single disruption shot NPZ file and return its contents as a dict.

    Parameters
    ----------
    shot_name_or_path
        Either a bare shot name (e.g. ``"shot_155916_locked_mode"``) which
        is resolved relative to *disruption_dir*, or an absolute/relative
        path to a ``.npz`` file.
    disruption_dir
        Directory containing NPZ files.  Defaults to
        ``validation/reference_data/diiid/disruption_shots/``.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys: ``time_s``, ``Ip_MA``, ``BT_T``, ``beta_N``,
        ``q95``, ``ne_1e19``, ``n1_amp``, ``n2_amp``, ``locked_mode_amp``,
        ``dBdt_gauss_per_s``, ``vertical_position_m`` (all ``NDArray[float64]``
        of shape ``(1000,)``), plus scalar metadata ``is_disruption`` (bool),
        ``disruption_time_idx`` (int), and ``disruption_type`` (str).

    Raises
    ------
    FileNotFoundError
        If the NPZ file does not exist.
    ValueError
        If the file is missing required keys.
    """
    p = Path(shot_name_or_path)
    if not p.suffix:
        d = disruption_dir if disruption_dir is not None else DEFAULT_DISRUPTION_DIR
        p = Path(d) / f"{p.name}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Disruption shot file not found: {p}")

    raw = np.load(str(p), allow_pickle=True)

    required_array_keys = {
        "time_s", "Ip_MA", "BT_T", "beta_N", "q95", "ne_1e19",
        "n1_amp", "n2_amp", "locked_mode_amp", "dBdt_gauss_per_s",
        "vertical_position_m",
    }
    required_scalar_keys = {"is_disruption", "disruption_time_idx", "disruption_type"}
    all_required = required_array_keys | required_scalar_keys
    present = set(raw.files)
    missing = all_required - present
    if missing:
        raise ValueError(f"NPZ file {p.name} missing keys: {sorted(missing)}")

    result: dict[str, Any] = {}
    for k in required_array_keys:
        result[k] = np.asarray(raw[k], dtype=np.float64)
    result["is_disruption"] = bool(raw["is_disruption"])
    result["disruption_time_idx"] = int(raw["disruption_time_idx"])
    result["disruption_type"] = str(raw["disruption_type"])
    return result


# ---------------------------------------------------------------------------
# Synthetic multi-machine shot database (ITER / SPARC / DIII-D / EAST)
# ---------------------------------------------------------------------------

DEFAULT_SYNTHETIC_DIR = REPO_ROOT / "validation" / "reference_data" / "synthetic_shots"

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
        peak * ((1.0 - rho ** 2) ** alpha * (1.0 - edge_fraction) + edge_fraction),
        dtype=np.float64,
    )


def generate_synthetic_shot_database(
    *,
    n_shots: int = 60,
    output_dir: Path | None = None,
    seed: int = 20260218,
) -> list[dict[str, Any]]:
    """
    Generate synthetic-realistic tokamak shot profiles for ITER/SPARC/DIII-D/EAST.

    Creates NPZ files with physically motivated profile data covering four
    machine parameter ranges.  Approximately 20 %% of shots are flagged as
    disruptions (elevated beta_N, lowered q95).

    Parameters
    ----------
    n_shots
        Minimum total number of shots to generate (default 60).  The actual
        count may be slightly higher to satisfy per-machine minimums.
    output_dir
        Directory to write NPZ files into.  Defaults to
        ``validation/reference_data/synthetic_shots/``.
    seed
        Random seed for reproducibility.

    Returns
    -------
    list[dict[str, Any]]
        One metadata dict per generated shot with keys: ``file``, ``machine``,
        ``shot_id``, ``disruption_label``, ``Ip_MA``, ``BT_T``, ``ne0_1e19``,
        ``Te0_keV``, ``q95``, ``beta_N``.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(output_dir) if output_dir is not None else DEFAULT_SYNTHETIC_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Distribute shots across machines, respecting spec defaults but scaling
    # up proportionally if the caller requests more than sum-of-defaults.
    spec_total = sum(s["n_shots"] for s in _MACHINE_SPECS.values())
    scale = max(1.0, n_shots / spec_total)
    machine_counts: dict[str, int] = {
        m: max(int(round(s["n_shots"] * scale)), s["n_shots"])
        for m, s in _MACHINE_SPECS.items()
    }

    rho = np.linspace(0.0, 1.0, 1000, dtype=np.float64)
    time_s = np.linspace(0.0, 10.0, 1000, dtype=np.float64)

    catalogue: list[dict[str, Any]] = []
    shot_counter = 0

    for machine, spec in _MACHINE_SPECS.items():
        count = machine_counts[machine]
        for idx in range(count):
            shot_counter += 1
            is_disruption = bool(rng.random() < 0.20)

            # Sample scalar parameters uniformly within machine range.
            ip = float(rng.uniform(*spec["Ip_MA"]))
            bt = float(rng.uniform(*spec["BT_T"]))
            ne0 = float(rng.uniform(*spec["ne_1e19"]))
            te0 = float(rng.uniform(*spec["Te_keV"]))
            ti_ratio = float(rng.uniform(*spec["Ti_keV_ratio"]))
            ti0 = te0 * ti_ratio
            q95_val = float(rng.uniform(*spec["q95"]))
            beta_n = float(rng.uniform(*spec["beta_N"]))

            # Disruption shots: push beta_N up, q95 down.
            if is_disruption:
                beta_n = float(np.clip(beta_n * 1.4, spec["beta_N"][0], spec["beta_N"][1] * 1.5))
                q95_val = float(np.clip(q95_val * 0.75, 1.5, spec["q95"][1]))

            # Build radial profiles on rho grid (1000 pts).
            ne_profile = _parabolic_profile(rho, ne0, edge_fraction=0.12, alpha=0.5)
            te_profile = _parabolic_profile(rho, te0, edge_fraction=0.08, alpha=0.6)
            ti_profile = _parabolic_profile(rho, ti0, edge_fraction=0.08, alpha=0.55)

            # Time-dependent scalar traces (1000 pts).
            ip_trace = np.full(1000, ip, dtype=np.float64)
            bt_trace = np.full(1000, bt, dtype=np.float64)
            q95_trace = np.full(1000, q95_val, dtype=np.float64)
            beta_n_trace = np.full(1000, beta_n, dtype=np.float64)

            # Add small temporal noise.
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
            catalogue.append({
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
            })

    return catalogue


def list_synthetic_shots(
    *,
    synthetic_dir: Path | None = None,
) -> list[str]:
    """
    Return names of all available synthetic shot NPZ files (without extension).

    Parameters
    ----------
    synthetic_dir
        Directory containing NPZ files.  Defaults to
        ``validation/reference_data/synthetic_shots/``.

    Returns
    -------
    list[str]
        Sorted list of shot names (stem only).
    """
    d = Path(synthetic_dir) if synthetic_dir is not None else DEFAULT_SYNTHETIC_DIR
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.npz"))


def load_synthetic_shot(
    shot_name_or_path: str | Path,
    *,
    synthetic_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Load a single synthetic shot NPZ file and return its contents as a dict.

    Parameters
    ----------
    shot_name_or_path
        Either a bare shot name (e.g. ``"shot_ITER_0001"``) which is resolved
        relative to *synthetic_dir*, or an absolute/relative path to a ``.npz``
        file.
    synthetic_dir
        Directory containing NPZ files.  Defaults to
        ``validation/reference_data/synthetic_shots/``.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys: ``time_s``, ``Ip_MA``, ``BT_T``, ``ne_1e19``,
        ``Te_keV``, ``Ti_keV``, ``q95``, ``beta_N`` (all ``NDArray[float64]``
        of shape ``(1000,)``), plus ``disruption_label`` (bool) and
        ``machine`` (str).

    Raises
    ------
    FileNotFoundError
        If the NPZ file does not exist.
    ValueError
        If the file is missing required keys.
    """
    p = Path(shot_name_or_path)
    if not p.suffix:
        d = Path(synthetic_dir) if synthetic_dir is not None else DEFAULT_SYNTHETIC_DIR
        p = d / f"{p.name}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Synthetic shot file not found: {p}")

    raw = np.load(str(p), allow_pickle=True)

    required_array_keys = {
        "time_s", "Ip_MA", "BT_T", "ne_1e19", "Te_keV", "Ti_keV",
        "q95", "beta_N",
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
    result["disruption_label"] = bool(raw["disruption_label"])
    result["machine"] = str(raw["machine"])
    return result
