# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — MAST SNN Validation Tool

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.io.mast_ingestor import MastIngestor, default_mast_cache_dir


@dataclass(frozen=True)
class ShotTrace:
    """Time-aligned MAST summary and magnetic trace."""

    time_s: NDArray[np.float64]
    plasma_current_a: NDArray[np.float64]
    magnetic_trace_t: NDArray[np.float64]
    disruption_time_s: float
    source: str


class HardwareSNN:
    """Small deterministic leaky-integrate-and-fire detector."""

    def __init__(
        self,
        n_neurons: int = 64,
        dt_s: float = 1e-4,
        threshold: float = 0.3,
        weights: NDArray[np.float64] | None = None,
    ) -> None:
        if n_neurons <= 0:
            raise ValueError("n_neurons must be positive")
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")

        self.n_neurons = n_neurons
        self.alpha = dt_s / 0.05e-3
        self.threshold = threshold
        self.weights = (
            np.asarray(weights, dtype=np.float64)
            if weights is not None
            else np.ones(n_neurons, dtype=np.float64)
        )
        if self.weights.shape != (n_neurons,):
            raise ValueError("weights must have shape (n_neurons,)")
        self.v = np.zeros(n_neurons, dtype=np.float64)

    def reset(self) -> None:
        self.v = np.zeros(self.n_neurons, dtype=np.float64)

    def step(self, signal: float) -> float:
        current = signal * self.weights * 10.0 + 0.01
        self.v += self.alpha * (-self.v + current)
        spikes = self.v >= self.threshold
        self.v[spikes] = 0.0
        return float(np.sum(spikes) / self.n_neurons)


def resolve_mast_cache_dir() -> Path:
    """Resolve the MAST cache directory from environment or repo defaults."""
    override = os.environ.get("SCPN_MAST_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return default_mast_cache_dir()


def initialise_base_weights(n_neurons: int = 64, seed: int = 1729) -> NDArray[np.float64]:
    """Create deterministic base sensitivities for local validation runs."""
    if n_neurons <= 0:
        raise ValueError("n_neurons must be positive")
    rng = np.random.default_rng(seed)
    return rng.normal(5.0, 0.5, n_neurons).astype(np.float64)


def load_shot(ingestor: MastIngestor, shot_id: int) -> ShotTrace | None:
    """Load and align the summary current and first available magnetic trace."""
    try:
        summary = ingestor.load_shot_summary(shot_id)
        magnetics = ingestor.load_magnetic_probes(shot_id)

        time_s = np.asarray(summary["time"], dtype=np.float64)
        plasma_current_a = np.asarray(summary["ip"], dtype=np.float64)
        max_ip = float(np.max(plasma_current_a))

        flattop = np.where(plasma_current_a > 0.8 * max_ip)[0]
        if len(flattop) == 0:
            return None

        disruption_idx = int(flattop[-1])
        for idx in range(disruption_idx, len(plasma_current_a)):
            if plasma_current_a[idx] < 0.2 * max_ip:
                disruption_idx = idx
                break

        magnetic_keys = [key for key in magnetics if key != "time"]
        if not magnetic_keys:
            return None
        magnetic_trace_t = np.asarray(magnetics[magnetic_keys[0]], dtype=np.float64).flatten()
        if len(magnetic_trace_t) != len(time_s):
            sample_idx = np.linspace(0, len(magnetic_trace_t) - 1, len(time_s)).astype(int)
            magnetic_trace_t = magnetic_trace_t[sample_idx]

        return ShotTrace(
            time_s=time_s,
            plasma_current_a=plasma_current_a,
            magnetic_trace_t=magnetic_trace_t,
            disruption_time_s=float(time_s[disruption_idx]),
            source="fair_mast_zarr",
        )
    except Exception:
        return None


def _resample_to_summary_time(trace: NDArray[np.float64], n_time: int) -> NDArray[np.float64]:
    """Resample a magnetic trace to the summary-current time base."""
    flat = np.asarray(trace, dtype=np.float64).reshape(-1)
    if len(flat) == n_time:
        return np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    if len(flat) == 0:
        return np.zeros(n_time, dtype=np.float64)
    sample_idx = np.linspace(0, len(flat) - 1, n_time).astype(int)
    return np.nan_to_num(flat[sample_idx], nan=0.0, posinf=0.0, neginf=0.0)


def load_local_npz_shot(cache_dir: Path, shot_id: int) -> ShotTrace | None:
    """Load a locally materialised MAST NPZ shot artifact when available."""
    path = cache_dir / f"mast_shot_{shot_id}.npz"
    if not path.exists():
        return None

    try:
        data = np.load(path, allow_pickle=False)
        time_s = np.asarray(data["time"], dtype=np.float64)
        plasma_current_a = np.asarray(data["ip"], dtype=np.float64)
        if time_s.ndim != 1 or plasma_current_a.shape != time_s.shape:
            return None

        max_ip = float(np.max(np.abs(plasma_current_a)))
        if max_ip <= 0.0:
            return None
        flattop = np.where(np.abs(plasma_current_a) > 0.8 * max_ip)[0]
        if len(flattop) == 0:
            return None

        disruption_idx = int(flattop[-1])
        for idx in range(disruption_idx, len(plasma_current_a)):
            if abs(float(plasma_current_a[idx])) < 0.2 * max_ip:
                disruption_idx = idx
                break

        magnetic_keys = [
            key for key in data.files if key.startswith("mag_") and key.endswith("_field")
        ]
        if not magnetic_keys:
            return None
        magnetic_trace_t = _resample_to_summary_time(
            np.asarray(data[magnetic_keys[0]], dtype=np.float64), len(time_s)
        )

        return ShotTrace(
            time_s=time_s,
            plasma_current_a=plasma_current_a,
            magnetic_trace_t=magnetic_trace_t,
            disruption_time_s=float(time_s[disruption_idx]),
            source=f"local_npz:{path.name}",
        )
    except Exception:
        return None


def load_shot_from_sources(
    cache_dir: Path,
    shot_id: int,
    *,
    ingestor: MastIngestor | None,
) -> ShotTrace | None:
    """Load a shot from local NPZ evidence first, then FAIR MAST when available."""
    local = load_local_npz_shot(cache_dir, shot_id)
    if local is not None:
        return local
    if ingestor is None:
        return None
    return load_shot(ingestor, shot_id)


def fine_tune_weights(
    ingestor: MastIngestor,
    train_shots: list[int],
    *,
    n_neurons: int,
    seed: int,
    epochs: int,
) -> NDArray[np.float64]:
    """Run deterministic sensitivity adaptation over available training shots."""
    weights = initialise_base_weights(n_neurons=n_neurons, seed=seed)
    for _epoch in range(epochs):
        for shot_id in train_shots:
            if load_shot(ingestor, shot_id) is not None:
                weights *= 1.02
    return weights


def adapt_weights_from_sources(
    cache_dir: Path,
    train_shots: list[int],
    *,
    ingestor: MastIngestor | None,
    n_neurons: int,
    seed: int,
    epochs: int,
) -> tuple[NDArray[np.float64], int]:
    """Adapt deterministic sensitivities only over shots that can be loaded."""
    weights = initialise_base_weights(n_neurons=n_neurons, seed=seed)
    available = [
        shot_id
        for shot_id in train_shots
        if load_shot_from_sources(cache_dir, shot_id, ingestor=ingestor) is not None
    ]
    for _epoch in range(epochs):
        for _shot_id in available:
            weights *= 1.02
    return weights, len(available)


def evaluate_lead_times(
    ingestor: MastIngestor,
    val_shots: list[int],
    *,
    weights: NDArray[np.float64],
) -> list[dict[str, float | int | str]]:
    """Evaluate local lead-time evidence for the supplied validation shots."""
    detector = HardwareSNN(n_neurons=len(weights), weights=weights)
    report: list[dict[str, float | int | str]] = []

    for shot_id in val_shots:
        trace = load_shot(ingestor, shot_id)
        if trace is None:
            report.append({"shot_id": shot_id, "status": "unavailable"})
            continue

        detector.reset()
        dt_s = float(np.mean(np.diff(trace.time_s)))
        magnetic_scale_t = max(float(np.max(np.abs(trace.magnetic_trace_t))), 1.0)
        flattop_start = int(
            np.where(trace.plasma_current_a > 0.8 * np.max(trace.plasma_current_a))[0][0]
        )
        alarm_time_s: float | None = None

        for idx in range(1, len(trace.time_s)):
            if trace.time_s[idx] >= trace.disruption_time_s:
                break
            db_dt = (trace.magnetic_trace_t[idx] - trace.magnetic_trace_t[idx - 1]) / dt_s
            score = detector.step(abs(db_dt / magnetic_scale_t))
            if score > 0.85 and idx > flattop_start:
                alarm_time_s = float(trace.time_s[idx])
                break

        if alarm_time_s is None:
            report.append({"shot_id": shot_id, "status": "no_detection"})
            continue

        report.append(
            {
                "shot_id": shot_id,
                "status": "detected",
                "disruption_time_s": trace.disruption_time_s,
                "alarm_time_s": alarm_time_s,
                "lead_time_ms": (trace.disruption_time_s - alarm_time_s) * 1000.0,
            }
        )

    return report


def evaluate_lead_times_from_sources(
    cache_dir: Path,
    val_shots: list[int],
    *,
    ingestor: MastIngestor | None,
    weights: NDArray[np.float64],
) -> list[dict[str, float | int | str]]:
    """Evaluate lead-time evidence from local NPZ and FAIR MAST sources."""
    detector = HardwareSNN(n_neurons=len(weights), weights=weights)
    report: list[dict[str, float | int | str]] = []

    for shot_id in val_shots:
        trace = load_shot_from_sources(cache_dir, shot_id, ingestor=ingestor)
        if trace is None:
            report.append({"shot_id": shot_id, "status": "unavailable"})
            continue

        detector.reset()
        dt_s = float(np.mean(np.diff(trace.time_s)))
        magnetic_scale_t = max(float(np.max(np.abs(trace.magnetic_trace_t))), 1.0)
        flattop_candidates = np.where(
            np.abs(trace.plasma_current_a) > 0.8 * np.max(np.abs(trace.plasma_current_a))
        )[0]
        if len(flattop_candidates) == 0:
            report.append({"shot_id": shot_id, "status": "no_flattop", "source": trace.source})
            continue
        flattop_start = int(flattop_candidates[0])
        alarm_time_s: float | None = None

        for idx in range(1, len(trace.time_s)):
            if trace.time_s[idx] >= trace.disruption_time_s:
                break
            db_dt = (trace.magnetic_trace_t[idx] - trace.magnetic_trace_t[idx - 1]) / dt_s
            score = detector.step(abs(db_dt / magnetic_scale_t))
            if score > 0.85 and idx > flattop_start:
                alarm_time_s = float(trace.time_s[idx])
                break

        if alarm_time_s is None:
            report.append({"shot_id": shot_id, "status": "no_detection", "source": trace.source})
            continue

        report.append(
            {
                "shot_id": shot_id,
                "status": "detected",
                "source": trace.source,
                "disruption_time_s": trace.disruption_time_s,
                "alarm_time_s": alarm_time_s,
                "lead_time_ms": (trace.disruption_time_s - alarm_time_s) * 1000.0,
            }
        )

    return report


def classify_full_fidelity_status(
    *,
    train_available_count: int,
    validation_report: list[dict[str, float | int | str]],
    min_train_shots: int,
    min_validation_shots: int,
) -> str:
    """Classify the MAST SNN report without accepting local-only evidence."""
    detected = [row for row in validation_report if row.get("status") == "detected"]
    if train_available_count < min_train_shots:
        return "blocked_insufficient_training_shots"
    if len(detected) < min_validation_shots:
        return "blocked_insufficient_detected_validation_shots"
    return "blocked_local_mast_snn_not_physics_validation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=resolve_mast_cache_dir())
    parser.add_argument(
        "--train-shots", nargs="+", type=int, default=[30471, 30470, 30469, 30468, 30467]
    )
    parser.add_argument("--val-shots", nargs="+", type=int, default=[30466, 30465, 30463, 30462])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--min-train-shots", type=int, default=3)
    parser.add_argument("--min-validation-shots", type=int, default=3)
    parser.add_argument(
        "--enable-fair-mast",
        action="store_true",
        help="Enable FAIR MAST S3/Zarr fallback when local NPZ shots are absent.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("validation/reports/mast_snn_local_validation.json")
    )
    args = parser.parse_args()

    ingestor: MastIngestor | None = None
    if args.enable_fair_mast:
        try:
            ingestor = MastIngestor(cache_dir=args.cache_dir)
        except ImportError:
            ingestor = None

    weights, train_available_count = adapt_weights_from_sources(
        args.cache_dir,
        args.train_shots,
        ingestor=ingestor,
        n_neurons=64,
        seed=args.seed,
        epochs=args.epochs,
    )
    report = evaluate_lead_times_from_sources(
        args.cache_dir,
        args.val_shots,
        ingestor=ingestor,
        weights=weights,
    )
    detected = [row["lead_time_ms"] for row in report if row.get("status") == "detected"]
    status = classify_full_fidelity_status(
        train_available_count=train_available_count,
        validation_report=report,
        min_train_shots=args.min_train_shots,
        min_validation_shots=args.min_validation_shots,
    )
    payload = {
        "status": status,
        "accepted_full_fidelity_ready": False,
        "claim_boundary": (
            "MAST SNN reports are local disruption-detection diagnostics only. "
            "Full-fidelity claims require same-case magnetic-geometry validation, "
            "shot provenance review, and independent acceptance gates beyond local "
            "train and detected-validation counts."
        ),
        "cache_dir": str(args.cache_dir),
        "fair_mast_enabled": bool(args.enable_fair_mast and ingestor is not None),
        "train_shots": args.train_shots,
        "val_shots": args.val_shots,
        "train_available_count": train_available_count,
        "detected_validation_count": len(detected),
        "local_count_gate_passed": (
            train_available_count >= args.min_train_shots
            and len(detected) >= args.min_validation_shots
        ),
        "epochs": args.epochs,
        "seed": args.seed,
        "average_lead_time_ms": float(np.mean(detected)) if detected else None,
        "shots": report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote local MAST validation report to {args.out}")


if __name__ == "__main__":
    main()
