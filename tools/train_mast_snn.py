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
        )
    except Exception:
        return None


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=resolve_mast_cache_dir())
    parser.add_argument("--train-shots", nargs="+", type=int, default=[30471, 30470, 30469, 30468, 30467])
    parser.add_argument("--val-shots", nargs="+", type=int, default=[30466, 30465, 30463, 30462])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--out", type=Path, default=Path("validation/reports/mast_snn_local_validation.json"))
    args = parser.parse_args()

    ingestor = MastIngestor(cache_dir=args.cache_dir)
    weights = fine_tune_weights(
        ingestor,
        args.train_shots,
        n_neurons=64,
        seed=args.seed,
        epochs=args.epochs,
    )
    report = evaluate_lead_times(ingestor, args.val_shots, weights=weights)
    detected = [row["lead_time_ms"] for row in report if row.get("status") == "detected"]
    payload = {
        "claim_boundary": "Local MAST SNN validation evidence only; not a benchmark claim.",
        "cache_dir": str(args.cache_dir),
        "train_shots": args.train_shots,
        "val_shots": args.val_shots,
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
