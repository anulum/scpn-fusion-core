#!/usr/bin/env python
"""Generate/check disruption-risk calibration config and holdout report."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scpn_fusion.control.disruption_predictor import (  # noqa: E402
    DEFAULT_DISRUPTION_RISK_BIAS,
    DEFAULT_DISRUPTION_RISK_THRESHOLD,
    DISRUPTION_RISK_LINEAR_WEIGHTS,
    predict_disruption_risk,
)


DEFAULT_SHOT_DIR = REPO_ROOT / "validation" / "reference_data" / "diiid" / "disruption_shots"
DEFAULT_MANIFEST = (
    REPO_ROOT / "validation" / "reference_data" / "diiid" / "disruption_shots_manifest.json"
)
DEFAULT_SPLITS = (
    REPO_ROOT / "validation" / "reference_data" / "diiid" / "disruption_shot_splits.json"
)
DEFAULT_CALIBRATION = (
    REPO_ROOT / "validation" / "reference_data" / "diiid" / "disruption_risk_calibration.json"
)
DEFAULT_REPORT_MD = REPO_ROOT / "validation" / "reports" / "disruption_risk_holdout_report.md"
VALIDATE_REAL_SHOTS_PATH = REPO_ROOT / "validation" / "validate_real_shots.py"
_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_SPLIT_IDS_PER_SET = 200_000
_MAX_MANIFEST_SHOTS = 50_000
_MAX_SIGNAL_SAMPLES_PER_SHOT = 200_000


def _load_json_text(path: Path) -> str:
    size = int(path.stat().st_size)
    if size > _MAX_JSON_BYTES:
        raise ValueError(
            f"{path} exceeds max JSON size "
            f"({_MAX_JSON_BYTES} bytes)."
        )
    return path.read_text(encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(_load_json_text(path))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a top-level object.")
    return data


def _resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _parse_split_ids(name: str, value: Any) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Split '{name}' must be a non-empty list of integer shot ids.")
    if len(value) > _MAX_SPLIT_IDS_PER_SET:
        raise ValueError(
            f"Split '{name}' has {len(value)} ids, exceeding max "
            f"{_MAX_SPLIT_IDS_PER_SET}."
        )
    out: list[int] = []
    for i, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"Split '{name}[{i}]' must be a positive integer shot id.")
        out.append(int(item))
    if len(out) != len(set(out)):
        raise ValueError(f"Split '{name}' contains duplicate shot ids.")
    return out


def _load_split_map(splits_path: Path) -> dict[int, str]:
    split_data = _load_json(splits_path)
    train = _parse_split_ids("train", split_data.get("train"))
    val = _parse_split_ids("val", split_data.get("val"))
    test = _parse_split_ids("test", split_data.get("test"))

    overlap = (set(train) & set(val)) | (set(train) & set(test)) | (set(val) & set(test))
    if overlap:
        raise ValueError(f"Split overlap detected: {sorted(overlap)}")

    split_map: dict[int, str] = {}
    for shot in train:
        split_map[shot] = "train"
    for shot in val:
        split_map[shot] = "val"
    for shot in test:
        split_map[shot] = "test"
    return split_map


def _load_payload_loader() -> Any:
    spec = importlib.util.spec_from_file_location(
        "validate_real_shots_calibration_loader",
        VALIDATE_REAL_SHOTS_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {VALIDATE_REAL_SHOTS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader = getattr(module, "load_disruption_shot_payload", None)
    if loader is None:
        raise RuntimeError("validate_real_shots.py is missing load_disruption_shot_payload().")
    return loader


def _load_samples(
    *,
    shot_dir: Path,
    manifest_path: Path,
    split_map: dict[int, str],
    window_size: int,
) -> list[dict[str, Any]]:
    manifest = _load_json(manifest_path)
    items = manifest.get("shots")
    if not isinstance(items, list) or not items:
        raise ValueError("Manifest must contain non-empty 'shots' list.")
    if len(items) > _MAX_MANIFEST_SHOTS:
        raise ValueError(
            f"Manifest includes {len(items)} shots, exceeding max "
            f"{_MAX_MANIFEST_SHOTS}."
        )

    loader = _load_payload_loader()
    samples: list[dict[str, Any]] = []
    seen_files: set[str] = set()
    for item in sorted(items, key=lambda x: str(x.get("file", ""))):
        if not isinstance(item, dict):
            raise ValueError("Manifest shot entries must be objects.")
        file_name = item.get("file")
        shot = item.get("shot")
        if not isinstance(file_name, str) or not file_name.endswith(".npz"):
            raise ValueError("Manifest shot entry missing valid 'file'.")
        if isinstance(shot, bool) or not isinstance(shot, int) or shot <= 0:
            raise ValueError(f"Manifest file '{file_name}' has invalid positive integer shot id.")
        if file_name in seen_files:
            raise ValueError(f"Manifest includes duplicate file entry: {file_name}")
        seen_files.add(file_name)
        if shot not in split_map:
            raise ValueError(f"Shot {shot} from manifest is missing in split definitions.")

        npz_path = shot_dir / file_name
        if not npz_path.exists():
            raise FileNotFoundError(f"Shot file missing from shot directory: {npz_path}")

        payload = loader(npz_path)
        signal = np.asarray(payload["signal"], dtype=np.float64)
        if signal.size > _MAX_SIGNAL_SAMPLES_PER_SHOT:
            raise ValueError(
                f"{file_name}: signal length {signal.size} exceeds max "
                f"{_MAX_SIGNAL_SAMPLES_PER_SHOT}."
            )
        n1_amp = np.asarray(payload["n1_amp"], dtype=np.float64)
        n2_amp = payload["n2_amp"]

        start_idx = min(max(window_size, 2), signal.size - 1)
        logits: list[float] = []
        for t in range(start_idx, signal.size):
            window = signal[t - start_idx : t]
            toroidal = {
                "toroidal_n1_amp": float(n1_amp[t]),
                "toroidal_n2_amp": float(n2_amp[t]) if n2_amp is not None else 0.05,
                "toroidal_n3_amp": 0.02,
            }
            risk = float(predict_disruption_risk(window, toroidal))
            risk = float(np.clip(risk, 1e-9, 1.0 - 1e-9))
            logits.append(float(np.log(risk / (1.0 - risk))))
        if not logits:
            raise ValueError(f"{file_name}: no sliding-window logits generated.")

        samples.append(
            {
                "file": file_name,
                "shot": int(shot),
                "split": split_map[int(shot)],
                "is_disruption": bool(payload["is_disruption"]),
                "disruption_time_idx": int(payload["disruption_time_idx"]),
                "base_logits": np.asarray(logits, dtype=np.float64),
            }
        )
    return samples


def _evaluate_subset(
    samples: list[dict[str, Any]],
    *,
    risk_threshold: float,
    bias_delta: float,
) -> dict[str, Any]:
    if not np.isfinite(risk_threshold) or not (0.0 < risk_threshold < 1.0):
        raise ValueError("risk_threshold must be finite in (0, 1).")
    if not np.isfinite(bias_delta):
        raise ValueError("bias_delta must be finite.")

    logit_threshold = float(np.log(risk_threshold / (1.0 - risk_threshold)))
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for sample in samples:
        detected = bool(np.any(sample["base_logits"] + bias_delta > logit_threshold))
        is_disruption = bool(sample["is_disruption"])
        disruption_idx = int(sample["disruption_time_idx"])
        if is_disruption and disruption_idx > 0:
            if detected:
                tp += 1
            else:
                fn += 1
        elif not is_disruption:
            if detected:
                fp += 1
            else:
                tn += 1

    n_disruptions = tp + fn
    n_safe = tn + fp
    recall = tp / max(n_disruptions, 1)
    fpr = fp / max(n_safe, 1)
    return {
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "n_disruptions": n_disruptions,
        "n_safe": n_safe,
        "recall": round(float(recall), 4),
        "false_positive_rate": round(float(fpr), 4),
    }


def _select_calibration(
    *,
    train_val_samples: list[dict[str, Any]],
    target_recall: float,
    target_fpr: float,
    threshold_values: np.ndarray,
    bias_values: np.ndarray,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for bias_delta in bias_values.tolist():
        for threshold in threshold_values.tolist():
            metrics = _evaluate_subset(
                train_val_samples,
                risk_threshold=float(threshold),
                bias_delta=float(bias_delta),
            )
            candidates.append(
                {
                    "risk_threshold": round(float(threshold), 4),
                    "bias_delta": round(float(bias_delta), 4),
                    "train_val": metrics,
                    "train_val_pass": bool(
                        metrics["recall"] >= target_recall
                        and metrics["false_positive_rate"] <= target_fpr
                    ),
                }
            )
    if not candidates:
        raise ValueError("No calibration candidates generated.")

    feasible = [c for c in candidates if c["train_val_pass"]]
    pool = feasible if feasible else candidates
    selection_mode = "feasible" if feasible else "pareto_fallback"
    pool.sort(
        key=lambda c: (
            -c["train_val"]["recall"],
            c["train_val"]["false_positive_rate"],
            abs(c["bias_delta"]),
            abs(c["risk_threshold"] - DEFAULT_DISRUPTION_RISK_THRESHOLD),
            -c["risk_threshold"],
        )
    )
    selected = dict(pool[0])
    selected["selection_mode"] = selection_mode
    selected["candidate_count"] = len(candidates)
    selected["feasible_count"] = len(feasible)
    return selected


def _summarize_split(samples: list[dict[str, Any]]) -> dict[str, int]:
    n_disruptive = sum(1 for s in samples if bool(s["is_disruption"]))
    n_safe = len(samples) - n_disruptive
    return {
        "files": len(samples),
        "disruptive_files": n_disruptive,
        "safe_files": n_safe,
    }


def _generate(
    *,
    shot_dir: Path,
    manifest_path: Path,
    splits_path: Path,
    target_recall: float,
    target_fpr: float,
    threshold_values: np.ndarray,
    bias_values: np.ndarray,
    window_size: int,
) -> dict[str, Any]:
    split_map = _load_split_map(splits_path)
    samples = _load_samples(
        shot_dir=shot_dir,
        manifest_path=manifest_path,
        split_map=split_map,
        window_size=window_size,
    )
    train_val_samples = [s for s in samples if s["split"] in {"train", "val"}]
    holdout_samples = [s for s in samples if s["split"] == "test"]
    if not train_val_samples:
        raise ValueError("No train/val samples found for calibration.")
    if not holdout_samples:
        raise ValueError("No holdout test samples found for calibration.")

    selected = _select_calibration(
        train_val_samples=train_val_samples,
        target_recall=target_recall,
        target_fpr=target_fpr,
        threshold_values=threshold_values,
        bias_values=bias_values,
    )
    holdout_metrics = _evaluate_subset(
        holdout_samples,
        risk_threshold=float(selected["risk_threshold"]),
        bias_delta=float(selected["bias_delta"]),
    )
    baseline_train_val = _evaluate_subset(
        train_val_samples,
        risk_threshold=DEFAULT_DISRUPTION_RISK_THRESHOLD,
        bias_delta=0.0,
    )
    baseline_holdout = _evaluate_subset(
        holdout_samples,
        risk_threshold=DEFAULT_DISRUPTION_RISK_THRESHOLD,
        bias_delta=0.0,
    )
    holdout_pass = bool(
        holdout_metrics["recall"] >= target_recall
        and holdout_metrics["false_positive_rate"] <= target_fpr
    )
    train_val_pass = bool(selected["train_val_pass"])

    output = {
        "version": "diiid-disruption-risk-calibration-v1",
        "sources": {
            "shot_dir": _display_path(shot_dir),
            "manifest": _display_path(manifest_path),
            "splits": _display_path(splits_path),
            "predictor_module": "src/scpn_fusion/control/disruption_predictor.py",
            "ingest_contract_module": "validation/validate_real_shots.py",
        },
        "targets": {
            "recall_min": round(float(target_recall), 4),
            "false_positive_rate_max": round(float(target_fpr), 4),
        },
        "grid": {
            "risk_threshold_min": round(float(threshold_values.min()), 4),
            "risk_threshold_max": round(float(threshold_values.max()), 4),
            "risk_threshold_step": round(float(threshold_values[1] - threshold_values[0]), 4)
            if len(threshold_values) > 1
            else 0.0,
            "bias_delta_min": round(float(bias_values.min()), 4),
            "bias_delta_max": round(float(bias_values.max()), 4),
            "bias_delta_step": round(float(bias_values[1] - bias_values[0]), 4)
            if len(bias_values) > 1
            else 0.0,
            "window_size": int(window_size),
        },
        "dataset": {
            "all": _summarize_split(samples),
            "train_val": _summarize_split(train_val_samples),
            "holdout_test": _summarize_split(holdout_samples),
        },
        "model": {
            "base_bias": round(float(DEFAULT_DISRUPTION_RISK_BIAS), 4),
            "linear_weights": {
                key: round(float(value), 6)
                for key, value in sorted(DISRUPTION_RISK_LINEAR_WEIGHTS.items())
            },
            "baseline_risk_threshold": round(float(DEFAULT_DISRUPTION_RISK_THRESHOLD), 4),
        },
        "selection": {
            "mode": selected["selection_mode"],
            "candidate_count": int(selected["candidate_count"]),
            "feasible_count": int(selected["feasible_count"]),
            "risk_threshold": round(float(selected["risk_threshold"]), 4),
            "bias_delta": round(float(selected["bias_delta"]), 4),
            "effective_bias": round(
                float(DEFAULT_DISRUPTION_RISK_BIAS + selected["bias_delta"]),
                4,
            ),
        },
        "metrics": {
            "selected_train_val": selected["train_val"],
            "selected_holdout_test": holdout_metrics,
            "baseline_train_val": baseline_train_val,
            "baseline_holdout_test": baseline_holdout,
        },
        "gates": {
            "train_val_pass": train_val_pass,
            "holdout_test_pass": holdout_pass,
            "overall_pass": bool(train_val_pass and holdout_pass),
        },
    }
    return output


def _render_markdown(data: dict[str, Any]) -> str:
    selection = data["selection"]
    metrics = data["metrics"]
    gates = data["gates"]
    targets = data["targets"]
    return (
        "# Disruption Risk Calibration Report\n\n"
        f"- Calibration version: `{data['version']}`\n"
        f"- Selection mode: `{selection['mode']}`\n"
        f"- Selected threshold: `{selection['risk_threshold']}`\n"
        f"- Selected bias delta: `{selection['bias_delta']}`\n"
        f"- Effective bias: `{selection['effective_bias']}`\n"
        f"- Targets: recall >= `{targets['recall_min']}`, FPR <= `{targets['false_positive_rate_max']}`\n\n"
        "## Train/Val Metrics (Selected)\n\n"
        f"- Recall: `{metrics['selected_train_val']['recall']}`\n"
        f"- False positive rate: `{metrics['selected_train_val']['false_positive_rate']}`\n\n"
        "## Holdout Metrics (Selected)\n\n"
        f"- Recall: `{metrics['selected_holdout_test']['recall']}`\n"
        f"- False positive rate: `{metrics['selected_holdout_test']['false_positive_rate']}`\n\n"
        "## Baseline Comparison (threshold=0.50, bias_delta=0.0)\n\n"
        f"- Train/Val recall: `{metrics['baseline_train_val']['recall']}`\n"
        f"- Train/Val FPR: `{metrics['baseline_train_val']['false_positive_rate']}`\n"
        f"- Holdout recall: `{metrics['baseline_holdout_test']['recall']}`\n"
        f"- Holdout FPR: `{metrics['baseline_holdout_test']['false_positive_rate']}`\n\n"
        "## Gate Status\n\n"
        f"- Train/Val gate: `{'PASS' if gates['train_val_pass'] else 'FAIL'}`\n"
        f"- Holdout gate: `{'PASS' if gates['holdout_test_pass'] else 'FAIL'}`\n"
        f"- Overall: `{'PASS' if gates['overall_pass'] else 'FAIL'}`\n"
    )


def _write_outputs(calibration_path: Path, report_path: Path, data: dict[str, Any]) -> None:
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_markdown(data), encoding="utf-8")


def _check_outputs(calibration_path: Path, report_path: Path, data: dict[str, Any]) -> int:
    if not calibration_path.exists():
        print(f"Calibration file missing: {calibration_path}")
        return 1
    if not report_path.exists():
        print(f"Calibration report missing: {report_path}")
        return 1

    expected_json = json.dumps(data, indent=2, sort_keys=True) + "\n"
    current_json = calibration_path.read_text(encoding="utf-8")
    if current_json != expected_json:
        print(f"Calibration drift detected: {calibration_path}")
        print("Run tools/generate_disruption_risk_calibration.py to refresh outputs.")
        return 1

    expected_md = _render_markdown(data)
    current_md = report_path.read_text(encoding="utf-8")
    if current_md != expected_md:
        print(f"Calibration report drift detected: {report_path}")
        print("Run tools/generate_disruption_risk_calibration.py to refresh outputs.")
        return 1

    print(f"Disruption risk calibration is up to date: {calibration_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot-dir", default=str(DEFAULT_SHOT_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS))
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--target-recall", type=float, default=0.80)
    parser.add_argument("--target-fpr", type=float, default=0.30)
    parser.add_argument("--threshold-min", type=float, default=0.20)
    parser.add_argument("--threshold-max", type=float, default=0.80)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--bias-min", type=float, default=-0.50)
    parser.add_argument("--bias-max", type=float, default=0.50)
    parser.add_argument("--bias-step", type=float, default=0.05)
    parser.add_argument("--skip-gates", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    shot_dir = _resolve_repo_path(args.shot_dir)
    manifest_path = _resolve_repo_path(args.manifest)
    splits_path = _resolve_repo_path(args.splits)
    calibration_path = _resolve_repo_path(args.calibration)
    report_path = _resolve_repo_path(args.report_md)
    window_size = int(args.window_size)
    if window_size < 2:
        raise ValueError("window_size must be >= 2.")
    target_recall = float(args.target_recall)
    target_fpr = float(args.target_fpr)
    if not np.isfinite(target_recall) or target_recall < 0.0 or target_recall > 1.0:
        raise ValueError("target_recall must be finite in [0, 1].")
    if not np.isfinite(target_fpr) or target_fpr < 0.0 or target_fpr > 1.0:
        raise ValueError("target_fpr must be finite in [0, 1].")
    threshold_step = float(args.threshold_step)
    bias_step = float(args.bias_step)
    if not np.isfinite(threshold_step) or threshold_step <= 0.0:
        raise ValueError("threshold_step must be finite and > 0.")
    if not np.isfinite(bias_step) or bias_step <= 0.0:
        raise ValueError("bias_step must be finite and > 0.")

    threshold_values = np.arange(
        float(args.threshold_min),
        float(args.threshold_max) + 0.5 * threshold_step,
        threshold_step,
        dtype=np.float64,
    )
    bias_values = np.arange(
        float(args.bias_min),
        float(args.bias_max) + 0.5 * bias_step,
        bias_step,
        dtype=np.float64,
    )
    if threshold_values.size == 0 or np.any((threshold_values <= 0.0) | (threshold_values >= 1.0)):
        raise ValueError("Threshold sweep must generate finite values strictly inside (0, 1).")
    if bias_values.size == 0:
        raise ValueError("Bias sweep must generate at least one value.")

    data = _generate(
        shot_dir=shot_dir,
        manifest_path=manifest_path,
        splits_path=splits_path,
        target_recall=target_recall,
        target_fpr=target_fpr,
        threshold_values=threshold_values,
        bias_values=bias_values,
        window_size=window_size,
    )

    if args.check:
        rc = _check_outputs(calibration_path, report_path, data)
        if rc != 0:
            return rc
    else:
        _write_outputs(calibration_path, report_path, data)
        print(f"Wrote calibration: {calibration_path}")
        print(f"Wrote report: {report_path}")

    if not bool(args.skip_gates):
        if not bool(data["gates"]["overall_pass"]):
            print(
                "Disruption calibration holdout gate FAILED: "
                f"train_val_pass={data['gates']['train_val_pass']}, "
                f"holdout_test_pass={data['gates']['holdout_test_pass']}"
            )
            return 1
    print("Disruption calibration holdout gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
