#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Onboard DIII-D raw-derived disruption shots into validation/reference_data."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.download_diiid_data import download_shot_data
except ImportError:  # pragma: no cover - direct script execution fallback
    from download_diiid_data import download_shot_data  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = (
    REPO_ROOT
    / "validation"
    / "reference_data"
    / "diiid"
    / "raw_disruption_onboarding_spec.example.json"
)
DEFAULT_SHOT_DIR = REPO_ROOT / "validation" / "reference_data" / "diiid" / "disruption_shots"
DEFAULT_METADATA = (
    REPO_ROOT / "validation" / "reference_data" / "diiid" / "disruption_shot_metadata.json"
)
DEFAULT_MANIFEST = (
    REPO_ROOT / "validation" / "reference_data" / "diiid" / "disruption_shots_manifest.json"
)
_SCENARIO_SANITIZE = re.compile(r"[^a-z0-9_]+")


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object payload")
    return payload


def _sanitize_scenario(value: str) -> str:
    sanitized = _SCENARIO_SANITIZE.sub("_", value.strip().lower()).strip("_")
    if not sanitized:
        raise ValueError("scenario must contain at least one alphanumeric character")
    return sanitized


def _ensure_monotonic_timebase(raw_time: np.ndarray) -> np.ndarray:
    time_s = np.asarray(raw_time, dtype=np.float64).reshape(-1)
    if time_s.size < 2:
        return np.arange(max(time_s.size, 2), dtype=np.float64)
    if not np.all(np.isfinite(time_s)):
        return np.arange(time_s.size, dtype=np.float64)
    diffs = np.diff(time_s)
    if np.any(diffs <= 0.0):
        return np.arange(time_s.size, dtype=np.float64)
    return time_s


def _derive_disruption_signal(data: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    base = np.asarray(data, dtype=np.float64).reshape(-1)
    if base.size < 2:
        raise ValueError("signal must contain at least 2 samples")
    if not np.all(np.isfinite(base)):
        raise ValueError("signal contains non-finite values")
    gradient = np.gradient(base, time_s, edge_order=1)
    if not np.all(np.isfinite(gradient)):
        gradient = np.gradient(base, edge_order=1)
    scale = float(np.percentile(np.abs(gradient), 95))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(gradient))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return np.asarray(gradient / scale, dtype=np.float64)


def _choose_disruption_index(
    *,
    n_samples: int,
    is_disruption: bool,
    disruption_time_idx: int | None,
    disruption_time_s: float | None,
    time_s: np.ndarray,
) -> int:
    if not is_disruption:
        return -1
    if disruption_time_idx is not None:
        idx = int(disruption_time_idx)
    elif disruption_time_s is not None:
        idx = int(np.searchsorted(time_s, float(disruption_time_s), side="left"))
    else:
        idx = int(round(0.8 * n_samples))
    return int(min(max(idx, 1), n_samples - 1))


def _derive_source_type(source: str) -> str:
    normalized = source.strip().lower()
    if normalized == "mdsplus":
        return "raw_diiid_mdsplus_proxy"
    if normalized == "cache":
        return "raw_diiid_cache_proxy"
    if normalized == "reference":
        return "reference_diiid_proxy"
    return "unknown_diiid_proxy"


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"manifest_overrides": {}, "shot_overrides": {}}
    payload = _load_json(path)
    manifest_overrides = payload.get("manifest_overrides", {})
    shot_overrides = payload.get("shot_overrides", {})
    if not isinstance(manifest_overrides, dict):
        raise ValueError("metadata.manifest_overrides must be an object")
    if not isinstance(shot_overrides, dict):
        raise ValueError("metadata.shot_overrides must be an object")
    return {
        "manifest_overrides": dict(manifest_overrides),
        "shot_overrides": dict(shot_overrides),
    }


def _save_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _refresh_manifest(
    *,
    shot_dir: Path,
    metadata_path: Path,
    manifest_path: Path,
) -> None:
    subprocess.run(
        [
            sys.executable,
            "tools/generate_disruption_shot_manifest.py",
            "--shot-dir",
            str(shot_dir),
            "--metadata",
            str(metadata_path),
            "--manifest",
            str(manifest_path),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def onboard_shots(
    *,
    spec: dict[str, Any],
    shot_dir: Path,
    metadata_path: Path,
    cache_dir: Path,
    force_download: bool,
    refresh_manifest: bool,
    manifest_path: Path,
) -> dict[str, Any]:
    shots_raw = spec.get("shots", [])
    if not isinstance(shots_raw, list) or not shots_raw:
        raise ValueError("spec must contain non-empty 'shots' list")

    metadata = _load_metadata(metadata_path)
    manifest_overrides = metadata.setdefault("manifest_overrides", {})
    shot_overrides = metadata.setdefault("shot_overrides", {})
    if not isinstance(manifest_overrides, dict) or not isinstance(shot_overrides, dict):
        raise ValueError("metadata payload malformed")

    shot_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    failures: list[dict[str, Any]] = []

    for i, item in enumerate(shots_raw):
        if not isinstance(item, dict):
            raise ValueError(f"spec.shots[{i}] must be an object")
        shot = item.get("shot")
        if isinstance(shot, bool) or not isinstance(shot, int) or shot <= 0:
            raise ValueError(f"spec.shots[{i}].shot must be a positive integer")
        scenario_raw = str(item.get("scenario", f"raw_{shot}"))
        scenario = _sanitize_scenario(scenario_raw)
        label = str(item.get("label", "disruptive")).strip().lower()
        if label not in {"disruptive", "safe"}:
            raise ValueError(f"spec.shots[{i}].label must be 'disruptive' or 'safe'")
        is_disruption = bool(item.get("is_disruption", label == "disruptive"))
        machine = str(item.get("machine", "DIII-D")).strip() or "DIII-D"
        signals_raw = item.get("signals", ["Ip"])
        if isinstance(signals_raw, str):
            signals = [v.strip() for v in signals_raw.split(",") if v.strip()]
        elif isinstance(signals_raw, list):
            signals = [str(v).strip() for v in signals_raw if str(v).strip()]
        else:
            raise ValueError(f"spec.shots[{i}].signals must be a list or CSV string")
        if not signals:
            raise ValueError(f"spec.shots[{i}].signals cannot be empty")

        try:
            download_result = download_shot_data(
                machine=machine,
                shot=shot,
                signals=signals,
                cache_dir=cache_dir,
                host=item.get("host"),
                tree=item.get("tree"),
                force_download=force_download,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"shot": shot, "error": f"download failed: {exc}"})
            continue

        if not download_result.signals:
            failures.append(
                {
                    "shot": shot,
                    "error": "no signals returned",
                    "source": download_result.source,
                }
            )
            continue

        preferred_signal = str(item.get("primary_signal", "Ip")).strip()
        if preferred_signal and preferred_signal in download_result.signals:
            signal_record = download_result.signals[preferred_signal]
        else:
            first_key = sorted(download_result.signals.keys())[0]
            signal_record = download_result.signals[first_key]
            preferred_signal = first_key

        data = np.asarray(signal_record.data, dtype=np.float64).reshape(-1)
        time_s = _ensure_monotonic_timebase(np.asarray(signal_record.time, dtype=np.float64))
        n = min(data.size, time_s.size)
        data = data[:n]
        time_s = time_s[:n]
        if n < 16:
            failures.append({"shot": shot, "error": "insufficient samples (<16)"})
            continue

        try:
            dBdt = _derive_disruption_signal(data, time_s)
        except ValueError as exc:
            failures.append({"shot": shot, "error": str(exc)})
            continue

        n1_amp = np.asarray(np.abs(dBdt), dtype=np.float64)
        n2_amp = np.asarray(0.25 * n1_amp, dtype=np.float64)
        disruption_idx = _choose_disruption_index(
            n_samples=n,
            is_disruption=is_disruption,
            disruption_time_idx=(
                int(item["disruption_time_idx"]) if "disruption_time_idx" in item else None
            ),
            disruption_time_s=(
                float(item["disruption_time_s"]) if "disruption_time_s" in item else None
            ),
            time_s=time_s,
        )

        out_name = f"shot_{shot}_{scenario}.npz"
        out_path = shot_dir / out_name
        np.savez(
            out_path,
            time_s=time_s,
            dBdt_gauss_per_s=dBdt,
            n1_amp=n1_amp,
            n2_amp=n2_amp,
            is_disruption=np.array(is_disruption),
            disruption_time_idx=np.array(disruption_idx),
        )

        shot_overrides[out_name] = {
            "shot": shot,
            "scenario": scenario,
            "label": label,
            "source_type": _derive_source_type(str(download_result.source)),
            "generator": "tools/onboard_diiid_raw_disruption_shots.py",
            "license": str(item.get("license", "facility-restricted-not-redistributable")),
        }
        created.append(out_name)
        print(
            f"Onboarded {out_name}: source={download_result.source} "
            f"primary_signal={preferred_signal} samples={n}"
        )

    if created:
        if "data_license" not in manifest_overrides:
            manifest_overrides["data_license"] = "mixed-v1"
        if "real_data_notice" not in manifest_overrides:
            manifest_overrides["real_data_notice"] = (
                "Bundled files may include synthetic and raw-derived DIII-D profiles. "
                "Raw data access/reuse follows DOE/GA facility terms."
            )
        manifest_overrides["generator_reference"] = (
            "tools/generate_disruption_profiles.py + tools/onboard_diiid_raw_disruption_shots.py"
        )

    _save_metadata(metadata_path, metadata)
    if refresh_manifest:
        _refresh_manifest(
            shot_dir=shot_dir,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
        )

    return {
        "created_count": len(created),
        "created_files": created,
        "failed_count": len(failures),
        "failures": failures,
        "metadata_path": str(metadata_path),
        "manifest_path": str(manifest_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        default=str(DEFAULT_SPEC),
        help="Onboarding spec JSON path.",
    )
    parser.add_argument(
        "--shot-dir",
        default=str(DEFAULT_SHOT_DIR),
        help="Output disruption shot directory.",
    )
    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_METADATA),
        help="Metadata override JSON path consumed by generate_disruption_shot_manifest.py.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Disruption shot manifest output path.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(REPO_ROOT / "validation" / "reference_data" / "diiid"),
        help="Cache directory passed to tools/download_diiid_data.py.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download in tools/download_diiid_data.py.",
    )
    parser.add_argument(
        "--skip-refresh-manifest",
        action="store_true",
        help="Do not regenerate disruption_shots_manifest.json after onboarding.",
    )
    parser.add_argument(
        "--summary-json",
        default="artifacts/raw_disruption_onboarding_summary.json",
        help="Output summary JSON path.",
    )
    args = parser.parse_args(argv)

    spec_path = _resolve(args.spec)
    shot_dir = _resolve(args.shot_dir)
    metadata_path = _resolve(args.metadata)
    manifest_path = _resolve(args.manifest)
    cache_dir = _resolve(args.cache_dir)
    summary_path = _resolve(args.summary_json)

    if not spec_path.exists():
        raise FileNotFoundError(f"Onboarding spec not found: {spec_path}")

    summary = onboard_shots(
        spec=_load_json(spec_path),
        shot_dir=shot_dir,
        metadata_path=metadata_path,
        cache_dir=cache_dir,
        force_download=bool(args.force_download),
        refresh_manifest=not args.skip_refresh_manifest,
        manifest_path=manifest_path,
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Raw disruption onboarding complete: "
        f"created={summary['created_count']} failed={summary['failed_count']}"
    )
    if int(summary["failed_count"]) > 0:
        return 2 if int(summary["created_count"]) > 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
