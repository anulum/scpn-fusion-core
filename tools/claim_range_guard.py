#!/usr/bin/env python
"""Guard headline claim metrics using range/equality checks on JSON artifacts."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "validation" / "claim_range_thresholds.json"


PathToken = str | int


@dataclass(frozen=True)
class RatioPath:
    numerator: tuple[PathToken, ...]
    denominator: tuple[PathToken, ...]


@dataclass(frozen=True)
class RangeCheck:
    check_id: str
    file: str
    path: tuple[PathToken, ...] | None
    ratio: RatioPath | None
    minimum: float | None
    maximum: float | None
    equals: Any
    description: str


def _require_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _coerce_finite_float(name: str, value: Any) -> float:
    if not _is_number(value):
        raise ValueError(f"{name} must be a finite number.")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def _parse_path(name: str, value: Any) -> tuple[PathToken, ...]:
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError(f"{name} must be a non-empty list of path tokens.")
    tokens: list[PathToken] = []
    for i, token in enumerate(value):
        if isinstance(token, bool):
            raise ValueError(f"{name}[{i}] must be a string or integer token.")
        if isinstance(token, int):
            tokens.append(token)
            continue
        if isinstance(token, str) and token:
            tokens.append(token)
            continue
        raise ValueError(f"{name}[{i}] must be a string or integer token.")
    return tuple(tokens)


def _parse_ratio(name: str, value: Any) -> RatioPath:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object.")
    return RatioPath(
        numerator=_parse_path(f"{name}.numerator", value.get("numerator")),
        denominator=_parse_path(f"{name}.denominator", value.get("denominator")),
    )


def _parse_check(index: int, value: Any) -> RangeCheck:
    if not isinstance(value, dict):
        raise ValueError(f"checks[{index}] must be an object.")

    check_id = _require_str(f"checks[{index}].id", value.get("id"))
    file = _require_str(f"checks[{index}].file", value.get("file"))

    path_raw = value.get("path")
    ratio_raw = value.get("ratio")
    if path_raw is None and ratio_raw is None:
        raise ValueError(f"checks[{index}] must define either 'path' or 'ratio'.")
    if path_raw is not None and ratio_raw is not None:
        raise ValueError(f"checks[{index}] cannot define both 'path' and 'ratio'.")

    path = _parse_path(f"checks[{index}].path", path_raw) if path_raw is not None else None
    ratio = _parse_ratio(f"checks[{index}].ratio", ratio_raw) if ratio_raw is not None else None

    minimum = value.get("min")
    maximum = value.get("max")
    equals = value.get("equals", None)
    if minimum is None and maximum is None and equals is None:
        raise ValueError(
            f"checks[{index}] must define at least one of 'min', 'max', or 'equals'."
        )

    min_float = _coerce_finite_float(f"checks[{index}].min", minimum) if minimum is not None else None
    max_float = _coerce_finite_float(f"checks[{index}].max", maximum) if maximum is not None else None
    if min_float is not None and max_float is not None and min_float > max_float:
        raise ValueError(
            f"checks[{index}] has invalid bounds: min ({min_float}) > max ({max_float})."
        )

    if _is_number(equals):
        equals = _coerce_finite_float(f"checks[{index}].equals", equals)
    elif equals is not None and not isinstance(equals, (bool, str)):
        raise ValueError(
            f"checks[{index}].equals must be null, bool, string, or finite number."
        )

    description = str(value.get("description") or "").strip()
    return RangeCheck(
        check_id=check_id,
        file=file,
        path=path,
        ratio=ratio,
        minimum=min_float,
        maximum=max_float,
        equals=equals,
        description=description,
    )


def load_checks(config_path: Path) -> tuple[RangeCheck, ...]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Claim range config must be a JSON object.")
    checks_raw = payload.get("checks")
    if not isinstance(checks_raw, list) or len(checks_raw) == 0:
        raise ValueError("Claim range config must contain non-empty 'checks'.")

    checks: list[RangeCheck] = []
    seen: set[str] = set()
    for idx, check_raw in enumerate(checks_raw):
        check = _parse_check(idx, check_raw)
        if check.check_id in seen:
            raise ValueError(f"Duplicate check id: {check.check_id}")
        seen.add(check.check_id)
        checks.append(check)
    return tuple(checks)


def _resolve_path(payload: Any, path: tuple[PathToken, ...], *, label: str) -> Any:
    current = payload
    for token in path:
        token_desc = repr(token)
        if isinstance(token, str):
            if not isinstance(current, dict):
                raise KeyError(f"{label}: expected object before key {token_desc}")
            if token not in current:
                raise KeyError(f"{label}: missing key {token_desc}")
            current = current[token]
            continue
        if not isinstance(current, list):
            raise KeyError(f"{label}: expected list before index {token_desc}")
        if token < 0 or token >= len(current):
            raise KeyError(f"{label}: index out of range {token_desc}")
        current = current[token]
    return current


def _coerce_observed_number(check_id: str, value: Any) -> float:
    if not _is_number(value):
        raise ValueError(f"[{check_id}] observed value is not numeric: {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"[{check_id}] observed value is not finite: {value!r}")
    return out


def run_checks(
    checks: tuple[RangeCheck, ...],
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    summary_rows: list[dict[str, Any]] = []
    json_cache: dict[Path, Any] = {}

    def load_json(rel_file: str) -> Any:
        file_path = repo_root / rel_file
        if file_path in json_cache:
            return json_cache[file_path]
        if not file_path.exists():
            raise FileNotFoundError(rel_file)
        data = json.loads(file_path.read_text(encoding="utf-8"))
        json_cache[file_path] = data
        return data

    for check in checks:
        row: dict[str, Any] = {
            "id": check.check_id,
            "file": check.file,
            "status": "pass",
            "description": check.description,
        }
        try:
            payload = load_json(check.file)
            if check.path is not None:
                observed = _resolve_path(
                    payload,
                    check.path,
                    label=f"[{check.check_id}] {check.file}",
                )
                row["path"] = list(check.path)
            else:
                assert check.ratio is not None
                numerator = _resolve_path(
                    payload,
                    check.ratio.numerator,
                    label=f"[{check.check_id}] {check.file} numerator",
                )
                denominator = _resolve_path(
                    payload,
                    check.ratio.denominator,
                    label=f"[{check.check_id}] {check.file} denominator",
                )
                num_value = _coerce_observed_number(check.check_id, numerator)
                den_value = _coerce_observed_number(check.check_id, denominator)
                if den_value == 0.0:
                    raise ValueError(f"[{check.check_id}] denominator is zero.")
                observed = num_value / den_value
                row["ratio"] = {
                    "numerator": list(check.ratio.numerator),
                    "denominator": list(check.ratio.denominator),
                }

            row["observed"] = observed
            if check.equals is not None:
                expected = check.equals
                if _is_number(expected):
                    observed_num = _coerce_observed_number(check.check_id, observed)
                    if not math.isclose(observed_num, float(expected), rel_tol=0.0, abs_tol=1e-12):
                        raise ValueError(
                            f"[{check.check_id}] expected == {expected}, got {observed_num}"
                        )
                elif observed != expected:
                    raise ValueError(
                        f"[{check.check_id}] expected == {expected!r}, got {observed!r}"
                    )

            if check.minimum is not None:
                observed_num = _coerce_observed_number(check.check_id, observed)
                if observed_num < check.minimum:
                    raise ValueError(
                        f"[{check.check_id}] expected >= {check.minimum}, got {observed_num}"
                    )
            if check.maximum is not None:
                observed_num = _coerce_observed_number(check.check_id, observed)
                if observed_num > check.maximum:
                    raise ValueError(
                        f"[{check.check_id}] expected <= {check.maximum}, got {observed_num}"
                    )
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            row["status"] = "fail"
            row["error"] = str(exc)
            errors.append(str(exc))
        summary_rows.append(row)

    summary = {
        "total_checks": len(checks),
        "failed_checks": len(errors),
        "checks": summary_rows,
    }
    return errors, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to claim range threshold config JSON.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional output path for a machine-readable summary JSON.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Claim range config not found: {config_path}")

    checks = load_checks(config_path)
    errors, summary = run_checks(checks, repo_root=REPO_ROOT)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        if not summary_path.is_absolute():
            summary_path = REPO_ROOT / summary_path
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if errors:
        print(f"Claim range guard FAILED ({len(errors)} issue(s))")
        for error in errors:
            print(f" - {error}")
        return 1

    print(f"Claim range guard passed for {len(checks)} checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
