#!/usr/bin/env python
"""Run mypy with CI-parity settings for deterministic local type-gate checks."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_MYPY_TIMEOUT_SECONDS = 1200.0


def _normalize_timeout_seconds(timeout_s: float) -> float:
    timeout = float(timeout_s)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("timeout_seconds must be finite and > 0.")
    return timeout


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"
    )

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_MYPY_TIMEOUT_SECONDS,
    )
    args, mypy_args = parser.parse_known_args(sys.argv[1:])
    try:
        timeout_s = _normalize_timeout_seconds(args.timeout_seconds)
    except ValueError as exc:
        print(f"[run_mypy_strict] {exc}", file=sys.stderr)
        return 2

    cmd = [
        sys.executable,
        "-m",
        "mypy",
        "--no-incremental",
        "--no-warn-unused-configs",
        *mypy_args,
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        print(
            f"[run_mypy_strict] mypy timed out after {timeout_s:.1f}s",
            file=sys.stderr,
        )
        return 124
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
