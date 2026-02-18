#!/usr/bin/env python
"""Run mypy with CI-parity settings for deterministic local type-gate checks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"
    )

    cmd = [
        sys.executable,
        "-m",
        "mypy",
        "--no-incremental",
        "--no-warn-unused-configs",
        *sys.argv[1:],
    ]
    return subprocess.call(cmd, cwd=repo_root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
