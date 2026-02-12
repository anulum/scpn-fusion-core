# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Kernel Profiler
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Profile Grad-Shafranov equilibrium solve with cProfile."""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scpn_fusion.core.fusion_kernel import FusionKernel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile SCPN FusionKernel.solve_equilibrium.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "validation" / "iter_validated_config.json"),
        help="Path to reactor config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "profiling"),
        help="Directory for profile artifacts.",
    )
    parser.add_argument(
        "--sort",
        default="cumtime",
        choices=["calls", "cumulative", "cumtime", "filename", "line", "name", "nfl", "pcalls", "stdname", "time", "tottime"],
        help="pstats sorting key.",
    )
    parser.add_argument("--top", type=int, default=40, help="Top N lines for text report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_path = output_dir / "kernel_solve.prof"
    report_path = output_dir / "kernel_solve.txt"

    kernel = FusionKernel(args.config)
    profiler = cProfile.Profile()

    t0 = time.perf_counter()
    profiler.enable()
    kernel.solve_equilibrium()
    profiler.disable()
    elapsed = time.perf_counter() - t0

    profiler.dump_stats(str(profile_path))

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(args.sort)
    stats.print_stats(args.top)
    report_path.write_text(stream.getvalue(), encoding="utf-8")

    print("Kernel profiling complete.")
    print(f"Elapsed: {elapsed:.3f}s")
    print(f"Profile: {profile_path}")
    print(f"Report:  {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
