# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Geometry 3D Profiler
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Profile 3D LCFS mesh generation with cProfile."""

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

from scpn_fusion.core.geometry_3d import Reactor3DBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile SCPN 3D geometry generation.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "validation" / "iter_validated_config.json"),
        help="Path to reactor config JSON.",
    )
    parser.add_argument("--toroidal", type=int, default=48, help="Toroidal resolution.")
    parser.add_argument("--poloidal", type=int, default=48, help="Poloidal resolution.")
    parser.add_argument("--radial-steps", type=int, default=384, help="Ray marching resolution.")
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
    parser.add_argument(
        "--export-obj",
        default="",
        help="Optional OBJ output path relative to repository root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_path = output_dir / "geometry_3d.prof"
    report_path = output_dir / "geometry_3d.txt"

    builder = Reactor3DBuilder(args.config)
    profiler = cProfile.Profile()

    t0 = time.perf_counter()
    profiler.enable()
    vertices, faces = builder.generate_plasma_surface(
        resolution_toroidal=args.toroidal,
        resolution_poloidal=args.poloidal,
        radial_steps=args.radial_steps,
    )
    profiler.disable()
    elapsed = time.perf_counter() - t0

    if args.export_obj:
        out_path = ROOT / args.export_obj
        builder.export_obj(vertices, faces, out_path)
        print(f"OBJ: {out_path}")

    profiler.dump_stats(str(profile_path))
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(args.sort)
    stats.print_stats(args.top)
    report_path.write_text(stream.getvalue(), encoding="utf-8")

    print("Geometry 3D profiling complete.")
    print(f"Elapsed: {elapsed:.3f}s")
    print(f"Vertices: {len(vertices)} | Faces: {len(faces)}")
    print(f"Profile: {profile_path}")
    print(f"Report:  {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
