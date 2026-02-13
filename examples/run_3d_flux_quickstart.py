# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — 3D Flux Quickstart
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Generate a 3D LCFS mesh in one command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    root = _repo_root()
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from scpn_fusion.core.geometry_3d import Reactor3DBuilder
    from scpn_fusion.core.equilibrium_3d import FourierMode3D

    parser = argparse.ArgumentParser(description="SCPN 3D flux-surface quickstart.")
    parser.add_argument(
        "--config",
        default=str(root / "validation" / "iter_validated_config.json"),
        help="Path to reactor configuration JSON.",
    )
    parser.add_argument(
        "--output",
        default=str(root / "artifacts" / "SCPN_Plasma_3D_quickstart.obj"),
        help="Output OBJ path.",
    )
    parser.add_argument("--toroidal", type=int, default=48, help="Toroidal resolution.")
    parser.add_argument("--poloidal", type=int, default=48, help="Poloidal resolution.")
    parser.add_argument(
        "--vmec-like",
        action="store_true",
        help="Generate surface from reduced VMEC-like 3D equilibrium coordinates.",
    )
    parser.add_argument(
        "--n1-amplitude",
        type=float,
        default=0.0,
        help="Optional amplitude for a single n=1 harmonic in VMEC-like mode.",
    )
    parser.add_argument(
        "--preview-png",
        default="",
        help="Optional preview PNG path.",
    )
    args = parser.parse_args(argv)

    builder = Reactor3DBuilder(args.config)
    if args.vmec_like:
        modes: list[FourierMode3D] = []
        if abs(args.n1_amplitude) > 0.0:
            amp = float(args.n1_amplitude)
            modes.append(FourierMode3D(m=1, n=1, r_cos=amp, z_sin=0.5 * amp))
        builder.equilibrium_3d = builder.build_vmec_like_equilibrium(
            toroidal_modes=modes
        )

    vertices, faces = builder.generate_plasma_surface(
        resolution_toroidal=args.toroidal,
        resolution_poloidal=args.poloidal,
    )
    output_path = builder.export_obj(vertices, faces, args.output)
    if args.preview_png:
        png_path = builder.export_preview_png(vertices, faces, args.preview_png)
        print(f"Preview PNG: {png_path}")

    print("SCPN 3D quickstart complete.")
    print(f"Config: {args.config}")
    print(f"VMEC-like mode: {args.vmec_like}")
    print(f"Output: {output_path}")
    print(f"Vertices: {len(vertices)} | Faces: {len(faces)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
