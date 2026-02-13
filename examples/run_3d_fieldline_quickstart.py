# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — 3D Field-Line Quickstart
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Generate reduced 3D field-line trajectory and Poincare sections."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_phi_planes(raw: str) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        return [0.0]
    return values


def main(argv: list[str] | None = None) -> int:
    root = _repo_root()
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from scpn_fusion.core.equilibrium_3d import FourierMode3D
    from scpn_fusion.core.geometry_3d import Reactor3DBuilder

    parser = argparse.ArgumentParser(
        description="SCPN 3D field-line and Poincare quickstart."
    )
    parser.add_argument(
        "--config",
        default=str(root / "validation" / "iter_validated_config.json"),
        help="Path to reactor configuration JSON.",
    )
    parser.add_argument(
        "--output",
        default=str(root / "artifacts" / "SCPN_3D_Poincare_quickstart.npz"),
        help="Output NPZ path.",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=12,
        help="Toroidal turns for field-line tracing.",
    )
    parser.add_argument(
        "--steps-per-turn",
        type=int,
        default=256,
        help="Integration steps per toroidal turn.",
    )
    parser.add_argument(
        "--phi-planes",
        default="0.0,1.5707963267948966",
        help="Comma-separated toroidal planes in radians.",
    )
    parser.add_argument(
        "--vmec-like",
        action="store_true",
        help="Use reduced VMEC-like 3D equilibrium interface.",
    )
    parser.add_argument(
        "--n1-amplitude",
        type=float,
        default=0.0,
        help="Optional amplitude for a single n=1 harmonic in VMEC-like mode.",
    )
    args = parser.parse_args(argv)

    planes = _parse_phi_planes(args.phi_planes)
    builder = Reactor3DBuilder(args.config)
    if args.vmec_like:
        modes: list[FourierMode3D] = []
        if abs(args.n1_amplitude) > 0.0:
            amp = float(args.n1_amplitude)
            modes.append(FourierMode3D(m=1, n=1, r_cos=amp, z_sin=0.5 * amp))
        builder.equilibrium_3d = builder.build_vmec_like_equilibrium(
            toroidal_modes=modes
        )

    trace, poincare = builder.generate_poincare_map(
        toroidal_turns=args.turns,
        steps_per_turn=args.steps_per_turn,
        phi_planes=planes,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, np.ndarray | float] = {
        "trace_xyz": trace.xyz,
        "trace_rho": trace.rho,
        "trace_theta": trace.theta,
        "trace_phi": trace.phi,
    }
    for idx, plane in enumerate(sorted(poincare)):
        payload[f"plane_{idx}_phi"] = float(plane)
        payload[f"plane_{idx}_rz"] = poincare[plane]
    np.savez(output_path, **payload)

    print("SCPN 3D field-line quickstart complete.")
    print(f"Config: {args.config}")
    print(f"VMEC-like mode: {args.vmec_like}")
    print(f"Output: {output_path}")
    print(f"Trace samples: {len(trace.phi)}")
    for idx, plane in enumerate(sorted(poincare)):
        print(f"Plane[{idx}] phi={plane:.6f} -> {len(poincare[plane])} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
