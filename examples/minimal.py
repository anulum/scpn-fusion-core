# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Minimal API Example
"""Run one fast equilibrium solve and one SCPN controller step."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path(root: Path) -> None:
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def _prepare_minimal_kernel_config(
    root: Path,
    *,
    grid: int,
    equilibrium_iters: int,
) -> dict[str, Any]:
    base_path = root / "src" / "scpn_fusion" / "core" / "default_config.json"
    cfg = json.loads(base_path.read_text(encoding="utf-8"))
    cfg["reactor_name"] = "SCPN-Minimal-Demo"
    cfg["grid_resolution"] = [grid, grid]
    cfg.setdefault("solver", {})
    cfg["solver"]["max_iterations"] = int(equilibrium_iters)
    cfg["solver"]["convergence_threshold"] = float(cfg["solver"].get("convergence_threshold", 1e-4))
    cfg["solver"]["relaxation_factor"] = float(cfg["solver"].get("relaxation_factor", 0.1))
    return cfg


def _run_equilibrium(
    root: Path,
    *,
    grid: int,
    equilibrium_iters: int,
) -> dict[str, float | int | bool]:
    from scpn_fusion.core.fusion_kernel import FusionKernel

    cfg = _prepare_minimal_kernel_config(
        root,
        grid=grid,
        equilibrium_iters=equilibrium_iters,
    )
    with tempfile.TemporaryDirectory(prefix="scpn-minimal-") as tmpdir:
        cfg_path = Path(tmpdir) / "minimal_config.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        kernel = FusionKernel(cfg_path)
        result = kernel.solve_equilibrium()

    return {
        "converged": bool(result.get("converged", False)),
        "iterations": int(result.get("iterations", 0)),
        "residual": float(result.get("residual", 0.0)),
    }


def _build_minimal_controller_artifact(seed: int) -> Any:
    from scpn_fusion.scpn.compiler import FusionCompiler
    from scpn_fusion.scpn.structure import StochasticPetriNet

    net = StochasticPetriNet()
    net.add_place("x_R_pos", initial_tokens=0.0)
    net.add_place("x_R_neg", initial_tokens=0.0)
    net.add_place("x_Z_pos", initial_tokens=0.0)
    net.add_place("x_Z_neg", initial_tokens=0.0)
    net.add_place("a_R_pos", initial_tokens=0.0)
    net.add_place("a_R_neg", initial_tokens=0.0)
    net.add_place("a_Z_pos", initial_tokens=0.0)
    net.add_place("a_Z_neg", initial_tokens=0.0)

    net.add_transition("T_Rp", threshold=0.1)
    net.add_transition("T_Rn", threshold=0.1)
    net.add_transition("T_Zp", threshold=0.1)
    net.add_transition("T_Zn", threshold=0.1)

    net.add_arc("x_R_pos", "T_Rp", weight=1.0)
    net.add_arc("x_R_neg", "T_Rn", weight=1.0)
    net.add_arc("x_Z_pos", "T_Zp", weight=1.0)
    net.add_arc("x_Z_neg", "T_Zn", weight=1.0)

    net.add_arc("T_Rp", "a_R_pos", weight=1.0)
    net.add_arc("T_Rn", "a_R_neg", weight=1.0)
    net.add_arc("T_Zp", "a_Z_pos", weight=1.0)
    net.add_arc("T_Zn", "a_Z_neg", weight=1.0)
    net.compile()

    compiled = FusionCompiler(bitstream_length=256, seed=seed).compile(
        net,
        firing_mode="binary",
        firing_margin=0.05,
    )

    return compiled.export_artifact(
        name="minimal_controller",
        dt_control_s=0.001,
        readout_config={
            "actions": [
                {"name": "dI_PF3_A", "pos_place": 4, "neg_place": 5},
                {"name": "dI_PF_topbot_A", "pos_place": 6, "neg_place": 7},
            ],
            "gains": [1000.0, 1000.0],
            "abs_max": [5000.0, 5000.0],
            "slew_per_s": [1e6, 1e6],
        },
        injection_config=[
            {
                "place_id": 0,
                "source": "x_R_pos",
                "scale": 1.0,
                "offset": 0.0,
                "clamp_0_1": True,
            },
            {
                "place_id": 1,
                "source": "x_R_neg",
                "scale": 1.0,
                "offset": 0.0,
                "clamp_0_1": True,
            },
            {
                "place_id": 2,
                "source": "x_Z_pos",
                "scale": 1.0,
                "offset": 0.0,
                "clamp_0_1": True,
            },
            {
                "place_id": 3,
                "source": "x_Z_neg",
                "scale": 1.0,
                "offset": 0.0,
                "clamp_0_1": True,
            },
        ],
    )


def _run_controller_step(seed: int, obs_r: float, obs_z: float) -> Dict[str, Any]:
    from scpn_fusion.scpn.contracts import ControlScales, ControlTargets
    from scpn_fusion.scpn.controller import NeuroSymbolicController

    artifact = _build_minimal_controller_artifact(seed=seed)
    controller = NeuroSymbolicController(
        artifact=artifact,
        seed_base=seed,
        targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
        scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
    )
    action = controller.step({"R_axis_m": obs_r, "Z_axis_m": obs_z}, k=0)
    return {
        "runtime_backend": controller.runtime_backend_name,
        "runtime_profile": controller.runtime_profile_name,
        "action": dict(action),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a minimal SCPN Fusion Core equilibrium + controller demo.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=17,
        help="Square equilibrium grid size (>= 4).",
    )
    parser.add_argument(
        "--equilibrium-iters",
        type=int,
        default=4,
        help="Maximum equilibrium iterations for the fast demo path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for controller compilation/runtime.",
    )
    parser.add_argument("--obs-r", type=float, default=6.26, help="Observed R_axis_m.")
    parser.add_argument("--obs-z", type=float, default=0.03, help="Observed Z_axis_m.")
    parser.add_argument(
        "--skip-equilibrium",
        action="store_true",
        help="Skip equilibrium solve and run controller-only path.",
    )
    parser.add_argument(
        "--skip-controller",
        action="store_true",
        help="Skip controller step and run equilibrium-only path.",
    )
    args = parser.parse_args(argv)

    if args.grid < 4:
        raise SystemExit("--grid must be >= 4.")
    if args.equilibrium_iters < 1:
        raise SystemExit("--equilibrium-iters must be >= 1.")
    if args.skip_equilibrium and args.skip_controller:
        raise SystemExit("At least one lane must run (remove one skip flag).")

    root = _repo_root()
    _ensure_src_on_path(root)

    summary: Dict[str, Any] = {}

    if not args.skip_equilibrium:
        equilibrium = _run_equilibrium(
            root,
            grid=args.grid,
            equilibrium_iters=args.equilibrium_iters,
        )
        summary["equilibrium"] = equilibrium
        print(
            "equilibrium_converged="
            f"{equilibrium['converged']} "
            f"iterations={equilibrium['iterations']} "
            f"residual={equilibrium['residual']:.6e}"
        )

    if not args.skip_controller:
        controller = _run_controller_step(
            seed=args.seed,
            obs_r=float(args.obs_r),
            obs_z=float(args.obs_z),
        )
        summary["controller"] = controller
        print(
            "controller_backend="
            f"{controller['runtime_backend']} "
            f"profile={controller['runtime_profile']} "
            f"action={controller['action']}"
        )

    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
