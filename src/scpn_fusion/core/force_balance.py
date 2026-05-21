# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Force Balance
"""Newton-Raphson PF-coil force-balance adjustment for equilibrium configs."""

from __future__ import annotations

import numpy as np
import json
from pathlib import Path

from scpn_fusion.core.stability_analyzer import StabilityAnalyzer


class ForceBalanceSolver:
    """
    Automatic Newton-Raphson Solver to find Coil Currents that result in
    ZERO Radial Force on the plasma at the target Radius.
    This guarantees static equilibrium at R_target.
    """

    def __init__(self, config_path):
        self.config_path = config_path
        self.analyzer = StabilityAnalyzer(config_path)
        self._control_indices = (2, 3)

    def _validate_kernel_config(self) -> None:
        cfg = self.analyzer.kernel.cfg
        if "physics" not in cfg or "plasma_current_target" not in cfg["physics"]:
            raise ValueError("Kernel config missing physics.plasma_current_target.")
        coils = cfg.get("coils")
        if not isinstance(coils, list) or len(coils) <= max(self._control_indices):
            raise ValueError(
                "Kernel config must define at least 4 coils for PF3/PF4 force-balance control."
            )
        for idx in self._control_indices:
            cur = float(coils[idx]["current"])
            if not np.isfinite(cur):
                raise ValueError(f"Coil current at index {idx} must be finite.")

    def solve_for_equilibrium(
        self,
        target_R=6.2,
        target_Z=0.0,
        *,
        max_iterations: int = 10,
        jacobian_floor: float = 1e-12,
    ) -> dict[str, float | int | bool]:
        """Iteratively adjust paired PF currents until radial force is balanced."""
        self._validate_kernel_config()
        if int(max_iterations) < 1:
            raise ValueError("max_iterations must be >= 1.")
        max_iterations = int(max_iterations)
        jacobian_floor = float(jacobian_floor)
        if not np.isfinite(jacobian_floor) or jacobian_floor <= 0.0:
            raise ValueError("jacobian_floor must be finite and > 0.")
        target_R = float(target_R)
        target_Z = float(target_Z)
        if not np.isfinite(target_R) or not np.isfinite(target_Z):
            raise ValueError("target_R and target_Z must be finite.")

        print("--- FORCE BALANCE SOLVER (Newton-Raphson) ---")
        print(f"Target Equilibrium: R={target_R}m, Z={target_Z}m")

        # We control PF3 and PF4 (outer coils), constrained symmetrically.
        i_pf3, i_pf4 = self._control_indices

        # Load Physics Params
        Ip = self.analyzer.kernel.cfg["physics"]["plasma_current_target"]
        converged = False
        final_fr = float("nan")
        last_delta_i = 0.0

        # Newton Loop
        for iter in range(max_iterations):
            # 1. Calculate Current Force
            Fr, Fz, n_idx = self.analyzer.calculate_forces(target_R, target_Z, Ip)
            final_fr = float(Fr)
            print(f"Iter {iter}: Radial Force = {Fr / 1e6:.2f} MN")

            if abs(Fr) < 1e4:  # Tolerance 10 kN
                print("  -> CONVERGED. Force Balance Achieved.")
                converged = True
                break

            # 2. Calculate Jacobian dF/dI (Numerical Derivative)
            # We perturb PF3/PF4 together
            dI = 0.1  # MA perturbation

            # Apply perturbation
            currents = [c["current"] for c in self.analyzer.kernel.cfg["coils"]]
            original_I3 = float(currents[i_pf3])

            # Modify config in memory
            self.analyzer.kernel.cfg["coils"][i_pf3]["current"] = original_I3 + dI
            self.analyzer.kernel.cfg["coils"][i_pf4]["current"] = original_I3 + dI

            # Recalculate Vacuum Field (Expensive part, but necessary)
            self.analyzer.Psi_vac = self.analyzer.kernel.calculate_vacuum_field()

            # Calculate new Force
            Fr_new, _, _ = self.analyzer.calculate_forces(target_R, target_Z, Ip)

            # Jacobian J = dF/dI (MN / MA)
            J = (Fr_new - Fr) / dI
            print(f"  Jacobian dF/dI_PF34: {J / 1e6:.2f} MN/MA")

            # 3. Newton Step: I_new = I_old - F / J
            # We want F_target = 0
            # 0 = F_old + J * delta_I
            # delta_I = - F_old / J
            if not np.isfinite(J) or abs(J) < jacobian_floor:
                # Singular/ill-conditioned update; take a conservative directional step.
                delta_I = -0.25 * np.sign(Fr) if Fr != 0.0 else 0.0
                print(
                    "  Jacobian near-singular; applying conservative "
                    f"directional step ΔI={delta_I:.3f} MA"
                )
            else:
                delta_I = -Fr / J

            # Safety Clamp (don't jump more than 5 MA at once)
            delta_I = np.clip(delta_I, -5.0, 5.0)
            last_delta_i = float(delta_I)

            print(f"  Correction: Delta I = {delta_I:.3f} MA")

            # Apply correction
            new_I = original_I3 + delta_I

            # Reset Config to new values
            self.analyzer.kernel.cfg["coils"][i_pf3]["current"] = float(new_I)
            self.analyzer.kernel.cfg["coils"][i_pf4]["current"] = float(new_I)

            # Update Vacuum field for next iteration base
            self.analyzer.Psi_vac = self.analyzer.kernel.calculate_vacuum_field()

        # Save Result
        self.save_config()
        return {
            "converged": converged,
            "iterations_executed": iter + 1,
            "final_radial_force_n": final_fr,
            "last_delta_i_ma": last_delta_i,
            "pf3_current_ma": float(self.analyzer.kernel.cfg["coils"][i_pf3]["current"]),
            "pf4_current_ma": float(self.analyzer.kernel.cfg["coils"][i_pf4]["current"]),
        }

    def save_config(self, output_path: str | Path | None = None):
        """Write the current balanced kernel configuration to disk."""
        # Save to validation folder by default.
        repo_root = Path(__file__).resolve().parents[3]
        out_path = (
            Path(output_path)
            if output_path is not None
            else (repo_root / "validation" / "iter_force_balanced.json")
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(self.analyzer.kernel.cfg, indent=4),
            encoding="utf-8",
        )
        print(f"Balanced Config Saved: {out_path}")


if __name__ == "__main__":
    # Start from the last "best" config (Genetic or Validated)
    base_cfg = "03_CODE/SCPN-Fusion-Core/validation/iter_validated_config.json"

    solver = ForceBalanceSolver(base_cfg)
    solver.solve_for_equilibrium(target_R=6.2)
