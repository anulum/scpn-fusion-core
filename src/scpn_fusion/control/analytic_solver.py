# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Analytic Solver
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    try:
        from scpn_fusion.core.fusion_kernel import FusionKernel
    except ImportError as exc:  # pragma: no cover - import-guard path
        raise ImportError(
            "Unable to import FusionKernel. Run with PYTHONPATH=src "
            "or use `python -m scpn_fusion.control.analytic_solver`."
        ) from exc


class AnalyticEquilibriumSolver:
    """
    Analytic vertical-field target and least-norm coil-current solve.
    """

    def __init__(
        self,
        config_path: str,
        *,
        kernel_factory: Callable[[str], Any] = FusionKernel,
        verbose: bool = True,
    ) -> None:
        self.kernel = kernel_factory(str(config_path))
        self.config_path = str(config_path)
        self.verbose = bool(verbose)

    def _log(self, message: str) -> None:
        if self.verbose:
            logger.info(message)

    def calculate_required_Bv(
        self,
        R_geo: float,
        a_min: float,
        Ip_MA: float,
        *,
        beta_p: float = 0.5,
        li: float = 0.8,
    ) -> float:
        """
        Shafranov radial-force balance vertical field estimate.
        """
        R_geo = float(R_geo)
        a_min = float(a_min)
        Ip_MA = float(Ip_MA)
        beta_p = float(beta_p)
        li = float(li)
        if R_geo <= 0.0 or a_min <= 0.0 or Ip_MA <= 0.0:
            raise ValueError("R_geo, a_min and Ip_MA must be > 0.")

        mu0 = 4.0 * np.pi * 1e-7
        Ip = Ip_MA * 1e6
        term_log = np.log(8.0 * R_geo / a_min)
        term_physics = beta_p + (li / 2.0) - 1.5
        Bv = -((mu0 * Ip) / (4.0 * np.pi * R_geo)) * (term_log + term_physics)

        self._log("--- SHAFRANOV EQUILIBRIUM CHECK ---")
        self._log(f"Target Radius: {R_geo:.3f} m")
        self._log(f"Plasma Current: {Ip_MA:.3f} MA")
        self._log(f"Required Vertical Field (Bv): {Bv:.6f} Tesla")
        return float(Bv)

    def compute_coil_efficiencies(
        self,
        target_R: float,
        *,
        target_Z: float = 0.0,
    ) -> np.ndarray:
        """
        Compute dBz/dI per coil at target location using kernel vacuum-field map.
        """
        coils = self.kernel.cfg.get("coils", [])
        n_coils = len(coils)
        if n_coils == 0:
            raise ValueError("Kernel config has no coils.")

        target_R = float(target_R)
        target_Z = float(target_Z)
        if target_R <= 0.0:
            raise ValueError("target_R must be > 0.")

        original_currents = [float(c.get("current", 0.0)) for c in coils]
        eff = np.zeros(n_coils, dtype=np.float64)

        idx_r = int(np.argmin(np.abs(np.asarray(self.kernel.R, dtype=np.float64) - target_R)))
        idx_z = int(np.argmin(np.abs(np.asarray(self.kernel.Z, dtype=np.float64) - target_Z)))
        idx_r = int(np.clip(idx_r, 1, len(self.kernel.R) - 2))
        dR = float(getattr(self.kernel, "dR", float(self.kernel.R[1] - self.kernel.R[0])))
        if dR <= 0.0:
            raise ValueError("Kernel grid spacing dR must be > 0.")

        self._log("\nCalculating Coil Influence Matrix (Green's Functions)...")
        try:
            for i in range(n_coils):
                for c in coils:
                    c["current"] = 0.0
                coils[i]["current"] = 1.0

                psi_vac = np.asarray(self.kernel.calculate_vacuum_field(), dtype=np.float64)
                dpsi = (psi_vac[idx_z, idx_r + 1] - psi_vac[idx_z, idx_r - 1]) / (2.0 * dR)
                bz_unit = float((1.0 / target_R) * dpsi)
                eff[i] = bz_unit

                name = str(coils[i].get("name", f"coil_{i}"))
                self._log(f"  Coil {name} Efficiency: {bz_unit:.6f} T/MA")
        finally:
            for c, current in zip(coils, original_currents):
                c["current"] = float(current)

        return eff

    def solve_coil_currents(
        self,
        target_Bv: float,
        target_R: float,
        *,
        target_Z: float = 0.0,
        ridge_lambda: float = 0.0,
    ) -> np.ndarray:
        """
        Solve least-norm coil currents for desired vertical field target.
        """
        eff = self.compute_coil_efficiencies(target_R, target_Z=target_Z)
        target_Bv = float(target_Bv)
        ridge_lambda = max(float(ridge_lambda), 0.0)

        g = eff.reshape(1, -1)
        if ridge_lambda > 0.0:
            gg = float(np.dot(eff, eff))
            denom = max(gg + ridge_lambda, 1e-12)
            currents = (eff * target_Bv) / denom
        else:
            currents = np.linalg.pinv(g).dot(np.array([target_Bv], dtype=np.float64)).reshape(-1)

        self._log("\n--- ANALYTIC SOLUTION (Least Norm) ---")
        for i, val in enumerate(currents):
            name = str(self.kernel.cfg["coils"][i].get("name", f"coil_{i}"))
            self._log(f"  {name}: {float(val):.6f} MA")
        return np.asarray(currents, dtype=np.float64)

    def apply_currents(self, currents: np.ndarray) -> None:
        arr = np.asarray(currents, dtype=np.float64).reshape(-1)
        coils = self.kernel.cfg.get("coils", [])
        if arr.size != len(coils):
            raise ValueError("Current vector length mismatch with kernel coils.")
        for i, val in enumerate(arr):
            coils[i]["current"] = float(val)

    def apply_and_save(
        self,
        currents: np.ndarray,
        output_path: Optional[str] = None,
    ) -> str:
        self.apply_currents(currents)
        if output_path is None:
            repo_root = Path(__file__).resolve().parents[3]
            out_path = repo_root / "validation" / "iter_analytic_config.json"
        else:
            out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(self.kernel.cfg, f, indent=4)
        self._log(f"Saved analytic configuration: {out_path}")
        return str(out_path)


def run_analytic_solver(
    config_path: Optional[str] = None,
    *,
    target_r: float = 6.2,
    target_z: float = 0.0,
    a_minor: float = 2.0,
    ip_target_ma: float = 15.0,
    beta_p: float = 0.5,
    li: float = 0.8,
    ridge_lambda: float = 0.0,
    save_config: bool = True,
    output_config_path: Optional[str] = None,
    verbose: bool = True,
    kernel_factory: Callable[[str], Any] = FusionKernel,
) -> Dict[str, Any]:
    """
    Run analytic equilibrium solve and return deterministic summary.
    """
    repo_root = Path(__file__).resolve().parents[3]
    if config_path is None:
        preferred = repo_root / "calibration" / "iter_genetic_temp.json"
        fallback = repo_root / "validation" / "iter_validated_config.json"
        config_path = str(preferred if preferred.exists() else fallback)

    solver = AnalyticEquilibriumSolver(
        str(config_path),
        kernel_factory=kernel_factory,
        verbose=verbose,
    )
    required_bv = solver.calculate_required_Bv(
        target_r,
        a_minor,
        ip_target_ma,
        beta_p=beta_p,
        li=li,
    )
    currents = solver.solve_coil_currents(
        required_bv,
        target_r,
        target_Z=target_z,
        ridge_lambda=ridge_lambda,
    )

    written_path: Optional[str] = None
    if save_config:
        written_path = solver.apply_and_save(currents, output_path=output_config_path)
    else:
        solver.apply_currents(currents)

    names = [str(c.get("name", f"coil_{i}")) for i, c in enumerate(solver.kernel.cfg["coils"])]
    summary_currents = {name: float(currents[i]) for i, name in enumerate(names)}
    return {
        "config_path": str(config_path),
        "output_config_path": written_path,
        "target_r_m": float(target_r),
        "target_z_m": float(target_z),
        "a_minor_m": float(a_minor),
        "ip_target_ma": float(ip_target_ma),
        "required_bv_t": float(required_bv),
        "coil_currents_ma": summary_currents,
        "coil_current_l2_norm": float(np.linalg.norm(currents)),
        "max_abs_coil_current_ma": float(np.max(np.abs(currents))) if currents.size else 0.0,
    }


if __name__ == "__main__":
    run_analytic_solver()
