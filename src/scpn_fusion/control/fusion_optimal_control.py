# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Fusion Optimal Control
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel

# --- MISSION PARAMETERS ---
TARGET_R = 6.0
TARGET_Z = 0.0
SHOT_STEPS = 50


def _normalize_bounds(bounds: Tuple[float, float], name: str) -> Tuple[float, float]:
    lo = float(bounds[0])
    hi = float(bounds[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError(f"{name} must be finite with lower < upper.")
    return lo, hi


class OptimalController:
    """
    MIMO controller using response-matrix inversion with bounded actuation.
    """

    def __init__(
        self,
        config_file: str,
        *,
        kernel_factory: Callable[[str], Any] = FusionKernel,
        verbose: bool = True,
        correction_limit: float = 5.0,
        coil_current_limits: Tuple[float, float] = (-40.0, 40.0),
        current_target_limits: Tuple[float, float] = (5.0, 16.0),
    ) -> None:
        self.kernel = kernel_factory(config_file)
        self.verbose = bool(verbose)
        self.n_coils = len(self.kernel.cfg["coils"])
        self.coil_names = [str(c["name"]) for c in self.kernel.cfg["coils"]]
        self.response_matrix = np.zeros((2, self.n_coils), dtype=np.float64)
        self.correction_limit = max(float(correction_limit), 1e-9)
        self.coil_current_limits = _normalize_bounds(
            coil_current_limits, "coil_current_limits"
        )
        self.current_target_limits = _normalize_bounds(
            current_target_limits, "current_target_limits"
        )
        self.history: Dict[str, list[float]] = {
            "t": [],
            "R_axis": [],
            "Z_axis": [],
            "Ip": [],
            "error_norm": [],
            "max_abs_delta_i": [],
            "max_abs_coil_current": [],
        }

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def identify_system(self, perturbation: float = 0.5) -> None:
        """
        Perturb each coil and measure plasma-axis response to build Jacobian.
        """
        self._log("[OptControl] Identifying System Response Matrix...")
        self.kernel.solve_equilibrium()
        base_r, base_z = self.get_plasma_pos()
        self._log(f"  Base Position: R={base_r:.3f}, Z={base_z:.3f}")

        p = max(float(perturbation), 1e-9)
        for i in range(self.n_coils):
            orig_i = float(self.kernel.cfg["coils"][i].get("current", 0.0))

            self.kernel.cfg["coils"][i]["current"] = orig_i + p
            self.kernel.solve_equilibrium()
            pos_plus = self.get_plasma_pos()

            self.kernel.cfg["coils"][i]["current"] = orig_i - p
            self.kernel.solve_equilibrium()
            pos_minus = self.get_plasma_pos()

            self.kernel.cfg["coils"][i]["current"] = orig_i
            self.kernel.solve_equilibrium()

            d_r = float((pos_plus[0] - pos_minus[0]) / (2.0 * p))
            d_z = float((pos_plus[1] - pos_minus[1]) / (2.0 * p))
            self.response_matrix[0, i] = d_r
            self.response_matrix[1, i] = d_z
            self._log(f"  Coil {self.coil_names[i]}: dR/dI={d_r:.4f}, dZ/dI={d_z:.4f}")

        self._log("[OptControl] System Identification Complete.")

    def get_plasma_pos(self) -> np.ndarray:
        """Return current plasma-axis position [R, Z]."""
        idx_max = int(np.argmax(self.kernel.Psi))
        iz, ir = np.unravel_index(idx_max, self.kernel.Psi.shape)
        return np.array([self.kernel.R[ir], self.kernel.Z[iz]], dtype=np.float64)

    def compute_optimal_correction(
        self,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
        regularization_limit: float = 1e-2,
    ) -> np.ndarray:
        """
        Solve Error = J * Delta_I using damped pseudoinverse.
        """
        cur = np.asarray(current_pos, dtype=np.float64).reshape(2)
        tgt = np.asarray(target_pos, dtype=np.float64).reshape(2)
        error = tgt - cur

        u, s, vt = np.linalg.svd(self.response_matrix, full_matrices=False)
        cut = max(float(regularization_limit), 0.0)
        s_inv = np.zeros_like(s, dtype=np.float64)
        nz = s > cut
        s_inv[nz] = 1.0 / s[nz]
        j_inv = vt.T @ np.diag(s_inv) @ u.T
        delta_currents = np.asarray(j_inv @ error, dtype=np.float64)
        return np.clip(delta_currents, -self.correction_limit, self.correction_limit)

    def _apply_corrections(self, delta_currents: np.ndarray, gain: float) -> None:
        lo, hi = self.coil_current_limits
        g = float(gain)
        for i in range(self.n_coils):
            old = float(self.kernel.cfg["coils"][i].get("current", 0.0))
            upd = old + g * float(delta_currents[i])
            self.kernel.cfg["coils"][i]["current"] = float(np.clip(upd, lo, hi))

    def run_optimal_shot(
        self,
        shot_steps: int = SHOT_STEPS,
        target_r: float = TARGET_R,
        target_z: float = TARGET_Z,
        gain: float = 0.8,
        ip_start_ma: float = 10.0,
        ip_span_ma: float = 5.0,
        identify_first: bool = False,
        save_plot: bool = True,
        output_path: str = "Optimal_Control_Result.png",
    ) -> Dict[str, Any]:
        self._log("\n--- INITIATING OPTIMAL CONTROL SHOT ---")
        if identify_first:
            self.identify_system()

        steps = max(int(shot_steps), 1)
        target_vec = np.array([float(target_r), float(target_z)], dtype=np.float64)
        self.history = {k: [] for k in self.history}

        self.kernel.solve_equilibrium()
        lo_ip, hi_ip = self.current_target_limits
        for t in range(steps):
            frac = float(t) / float(max(steps, 1))
            target_ip = float(np.clip(ip_start_ma + ip_span_ma * frac, lo_ip, hi_ip))
            self.kernel.cfg.setdefault("physics", {})["plasma_current_target"] = target_ip

            curr_pos = self.get_plasma_pos()
            d_i = self.compute_optimal_correction(curr_pos, target_vec)
            self._apply_corrections(d_i, gain=float(gain))
            self.kernel.solve_equilibrium()

            err = float(np.linalg.norm(target_vec - curr_pos))
            max_abs_delta_i = float(np.max(np.abs(d_i))) if d_i.size > 0 else 0.0
            max_abs_coil_current = float(
                np.max(
                    np.abs(
                        np.asarray(
                            [
                                float(c.get("current", 0.0))
                                for c in self.kernel.cfg.get("coils", [])
                            ],
                            dtype=np.float64,
                        )
                    )
                )
            )
            self.history["t"].append(float(t))
            self.history["R_axis"].append(float(curr_pos[0]))
            self.history["Z_axis"].append(float(curr_pos[1]))
            self.history["Ip"].append(target_ip)
            self.history["error_norm"].append(err)
            self.history["max_abs_delta_i"].append(max_abs_delta_i)
            self.history["max_abs_coil_current"].append(max_abs_coil_current)

            self._log(
                f"Step {t}: R={curr_pos[0]:.3f} (Tgt {target_r}), "
                f"Z={curr_pos[1]:.3f} (Tgt {target_z}) | Err={err:.4f} | "
                f"Max dI={max_abs_delta_i:.2f}"
            )

        plot_saved = False
        plot_error: Optional[str] = None
        if save_plot:
            plot_saved, plot_error = self.plot_telemetry(output_path=output_path)

        r_arr = np.asarray(self.history["R_axis"], dtype=np.float64)
        z_arr = np.asarray(self.history["Z_axis"], dtype=np.float64)
        e_arr = np.asarray(self.history["error_norm"], dtype=np.float64)
        di_arr = np.asarray(self.history["max_abs_delta_i"], dtype=np.float64)
        coil_arr = np.asarray(self.history["max_abs_coil_current"], dtype=np.float64)
        return {
            "steps": int(steps),
            "final_target_ip_ma": float(self.history["Ip"][-1]) if self.history["Ip"] else 0.0,
            "final_axis_r": float(r_arr[-1]) if r_arr.size else 0.0,
            "final_axis_z": float(z_arr[-1]) if z_arr.size else 0.0,
            "mean_abs_r_error": float(np.mean(np.abs(r_arr - float(target_r))))
            if r_arr.size
            else 0.0,
            "mean_abs_z_error": float(np.mean(np.abs(z_arr - float(target_z))))
            if z_arr.size
            else 0.0,
            "mean_error_norm": float(np.mean(e_arr)) if e_arr.size else 0.0,
            "max_abs_delta_i": float(np.max(di_arr)) if di_arr.size else 0.0,
            "max_abs_coil_current": float(np.max(coil_arr)) if coil_arr.size else 0.0,
            "plot_saved": bool(plot_saved),
            "plot_error": plot_error,
        }

    def plot_telemetry(
        self,
        output_path: str = "Optimal_Control_Result.png",
    ) -> Tuple[bool, Optional[str]]:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.set_title("Optimal Position Control (SVD-MIMO)")
            ax1.plot(self.history["t"], self.history["R_axis"], "b-o", label="R Axis")
            ax1.plot(self.history["t"], self.history["Z_axis"], "r-s", label="Z Axis")
            ax1.axhline(TARGET_R, color="b", linestyle="--", alpha=0.5)
            ax1.axhline(TARGET_Z, color="r", linestyle="--", alpha=0.5)
            ax1.legend()
            ax1.grid(True)

            ax2.set_title("Final Plasma State")
            if hasattr(self.kernel, "RR") and hasattr(self.kernel, "ZZ"):
                ax2.contour(self.kernel.RR, self.kernel.ZZ, self.kernel.Psi, levels=20, colors="k")
            if hasattr(self.kernel, "J_phi"):
                ax2.imshow(
                    self.kernel.J_phi,
                    extent=[1, 9, -5, 5],
                    origin="lower",
                    cmap="hot",
                    alpha=0.5,
                )
            for c in self.kernel.cfg.get("coils", []):
                r = float(c.get("r", 0.0))
                z = float(c.get("z", 0.0))
                cur = float(c.get("current", 0.0))
                ax2.plot(r, z, "rx" if cur > 0 else "bx", markersize=8)

            plt.tight_layout()
            plt.savefig(output_path)
            plt.close(fig)
            self._log(f"Analysis saved: {output_path}")
            return True, None
        except Exception as exc:
            return False, str(exc)


def run_optimal_control(
    config_file: Optional[str] = None,
    shot_steps: int = SHOT_STEPS,
    target_r: float = TARGET_R,
    target_z: float = TARGET_Z,
    seed: int = 42,
    save_plot: bool = True,
    output_path: str = "Optimal_Control_Result.png",
    verbose: bool = True,
    kernel_factory: Callable[[str], Any] = FusionKernel,
    coil_current_limits: Tuple[float, float] = (-40.0, 40.0),
    current_target_limits: Tuple[float, float] = (5.0, 16.0),
) -> Dict[str, Any]:
    """
    Run bounded optimal-control shot and return deterministic summary.
    """
    seed_int = int(seed)
    if config_file is None:
        repo_root = Path(__file__).resolve().parents[3]
        config_file = str(repo_root / "iter_config.json")

    pilot = OptimalController(
        str(config_file),
        kernel_factory=kernel_factory,
        verbose=verbose,
        coil_current_limits=coil_current_limits,
        current_target_limits=current_target_limits,
    )
    pilot.identify_system()
    summary = pilot.run_optimal_shot(
        shot_steps=shot_steps,
        target_r=target_r,
        target_z=target_z,
        save_plot=save_plot,
        output_path=output_path,
    )
    summary["seed"] = seed_int
    summary["config_path"] = str(config_file)
    return summary


if __name__ == "__main__":
    run_optimal_control()
