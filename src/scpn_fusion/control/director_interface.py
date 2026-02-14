# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Director Interface
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    from director_module import DirectorModule

    DIRECTOR_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency path
    DIRECTOR_AVAILABLE = False
    DirectorModule = None

try:
    from scpn_fusion.core._rust_compat import FusionKernel, RUST_BACKEND
except ImportError:
    try:
        from scpn_fusion.core.fusion_kernel import FusionKernel

        RUST_BACKEND = False
    except ImportError as exc:  # pragma: no cover - import-guard path
        raise ImportError(
            "Unable to import FusionKernel. Run with PYTHONPATH=src "
            "or use `python -m scpn_fusion.control.director_interface`."
        ) from exc

from scpn_fusion.control.neuro_cybernetic_controller import NeuroCyberneticController


class _RuleBasedDirector:
    """Deterministic fallback director when external DIRECTOR_AI is unavailable."""

    def __init__(self, entropy_threshold: float = 0.3, history_window: int = 10) -> None:
        self.entropy_threshold = max(float(entropy_threshold), 1e-6)
        self.history_window = max(int(history_window), 1)
        self._scores: list[float] = []

    def review_action(self, prompt: str, proposed_action: str) -> tuple[bool, float]:
        del proposed_action
        m_stability = re.search(r"Stability=([A-Za-z]+)", prompt)
        stability = m_stability.group(1) if m_stability else "Unstable"

        m_entropy = re.search(r"BrainEntropy=([0-9.]+)", prompt)
        entropy = float(m_entropy.group(1)) if m_entropy else self.entropy_threshold * 2.0

        sec_score = float(np.clip(entropy / self.entropy_threshold, 0.0, 10.0))
        self._scores.append(sec_score)
        if len(self._scores) > self.history_window:
            self._scores = self._scores[-self.history_window :]

        rolling = float(np.mean(self._scores))
        approved = stability == "Stable" and rolling <= 1.0
        return bool(approved), float(sec_score)


class DirectorInterface:
    """
    Interfaces the 'Director' (Layer 16: Coherence Oversight) with the Fusion Reactor.

    Role:
    The Director does NOT control the coils (Layer 2 does that).
    The Director controls the *Controller*. It sets the strategy and monitors for "Backfire".

    Mechanism:
    1. Sample System State (Physics + Neural Activity).
    2. Format as a "Prompt" for the Director.
    3. Director calculates Entropy/Risk.
    4. If Safe: Director updates Target Parameters.
    5. If Unsafe: Director triggers corrective action.
    """

    def __init__(
        self,
        config_path: str,
        *,
        allow_fallback: bool = True,
        director: Optional[Any] = None,
        controller_factory: Callable[[str], Any] = NeuroCyberneticController,
        entropy_threshold: float = 0.3,
        history_window: int = 10,
    ) -> None:
        self.nc = controller_factory(config_path)
        if director is not None:
            self.director = director
            self.director_backend = "injected"
        elif DIRECTOR_AVAILABLE and DirectorModule is not None:
            self.director = DirectorModule(
                entropy_threshold=float(entropy_threshold),
                history_window=int(history_window),
            )
            self.director_backend = "director_module"
        elif allow_fallback:
            self.director = _RuleBasedDirector(
                entropy_threshold=float(entropy_threshold),
                history_window=int(history_window),
            )
            self.director_backend = "fallback_rule_based"
        else:
            raise ImportError(
                "Cannot initialize DirectorInterface without DIRECTOR_AI module "
                "when allow_fallback=False."
            )

        self.step_count = 0
        self.log: list[dict[str, float]] = []

    def format_state_for_director(
        self,
        t: int,
        ip: float,
        err_r: float,
        err_z: float,
        brain_activity: list[float],
    ) -> str:
        """
        Translate physical telemetry into a semantic prompt for the Director.
        """
        stability = "Stable"
        if abs(err_r) > 0.1 or abs(err_z) > 0.1:
            stability = "Unstable"
        if abs(err_r) > 0.5:
            stability = "Critical"

        neural_entropy = float(np.std(np.asarray(brain_activity, dtype=np.float64)))
        return (
            f"Time={t}, Ip={ip:.1f}, Stability={stability}, "
            f"BrainEntropy={neural_entropy:.2f}"
        )

    def run_directed_mission(
        self,
        duration: int = 100,
        *,
        use_quantum: bool = True,
        glitch_start_step: int = 50,
        glitch_std: float = 500.0,
        rng_seed: int = 42,
        save_plot: bool = True,
        output_path: str = "Director_Interface_Result.png",
        verbose: bool = True,
    ) -> dict[str, Any]:
        duration = max(int(duration), 1)
        glitch_start_step = max(int(glitch_start_step), 0)
        glitch_std = max(float(glitch_std), 0.0)
        rng = np.random.default_rng(int(rng_seed))

        if verbose:
            print("--- DIRECTOR-GHOSTED FUSION MISSION ---")
            print("Layer 16 (Director) is now overseeing Layer 2 (Neurocore).")
            print(f"Director backend: {self.director_backend}")

        self.nc.kernel.solve_equilibrium()
        self.nc.initialize_brains(use_quantum=bool(use_quantum))

        current_target_ip = 5.0
        self.log = []

        for t in range(duration):
            self.nc.kernel.cfg["physics"]["plasma_current_target"] = current_target_ip

            if t >= glitch_start_step and glitch_std > 0.0:
                self.nc.kernel.cfg["coils"][2]["current"] += float(
                    rng.normal(0.0, glitch_std)
                )

            idx_max = int(np.argmax(self.nc.kernel.Psi))
            iz, ir = np.unravel_index(idx_max, self.nc.kernel.Psi.shape)
            curr_r = float(self.nc.kernel.R[ir])
            curr_z = float(self.nc.kernel.Z[iz])

            err_r = 6.2 - curr_r
            err_z = 0.0 - curr_z

            ctrl_r = float(self.nc.brain_R.step(err_r))
            ctrl_z = float(self.nc.brain_Z.step(err_z))

            self.nc.kernel.cfg["coils"][2]["current"] += ctrl_r
            self.nc.kernel.cfg["coils"][0]["current"] -= ctrl_z
            self.nc.kernel.cfg["coils"][4]["current"] += ctrl_z

            approved = True
            if t % 5 == 0:
                brain_activity = [ctrl_r, ctrl_z]
                proposed_action = f"Increase Ip to {current_target_ip + 1.0}"
                prompt = self.format_state_for_director(
                    t, current_target_ip, err_r, err_z, brain_activity
                )
                approved, sec_score = self.director.review_action(prompt, proposed_action)

                if verbose:
                    status = "APPROVED" if approved else "DENIED"
                    print(
                        f"[Director] T={t} | State: {prompt} | Proposal: "
                        f"{proposed_action} -> {status} (SEC={sec_score:.2f})"
                    )

                if approved:
                    current_target_ip += 1.0
                else:
                    if verbose:
                        print("[Director] INTERVENTION: Reducing Power to restore Coherence.")
                    current_target_ip = max(1.0, current_target_ip - 2.0)

            self.nc.kernel.solve_equilibrium()
            self.log.append(
                {
                    "t": float(t),
                    "Ip": float(current_target_ip),
                    "Err_R": float(err_r),
                    "Director_Intervention": float(0 if approved else 1),
                }
            )

        self.step_count = duration
        plot_error: Optional[str] = None
        plot_saved = False
        if save_plot:
            try:
                self.visualize(output_path=output_path)
                plot_saved = True
            except Exception as exc:  # pragma: no cover - backend-dependent
                plot_error = f"{exc.__class__.__name__}: {exc}"

        err = np.array([x["Err_R"] for x in self.log], dtype=np.float64)
        interventions = np.array(
            [x["Director_Intervention"] for x in self.log], dtype=np.float64
        )
        return {
            "backend": self.director_backend,
            "steps": int(duration),
            "final_target_ip": float(current_target_ip),
            "mean_abs_err_r": float(np.mean(np.abs(err))) if err.size > 0 else 0.0,
            "intervention_count": int(np.sum(interventions)),
            "plot_saved": bool(plot_saved),
            "plot_error": plot_error,
        }

    def visualize(self, output_path: str = "Director_Interface_Result.png") -> str:
        t = [x["t"] for x in self.log]
        ip = [x["Ip"] for x in self.log]
        err = [x["Err_R"] for x in self.log]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_title("Director-Mediated Fusion Control")
        ax1.plot(t, ip, "b-", label="Plasma Current Target (Director Controlled)")
        ax1.set_ylabel("Current (MA)", color="b")
        ax1.tick_params(axis="y", labelcolor="b")

        ax2 = ax1.twinx()
        ax2.plot(t, err, "r--", label="Radial Error (Instability)")
        ax2.set_ylabel("Error (m)", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

        plt.axvline(50, color="k", linestyle=":", label="External Disturbance")
        fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.85))
        plt.tight_layout()
        plt.savefig(output_path)
        return str(output_path)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    cfg = repo_root / "iter_config.json"
    di = DirectorInterface(str(cfg))
    di.run_directed_mission()
