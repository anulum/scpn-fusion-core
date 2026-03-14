# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Full Ballooning Equation Solver
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from scpn_fusion.core.stability_mhd import QProfile


@dataclass
class BallooningEigenResult:
    """Result of solving the ballooning equation for a single (s, alpha) pair."""

    theta: np.ndarray
    xi: np.ndarray
    is_stable: bool


class BallooningEquation:
    """
    Second-order ODE for ideal MHD ballooning stability in the s-alpha model.
    """

    def __init__(self, s: float, alpha: float, theta_max: float = 20 * np.pi, n_theta: int = 2001):
        self.s = s
        self.alpha = alpha
        self.theta_max = theta_max
        self.n_theta = n_theta

    def f(self, theta: float) -> float:
        """Field-line bending term coefficient."""
        return float(1.0 + (self.s * theta - self.alpha * np.sin(theta)) ** 2)

    def g(self, theta: float) -> float:
        """Curvature drive term coefficient."""
        return float(
            self.alpha
            * (np.cos(theta) + (self.s * theta - self.alpha * np.sin(theta)) * np.sin(theta))
        )

    def solve(self) -> BallooningEigenResult:
        """
        Solve the boundary-value problem via shooting method.
        The mode is unstable if the solution xi does not cross zero (oscillate).
        If it crosses zero, it is stable (oscillatory decay).
        """

        def eqs(t: float, y: np.ndarray) -> list[float]:
            u1, u2 = y
            du1 = u2 / self.f(t)
            du2 = -self.g(t) * u1
            return [du1, du2]

        class _ZeroCrossing:
            terminal = True
            direction = -1

            def __call__(self, t: float, y: np.ndarray) -> float:
                return float(y[0])

        t_span = (0.0, self.theta_max)
        # solve_ivp without t_eval to let it stop early smoothly, or with t_eval
        # and it will return up to the event.
        sol = solve_ivp(
            eqs,
            t_span,
            [1.0, 0.0],
            events=_ZeroCrossing(),
            rtol=1e-5,
            atol=1e-5,
        )

        # If the event triggered, the solution crossed zero -> unstable
        # Thus, stable if NO events triggered.
        is_stable = len(sol.t_events[0]) == 0

        return BallooningEigenResult(
            theta=sol.t,
            xi=sol.y[0],
            is_stable=is_stable,
        )


def find_marginal_stability(
    s: float, alpha_min: float = 0.0, alpha_max: float = 2.0, tol: float = 1e-3
) -> float:
    """
    Binary search for alpha_crit at fixed shear s.
    Returns the critical alpha (first stability boundary).
    """
    amin = alpha_min

    # Check lower bound
    if s <= 0.0:
        return 0.0

    eq_min = BallooningEquation(s, amin)
    if not eq_min.solve().is_stable:
        return 0.0  # Unstable even at min alpha

    # Find a valid unstable upper bound (don't jump to the 2nd stability region)
    amax = amin + 0.1
    found_unstable = False
    for _ in range(20):
        if not BallooningEquation(s, amax).solve().is_stable:
            found_unstable = True
            break
        amax += 0.2
        if amax > 3.0:
            break

    if not found_unstable:
        # Stable everywhere up to 3.0
        return alpha_max

    while (amax - amin) > tol:
        mid = (amin + amax) / 2.0
        eq = BallooningEquation(s, mid)
        if eq.solve().is_stable:
            amin = mid
        else:
            amax = mid

    return (amin + amax) / 2.0


def compute_stability_diagram(
    s_range: np.ndarray, alpha_min: float = 0.0, alpha_max: float = 2.0
) -> np.ndarray:
    """
    Compute alpha_crit(s) for an array of shear values.
    """
    alpha_crit = np.zeros_like(s_range)
    for i, s in enumerate(s_range):
        alpha_crit[i] = find_marginal_stability(s, alpha_min, alpha_max)
    return alpha_crit


class BallooningStabilityAnalysis:
    """
    Performs ballooning stability analysis given a QProfile.
    """

    def analyze(self, q_profile: QProfile) -> np.ndarray:
        """
        Extracts (s, alpha) at each radial point and returns per-radius stability margin.
        Positive margin means stable.
        margin = alpha_crit(s) - alpha_actual
        """
        margin = np.zeros_like(q_profile.rho)
        for i in range(len(q_profile.rho)):
            s = q_profile.shear[i]
            alpha = q_profile.alpha_mhd[i]
            if s <= 0.0:
                # s <= 0 is often stable to ideal ballooning, but standard s-alpha
                # breaks down. Let's say alpha_crit = 0.0 for s=0.
                alpha_crit = 0.0
            else:
                alpha_crit = find_marginal_stability(s)

            margin[i] = alpha_crit - alpha
        return margin
