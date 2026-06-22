# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Analytic Solver
"""Analytic vertical-field equilibrium helper for coil-current initialisation."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray
from scpn_fusion.fallback_telemetry import record_fallback_event

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

FusionKernel: type[Any]
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


def shafranov_bv(
    r_geo: float,
    a_min: float,
    ip_ma: float,
    *,
    beta_p: float = 0.5,
    li: float = 0.8,
) -> float:
    """Required vertical field from the Shafranov radial-force balance.

    Canonical free-function reference for the ``shafranov_bv`` dispatch kernel
    (:mod:`scpn_fusion.core._multi_compat`). The Rust tier
    (``scpn_fusion_rs.shafranov_bv``) and this NumPy tier are bit-exact
    interchangeable for the returned field. :meth:`AnalyticEquilibriumSolver.calculate_required_Bv`
    delegates here so the physics lives in exactly one place.

    Parameters
    ----------
    r_geo : float
        Plasma geometric major radius :math:`R_0` [m]; must be strictly positive.
    a_min : float
        Plasma minor radius :math:`a` [m]; must be strictly positive.
    ip_ma : float
        Plasma current :math:`I_p` [MA]; must be strictly positive.
    beta_p : float, optional
        Poloidal beta :math:`\\beta_p`, by default 0.5.
    li : float, optional
        Internal inductance :math:`l_i`, by default 0.8.

    Returns
    -------
    float
        Required vertical field :math:`B_v` [T], negative for positive
        :math:`I_p` (field points downward).

    Raises
    ------
    ValueError
        If ``r_geo``, ``a_min`` or ``ip_ma`` are not strictly positive.

    Notes
    -----
    From radial force balance of a large-aspect-ratio tokamak [1]_:

    .. math::

        B_v = -\\frac{\\mu_0 I_p}{4\\pi R_0}
        \\left[\\ln\\!\\frac{8 R_0}{a} + \\beta_p + \\frac{l_i}{2} - \\frac{3}{2}\\right]

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press, 2011, §3.6.
    """
    r = float(r_geo)
    a = float(a_min)
    ip = float(ip_ma)
    beta = float(beta_p)
    inductance = float(li)
    if r <= 0.0 or a <= 0.0 or ip <= 0.0:
        raise ValueError("r_geo, a_min and ip_ma must be > 0.")

    mu0 = 4.0 * np.pi * 1e-7
    ip_amp = ip * 1e6
    term_log = float(np.log(8.0 * r / a))
    term_physics = beta + (inductance / 2.0) - 1.5
    return float(-((mu0 * ip_amp) / (4.0 * np.pi * r)) * (term_log + term_physics))


def solve_coil_currents(
    green_func: FloatArray | Sequence[float],
    target_bv: float,
    *,
    ridge_lambda: float = 0.0,
) -> FloatArray:
    r"""Least-norm coil currents for a desired vertical field.

    Canonical free-function reference for the ``solve_coil_currents`` dispatch
    kernel (:mod:`scpn_fusion.core._multi_compat`); numerically equivalent to the
    Rust tier (``scpn_fusion_rs.solve_coil_currents``) for both the plain
    minimum-norm and the ridge-regularised solve. The agreement is tolerance-aware
    (not bit-exact): the Green's-norm reduction :math:`\sum_j g_j^2` is summed
    sequentially in Rust but via ``numpy.dot`` here, which can differ by a unit in
    the last place. :meth:`AnalyticEquilibriumSolver.solve_coil_currents` computes
    the per-coil efficiencies and then delegates the linear solve here.

    Parameters
    ----------
    green_func : array_like
        Per-coil vertical-field efficiency :math:`\partial B_z/\partial I`
        [T/MA]; must be non-empty and finite.
    target_bv : float
        Required vertical field :math:`B_v` [T]; must be finite.
    ridge_lambda : float, optional
        Tikhonov regularisation added to :math:`G G^\top`, by default 0.0 (plain
        minimum norm). Negative values are clamped to zero.

    Returns
    -------
    numpy.ndarray
        Minimum-norm coil currents [MA] satisfying :math:`G \cdot I \approx B_v`.

    Raises
    ------
    ValueError
        If ``green_func`` is empty or non-finite, if ``target_bv`` is non-finite,
        or if the unregularised Green's norm is too small for a stable solve.

    Notes
    -----
    For the underdetermined :math:`1 \times N` system :math:`G I = B_v` the
    minimum-norm solution is :math:`I = G^\top (G G^\top + \lambda)^{-1} B_v`,
    which for a row vector reduces to
    :math:`I_i = g_i B_v / (\sum_j g_j^2 + \lambda)`. The direct form (rather than
    a pseudo-inverse) is used so the field is bit-identical to the Rust tier.
    """
    eff = np.asarray(green_func, dtype=np.float64).reshape(-1)
    if eff.size == 0:
        raise ValueError("green_func must be non-empty.")
    if not np.all(np.isfinite(eff)):
        raise ValueError("green_func must contain only finite values.")
    target = float(target_bv)
    if not np.isfinite(target):
        raise ValueError("target_bv must be finite.")
    lam_raw = float(ridge_lambda)
    if not np.isfinite(lam_raw):
        raise ValueError("ridge_lambda must be finite.")

    lam = max(lam_raw, 0.0)
    gg = float(np.dot(eff, eff))
    if lam > 0.0:
        denom = max(gg + lam, 1e-12)
    else:
        if gg < 1e-20:
            raise ValueError("green_func norm is too small for a stable solve.")
        denom = gg
    return np.asarray(eff * (target / denom), dtype=np.float64)


class AnalyticEquilibriumSolver:
    """Analytic vertical-field target and least-norm coil-current solve."""

    def __init__(
        self,
        config_path: str,
        *,
        kernel_factory: Callable[[str], Any] = FusionKernel,
        verbose: bool = True,
    ) -> None:
        """Instantiate the kernel from the config and record verbosity."""
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
        """Estimate the vertical field from Shafranov radial-force balance."""
        R_geo = float(R_geo)
        Ip_MA = float(Ip_MA)
        Bv = shafranov_bv(R_geo, a_min, Ip_MA, beta_p=beta_p, li=li)

        self._log("--- SHAFRANOV EQUILIBRIUM CHECK ---")
        self._log(f"Target Radius: {R_geo:.3f} m")
        self._log(f"Plasma Current: {Ip_MA:.3f} MA")
        self._log(f"Required Vertical Field (Bv): {Bv:.6f} Tesla")
        return Bv

    def compute_coil_efficiencies(
        self,
        target_R: float,
        *,
        target_Z: float = 0.0,
    ) -> FloatArray:
        """Compute dBz/dI per coil at target location using kernel vacuum-field map."""
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
    ) -> FloatArray:
        """Solve least-norm coil currents for desired vertical field target."""
        eff = self.compute_coil_efficiencies(target_R, target_Z=target_Z)
        currents = solve_coil_currents(eff, target_Bv, ridge_lambda=ridge_lambda)

        self._log("\n--- ANALYTIC SOLUTION (Least Norm) ---")
        for i, val in enumerate(currents):
            name = str(self.kernel.cfg["coils"][i].get("name", f"coil_{i}"))
            self._log(f"  {name}: {float(val):.6f} MA")
        return currents

    def apply_currents(self, currents: FloatArray) -> None:
        """Write a coil-current vector into the solver kernel configuration."""
        arr = np.asarray(currents, dtype=np.float64).reshape(-1)
        coils = self.kernel.cfg.get("coils", [])
        if arr.size != len(coils):
            raise ValueError("Current vector length mismatch with kernel coils.")
        for i, val in enumerate(arr):
            coils[i]["current"] = float(val)

    def apply_and_save(
        self,
        currents: FloatArray,
        output_path: Optional[str] = None,
    ) -> str:
        """Apply coil currents and persist the resulting kernel configuration."""
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


def _resolve_default_config_path(
    repo_root: Path,
    *,
    allow_validation_fallback: bool = True,
) -> tuple[str, str, bool]:
    """Resolve the default analytic solver config with explicit fallback policy."""
    preferred = repo_root / "calibration" / "iter_genetic_temp.json"
    fallback = repo_root / "validation" / "iter_validated_config.json"

    if preferred.exists():
        return str(preferred), "preferred_default", False

    if fallback.exists():
        if not allow_validation_fallback:
            raise FileNotFoundError(
                "Preferred default config is missing and validation fallback is disabled: "
                f"{preferred}"
            )
        record_fallback_event(
            "analytic_solver",
            "default_config_validation_fallback",
            context={
                "preferred_config": str(preferred),
                "fallback_config": str(fallback),
            },
        )
        logger.warning(
            "Preferred analytic config missing; using validation fallback: %s",
            fallback,
        )
        return str(fallback), "validation_fallback_default", True

    raise FileNotFoundError(
        f"No default analytic config found. Checked:\n- {preferred}\n- {fallback}"
    )


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
    allow_validation_fallback: bool = True,
    verbose: bool = True,
    kernel_factory: Callable[[str], Any] = FusionKernel,
) -> Dict[str, Any]:
    """Run analytic equilibrium solve and return deterministic summary."""
    repo_root = Path(__file__).resolve().parents[3]
    config_source = "explicit"
    fallback_used = False
    if config_path is None:
        config_path, config_source, fallback_used = _resolve_default_config_path(
            repo_root,
            allow_validation_fallback=allow_validation_fallback,
        )

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
        "config_source": str(config_source),
        "fallback_used": bool(fallback_used),
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
