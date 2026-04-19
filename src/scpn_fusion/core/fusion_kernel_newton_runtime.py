# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Newton Solver Runtime Helpers
"""Shared Newton-solver option parsing and telemetry helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NewtonDispatchConfig:
    max_iter: int
    tol: float
    picard_alpha: float
    fail_on_diverge: bool
    require_gs_residual: bool
    gs_tol: float
    mu0: float
    warmup_steps: int
    newton_alpha: float
    use_newton_line_search: bool
    line_search_c: float
    max_backtracks: int
    gmres_preconditioner_mode: str
    ilu_drop_tol: float
    ilu_fill_factor: float
    gmres_fail_on_breakdown: bool
    gmres_nonconverged_budget: int | None


@dataclass
class GmresTelemetry:
    nonconverged_count: int = 0
    breakdown_count: int = 0
    last_info: int = 0

    def record(
        self,
        *,
        info: int,
        iter_idx: int,
        fail_on_breakdown: bool,
        nonconverged_budget: int | None,
    ) -> None:
        self.last_info = int(info)
        if info == 0:
            return

        if info < 0:
            self.breakdown_count += 1
            if fail_on_breakdown:
                raise RuntimeError(f"GMRES breakdown at Newton iter={iter_idx} (info={info})")
            return

        self.nonconverged_count += 1
        if nonconverged_budget is not None and self.nonconverged_count > nonconverged_budget:
            raise RuntimeError(
                "GMRES non-convergence budget exceeded: "
                f"{self.nonconverged_count}>{nonconverged_budget} "
                f"(last info={info}, iter={iter_idx})"
            )


def parse_newton_dispatch_config(cfg: dict[str, Any]) -> NewtonDispatchConfig:
    """Parse and validate Newton-dispatch options from kernel config."""
    solver_cfg = cfg["solver"]
    physics_cfg = cfg["physics"]

    max_iter = int(solver_cfg["max_iterations"])
    tol = float(solver_cfg["convergence_threshold"])
    picard_alpha = float(solver_cfg.get("relaxation_factor", 0.1))
    fail_on_diverge = bool(solver_cfg.get("fail_on_diverge", False))
    require_gs_residual = bool(solver_cfg.get("require_gs_residual", False))
    gs_tol = float(solver_cfg.get("gs_residual_threshold", tol))
    if require_gs_residual and gs_tol <= 0.0:
        raise ValueError("solver.gs_residual_threshold must be > 0")

    mu0 = float(physics_cfg["vacuum_permeability"])
    warmup_steps = min(15, max_iter // 2)
    newton_alpha = 0.5

    use_newton_line_search = bool(solver_cfg.get("newton_line_search", False))
    line_search_c = float(solver_cfg.get("newton_line_search_c", 1e-4))
    if (not np.isfinite(line_search_c)) or line_search_c <= 0.0 or line_search_c >= 1.0:
        raise ValueError("solver.newton_line_search_c must be finite and in (0, 1)")
    max_backtracks = int(solver_cfg.get("newton_line_search_max_backtracks", 6))
    if max_backtracks <= 0:
        raise ValueError("solver.newton_line_search_max_backtracks must be >= 1")

    gmres_preconditioner_mode = str(solver_cfg.get("gmres_preconditioner", "none")).strip().lower()
    if gmres_preconditioner_mode in {"", "auto", "default", "legacy"}:
        gmres_preconditioner_mode = "none"
    if gmres_preconditioner_mode == "none" and bool(
        solver_cfg.get("gmres_diagonal_preconditioner", False)
    ):
        gmres_preconditioner_mode = "diagonal"
    if gmres_preconditioner_mode not in {"none", "diagonal", "ilu"}:
        raise ValueError("solver.gmres_preconditioner must be one of {'none', 'diagonal', 'ilu'}")

    ilu_drop_tol = float(solver_cfg.get("gmres_ilu_drop_tol", 1e-4))
    ilu_fill_factor = float(solver_cfg.get("gmres_ilu_fill_factor", 8.0))
    if (not np.isfinite(ilu_drop_tol)) or ilu_drop_tol <= 0.0:
        raise ValueError("solver.gmres_ilu_drop_tol must be finite and > 0")
    if (not np.isfinite(ilu_fill_factor)) or ilu_fill_factor < 1.0:
        raise ValueError("solver.gmres_ilu_fill_factor must be finite and >= 1")

    gmres_fail_on_breakdown = bool(solver_cfg.get("gmres_fail_on_breakdown", True))
    gmres_nonconverged_budget_raw = solver_cfg.get("gmres_nonconverged_budget")
    gmres_nonconverged_budget: int | None
    if gmres_nonconverged_budget_raw is None:
        gmres_nonconverged_budget = None
    else:
        if isinstance(gmres_nonconverged_budget_raw, bool):
            raise ValueError("solver.gmres_nonconverged_budget must be an int >= 0")
        gmres_nonconverged_budget = int(gmres_nonconverged_budget_raw)
        if gmres_nonconverged_budget < 0:
            raise ValueError("solver.gmres_nonconverged_budget must be >= 0")

    return NewtonDispatchConfig(
        max_iter=max_iter,
        tol=tol,
        picard_alpha=picard_alpha,
        fail_on_diverge=fail_on_diverge,
        require_gs_residual=require_gs_residual,
        gs_tol=gs_tol,
        mu0=mu0,
        warmup_steps=warmup_steps,
        newton_alpha=newton_alpha,
        use_newton_line_search=use_newton_line_search,
        line_search_c=line_search_c,
        max_backtracks=max_backtracks,
        gmres_preconditioner_mode=gmres_preconditioner_mode,
        ilu_drop_tol=ilu_drop_tol,
        ilu_fill_factor=ilu_fill_factor,
        gmres_fail_on_breakdown=gmres_fail_on_breakdown,
        gmres_nonconverged_budget=gmres_nonconverged_budget,
    )


__all__ = [
    "GmresTelemetry",
    "NewtonDispatchConfig",
    "parse_newton_dispatch_config",
]
