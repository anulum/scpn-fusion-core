#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Ideal-Ballooning Reference Runner (pyrokinetics)
"""Generate a shaped-geometry ballooning-critical-alpha reference with pyrokinetics.

Runs the published infinite-n ideal ballooning solver shipped in pyrokinetics
(``Diagnostics.ideal_ballooning_solver``, adapted from R. Gaur's
``ideal-ballooning-solver``) over a Miller local equilibrium, sweeping the
MHD ballooning drive ``alpha = -q^2 (R0/a) beta'`` at fixed magnetic shear to
locate the first-stability boundary and to detect second-stability access.

Executes inside the dedicated ``.venv-pyrokinetics`` virtual environment, NOT
the project venv: pyrokinetics pulls numpy>=2 and a large scientific stack that
is incompatible with the project pins. The exported JSON carries full
provenance (pyrokinetics version, solver source, runner SHA-256, runtime
metadata) so the acquired reference is auditable and the pedestal comparison
lane in the main venv stays deterministic against the committed artifact.

pyrokinetics is LGPL-3.0-or-later: it is used here as an unmodified library via
its public API in an isolated environment, and only the derived numerical
reference (critical-alpha values, not pyrokinetics source) is redistributed
with attribution — licence-compatible with this AGPL-3.0 project.

Physics scope: the drive-to-alpha mapping ``alpha = -q^2 (R0/a) beta'`` and the
marginal boundary are validated in the near-circular limit against the classic
s-alpha trend (critical alpha rising with magnetic shear); the shaped rows
carry the DIII-D pedestal shaping (kappa, delta and their radial gradients plus
the Shafranov shift) that opens second-stability access. This is an infinite-n
*ideal* ballooning boundary, not the full ELITE peeling-ballooning eigenvalue
that EPED runs; the honest scope boundary is recorded in the artifact.

References
----------
Gaur R. et al. (2023) *J. Plasma Phys.* 89, 905890518 (ideal-ballooning-solver).
Miller R. L. et al. (1998) *Phys. Plasmas* 5, 973 (local equilibrium).
Snyder P. B. et al. (2009) *Phys. Plasmas* 16, 056118 (EPED1 methodology).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "validation" / "reference_data" / "ballooning" / "pyrokinetics_alpha_crit.json"
)
ARTIFACT_SCHEMA = "scpn-fusion-core.pyrokinetics-ballooning-alpha-crit.v1"

# DIII-D pedestal shaping for the EPED1 Ip-scan reference (Snyder APS-DPP slide:
# kappa=1.74, delta=0.3). The radial shaping gradients and Shafranov shift are
# pedestal-representative (steep-gradient edge); they are declared here because
# the slide does not publish them.
DIIID_SHAPING = {
    "kappa": 1.74,
    "delta": 0.3,
    "s_kappa": 0.5,
    "s_delta": 0.3,
    "shift": -0.5,
    "q": 4.0,
    "rho": 0.95,
    "Rmaj_over_a": 2.49,  # R0=1.67 m / a=0.67 m (assumed DIII-D geometry)
}
# Magnetic-shear grid spanning the pedestal-relevant range.
SHEAR_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
# Near-circular self-check shaping (recovers the s-alpha trend).
CIRCULAR_SHAPING = {
    "kappa": 1.0,
    "delta": 0.0,
    "s_kappa": 0.0,
    "s_delta": 0.0,
    "shift": 0.0,
    "q": 2.0,
    "rho": 0.5,
    "Rmaj_over_a": 3.0,
}
CIRCULAR_SHEAR_GRID = (0.5, 1.0, 2.0)

GammaFn = Callable[[float], float]


def find_alpha_crit(
    gamma_fn: GammaFn,
    *,
    alpha_max: float = 8.0,
    coarse_points: int = 17,
    bisect_iterations: int = 18,
    unstable_threshold: float = 1e-6,
) -> dict[str, Any]:
    """Locate the first-stability ballooning boundary from a drive sweep.

    Scans ``alpha`` upward from zero, brackets the first stable→unstable flip,
    and bisects to the marginal drive. Records whether the surface stays stable
    up to ``alpha_max`` (second-stability access at this shear).

    Parameters
    ----------
    gamma_fn : callable
        Maps ``alpha`` to the ideal-ballooning growth rate (unstable when
        ``> unstable_threshold``). Injected so the boundary logic is testable
        without pyrokinetics.
    alpha_max, coarse_points, bisect_iterations, unstable_threshold : float
        Sweep extent, coarse bracket density, bisection depth, and the growth
        rate above which a mode counts as unstable.

    Returns
    -------
    dict
        ``alpha_crit`` (float or ``None`` when stable to ``alpha_max``),
        ``second_stability_access`` (bool), ``gamma_max`` over the sweep, and
        ``evaluations``.
    """
    if alpha_max <= 0.0:
        raise ValueError("alpha_max must be > 0")
    if int(coarse_points) < 2:
        raise ValueError("coarse_points must be at least 2")
    if int(bisect_iterations) < 1:
        raise ValueError("bisect_iterations must be at least 1")

    step = alpha_max / (int(coarse_points) - 1)
    evaluations = 0
    gamma_max = -float("inf")
    prev_alpha = 0.0
    prev_unstable = False
    for i in range(int(coarse_points)):
        alpha = i * step
        gamma = float(gamma_fn(alpha))
        evaluations += 1
        gamma_max = max(gamma_max, gamma)
        unstable = gamma > unstable_threshold
        if unstable and not prev_unstable and i > 0:
            lo, hi = prev_alpha, alpha
            for _ in range(int(bisect_iterations)):
                mid = 0.5 * (lo + hi)
                evaluations += 1
                gmid = float(gamma_fn(mid))
                gamma_max = max(gamma_max, gmid)
                if gmid > unstable_threshold:
                    hi = mid
                else:
                    lo = mid
            return {
                "alpha_crit": 0.5 * (lo + hi),
                "second_stability_access": False,
                "alpha_max": float(alpha_max),
                "gamma_max": gamma_max,
                "evaluations": evaluations,
            }
        prev_alpha = alpha
        prev_unstable = unstable
    return {
        "alpha_crit": None,
        "second_stability_access": True,
        "alpha_max": float(alpha_max),
        "gamma_max": gamma_max,
        "evaluations": evaluations,
    }


def build_report(
    diiid_rows: Sequence[dict[str, Any]],
    circular_rows: Sequence[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the schema-versioned reference artifact.

    Parameters
    ----------
    diiid_rows, circular_rows : sequence of dict
        Per-shear ``alpha_crit`` rows for the DIII-D shaping and the circular
        self-check.
    provenance : dict
        Code identity and runtime metadata.

    Returns
    -------
    dict
        The full reference payload.
    """
    circular_monotonic = _crit_non_decreasing(circular_rows)
    return {
        "schema": ARTIFACT_SCHEMA,
        "provenance": provenance,
        "scope": {
            "kind": "infinite_n_ideal_ballooning_first_stability_boundary",
            "not_included": "ELITE peeling-ballooning eigenvalue / kinetic ballooning; this is the ideal-MHD ballooning boundary only",
            "drive_definition": "alpha = -q^2 (R0/a) beta_prime (MHD ballooning parameter)",
        },
        "diiid_shaping": dict(DIIID_SHAPING),
        "diiid_alpha_crit": list(diiid_rows),
        "circular_self_check": {
            "shaping": dict(CIRCULAR_SHAPING),
            "rows": list(circular_rows),
            "alpha_crit_non_decreasing_in_shear": circular_monotonic,
        },
    }


def _crit_non_decreasing(rows: Sequence[dict[str, Any]]) -> bool:
    """True when finite ``alpha_crit`` values do not decrease with shear."""
    finite = [r["alpha_crit"] for r in rows if r.get("alpha_crit") is not None]
    return all(b >= a - 1e-9 for a, b in zip(finite, finite[1:]))


def _self_sha256() -> str:
    """SHA-256 of this runner file, for artifact provenance."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _pyrokinetics_version() -> str:
    """Installed pyrokinetics version, read from metadata without importing it."""
    from importlib.metadata import version

    return version("pyrokinetics")


def _pyrokinetics_gamma_factory(shaping: dict[str, Any], shat: float) -> GammaFn:
    """Build a growth-rate callable for one (shaping, shear) via pyrokinetics.

    Imported lazily so the module (and its pure boundary logic) stays importable
    and unit-testable without pyrokinetics installed.
    """
    from pyrokinetics import Pyro, template_dir  # type: ignore[import-not-found]
    from pyrokinetics.diagnostics import Diagnostics  # type: ignore[import-not-found]

    pyro = Pyro(gk_file=template_dir / "input.gs2", gk_code="GS2")
    lg = pyro.local_geometry
    lg.q = float(shaping["q"])
    lg.shat = float(shat)
    lg.kappa = float(shaping["kappa"])
    lg.delta = float(shaping["delta"])
    lg.s_kappa = float(shaping["s_kappa"])
    lg.s_delta = float(shaping["s_delta"])
    lg.shift = float(shaping["shift"])
    lg.rho = float(shaping["rho"]) * lg.rho.units
    lg.Rmaj = float(shaping["Rmaj_over_a"]) * lg.Rmaj.units
    bp_units = lg.beta_prime.units
    q = float(lg.q)
    rmaj = float(lg.Rmaj.magnitude)

    def gamma(alpha: float) -> float:
        lg.beta_prime = (-alpha / (q**2 * rmaj)) * bp_units
        g = Diagnostics(pyro).ideal_ballooning_solver()
        return float(getattr(g, "magnitude", g))

    return gamma


def _scan_grid(
    shaping: dict[str, Any], shears: Sequence[float], alpha_max: float
) -> list[dict[str, Any]]:
    """Run the alpha-crit finder over a shear grid for one shaping."""
    rows: list[dict[str, Any]] = []
    for shat in shears:
        gamma_fn = _pyrokinetics_gamma_factory(shaping, shat)
        result = find_alpha_crit(gamma_fn, alpha_max=alpha_max)
        rows.append({"shat": float(shat), **result})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pyrokinetics ballooning scans and write the reference artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--diiid-alpha-max", type=float, default=8.0)
    parser.add_argument("--circular-alpha-max", type=float, default=5.0)
    args = parser.parse_args(argv)

    diiid_rows = _scan_grid(DIIID_SHAPING, SHEAR_GRID, args.diiid_alpha_max)
    circular_rows = _scan_grid(CIRCULAR_SHAPING, CIRCULAR_SHEAR_GRID, args.circular_alpha_max)

    provenance = {
        "code": "pyrokinetics",
        "code_url": "https://github.com/pyro-kinetics/pyrokinetics",
        "solver": "Diagnostics.ideal_ballooning_solver (adapted from rahulgaur104/ideal-ballooning-solver)",
        "licence": "LGPL-3.0-or-later",
        "pyrokinetics_version": _pyrokinetics_version(),
        "runner_sha256": _self_sha256(),
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    report = build_report(diiid_rows, circular_rows, provenance)

    if not report["circular_self_check"]["alpha_crit_non_decreasing_in_shear"]:
        print("circular self-check failed: alpha_crit not monotonic in shear", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output} ({len(diiid_rows)} DIII-D rows, {len(circular_rows)} circular rows)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
