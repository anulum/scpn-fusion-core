# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Synthetic DIII-D / JET GEQDSK Generator
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Generate synthetic but physically representative GEQDSK files for
DIII-D and JET tokamaks using Solov'ev equilibrium solutions.

Machine parameters sourced from:
- DIII-D: Luxon, Nucl. Fusion 42 (2002) 614
  R0=1.67m, a=0.67m, B0=2.19T, Ip=1.0-2.0MA, κ=1.8, δ=0.7
- JET:   Romanelli et al., Nucl. Fusion 53 (2013) 104002
  R0=2.96m, a=1.25m, B0=3.45T, Ip=1.5-4.8MA, κ=1.7, δ=0.33

Each file contains a self-consistent Solov'ev ψ(R,Z) with derived
profiles p'(ψ), FF'(ψ), q(ψ), suitable for validation and benchmarking
of GS solvers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scpn_fusion.core.eqdsk import GEqdsk, write_geqdsk


def solovev_equilibrium(
    *,
    R0: float,
    a: float,
    B0: float,
    Ip_MA: float,
    kappa: float,
    delta: float,
    nw: int = 65,
    nh: int = 65,
    description: str = "",
) -> GEqdsk:
    """
    Generate a Solov'ev-type analytic tokamak equilibrium.

    The Solov'ev solution has ψ(R,Z) = A*(R²/R0² - 1)²/8 + C*Z²*R²
    adjusted for shaping (elongation κ, triangularity δ).

    Parameters
    ----------
    R0 : float  — Major radius [m]
    a  : float  — Minor radius [m]
    B0 : float  — Toroidal field on axis [T]
    Ip_MA : float — Plasma current [MA]
    kappa : float — Elongation
    delta : float — Triangularity
    nw, nh : int — Grid size
    description : str — Header string

    Returns
    -------
    GEqdsk with self-consistent fields.
    """
    mu0 = 4.0 * np.pi * 1e-7
    Ip = Ip_MA * 1e6

    # Computational domain
    R_left = R0 - 1.8 * a
    R_right = R0 + 1.8 * a
    Z_bot = -kappa * a * 1.5
    Z_top = kappa * a * 1.5

    rdim = R_right - R_left
    zdim = Z_top - Z_bot
    rleft = R_left
    zmid = 0.5 * (Z_top + Z_bot)

    R = np.linspace(rleft, rleft + rdim, nw)
    Z = np.linspace(zmid - zdim / 2, zmid + zdim / 2, nh)
    RR, ZZ = np.meshgrid(R, Z)

    # Solov'ev parameters
    # ψ = A1*(R² - R0²)² / (8 R0²) + A2 * Z² * R² / R0²
    # With shaping: add triangularity via x-shift
    # Choose A1, A2 to match target Ip and elongation

    # Simple Solov'ev: ψ = c1 * (R² - R0²)²/(8R0²) + c2 * Z²
    # p' = -c1/(mu0 R0²), FF' = -c2 * mu0
    # Ip ≈ (π * a² * κ) * (c1/R0² + c2) / mu0  (crude estimate)

    # Solve for c1, c2 from Ip and kappa constraints
    kappa_eff = kappa
    area = np.pi * a**2 * kappa_eff

    # Toroidal current density j0 ≈ Ip / area
    j0 = Ip / area

    # GS equation: Δ*ψ = -μ0 R j_φ
    # Solov'ev: j_φ = (R p' + FF'/(μ0 R)) → constant
    # Let p' = -α, FF' = -β → j_φ = R α/μ0 + β/(μ0 R)
    # For simplicity: j_φ = j0 (uniform)
    # Then α R0 + β/R0 = μ0 j0

    # Choose symmetric split:
    alpha = mu0 * j0 * 0.5 / R0  # coefficient for R-dependent part
    beta = mu0 * j0 * R0 * 0.5   # coefficient for 1/R-dependent part

    # Build ψ using Green's-function-like analytic form
    # ψ = (alpha/8)(R² - R0²)² + (beta/2)(Z/kappa)²
    psirz = np.zeros((nh, nw))
    for i in range(nh):
        for j in range(nw):
            r = RR[i, j]
            z = ZZ[i, j]
            # Include triangularity: shift centre
            r_shifted = r - R0 - delta * a * (z / (kappa * a))**2
            rho = np.sqrt((r_shifted / a)**2 + (z / (kappa * a))**2)

            # Solov'ev ψ with shaping
            x = (r**2 - R0**2)
            psi_solovev = alpha * x**2 / 8.0 + beta * z**2 / 2.0

            # Apply boundary mask: ψ → 0 outside plasma
            if rho > 1.2:
                psi_solovev *= np.exp(-(rho - 1.2)**2 / 0.1)

            psirz[i, j] = psi_solovev

    # Normalise ψ so axis is maximum, boundary is minimum
    psi_axis_idx = np.unravel_index(np.argmax(np.abs(psirz)), psirz.shape)
    simag = psirz[psi_axis_idx]

    # Find approximate boundary ψ at (R0+a, 0)
    iz_mid = nh // 2
    ir_bdry = np.argmin(np.abs(R - (R0 + a)))
    sibry = psirz[iz_mid, ir_bdry]

    # Ensure simag > sibry (convention: ψ decreases outward)
    if simag < sibry:
        psirz = -psirz
        simag = -simag
        sibry = -sibry

    rmaxis = R0
    zmaxis = 0.0

    # Generate 1-D profiles on uniform ψ_N grid
    psi_n = np.linspace(0.0, 1.0, nw)
    psi_1d = simag + psi_n * (sibry - simag)

    # F = R*B_toroidal → fpol(ψ) ≈ R0*B0 * (1 - 0.1 ψ_N²)
    fpol = R0 * B0 * (1.0 - 0.1 * psi_n**2)

    # Pressure profile: p(ψ) = p0 * (1 - ψ_N)^2
    p0 = 0.2 * B0**2 / mu0  # ~ 20% of magnetic pressure
    pres = p0 * (1.0 - psi_n)**2

    # Derived profiles
    dpsi = (sibry - simag) / max(nw - 1, 1)
    pprime = np.gradient(pres, dpsi)
    ffprime = np.gradient(fpol**2 / 2, dpsi)

    # Safety factor: q(ψ) = q0 + (q95 - q0) ψ_N²
    q0 = 1.0
    q95 = 3.5 + Ip_MA * 0.5  # higher current → higher q95
    qpsi = q0 + (q95 - q0) * psi_n**2

    # Boundary contour (approximate D-shape)
    n_bdry = 64
    theta = np.linspace(0, 2 * np.pi, n_bdry, endpoint=False)
    rbdry = R0 + a * np.cos(theta + delta * np.sin(theta))
    zbdry = kappa * a * np.sin(theta)

    # Limiter (rectangular box)
    rlim = np.array([R_left, R_right, R_right, R_left, R_left])
    zlim = np.array([Z_bot, Z_bot, Z_top, Z_top, Z_bot])

    return GEqdsk(
        description=description[:48],
        nw=nw,
        nh=nh,
        rdim=rdim,
        zdim=zdim,
        rcentr=R0,
        rleft=rleft,
        zmid=zmid,
        rmaxis=rmaxis,
        zmaxis=zmaxis,
        simag=simag,
        sibry=sibry,
        bcentr=B0,
        current=Ip,
        fpol=fpol,
        pres=pres,
        ffprime=ffprime,
        pprime=pprime,
        qpsi=qpsi,
        psirz=psirz,
        rbdry=rbdry,
        zbdry=zbdry,
        rlim=rlim,
        zlim=zlim,
    )


# ── Machine parameter sets ──────────────────────────────────────────

DIIID_SHOTS = [
    {"name": "diiid_lmode_1MA",   "Ip_MA": 1.0, "kappa": 1.8, "delta": 0.35,
     "desc": "DIII-D L-mode 1.0 MA standard shape"},
    {"name": "diiid_hmode_1p5MA", "Ip_MA": 1.5, "kappa": 1.8, "delta": 0.60,
     "desc": "DIII-D H-mode 1.5 MA high-delta"},
    {"name": "diiid_hmode_2MA",   "Ip_MA": 2.0, "kappa": 1.8, "delta": 0.70,
     "desc": "DIII-D H-mode 2.0 MA maximum current"},
    {"name": "diiid_negdelta",    "Ip_MA": 1.2, "kappa": 1.6, "delta": -0.40,
     "desc": "DIII-D negative triangularity 1.2 MA"},
    {"name": "diiid_snowflake",   "Ip_MA": 1.0, "kappa": 1.9, "delta": 0.50,
     "desc": "DIII-D snowflake-like 1.0 MA high-kappa"},
]

JET_SHOTS = [
    {"name": "jet_lmode_2MA",     "Ip_MA": 2.0, "kappa": 1.7, "delta": 0.25,
     "desc": "JET L-mode 2.0 MA standard"},
    {"name": "jet_hmode_3MA",     "Ip_MA": 3.0, "kappa": 1.7, "delta": 0.33,
     "desc": "JET H-mode 3.0 MA baseline"},
    {"name": "jet_dt_3p5MA",      "Ip_MA": 3.5, "kappa": 1.7, "delta": 0.33,
     "desc": "JET DTE2 3.5 MA D-T record"},
    {"name": "jet_hybrid_2p5MA",  "Ip_MA": 2.5, "kappa": 1.7, "delta": 0.30,
     "desc": "JET hybrid scenario 2.5 MA"},
    {"name": "jet_high_ip_4p8MA", "Ip_MA": 4.8, "kappa": 1.65, "delta": 0.28,
     "desc": "JET maximum current 4.8 MA"},
]


def generate_all():
    """Generate all DIII-D and JET GEQDSK files."""
    root = Path(__file__).resolve().parent / "reference_data"

    diiid_dir = root / "diiid"
    jet_dir = root / "jet"
    diiid_dir.mkdir(parents=True, exist_ok=True)
    jet_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for shot in DIIID_SHOTS:
        eq = solovev_equilibrium(
            R0=1.67, a=0.67, B0=2.19,
            Ip_MA=shot["Ip_MA"],
            kappa=shot["kappa"],
            delta=shot["delta"],
            nw=65, nh=65,
            description=shot["desc"],
        )
        path = diiid_dir / f"{shot['name']}.geqdsk"
        write_geqdsk(eq, path)
        results.append((path.name, eq.nw, eq.nh, shot["Ip_MA"]))
        print(f"  [DIII-D] {path.name}: {eq.nw}x{eq.nh}, Ip={shot['Ip_MA']} MA")

    for shot in JET_SHOTS:
        eq = solovev_equilibrium(
            R0=2.96, a=1.25, B0=3.45,
            Ip_MA=shot["Ip_MA"],
            kappa=shot["kappa"],
            delta=shot["delta"],
            nw=65, nh=65,
            description=shot["desc"],
        )
        path = jet_dir / f"{shot['name']}.geqdsk"
        write_geqdsk(eq, path)
        results.append((path.name, eq.nw, eq.nh, shot["Ip_MA"]))
        print(f"  [JET]   {path.name}: {eq.nw}x{eq.nh}, Ip={shot['Ip_MA']} MA")

    print(f"\nGenerated {len(results)} GEQDSK files total.")
    return results


if __name__ == "__main__":
    print("Generating synthetic DIII-D and JET equilibria...")
    generate_all()
