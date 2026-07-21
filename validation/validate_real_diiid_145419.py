# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — GS validation against the real DIII-D shot 145419 EFIT reconstruction
"""Validate the SI Grad-Shafranov machinery against **real DIII-D data** (shot 145419, t=2100 ms).

Reference: ``g145419.02100`` — a real EFIT equilibrium reconstruction (129×129, Ip=1.508 MA)
openly redistributed by General Atomics in the ``omas`` package (see
``validation/reference_data/diiid/real_public/PROVENANCE.json``). EFIT output is itself a
model reconstruction constrained by measurements, not a raw measurement — stated honestly.

Two steps (the Milestone-B pattern, now on real data):

1. **Operator satisfaction** (pure evaluation, no solve): does the real ψ map satisfy *our*
   discretised GS operator with the g-file's own ``p'``/``FF'``?  Metric: interior residual
   ``Δ*ψ − S(ψ)`` relative to the ``max|Δ*ψ|`` scale (expected: discretisation-error level).
2. **Fixed-boundary reproduction on a coil-free sub-domain**: the g-file's 129² domain
   *contains PF-coil cross-sections* (empirically mapped as vacuum cells where ``|Δ*ψ|`` is
   large — F-coils at R≈0.85–0.88 m, divertor coils at R≈2.26–2.50 m); a full-domain re-solve
   with our zero-source vacuum model therefore diverges from the reference (documented
   honest negative: deep RMS ≈ 26 %). On a sub-rectangle verified free of external current
   (Dirichlet = real ψ on its edge, our source with the real profiles inside, normalisation
   levels anchored to the real axis/boundary values — reproduction mode, not blind
   prediction), the machinery reproduces the real interior to ≈ 0.1 % deep RMS.
3. **Full-domain reproduction with a measured external source**: outside the confined plasma
   (above the X-point, connected to the axis) the source is pinned to the *measured* ``Δ*ψ``
   (which is exactly ``−μ₀RJφ`` of the coils/legs/private flux); inside, our ``p'``/``FF'``
   model with Ip renormalised to the measured plasma-region current.

Fixed-point structure of that map (measured by the executable lanes below, 2026-07-22):
the map has (at least) two fixed points. Warm-started iteration reaches the **true branch**
under BOTH Anderson(m=8) (26 iterations) and plain relaxed Picard (ω = 0.3 or 0.5; slower,
no early stop within 200 iterations) — Anderson is the accelerator here, not the branch
selector. The second fixed point is the **zero-plasma absorbing state**: because the ψ_N
anchors are fixed reference values, a start whose interior carries no plasma current has its
``tanh`` LCFS cutoff exactly saturated, zero model current flows, and the map is exactly
stationary (deep RMS ≈ 127 % from the reference). The archived dev-time observation of a
"relaxed Picard → ≈ 127 % wrong attractor" matches this absorbing state to three digits; its
earlier interpretation (H-mode pedestal ``p''`` bistability) is therefore RETIRED and the
identification with the zero-plasma state recorded instead — data kept, reading corrected.

Disclosure (prominent, per distinct-eye finding D1): the reproduction lanes warm-start from
the EFIT ψ and anchor normalisation to reference values — they are fixed-point
consistency/reproduction checks, NOT independent branch selection. The cold-start lane
(``init="external"``) demonstrates precisely that limitation (it lands in the absorbing
state). Genuine cold-start branch selection needs iterate-derived anchors and an Ip ramp —
that is the Rung-1 *predictive* solver, validated on its own synthetic/FreeGS lane; running
it on this real shot requires the PF coil currents, which g-files do not carry (Rung 4,
facility data). All lanes are **regenerated on every run** (D2); the artefact JSON embeds
runtime provenance — reference SHA-256 and exact package versions (D3) — and is written with
a trailing newline so a rerun is hook-drift-free (D4).

COCOS note (explicit, not silent): the g-file stores ψ *descending* from axis to boundary
(``ψ_axis < ψ_bnd``, both negative).  The package convention has ψ *peaked* at the axis, so the
field AND the profile derivatives are sign-flipped together (``ψ → −ψ``, ``p' → −p'``,
``FF' → −FF'`` — an exact GS symmetry), and results are reported in the flipped frame.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from freeqdsk import geqdsk

from scpn_fusion.core.jax_free_boundary_gs import general_gs_source
from scpn_fusion.core.jax_free_boundary_gs_implicit import _laplacian_star

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "artifacts" / "real_diiid_145419"


def _find_gfile() -> Path:
    """Locate ``g145419.02100`` — the local cache first, else the installed omas package.

    The reference file is openly redistributed inside the ``omas`` PyPI package
    (``omas/samples/g145419.02100``), so ``pip install omas`` is sufficient to reproduce this
    validation from scratch — no private data access is required.
    """
    local = REPO / "validation" / "reference_data" / "diiid" / "real_public" / "g145419.02100"
    if local.exists():
        return local
    import omas

    packaged = Path(omas.__file__).parent / "samples" / "g145419.02100"
    if packaged.exists():
        return packaged
    raise FileNotFoundError(
        "g145419.02100 not found - install the 'omas' package (pip install omas), which "
        "openly ships this real DIII-D EFIT reconstruction in omas/samples/"
    )


GFILE = _find_gfile()


def load_gfile(path: Path) -> dict[str, np.ndarray | float | int]:
    with open(path) as fh:
        g = geqdsk.read(fh)
    nx, ny = int(g["nx"]), int(g["ny"])
    R = np.linspace(float(g["rleft"]), float(g["rleft"]) + float(g["rdim"]), nx)
    Z = np.linspace(
        float(g["zmid"]) - float(g["zdim"]) / 2, float(g["zmid"]) + float(g["zdim"]) / 2, ny
    )
    return {
        "R": R,
        "Z": Z,
        # geqdsk ψ is [nx, ny] = [R, Z] → package convention (NZ, NR); sign-flip to ψ-peaked
        "psi": -np.asarray(g["psi"], dtype=np.float64).T,
        "psin": np.linspace(0.0, 1.0, nx),
        "pprime": -np.asarray(g["pprime"], dtype=np.float64),
        "ffprime": -np.asarray(g["ffprime"], dtype=np.float64),
        "psi_axis": -float(g["simagx"]),
        "psi_bnd": -float(g["sibdry"]),
        "ip": float(g["cpasma"]),
    }


def operator_residual(d: dict) -> dict[str, float]:
    """Step 1 — evaluate our discrete GS residual on the real ψ with the real profiles."""
    psi = jnp.asarray(d["psi"])
    R, Z = jnp.asarray(d["R"]), jnp.asarray(d["Z"])
    src = general_gs_source(
        psi,
        R,
        jnp.asarray(d["psi_axis"]),
        jnp.asarray(d["psi_bnd"]),
        jnp.asarray(d["psin"]),
        jnp.asarray(d["pprime"]),
        jnp.asarray(d["ffprime"]),
    )
    lap = _laplacian_star(psi, R, R[1] - R[0], Z[1] - Z[0])
    res = np.asarray(lap - src)
    lap_scale = float(np.max(np.abs(np.asarray(lap)[2:-2, 2:-2])))
    psin_map = (np.asarray(d["psi"]) - d["psi_axis"]) / (d["psi_bnd"] - d["psi_axis"])
    deep = psin_map < 0.8
    deep[:2, :] = deep[-2:, :] = False
    deep[:, :2] = deep[:, -2:] = False
    return {
        "interior_max_rel": float(np.max(np.abs(res[2:-2, 2:-2]))) / lap_scale,
        "interior_rms_rel": float(np.sqrt(np.mean(res[2:-2, 2:-2] ** 2))) / lap_scale,
        "deep_max_rel": float(np.max(np.abs(res[deep]))) / lap_scale,
        "deep_rms_rel": float(np.sqrt(np.mean(res[deep] ** 2))) / lap_scale,
    }


def build_delta_star(R: np.ndarray, Z: np.ndarray) -> sp.csc_matrix:
    """5-point Δ* = ∂rr − (1/R)∂r + ∂zz with identity rows on the boundary (Dirichlet)."""
    nr, nz = R.size, Z.size
    dr, dz = R[1] - R[0], Z[1] - Z[0]
    n = nz * nr
    A = sp.lil_matrix((n, n))
    for iz in range(nz):
        for ir in range(nr):
            k = iz * nr + ir
            if iz in (0, nz - 1) or ir in (0, nr - 1):
                A[k, k] = 1.0
                continue
            A[k, k] = -2.0 / dr**2 - 2.0 / dz**2
            A[k, k - 1] = 1.0 / dr**2 + 1.0 / (2.0 * dr * R[ir])
            A[k, k + 1] = 1.0 / dr**2 - 1.0 / (2.0 * dr * R[ir])
            A[k, k - nr] = 1.0 / dz**2
            A[k, k + nr] = 1.0 / dz**2
    return A.tocsc()


def subdomain_reproduction(
    d: dict, n_iter: int = 400, omega: float = 0.7, tol: float = 1.0e-11
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Step 2 — fixed-boundary reproduction on an empirically verified coil-free sub-domain.

    The box (R ∈ [1.00, 2.24], Z ∈ [−1.15, 1.15]) contains the confined deep plasma but none
    of the external-current (coil) cells that live inside the full g-file domain; membership
    is *asserted*, not assumed. Dirichlet = real ψ on the box edge; the source uses the real
    profiles with normalisation levels anchored to the real axis/boundary, and the Picard
    iteration is **warm-started from the real ψ** — a reproduction/consistency lane by
    construction (disclosed; the basin evidence lives in the full-domain cold-start lane).
    Returns (ψ_fit, ψ_real_sub, R_sub, Z_sub, deep_mask, iterations).
    """
    R, Z, psi_real = d["R"], d["Z"], d["psi"]
    psin_map = (psi_real - d["psi_axis"]) / (d["psi_bnd"] - d["psi_axis"])
    lap = np.asarray(
        _laplacian_star(jnp.asarray(psi_real), jnp.asarray(R), R[1] - R[0], Z[1] - Z[0])
    )
    coil_cells = (psin_map >= 1.0) & (np.abs(lap) > 0.05 * np.max(np.abs(lap[2:-2, 2:-2])))
    ir0, ir1 = np.searchsorted(R, 1.00), np.searchsorted(R, 2.24)
    iz0, iz1 = np.searchsorted(Z, -1.15), np.searchsorted(Z, 1.15)
    box = np.s_[iz0 : iz1 + 1, ir0 : ir1 + 1]
    if coil_cells[box].any():
        raise RuntimeError("sub-domain contains external-current cells — box must be re-chosen")
    Rs, Zs = R[ir0 : ir1 + 1], Z[iz0 : iz1 + 1]
    ps_real = psi_real[box]
    nz, nr = Zs.size, Rs.size
    lu = spla.splu(build_delta_star(Rs, Zs))
    mask_bnd = np.zeros((nz, nr), dtype=bool)
    mask_bnd[0, :] = mask_bnd[-1, :] = mask_bnd[:, 0] = mask_bnd[:, -1] = True
    psi = ps_real.copy()
    for it in range(n_iter):
        src = np.asarray(
            general_gs_source(
                jnp.asarray(psi),
                jnp.asarray(Rs),
                jnp.asarray(d["psi_axis"]),
                jnp.asarray(d["psi_bnd"]),
                jnp.asarray(d["psin"]),
                jnp.asarray(d["pprime"]),
                jnp.asarray(d["ffprime"]),
            )
        ).copy()
        rhs = src.reshape(-1)
        rhs[mask_bnd.reshape(-1)] = ps_real[mask_bnd]
        psi_new = lu.solve(rhs).reshape(nz, nr)
        step = float(np.max(np.abs(psi_new - psi)))
        psi = (1.0 - omega) * psi + omega * psi_new
        if step < tol:
            break
    sub_psin = psin_map[box]
    deep = sub_psin < 0.8
    deep[:2, :] = deep[-2:, :] = False
    deep[:, :2] = deep[:, -2:] = False
    return psi, ps_real, Rs, Zs, deep, it + 1


def full_domain_reproduction(
    d: dict,
    m_depth: int = 8,
    n_iter: int = 200,
    tol: float = 1.0e-11,
    init: str = "reference",
    accel: str = "anderson",
    picard_omega: float = 0.5,
    external_source: str = "measured",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Step 3 — full 129² reproduction: measured external source + Anderson-accelerated model.

    The plasma region is the connected-to-axis component of ``ψ_N < 1`` restricted above the
    X-point (found as the min-``|∇ψ|`` near-separatrix cell below the axis); everywhere else
    the source is the measured ``Δ*ψ`` (coils, legs, private flux — exactly ``−μ₀RJφ`` there).
    The plasma model source is Ip-renormalised to the measured plasma-region current each
    iteration.

    Disclosed prominently: with ``init="reference"`` the iteration is **warm-started from the
    EFIT ψ itself**, and the normalisation anchors (ψ_axis, ψ_bnd, plasma-region Ip) are
    reference-derived throughout — this lane is a fixed-point *consistency/reproduction*
    check, not blind branch selection. The independent basin evidence is the
    ``init="external"`` lane: the start is the external-source-only solve (coils/legs flux,
    **zero plasma current** in the interior), so the plasma flux must be built up by the
    iteration itself.

    ``accel="picard"`` (with ``picard_omega``) measures the map's convergence without
    Anderson (measured: the warm-started Picard also reaches the true branch, only slower —
    Anderson contributes acceleration, not branch selection); ``external_source="zero"`` is
    the executable honest-negative lane exposing the domain-wide error of pretending the
    external-current cells are vacuum. Returns (ψ_fit, plasma_mask, deep_mask, iterations).
    """
    from scipy import ndimage

    R, Z, psi_real = d["R"], d["Z"], d["psi"]
    nz, nr = Z.size, R.size
    psin_map = (psi_real - d["psi_axis"]) / (d["psi_bnd"] - d["psi_axis"])
    rrg, zzg = np.meshgrid(R, Z)
    gz, gr = np.gradient(psi_real, Z[1] - Z[0], R[1] - R[0])
    g2 = gz**2 + gr**2
    iz_ax, ir_ax = np.unravel_index(np.argmax(psi_real[2:-2, 2:-2]), (nz - 4, nr - 4))
    iz_ax += 2
    ir_ax += 2
    sep = (np.abs(psin_map - 1.0) < 0.02) & (zzg < Z[iz_ax] - 0.3)
    iz_x, _ir_x = np.unravel_index(np.argmin(np.where(sep, g2, np.inf)), g2.shape)
    lab, _ = ndimage.label((psin_map < 1.0) & (zzg > Z[iz_x]))
    plasma = lab == lab[iz_ax, ir_ax]

    lap_real = np.asarray(
        _laplacian_star(jnp.asarray(psi_real), jnp.asarray(R), R[1] - R[0], Z[1] - Z[0])
    )
    lu = spla.splu(build_delta_star(R, Z))
    mask_bnd = np.zeros((nz, nr), dtype=bool)
    mask_bnd[0, :] = mask_bnd[-1, :] = mask_bnd[:, 0] = mask_bnd[:, -1] = True
    mu0 = 4.0e-7 * np.pi
    dA = float((R[1] - R[0]) * (Z[1] - Z[0]))
    ip_real_plasma = float(np.sum(-lap_real[plasma] / (mu0 * rrg[plasma])) * dA)

    external = lap_real if external_source == "measured" else np.zeros_like(lap_real)

    def step_map(x: np.ndarray) -> np.ndarray:
        psi = x.reshape(nz, nr)
        src_model = np.asarray(
            general_gs_source(
                jnp.asarray(psi),
                jnp.asarray(R),
                jnp.asarray(d["psi_axis"]),
                jnp.asarray(d["psi_bnd"]),
                jnp.asarray(d["psin"]),
                jnp.asarray(d["pprime"]),
                jnp.asarray(d["ffprime"]),
            )
        ).copy()
        ipm = float(np.sum(-src_model[plasma] / (mu0 * rrg[plasma])) * dA)
        scale = ip_real_plasma / ipm if ipm else 1.0
        src = np.where(plasma, src_model * scale, external)
        rhs = src.reshape(-1).copy()
        rhs[mask_bnd.reshape(-1)] = psi_real[mask_bnd]
        return np.asarray(lu.solve(rhs))

    if init == "reference":
        x = psi_real.reshape(-1).copy()
    elif init == "external":
        # Cold/neutral start: the external-source-only field (coils/legs/private flux, ZERO
        # plasma current inside the plasma mask) — the plasma flux must be built by the
        # iteration itself, so this lane probes basin selection, not just consistency.
        rhs0 = np.where(plasma, 0.0, lap_real).reshape(-1).copy()
        rhs0[mask_bnd.reshape(-1)] = psi_real[mask_bnd]
        x = np.asarray(lu.solve(rhs0))
    else:
        raise ValueError(f"unknown init {init!r} (use 'reference' or 'external')")
    hist_x: list[np.ndarray] = []
    hist_f: list[np.ndarray] = []
    for it in range(n_iter):
        f = step_map(x) - x
        if accel == "picard":
            x_new = x + picard_omega * f
        else:
            hist_x.append(x.copy())
            hist_f.append(f.copy())
            if len(hist_x) > m_depth:
                hist_x.pop(0)
                hist_f.pop(0)
            if len(hist_f) > 1:
                d_f = np.stack([hist_f[i + 1] - hist_f[i] for i in range(len(hist_f) - 1)], axis=1)
                d_x = np.stack([hist_x[i + 1] - hist_x[i] for i in range(len(hist_x) - 1)], axis=1)
                gamma, *_ = np.linalg.lstsq(d_f, f, rcond=None)
                x_new = x + f - (d_x + d_f) @ gamma
            else:
                x_new = x + 0.5 * f
        if not np.all(np.isfinite(x_new)):  # rank-deficient history guard → damped Picard
            x_new = x + 0.3 * f
        step = float(np.max(np.abs(x_new - x)))
        x = x_new
        if step < tol:
            break
    psi_fit = x.reshape(nz, nr)
    deep = (psin_map < 0.8) & plasma
    deep[:2, :] = deep[-2:, :] = False
    deep[:, :2] = deep[:, -2:] = False
    return psi_fit, plasma, deep, it + 1


def _provenance() -> dict:
    """Runtime-computed provenance: reference hash + the exact package versions used."""
    import freeqdsk
    import omas
    import scipy

    return {
        "reference_file": GFILE.name,
        "reference_sha256": hashlib.sha256(GFILE.read_bytes()).hexdigest(),
        "reference_source": (
            "omas PyPI package, omas/samples/ (gafusion/omas, MIT licence); real DIII-D EFIT "
            "reconstruction, shot 145419, t=2100 ms; see "
            "validation/reference_data/diiid/real_public/PROVENANCE.json"
        ),
        "packages": {
            "omas": omas.__version__,
            "freeqdsk": freeqdsk.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "jax": jax.__version__,
        },
        "generator": "validation/validate_real_diiid_145419.py",
        "pinned_environment": "requirements/full.txt (hash-pinned) for exact reproduction",
    }


def _full_domain_metrics(
    d: dict, psi_fit: np.ndarray, plasma: np.ndarray, deep: np.ndarray, iters: int, span: float
) -> dict[str, float | int]:
    diff = psi_fit - d["psi"]
    pl_i = plasma.copy()
    pl_i[:2, :] = pl_i[-2:, :] = False
    pl_i[:, :2] = pl_i[:, -2:] = False
    return {
        "iterations": iters,
        "deep_rms_rel_span": float(np.sqrt(np.mean(diff[deep] ** 2))) / span,
        "deep_max_rel_span": float(np.max(np.abs(diff[deep]))) / span,
        "plasma_rms_rel_span": float(np.sqrt(np.mean(diff[pl_i] ** 2))) / span,
        "axis_value_rel_err": abs(float(np.max(psi_fit[2:-2, 2:-2])) - d["psi_axis"]) / span,
        "global_max_rel_span": float(np.max(np.abs(diff))) / span,
    }


def main() -> None:
    d = load_gfile(GFILE)
    span = abs(d["psi_axis"] - d["psi_bnd"])
    print(
        f"g145419.02100: {d['Z'].size}x{d['R'].size}, Ip={d['ip'] / 1e6:.3f} MA, span={span:.4f} Wb"
    )

    op = operator_residual(d)
    print("STEP 1 — operator satisfaction (real psi + real profiles vs our discrete GS):")
    for k, v in op.items():
        print(f"  {k}: {v:.3e}")

    psi_fit, ps_real, Rs, Zs, deep, iters = subdomain_reproduction(d)
    span = abs(d["psi_axis"] - d["psi_bnd"])
    diff = psi_fit - ps_real
    metrics = {
        "picard_iterations": iters,
        "deep_rms_rel_span": float(np.sqrt(np.mean(diff[deep] ** 2))) / span,
        "deep_max_rel_span": float(np.max(np.abs(diff[deep]))) / span,
        "interior_rms_rel_span": float(np.sqrt(np.mean(diff[2:-2, 2:-2] ** 2))) / span,
        "axis_value_rel_err": abs(float(np.max(psi_fit[2:-2, 2:-2])) - d["psi_axis"]) / span,
    }
    print(f"STEP 2 — coil-free sub-domain reproduction ({iters} Picard iterations):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")

    psi_full, plasma, deep_f, iters_f = full_domain_reproduction(d)
    full_metrics = _full_domain_metrics(d, psi_full, plasma, deep_f, iters_f, span)
    full_metrics["anderson_iterations"] = full_metrics.pop("iterations")
    print("STEP 3 — full-domain w/ measured external source, Anderson(m=8), warm start:")
    for k, v in full_metrics.items():
        print(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")

    # STEP 3b — the map's fixed-point structure, measured (regenerated every run):
    # cold start (external-only field, zero plasma current) lands in the zero-plasma
    # ABSORBING state — the reproduction map cannot select the branch from a plasma-free
    # start (fixed ψ_N anchors ⇒ the tanh cutoff is exactly saturated ⇒ exactly stationary).
    psi_cold, _pl_c, deep_c, iters_c = full_domain_reproduction(d, init="external")
    cold_metrics = _full_domain_metrics(d, psi_cold, plasma, deep_c, iters_c, span)
    print("STEP 3b — COLD start (external-only field) -> zero-plasma absorbing state:")
    for k, v in cold_metrics.items():
        print(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")

    # Warm-started plain relaxed Picard reaches the TRUE branch (slower than Anderson —
    # acceleration, not branch selection, is what Anderson contributes on this map).
    psi_n2, _pl2, deep_n2, iters_n2 = full_domain_reproduction(d, accel="picard")
    picard_metrics = _full_domain_metrics(d, psi_n2, plasma, deep_n2, iters_n2, span)
    print("STEP 3c — relaxed Picard (omega=0.5), warm start -> true branch, slower:")
    print(f"  iters={iters_n2}  deep_rms_rel_span: {picard_metrics['deep_rms_rel_span']:.4g}")

    # Executable honest-negative lane (regenerated each run, not an archived constant):
    psi_n1, _pl1, deep_n1, iters_n1 = full_domain_reproduction(d, external_source="zero")
    neg_zero = _full_domain_metrics(d, psi_n1, plasma, deep_n1, iters_n1, span)
    print("NEGATIVE — zero external source (coil cells pretended vacuum):")
    print(f"  deep_rms_rel_span: {neg_zero['deep_rms_rel_span']:.4g}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "psi_fusion_145419_fulldomain.npz",
        psi_fusion=psi_full,
        psi_real=d["psi"],
        R_grid=d["R"],
        Z_grid=d["Z"],
        plasma_mask=plasma,
        deep_mask=deep_f,
    )
    np.savez_compressed(
        OUT_DIR / "psi_fusion_145419_subdomain.npz",
        psi_fusion=psi_fit,
        psi_real=ps_real,
        R_grid=Rs,
        Z_grid=Zs,
        deep_mask=deep,
    )
    map_structure = {
        "regenerated_each_run": True,
        "zero_plasma_absorbing_state": {
            "settings": "init='external' (external-source-only start, zero plasma current); "
            "identical outcome under Anderson and Picard, and at n_iter=400",
            "metrics": cold_metrics,
            "cause": "the psi_N anchors are FIXED reference values, so a plasma-free start "
            "has its tanh LCFS cutoff exactly saturated -> zero model current -> the map is "
            "exactly stationary; the reproduction map cannot select the branch from a "
            "plasma-free start (that requires the predictive solver's iterate-derived "
            "anchors + Ip ramp, which on real data needs PF coil currents - Rung 4)",
        },
        "picard_warm_reaches_true_branch": {
            "settings": "relaxed Picard omega=0.5 (omega=0.3 probed identically), warm "
            "start, measured external source; no early stop within 200 iterations",
            "metrics": picard_metrics,
            "note": "Anderson(m=8) contributes ACCELERATION on this map (26 vs >200 "
            "iterations), not branch selection",
        },
    }
    honest_negatives = {
        "regenerated_each_run": True,
        "zero_external_source": {
            "settings": "Anderson(m=8), warm start, external_source='zero'",
            "metrics": neg_zero,
            "cause": "the 129x129 g-file domain contains PF-coil cross-sections (empirically "
            "mapped external-current cells); a zero-source vacuum model is wrong there and "
            "the elliptic inverse propagates the mismatch domain-wide",
        },
        "archived_dev_observations": {
            "note": "development-time observations (session claude_2026-07-21T2226) measured "
            "deep RMS ~0.26 (zero source, different settings) and ~1.27 for a 'relaxed "
            "Picard wrong attractor'; the 1.27 matches the zero-plasma absorbing state to "
            "three digits, so its original interpretation (H-mode pedestal p'' bistability) "
            "is RETIRED and superseded by the map_structure evidence above - data kept, "
            "reading corrected 2026-07-22",
        },
    }
    record = {
        "provenance": _provenance(),
        "disclosure": (
            "reproduction lanes are warm-started from the EFIT psi and use reference-derived "
            "normalisation anchors (psi_axis, psi_bnd, plasma-region Ip) - they are "
            "consistency/reproduction checks, not blind prediction or independent branch "
            "selection; the cold-start lane demonstrates that limitation explicitly"
        ),
        "operator": op,
        "subdomain_reproduction": metrics,
        "full_domain_reproduction": full_metrics,
        "full_domain_cold_start": cold_metrics,
        "map_structure": map_structure,
        "honest_negatives": honest_negatives,
    }
    out_json = OUT_DIR / "real_145419_validation.json"
    out_json.write_text(json.dumps(record, indent=2) + "\n")  # trailing newline: no hook drift
    print(f"artefacts -> {OUT_DIR}")


if __name__ == "__main__":
    main()
