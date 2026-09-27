# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Native Julia Solvers
"""
    SCPNFusionSolvers

Native Julia fixed-boundary Grad-Shafranov reference solver and flux-derived
current diagnostics. The implementation is a numerical reference surface, not
facility-grade validation evidence.
"""
module SCPNFusionSolvers

export GradShafranovCase, GradShafranovResult, case_from_toml,
    grad_shafranov_delta_star, solve_grad_shafranov,
    toroidal_current_density_from_flux, total_toroidal_current_from_flux,
    total_toroidal_current_from_flux_masked,
    total_toroidal_current_from_flux_trapezoidal

using TOML

include("physical_case.jl")

"""Native Julia Grad-Shafranov solve result."""
struct GradShafranovResult
    psi::Matrix{Float64}
    residual_history::Vector{Float64}
end

function _linspace(start::Float64, stop::Float64, count::Int)::Vector{Float64}
    count == 1 && return [start]
    step = (stop - start) / (count - 1)
    return [start + step * (i - 1) for i in 1:count]
end

function _r_grid(case::GradShafranovCase)::Tuple{Vector{Float64}, Vector{Float64}, Matrix{Float64}, Float64, Float64}
    r = _linspace(case.R_min, case.R_max, case.NR)
    z = _linspace(case.Z_min, case.Z_max, case.NZ)
    rr = Matrix{Float64}(undef, case.NZ, case.NR)
    for iz in 1:case.NZ, ir in 1:case.NR
        rr[iz, ir] = r[ir]
    end
    return r, z, rr, r[2] - r[1], z[2] - z[1]
end

"""Check the Gaussian exponent before exponentiation can hide overflow."""
function _initial_psi(case::GradShafranovCase, rr::Matrix{Float64})::Matrix{Float64}
    r_center = 0.5 * (case.R_min + case.R_max)
    _require_numerics(r_center, "initial Gaussian centre")
    exponent = -((rr .- r_center) .^ 2) ./ 0.5
    _require_numerics(exponent, "initial Gaussian exponent")
    psi = exp.(exponent) .* 0.01
    psi[1, :] .= 0.0
    psi[end, :] .= 0.0
    psi[:, 1] .= 0.0
    psi[:, end] .= 0.0
    return psi
end

"""Refuse nonfinite accepted-input arithmetic rather than hiding it in a reduction."""
function _require_numerics(value::Union{Float64,AbstractArray{Float64}}, stage::String)::Nothing
    valid = value isa Float64 ? isfinite(value) : all(isfinite, value)
    valid || error("$stage became nonfinite")
    return nothing
end

"""Build the checked reference current profile while retaining its radius policy."""
function _compute_source(case::GradShafranovCase, psi::Matrix{Float64}, rr::Matrix{Float64}, dR::Float64, dZ::Float64)::Matrix{Float64}
    psi_axis = maximum(@view psi[2:end-1, 2:end-1])
    psi_boundary = 0.0
    denom = psi_boundary - psi_axis
    if abs(denom) < 1.0e-9
        denom = denom == 0.0 ? 1.0e-9 : sign(denom) * 1.0e-9
    end

    psi_norm = (psi .- psi_axis) ./ denom
    _require_numerics(psi_norm, "normalised flux")
    psi_norm = clamp.(psi_norm, 0.0, 1.0)
    profile = ifelse.((psi_norm .>= 0.0) .& (psi_norm .< 1.0), 1.0 .- psi_norm, 0.0)
    r_safe = max.(rr, 1.0e-10)
    j_p = rr .* profile
    denominator = case.mu0 .* r_safe
    _require_numerics(denominator, "source denominator")
    all(>(0.0), denominator) || error("source denominator became zero")
    j_f = profile ./ denominator
    _require_numerics(j_p, "pressure profile")
    _require_numerics(j_f, "poloidal-current profile")
    j_raw = case.beta_mix .* j_p .+ (1.0 - case.beta_mix) .* j_f
    _require_numerics(j_raw, "current profile")
    current = sum(j_raw) * dR * dZ
    _require_numerics(current, "profile current")
    scale = case.Ip_target / max(abs(current), 1.0e-9)
    _require_numerics(scale, "current scale")
    j_phi = j_raw .* scale
    _require_numerics(j_phi, "scaled current")
    source = -case.mu0 .* rr .* j_phi
    _require_numerics(source, "GS source")
    return source
end

"""Apply one Jacobi relaxation step with checked coefficients and flux arithmetic."""
function _jacobi_step(psi::Matrix{Float64}, source::Matrix{Float64}, rr::Matrix{Float64}, dR::Float64, dZ::Float64, omega_j::Float64)::Matrix{Float64}
    psi_new = copy(psi)
    dR2 = dR * dR
    dZ2 = dZ * dZ
    a_ns = 1.0 / dZ2
    a_c = 2.0 / dR2 + 2.0 / dZ2
    _require_numerics(a_ns, "Jacobi vertical coefficient")
    _require_numerics(a_c, "Jacobi center coefficient")
    a_ns > 0.0 && a_c > 0.0 || error("Jacobi coefficient became degenerate")

    nz, nr = size(psi)
    for iz in 2:nz-1, ir in 2:nr-1
        r_safe = max(rr[iz, ir], 1.0e-10)
        denominator = 2.0 * r_safe * dR
        _require_numerics(denominator, "radial Jacobi denominator")
        denominator > 0.0 || error("radial Jacobi denominator became zero")
        a_e = 1.0 / dR2 - 1.0 / denominator
        a_w = 1.0 / dR2 + 1.0 / denominator
        _require_numerics(a_e, "east Jacobi coefficient")
        _require_numerics(a_w, "west Jacobi coefficient")
        update = (a_e * psi[iz, ir + 1] + a_w * psi[iz, ir - 1] +
            a_ns * (psi[iz - 1, ir] + psi[iz + 1, ir]) - source[iz, ir]) / a_c
        psi_new[iz, ir] = (1.0 - omega_j) * psi[iz, ir] + omega_j * update
        _require_numerics(update, "Jacobi update")
        _require_numerics(psi_new[iz, ir], "Jacobi flux")
    end
    return psi_new
end

function _validate_flux_matrix(case::GradShafranovCase, psi::Matrix{Float64})::Nothing
    _validate_case(case)
    size(psi) == (case.NZ, case.NR) || throw(ArgumentError(
        "psi shape must match the Grad-Shafranov case grid"))
    all(isfinite, psi) || throw(ArgumentError("psi must contain only finite values"))
    return nothing
end

"""Evaluate the cylindrical Grad-Shafranov operator Delta*psi on the native grid."""
function grad_shafranov_delta_star(case::GradShafranovCase, psi::Matrix{Float64})::Matrix{Float64}
    _validate_flux_matrix(case, psi)
    r, _, _, dR, dZ = _r_grid(case)
    delta_star = zeros(Float64, case.NZ, case.NR)
    dR2 = dR * dR
    dZ2 = dZ * dZ

    for iz in 2:case.NZ-1, ir in 2:case.NR-1
        d2_dR2 = (psi[iz, ir + 1] - 2.0 * psi[iz, ir] + psi[iz, ir - 1]) / dR2
        d_dR_over_R = (psi[iz, ir + 1] - psi[iz, ir - 1]) / (2.0 * dR * r[ir])
        d2_dZ2 = (psi[iz + 1, ir] - 2.0 * psi[iz, ir] + psi[iz - 1, ir]) / dZ2
        delta_star[iz, ir] = d2_dR2 - d_dR_over_R + d2_dZ2
    end
    return delta_star
end

"""Return J_phi implied by Delta*psi = -mu0 R J_phi."""
function toroidal_current_density_from_flux(case::GradShafranovCase, psi::Matrix{Float64})::Matrix{Float64}
    _validate_flux_matrix(case, psi)
    r, _, _, _, _ = _r_grid(case)
    delta_star = grad_shafranov_delta_star(case, psi)
    current_density = zeros(Float64, case.NZ, case.NR)
    for iz in 2:case.NZ-1, ir in 2:case.NR-1
        current_density[iz, ir] = -delta_star[iz, ir] / (case.mu0 * r[ir])
    end
    return current_density
end

"""Integrate J_phi implied by a flux grid over the native R-Z grid."""
function total_toroidal_current_from_flux(case::GradShafranovCase, psi::Matrix{Float64})::Float64
    _validate_flux_matrix(case, psi)
    _, _, _, dR, dZ = _r_grid(case)
    current_density = toroidal_current_density_from_flux(case, psi)
    return sum(@view current_density[2:end-1, 2:end-1]) * dR * dZ
end

"""Integrate J_phi implied by a flux grid using full-domain trapezoidal weights."""
function total_toroidal_current_from_flux_trapezoidal(case::GradShafranovCase,
    psi::Matrix{Float64})::Float64
    _validate_flux_matrix(case, psi)
    _, _, _, dR, dZ = _r_grid(case)
    current_density = toroidal_current_density_from_flux(case, psi)
    total = 0.0
    for iz in 1:case.NZ, ir in 1:case.NR
        z_weight = (iz == 1 || iz == case.NZ) ? 0.5 : 1.0
        r_weight = (ir == 1 || ir == case.NR) ? 0.5 : 1.0
        total += current_density[iz, ir] * z_weight * r_weight * dR * dZ
    end
    isfinite(total) || throw(ArgumentError(
        "trapezoidal integrated toroidal current became non-finite"))
    return total
end

"""Integrate J_phi implied by a flux grid over an explicit R-Z domain mask."""
function total_toroidal_current_from_flux_masked(case::GradShafranovCase,
    psi::Matrix{Float64}, domain_mask::AbstractMatrix{Bool})::Float64
    _validate_flux_matrix(case, psi)
    size(domain_mask) == (case.NZ, case.NR) || throw(ArgumentError(
        "toroidal current mask shape must match the Grad-Shafranov case grid"))
    any(domain_mask) || throw(ArgumentError(
        "toroidal current mask must include at least one cell"))
    _, _, _, dR, dZ = _r_grid(case)
    current_density = toroidal_current_density_from_flux(case, psi)
    total = sum(current_density[domain_mask]) * dR * dZ
    isfinite(total) || throw(ArgumentError(
        "masked integrated toroidal current became non-finite"))
    return total
end

function _max_change(a::Matrix{Float64}, b::Matrix{Float64})::Float64
    return maximum(abs.(a .- b))
end

"""Run a validated reference solve; numerical failure throws before any result is returned."""
function solve_grad_shafranov(case::GradShafranovCase)::GradShafranovResult
    _validate_case(case)
    _, _, rr, dR, dZ = _r_grid(case)
    isfinite(dR*dR) && isfinite(dZ*dZ) && dR*dR > 0.0 && dZ*dZ > 0.0 ||
        error("solver spacing arithmetic became nonfinite or degenerate")
    psi = _initial_psi(case, rr)
    _require_numerics(psi, "initial flux")
    residual_history = Float64[]

    for _ in 1:case.n_picard
        source = _compute_source(case, psi, rr, dR, dZ)
        psi_elliptic = copy(psi)
        for _ in 1:case.n_jacobi
            psi_elliptic = _jacobi_step(psi_elliptic, source, rr, dR, dZ, case.omega_j)
        end
        psi_next = (1.0 - case.alpha) .* psi .+ case.alpha .* psi_elliptic
        _require_numerics(psi_next, "Picard flux")
        residual = _max_change(psi_next, psi)
        _require_numerics(residual, "update residual")
        push!(residual_history, residual)
        psi = psi_next
    end

    psi[1, :] .= 0.0
    psi[end, :] .= 0.0
    psi[:, 1] .= 0.0
    psi[:, end] .= 0.0
    return GradShafranovResult(psi, residual_history)
end

end
