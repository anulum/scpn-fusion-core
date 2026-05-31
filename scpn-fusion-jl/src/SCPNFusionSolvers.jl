# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Native Julia Solvers
module SCPNFusionSolvers

export GradShafranovCase, GradShafranovResult, case_from_toml,
    grad_shafranov_delta_star, solve_grad_shafranov,
    toroidal_current_density_from_flux, total_toroidal_current_from_flux,
    total_toroidal_current_from_flux_masked,
    total_toroidal_current_from_flux_trapezoidal

using TOML

const DEFAULT_MU0 = 4.0e-7 * pi

"""Fixed-boundary Grad-Shafranov Picard/Jacobi case definition."""
struct GradShafranovCase
    R_min::Float64
    R_max::Float64
    Z_min::Float64
    Z_max::Float64
    NR::Int
    NZ::Int
    Ip_target::Float64
    mu0::Float64
    n_picard::Int
    n_jacobi::Int
    alpha::Float64
    omega_j::Float64
    beta_mix::Float64
end

function GradShafranovCase(; R_min::Real=0.1, R_max::Real=2.0,
    Z_min::Real=-1.5, Z_max::Real=1.5, NR::Integer=33, NZ::Integer=33,
    Ip_target::Real=1.0e6, mu0::Real=DEFAULT_MU0, n_picard::Integer=80,
    n_jacobi::Integer=200, alpha::Real=0.1, omega_j::Real=2.0 / 3.0,
    beta_mix::Real=0.5)
    case = GradShafranovCase(Float64(R_min), Float64(R_max), Float64(Z_min), Float64(Z_max),
        Int(NR), Int(NZ), Float64(Ip_target), Float64(mu0), Int(n_picard), Int(n_jacobi),
        Float64(alpha), Float64(omega_j), Float64(beta_mix))
    _validate_case(case)
    return case
end

"""Native Julia Grad-Shafranov solve result."""
struct GradShafranovResult
    psi::Matrix{Float64}
    residual_history::Vector{Float64}
end

function _validate_case(case::GradShafranovCase)::Nothing
    finite_values = (case.R_min, case.R_max, case.Z_min, case.Z_max, case.Ip_target,
        case.mu0, case.alpha, case.omega_j, case.beta_mix)
    if any(value -> !isfinite(value), finite_values)
        throw(ArgumentError("Grad-Shafranov case contains a non-finite scalar"))
    end
    case.R_max > case.R_min || throw(ArgumentError("R_max must exceed R_min"))
    case.Z_max > case.Z_min || throw(ArgumentError("Z_max must exceed Z_min"))
    case.NR >= 3 || throw(ArgumentError("NR must be at least 3"))
    case.NZ >= 3 || throw(ArgumentError("NZ must be at least 3"))
    case.mu0 > 0.0 || throw(ArgumentError("mu0 must be positive"))
    case.n_picard >= 1 || throw(ArgumentError("n_picard must be at least 1"))
    case.n_jacobi >= 1 || throw(ArgumentError("n_jacobi must be at least 1"))
    0.0 < case.alpha <= 1.0 || throw(ArgumentError("alpha must be in (0, 1]"))
    0.0 < case.omega_j < 2.0 || throw(ArgumentError("omega_j must be in (0, 2)"))
    0.0 <= case.beta_mix <= 1.0 || throw(ArgumentError("beta_mix must be in [0, 1]"))
    return nothing
end

function case_from_toml(path::AbstractString)::GradShafranovCase
    data = TOML.parsefile(path)
    case_data = get(data, "grad_shafranov", data)
    return GradShafranovCase(; R_min=case_data["R_min"], R_max=case_data["R_max"],
        Z_min=case_data["Z_min"], Z_max=case_data["Z_max"], NR=case_data["NR"],
        NZ=case_data["NZ"], Ip_target=case_data["Ip_target"],
        mu0=get(case_data, "mu0", DEFAULT_MU0), n_picard=case_data["n_picard"],
        n_jacobi=case_data["n_jacobi"], alpha=case_data["alpha"],
        omega_j=case_data["omega_j"], beta_mix=case_data["beta_mix"])
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

function _initial_psi(case::GradShafranovCase, rr::Matrix{Float64})::Matrix{Float64}
    r_center = 0.5 * (case.R_min + case.R_max)
    psi = exp.(-((rr .- r_center) .^ 2) ./ 0.5) .* 0.01
    psi[1, :] .= 0.0
    psi[end, :] .= 0.0
    psi[:, 1] .= 0.0
    psi[:, end] .= 0.0
    return psi
end

function _compute_source(case::GradShafranovCase, psi::Matrix{Float64}, rr::Matrix{Float64}, dR::Float64, dZ::Float64)::Matrix{Float64}
    psi_axis = maximum(@view psi[2:end-1, 2:end-1])
    psi_boundary = 0.0
    denom = psi_boundary - psi_axis
    if abs(denom) < 1.0e-9
        denom = denom == 0.0 ? 1.0e-9 : sign(denom) * 1.0e-9
    end

    psi_norm = clamp.((psi .- psi_axis) ./ denom, 0.0, 1.0)
    profile = ifelse.((psi_norm .>= 0.0) .& (psi_norm .< 1.0), 1.0 .- psi_norm, 0.0)
    r_safe = max.(rr, 1.0e-10)
    j_p = rr .* profile
    j_f = profile ./ (case.mu0 .* r_safe)
    j_raw = case.beta_mix .* j_p .+ (1.0 - case.beta_mix) .* j_f
    current = sum(j_raw) * dR * dZ
    scale = case.Ip_target / max(abs(current), 1.0e-9)
    j_phi = j_raw .* scale
    return -case.mu0 .* rr .* j_phi
end

function _jacobi_step(psi::Matrix{Float64}, source::Matrix{Float64}, rr::Matrix{Float64}, dR::Float64, dZ::Float64, omega_j::Float64)::Matrix{Float64}
    psi_new = copy(psi)
    dR2 = dR * dR
    dZ2 = dZ * dZ
    a_ns = 1.0 / dZ2
    a_c = 2.0 / dR2 + 2.0 / dZ2

    nz, nr = size(psi)
    for iz in 2:nz-1, ir in 2:nr-1
        r_safe = max(rr[iz, ir], 1.0e-10)
        a_e = 1.0 / dR2 - 1.0 / (2.0 * r_safe * dR)
        a_w = 1.0 / dR2 + 1.0 / (2.0 * r_safe * dR)
        update = (a_e * psi[iz, ir + 1] + a_w * psi[iz, ir - 1] +
            a_ns * (psi[iz - 1, ir] + psi[iz + 1, ir]) - source[iz, ir]) / a_c
        psi_new[iz, ir] = (1.0 - omega_j) * psi[iz, ir] + omega_j * update
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

function solve_grad_shafranov(case::GradShafranovCase)::GradShafranovResult
    _validate_case(case)
    _, _, rr, dR, dZ = _r_grid(case)
    psi = _initial_psi(case, rr)
    residual_history = Float64[]

    for _ in 1:case.n_picard
        source = _compute_source(case, psi, rr, dR, dZ)
        psi_elliptic = copy(psi)
        for _ in 1:case.n_jacobi
            psi_elliptic = _jacobi_step(psi_elliptic, source, rr, dR, dZ, case.omega_j)
        end
        psi_next = (1.0 - case.alpha) .* psi .+ case.alpha .* psi_elliptic
        push!(residual_history, _max_change(psi_next, psi))
        psi = psi_next
    end

    psi[1, :] .= 0.0
    psi[end, :] .= 0.0
    psi[:, 1] .= 0.0
    psi[:, end] .= 0.0
    return GradShafranovResult(psi, residual_history)
end

end
