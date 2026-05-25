# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Julia Solver Tests
using SCPNFusionSolvers
using Test

@testset "fixed-boundary Grad-Shafranov Picard/Jacobi" begin
    case = GradShafranovCase(; R_min=1.0, R_max=3.0, Z_min=-1.2, Z_max=1.2,
        NR=17, NZ=17, Ip_target=1.0e6, n_picard=8, n_jacobi=16)
    result = solve_grad_shafranov(case)

    @test size(result.psi) == (17, 17)
    @test all(isfinite, result.psi)
    @test length(result.residual_history) == 8
    @test maximum(abs.(result.psi[1, :])) < 1.0e-14
    @test maximum(abs.(result.psi[end, :])) < 1.0e-14
    @test maximum(abs.(result.psi[:, 1])) < 1.0e-14
    @test maximum(abs.(result.psi[:, end])) < 1.0e-14
    @test maximum(abs.(result.psi .- reverse(result.psi, dims=1))) < 1.0e-14
    @test maximum(abs.(result.psi[2:end-1, 2:end-1])) > 1.0e-6
end

@testset "case validation" begin
    @test_throws ArgumentError GradShafranovCase(; NR=2)
    @test_throws ArgumentError GradShafranovCase(; alpha=0.0)
    @test_throws ArgumentError GradShafranovCase(; omega_j=2.0)
    @test_throws ArgumentError GradShafranovCase(; beta_mix=1.5)
end

@testset "operator-current closure" begin
    case = GradShafranovCase(; R_min=1.0, R_max=3.0, Z_min=-1.0, Z_max=1.0,
        NR=17, NZ=19, Ip_target=1.0e6, n_picard=2, n_jacobi=2)
    z = range(case.Z_min, case.Z_max; length=case.NZ)
    dR = (case.R_max - case.R_min) / (case.NR - 1)
    dZ = (case.Z_max - case.Z_min) / (case.NZ - 1)
    coeff = -0.25
    psi = [coeff * z[iz]^2 for iz in 1:case.NZ, _ in 1:case.NR]

    delta_star = grad_shafranov_delta_star(case, psi)
    current_density = toroidal_current_density_from_flux(case, psi)
    total_current = total_toroidal_current_from_flux(case, psi)

    expected_total = 0.0
    for iz in 2:case.NZ-1, ir in 2:case.NR-1
        expected_j = -2.0 * coeff / (case.mu0 * (case.R_min + (ir - 1) * dR))
        @test abs(delta_star[iz, ir] - 2.0 * coeff) < 1.0e-12
        @test abs(current_density[iz, ir] - expected_j) < 1.0e-6
        expected_total += expected_j * dR * dZ
    end
    @test abs((total_current - expected_total) / expected_total) < 1.0e-12

    radial_coeff = 0.03125
    vertical_coeff = -0.125
    r = range(case.R_min, case.R_max; length=case.NR)
    psi_radial = [radial_coeff * r[ir]^4 + vertical_coeff * z[iz]^2
                  for iz in 1:case.NZ, ir in 1:case.NR]
    delta_star_radial = grad_shafranov_delta_star(case, psi_radial)
    current_density_radial = toroidal_current_density_from_flux(case, psi_radial)

    for iz in 2:case.NZ-1, ir in 2:case.NR-1
        expected_delta = 8.0 * radial_coeff * r[ir]^2 + 2.0 * vertical_coeff -
                         2.0 * radial_coeff * dR^2
        expected_j = -expected_delta / (case.mu0 * r[ir])
        @test abs(delta_star_radial[iz, ir] - expected_delta) < 1.0e-12
        @test abs(current_density_radial[iz, ir] - expected_j) < 1.0e-6
    end
end
