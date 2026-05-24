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
