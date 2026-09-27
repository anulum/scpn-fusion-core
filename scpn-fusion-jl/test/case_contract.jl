# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Julia public physical case contract tests
using SCPNFusionSolvers
using Test
using TOML
using SHA

@testset "shared physical case corpus" begin
    folder = joinpath(@__DIR__, "../../validation/polyglot/case_contract")
    manifest = TOML.parsefile(joinpath(folder, "manifest.toml"))
    @test manifest["schema"] == "scpn.physical-case-corpus.v1"
    for fixture in manifest["case"]
        @testset "$(fixture["name"])" begin
            path = joinpath(folder, fixture["path"])
            @test bytes2hex(sha256(read(path))) == fixture["sha256"]
            if fixture["valid"]
                case = case_from_toml(path)
                for (key,value) in fixture["expected"]
                    @test getproperty(case, Symbol(key)) == value
                end
                if fixture["solve"]
                    result = solve_grad_shafranov(case)
                    @test size(result.psi) == (case.NZ,case.NR)
                    @test all(isfinite,result.psi)
                end
            else
                @test_throws Union{ArgumentError,TOML.ParserError} case_from_toml(path)
            end
        end
    end
end

@testset "direct physical case admission" begin
    @test_throws ArgumentError GradShafranovCase(; NR=true)
    @test_throws ArgumentError GradShafranovCase(; NR=17.0)
    @test_throws ArgumentError GradShafranovCase(; R_min=true)
    @test_throws ArgumentError GradShafranovCase(; Ip_target=9007199254740993)
    @test_throws ArgumentError GradShafranovCase(; n_jacobi=big(2)^100)
    @test_throws ArgumentError GradShafranovCase(1.,3.,-1.2,1.2,true,17,1e6,1.2566370614359173e-6,8,16,0.1,2/3,0.5)
    @test_throws ArgumentError GradShafranovCase(; NR=1026)
    @test_throws ArgumentError GradShafranovCase(; NR=10,NZ=10,n_picard=10000,n_jacobi=101)
    @test_throws ArgumentError GradShafranovCase(; R_min=0.)
    @test_throws ArgumentError GradShafranovCase(; R_max=nextfloat(0.1))
    @test_throws ArgumentError GradShafranovCase(; Z_min=-floatmax(Float64),Z_max=floatmax(Float64))
    @test_throws ArgumentError GradShafranovCase(; Z_min=0.,Z_max=nextfloat(0.))
end

@testset "accepted-input numerical failure" begin
    case = GradShafranovCase(; R_min=1e200,R_max=2e200,NR=17,NZ=17,n_picard=8,n_jacobi=16)
    @test_throws ErrorException solve_grad_shafranov(case)
end


@testset "hidden intermediate overflow" begin
    for (case, diagnostic) in [
        (GradShafranovCase(; R_min=1e154,R_max=1e155,n_picard=1,n_jacobi=1), "initial Gaussian"),
        (GradShafranovCase(; R_min=1e154,R_max=3e154,n_picard=1,n_jacobi=1), "initial Gaussian"),
        (GradShafranovCase(; R_min=1.0,R_max=1.01,Z_min=-0.01,Z_max=0.01,Ip_target=1e308,n_picard=1,n_jacobi=1), "scaled current")]
        failure = try
            solve_grad_shafranov(case)
            nothing
        catch error
            error
        end
        @test failure isa ErrorException
        @test failure isa ErrorException && occursin(diagnostic, failure.msg)
    end
end
