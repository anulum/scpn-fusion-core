# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Julia typed physical case admission

const DEFAULT_MU0 = 4.0e-7 * pi

"""
    GradShafranovCase

Validated fixed-boundary physical case: R/Z bounds in metres, target toroidal
current in amperes (signed), vacuum permeability in H/m and dimensionless
iteration/relaxation controls. Both positional and keyword construction reject
implicit numeric-kind conversion. Grid counts are 3:1025, iteration counts
1:10000 and total point iterations at most 100000000. Limits describe software
admission, not numerical accuracy or permission to execute that workload.
"""
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

    """Construct from delivered typed values before any narrowing or allocation."""
    function GradShafranovCase(R_min, R_max, Z_min, Z_max, NR, NZ,
        Ip_target, mu0, n_picard, n_jacobi, alpha, omega_j, beta_mix)
        case = new(_case_real(R_min, "R_min"), _case_real(R_max, "R_max"),
            _case_real(Z_min, "Z_min"), _case_real(Z_max, "Z_max"),
            _case_count(NR, "NR"), _case_count(NZ, "NZ"),
            _case_real(Ip_target, "Ip_target"), _case_real(mu0, "mu0"),
            _case_count(n_picard, "n_picard"), _case_count(n_jacobi, "n_jacobi"),
            _case_real(alpha, "alpha"), _case_real(omega_j, "omega_j"),
            _case_real(beta_mix, "beta_mix"))
        _validate_case(case)
        return case
    end
end

"""Construct a typed physical case from explicit keyword values or API defaults."""
function GradShafranovCase(; R_min=0.1, R_max=2.0, Z_min=-1.5, Z_max=1.5,
    NR=33, NZ=33, Ip_target=1.0e6, mu0=DEFAULT_MU0, n_picard=80,
    n_jacobi=200, alpha=0.1, omega_j=2.0 / 3.0, beta_mix=0.5)
    return GradShafranovCase(R_min, R_max, Z_min, Z_max, NR, NZ, Ip_target,
        mu0, n_picard, n_jacobi, alpha, omega_j, beta_mix)
end

"""Accept a finite Float64 or signed64 integer exactly representable in binary64."""
function _case_real(value, field::String)::Float64
    value isa Bool && throw(ArgumentError("$field must not be Boolean"))
    if value isa Integer
        typemin(Int64) <= value <= typemax(Int64) || throw(ArgumentError(
            "$field integer exceeds signed64"))
        converted = Float64(value)
        BigInt(converted) == BigInt(value) || throw(ArgumentError(
            "$field integer is not exactly representable in binary64"))
    elseif value isa Float64
        converted = value
    else
        throw(ArgumentError("$field requires Float64 or exact Integer"))
    end
    isfinite(converted) || throw(ArgumentError("$field must be finite"))
    return converted
end

"""Accept only an actual non-Boolean signed64 integer before native Int narrowing."""
function _case_count(value, field::String)::Int
    value isa Integer && !(value isa Bool) || throw(ArgumentError(
        "$field requires Integer, excluding Boolean and integral Float"))
    typemin(Int64) <= value <= min(typemax(Int64), typemax(Int)) ||
        throw(ArgumentError("$field is outside the supported integer range"))
    return Int(value)
end

"""Check finite physical domains, bounded point iterations and actual mesh axes."""
function _validate_case(case::GradShafranovCase)::Nothing
    finite_values = (case.R_min, case.R_max, case.Z_min, case.Z_max, case.Ip_target,
        case.mu0, case.alpha, case.omega_j, case.beta_mix)
    all(isfinite, finite_values) || throw(ArgumentError("case contains nonfinite scalar"))
    0.0 < case.R_min < case.R_max || throw(ArgumentError("require positive ordered R"))
    case.Z_min < case.Z_max || throw(ArgumentError("require ordered Z"))
    3 <= case.NR <= 1025 && 3 <= case.NZ <= 1025 ||
        throw(ArgumentError("grid counts must be in [3,1025]"))
    1 <= case.n_picard <= 10000 && 1 <= case.n_jacobi <= 10000 ||
        throw(ArgumentError("iteration counts must be in [1,10000]"))
    work = 1
    for count in (case.NR, case.NZ, case.n_picard, case.n_jacobi)
        count <= 100000000 ÷ work || throw(ArgumentError(
            "case exceeds 100000000 point iterations"))
        work *= count
    end
    case.mu0 > 0.0 || throw(ArgumentError("mu0 must be positive"))
    0.0 < case.alpha <= 1.0 || throw(ArgumentError("alpha must be in (0,1]"))
    0.0 < case.omega_j < 2.0 || throw(ArgumentError("omega_j must be in (0,2)"))
    0.0 <= case.beta_mix <= 1.0 || throw(ArgumentError("beta_mix must be in [0,1]"))
    for (start, stop, count) in ((case.R_min, case.R_max, case.NR),
        (case.Z_min, case.Z_max, case.NZ))
        step = (stop - start) / (count - 1)
        isfinite(step) && step > 0.0 || throw(ArgumentError(
            "grid spacing must be finite and positive"))
        stop > start + step * (count - 2) || throw(ArgumentError(
            "inclusive grid endpoint has collapsed adjacent nodes"))
        previous = start
        for index in 1:count-1
            current = start + step * index
            isfinite(current) && current > previous || throw(ArgumentError(
                "grid axis is nonfinite or has collapsed adjacent nodes"))
            previous = current
        end
    end
    return nothing
end

"""
    case_from_toml(path::AbstractString)::GradShafranovCase

Parse conforming TOML 1.0, then require exactly one `grad_shafranov` table with
all thirteen case fields and no unknown entries. Quoted/dotted/inline table
spellings share the same typed physical contract. There is no file-level mu0
default. Syntax errors retain TOML.ParserError; schema/kind/domain/resource
errors use ArgumentError. Malformed UTF-8 refuses before parsing.
"""
function case_from_toml(path::AbstractString)::GradShafranovCase
    text = read(path, String)
    isvalid(text) || throw(ArgumentError("case TOML must be valid UTF-8"))
    data = TOML.parse(text)
    Set(keys(data)) == Set(["grad_shafranov"]) || throw(ArgumentError(
        "expected only the grad_shafranov table"))
    fields = data["grad_shafranov"]
    fields isa AbstractDict || throw(ArgumentError("grad_shafranov must be a table"))
    expected = Set(string.(fieldnames(GradShafranovCase)))
    missing = sort!(collect(setdiff(expected, Set(keys(fields)))))
    isempty(missing) || throw(ArgumentError(
        "missing required Grad-Shafranov case field: " * join(missing, ", ")))
    Set(keys(fields)) == expected || throw(ArgumentError(
        "case requires exactly thirteen known fields"))
    return GradShafranovCase(; (Symbol(key) => value for (key,value) in fields)...)
end
