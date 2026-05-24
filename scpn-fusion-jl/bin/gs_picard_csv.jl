# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Julia Grad-Shafranov CSV CLI
using SCPNFusionSolvers
using DelimitedFiles

length(ARGS) == 1 || error("usage: julia --project=scpn-fusion-jl scpn-fusion-jl/bin/gs_picard_csv.jl CASE.toml")
case = case_from_toml(ARGS[1])
result = solve_grad_shafranov(case)
writedlm(stdout, result.psi, ',')
