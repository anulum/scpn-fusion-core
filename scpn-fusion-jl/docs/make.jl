# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Documenter
using SCPNFusionSolvers

makedocs(
    sitename = "SCPNFusionSolvers.jl",
    modules = [SCPNFusionSolvers],
    checkdocs = :exports,
    doctest = true,
    warnonly = false,
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "Overview" => "index.md",
        "API reference" => "api.md",
    ],
)
