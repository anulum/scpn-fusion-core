/-
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Fusion Core — Lean Solver Package
-/
import Lake
open Lake DSL

package «scpn-fusion-lean» where
  version := v!"0.1.0"

-- Every proof library is a default target so a bare `lake build` (the CI gate)
-- type-checks the entire contract surface, not only the modules imported by the
-- executable.
@[default_target]
lean_lib SCPNFusionSolvers where

@[default_target]
lean_lib SafetyProof where

@[default_target]
lean_lib PIDBoundedOutput where

@[default_target]
lean_lib SNNReachabilityPreservation where

@[default_target]
lean_lib PetriTokenBoundedness where

@[default_target]
lean_lib InterlockReplayInvariance where

@[default_target]
lean_exe gs_picard_csv where
  root := `Main
