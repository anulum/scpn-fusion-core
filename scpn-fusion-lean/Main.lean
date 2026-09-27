/-
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Fusion Core — Lean Grad-Shafranov CSV CLI
-/
import SCPNFusionSolvers
import SCPNFusionSolvers.CSV
import SafetyProof
import PIDBoundedOutput
import SNNReachabilityPreservation
import PetriTokenBoundedness

open SCPNFusionSolvers

/-- Solve the requested physical case and emit only finite, round-trippable CSV. -/
def main (args : List String) : IO UInt32 := do
  if args.length != 1 then
    IO.eprintln "usage: gs_picard_csv CASE.toml"
    return 2
  let casePath := System.FilePath.mk (args.head!)
  match ← caseFromToml casePath with
  | Except.error err =>
      IO.eprintln err
      return 1
  | Except.ok requestedCase =>
  match solveGradShafranov requestedCase with
  | Except.error err =>
      IO.eprintln err
      return 1
  | Except.ok result =>
      let encoded := result.psi.toList.mapM formatRow64
      match encoded with
      | .error error =>
        IO.eprintln error
        return 1
      | .ok rows =>
        for row in rows do IO.println row
        return 0
