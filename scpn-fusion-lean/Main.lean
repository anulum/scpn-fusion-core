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
import SafetyProof
import PIDBoundedOutput

open SCPNFusionSolvers

def padLeftZeros (s : String) (width : Nat) : String :=
  if s.length >= width then s else String.ofList (List.replicate (width - s.length) '0') ++ s

def formatFloat15 (value : Float) : String :=
  let scale : UInt64 := 1000000000000000
  let scaled := Float.toUInt64 ((Float.abs value) * 1000000000000000.0 + 0.5)
  let intPart := scaled / scale
  let fracPart := scaled % scale
  (if value < 0.0 then "-" else "") ++ toString intPart ++ "." ++ padLeftZeros (toString fracPart) 15

def formatRow (row : Array Float) : String :=
  String.intercalate "," ((row.toList).map formatFloat15)

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
      for row in result.psi do
        IO.println (formatRow row)
      return 0
