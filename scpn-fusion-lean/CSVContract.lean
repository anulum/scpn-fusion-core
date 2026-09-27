/-
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Fusion Core — Native binary64 CSV round-trip regressions
-/
import SCPNFusionSolvers.CSV
import Lake.Toml

open SCPNFusionSolvers Lake.Toml

/-- Check real serialization against exact input bits using the native decimal parser. -/
def checkRoundTrip (bits : UInt64) : IO Unit := do
  let value := Float.ofBits bits
  match formatFloat64 value with
  | .error error =>
    if value.isFinite then throw (IO.userError s!"finite encoding failed: {error}")
  | .ok encoded =>
    if !value.isFinite then throw (IO.userError "nonfinite value encoded successfully")
    let tree ← match ← loadToml (Lean.Parser.mkInputContext ("value = " ++ encoded) "CSV round trip") |>.toBaseIO with
      | .ok tree => pure tree
      | .error _ => throw (IO.userError s!"invalid decimal encoding: {encoded}")
    match tree.find? `value with
    | some (.float _ decoded) =>
      if decoded.toBits != bits then
        throw (IO.userError s!"binary64 round-trip mismatch: {bits} -> {encoded} -> {decoded.toBits}")
    | _ => throw (IO.userError "decimal encoding lost Float kind")

/-- Cover zero signs, normal/subnormal extremes, infinities/NaN and distributed real bit patterns. -/
def main : IO UInt32 := do
  let boundaries : Array UInt64 := #[
    0, 0x8000000000000000, 1, 0x8000000000000001,
    0x000fffffffffffff, 0x0010000000000000, 0x3ff0000000000000,
    0x4340000000000000, 0x7fefffffffffffff, 0xffefffffffffffff,
    0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000000001]
  for bits in boundaries do checkRoundTrip bits
  let mut state : UInt64 := 0x123456789abcdef0
  for _ in List.range 4096 do
    state := state * 6364136223846793005 + 1442695040888963407
    checkRoundTrip state
  match formatRow64 #[1.0, -0.0, Float.ofBits 1] with
  | .error error => throw (IO.userError s!"finite row failed: {error}")
  | .ok row =>
    if (row.splitOn ",").length != 3 then throw (IO.userError "wrong row cell count")
  match formatRow64 #[1.0, Float.ofBits 0x7ff0000000000000] with
  | .error _ => pure ()
  | .ok _ => throw (IO.userError "nonfinite row succeeded")
  IO.println "PASS: 4109 actual binary64 scalar patterns and finite/nonfinite rows"
  return 0
