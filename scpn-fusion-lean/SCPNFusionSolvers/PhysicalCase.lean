/-
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Fusion Core — Typed physical case admission
-/
import Lake.Toml

namespace SCPNFusionSolvers

/-- Thirteen SI physical and dimensionless iteration inputs; counts are typed Nat. -/
structure GradShafranovCase where
  rMin : Float
  rMax : Float
  zMin : Float
  zMax : Float
  nr : Nat
  nz : Nat
  ipTarget : Float
  mu0 : Float
  nPicard : Nat
  nJacobi : Nat
  alpha : Float
  omegaJ : Float
  betaMix : Float
  deriving Repr

/-- The affordable common reference case; file input never falls back to it. -/
def referenceCase : GradShafranovCase :=
  { rMin := 1.0, rMax := 3.0, zMin := -1.2, zMax := 1.2, nr := 17, nz := 17,
    ipTarget := 1000000.0, mu0 := 1.2566370614359173e-6, nPicard := 8,
    nJacobi := 16, alpha := 0.1, omegaJ := 0.6666666666666666, betaMix := 0.5 }


/-- Reject nonfinite/collapsed nominal and endpoint-inclusive coordinates. -/
def validateCaseAxis (start stop : Float) (count : Nat) : Except String Unit := do
  let step := (stop - start) / Float.ofNat (count - 1)
  if !step.isFinite || !(step > 0.0) then
    throw "grid spacing must be finite and positive"
  if !(stop > start + step * Float.ofNat (count - 2)) then
    throw "inclusive grid endpoint has collapsed adjacent nodes"
  let mut previous := start
  for index in List.range (count - 1) do
    let current := start + step * Float.ofNat (index + 1)
    if !current.isFinite || !(current > previous) then
      throw "grid axis is nonfinite or has collapsed adjacent nodes"
    previous := current

/-- Validate typed direct cases before allocating any full mesh or iteration state. -/
def validateCase (c : GradShafranovCase) : Except String Unit := do
  if !([c.rMin, c.rMax, c.zMin, c.zMax, c.ipTarget, c.mu0,
      c.alpha, c.omegaJ, c.betaMix].all Float.isFinite) then
    throw "case contains nonfinite scalar"
  if !(0.0 < c.rMin && c.rMin < c.rMax && c.zMin < c.zMax) then
    throw "require positive ordered R and ordered Z"
  if c.nr < 3 || c.nr > 1025 || c.nz < 3 || c.nz > 1025 then
    throw "grid counts must be in [3,1025]"
  if c.nPicard < 1 || c.nPicard > 10000 || c.nJacobi < 1 || c.nJacobi > 10000 then
    throw "iteration counts must be in [1,10000]"
  let mut work := 1
  for count in [c.nr, c.nz, c.nPicard, c.nJacobi] do
    if count > 100000000 / work then
      throw "case exceeds 100000000 point iterations"
    work := work * count
  if !(c.mu0 > 0.0) then
    throw "mu0 must be positive"
  if !(c.alpha > 0.0 && c.alpha <= 1.0 && c.omegaJ > 0.0 && c.omegaJ < 2.0 &&
      c.betaMix >= 0.0 && c.betaMix <= 1.0) then
    throw "invalid relaxation or profile scalar"
  validateCaseAxis c.rMin c.rMax c.nr
  validateCaseAxis c.zMin c.zMax c.nz

/-- Read a required table entry by its exact case-sensitive simple key. -/
def requiredCaseValue (table : Lake.Toml.Table) (key : String) : Except String Lake.Toml.Value :=
  match table.find? (Lean.Name.mkSimple key) with
  | some value => pure value
  | none => throw s!"missing required Grad-Shafranov case field: {key}"

/-- Accept actual signed64 integers before converting file counts to Nat. -/
def caseCountValue (table : Lake.Toml.Table) (key : String) : Except String Nat := do
  match ← requiredCaseValue table key with
  | .integer _ value =>
    if value < 0 || value > 9223372036854775807 then
      throw s!"{key} is outside the supported integer range"
    return value.toNat
  | _ => throw s!"{key} requires Integer, excluding Boolean and integral Float"

/-- Read finite binary64 or a mathematically exact signed64 integer, without saturation. -/
def caseRealValue (table : Lake.Toml.Table) (key : String) : Except String Float := do
  let value ← requiredCaseValue table key
  let number ← match value with
    | .float _ number => pure number
    | .integer _ integer => do
      if integer < -9223372036854775808 || integer > 9223372036854775807 then
        throw s!"{key} integer exceeds signed64"
      let absolute := Float.ofNat integer.natAbs
      if absolute.toUInt64.toNat != integer.natAbs then
        throw s!"{key} integer is not exactly representable in binary64"
      pure (if integer < 0 then -absolute else absolute)
    | _ => throw s!"{key} requires Float or exact Integer"
  if !number.isFinite then
    throw s!"{key} must be finite"
  return number

/-- Admit the exact thirteen-field table from a conforming TOML parsed tree. -/
def caseFromTable (root : Lake.Toml.Table) : Except String GradShafranovCase := do
  if root.size != 1 then
    throw "expected only the grad_shafranov table"
  let table ← match root.find? `grad_shafranov with
    | some (.table _ table) => pure table
    | _ => throw "expected the grad_shafranov table"
  for key in ["R_min", "R_max", "Z_min", "Z_max", "NR", "NZ", "Ip_target", "mu0",
      "n_picard", "n_jacobi", "alpha", "omega_j", "beta_mix"] do
    let _ ← requiredCaseValue table key
  if table.size != 13 then
    throw "expected exactly thirteen required case fields"
  let c : GradShafranovCase := {
    rMin := ← caseRealValue table "R_min", rMax := ← caseRealValue table "R_max",
    zMin := ← caseRealValue table "Z_min", zMax := ← caseRealValue table "Z_max",
    nr := ← caseCountValue table "NR", nz := ← caseCountValue table "NZ",
    ipTarget := ← caseRealValue table "Ip_target", mu0 := ← caseRealValue table "mu0",
    nPicard := ← caseCountValue table "n_picard", nJacobi := ← caseCountValue table "n_jacobi",
    alpha := ← caseRealValue table "alpha", omegaJ := ← caseRealValue table "omega_j",
    betaMix := ← caseRealValue table "beta_mix" }
  validateCase c
  return c

/-- Parse UTF-8 TOML1.0 before exact case schema/kind/domain/resource admission. -/
def caseFromToml (path : System.FilePath) : IO (Except String GradShafranovCase) := do
  let bytes ← IO.FS.readBinFile path
  let some content := String.fromUTF8? bytes
    | return .error "case TOML must be valid UTF-8"
  match ← Lake.Toml.loadToml (Lean.Parser.mkInputContext content path.toString) |>.toBaseIO with
  | .error _ => return .error "invalid TOML physical case syntax"
  | .ok root => return caseFromTable root

end SCPNFusionSolvers
