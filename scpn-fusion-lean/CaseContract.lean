/-
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Fusion Core — Real public physical case corpus regressions
-/
import SCPNFusionSolvers
import Lake.Toml

open SCPNFusionSolvers Lake.Toml

/-- Require a real test condition without substituting a fake runtime surface. -/
def check (condition : Bool) (label : String) : IO Unit :=
  unless condition do throw (IO.userError label)

/-- Read the literal metadata string from the corpus manifest. -/
def metadataString (table : Table) (key : Lean.Name) : IO String :=
  match table.find? key with
  | some (.string _ value) => pure value
  | _ => throw (IO.userError s!"missing manifest string: {key}")

/-- Read the literal metadata Boolean from the corpus manifest. -/
def metadataBool (table : Table) (key : Lean.Name) : IO Bool :=
  match table.find? key with
  | some (.boolean _ value) => pure value
  | _ => throw (IO.userError s!"missing manifest Boolean: {key}")

/-- Compare literal expected real values; the expected mapping is not a loader oracle. -/
def expectedReal (table : Table) (key : Lean.Name) (actual : Float) : Bool :=
  match table.find? key with
  | some (.float _ expected) => actual == expected
  | _ => false

/-- Compare literal expected integer counts without using case conversion helpers. -/
def expectedCount (table : Table) (key : Lean.Name) (actual : Nat) : Bool :=
  match table.find? key with
  | some (.integer _ expected) => expected == Int.ofNat actual
  | _ => false

/-- Compare all thirteen declared physical values against the public file result. -/
def matchesExpected (c : GradShafranovCase) (table : Table) : Bool :=
  table.size == 13 &&
  expectedReal table `R_min c.rMin && expectedReal table `R_max c.rMax &&
  expectedReal table `Z_min c.zMin && expectedReal table `Z_max c.zMax &&
  expectedCount table `NR c.nr && expectedCount table `NZ c.nz &&
  expectedReal table `Ip_target c.ipTarget && expectedReal table `mu0 c.mu0 &&
  expectedCount table `n_picard c.nPicard && expectedCount table `n_jacobi c.nJacobi &&
  expectedReal table `alpha c.alpha && expectedReal table `omega_j c.omegaJ &&
  expectedReal table `beta_mix c.betaMix

/-- Verify SHA256 fixture custody with the actual system hashing executable. -/
def checkFixtureHash (path : System.FilePath) (expected : String) : IO Unit := do
  let result ← IO.Process.output {cmd := "sha256sum", args := #[path.toString]}
  check (result.exitCode == 0) s!"hash executable failed: {path}"
  let actual := (result.stdout.splitOn " ").head!
  check (actual == expected) s!"fixture SHA mismatch: {path}"

/-- Execute affordable public solves and inspect every shape, finite cell and boundary. -/
def checkAffordableSolve (c : GradShafranovCase) (name : String) : IO Unit := do
  match solveGradShafranov c with
  | .error error => throw (IO.userError s!"{name}: public solve failed: {error}")
  | .ok result =>
    check (result.psi.size == c.nz) s!"{name}: wrong NZ"
    check (result.residualHistory.size == c.nPicard) s!"{name}: wrong history length"
    check (result.residualHistory.all Float.isFinite) s!"{name}: nonfinite history"
    for iz in List.range c.nz do
      let row := result.psi[iz]!
      check (row.size == c.nr) s!"{name}: wrong NR"
      for ir in List.range c.nr do
        let value := row[ir]!
        check value.isFinite s!"{name}: nonfinite flux"
        if iz == 0 || iz + 1 == c.nz || ir == 0 || ir + 1 == c.nr then
          check (value == 0.0) s!"{name}: nonzero boundary"

/-- Test typed direct admission and accepted-input arithmetic failures through public solve. -/
def checkDirectCases : IO Unit := do
  let invalid : List GradShafranovCase := [
    {referenceCase with nr := 2}, {referenceCase with nz := 1026},
    {referenceCase with nPicard := 0}, {referenceCase with nJacobi := 10001},
    {referenceCase with nPicard := 10000, nJacobi := 10000},
    {referenceCase with nr := 2^100}, {referenceCase with rMin := 0.0},
    {referenceCase with mu0 := 0.0}, {referenceCase with alpha := 0.0},
    {referenceCase with omegaJ := 2.0}, {referenceCase with betaMix := -1.0},
    {referenceCase with ipTarget := 0.0 / 0.0},
    {referenceCase with rMin := 5.0e-324, rMax := 1.5e-323, nr := 4}]
  for c in invalid do
    match solveGradShafranov c with
    | .error _ => pure ()
    | .ok _ => throw (IO.userError "invalid direct case solved successfully")
  let unstable : List GradShafranovCase := [
    {referenceCase with rMin := 1.0e200, rMax := 2.0e200, nr := 3, nz := 3, nPicard := 1, nJacobi := 1},
    {referenceCase with mu0 := 5.0e-324, nPicard := 1, nJacobi := 1},
    {referenceCase with rMin := 1.0e154, rMax := 1.0e155, nPicard := 1, nJacobi := 1},
    {referenceCase with rMin := 1.0e154, rMax := 3.0e154, nPicard := 1, nJacobi := 1},
    {referenceCase with rMin := 1.0, rMax := 1.01, zMin := -0.01, zMax := 0.01, ipTarget := 1.0e308, nPicard := 1, nJacobi := 1}]
  for c in unstable do
    match validateCase c with
    | .error error => throw (IO.userError s!"numerical witness failed admission: {error}")
    | .ok _ => pure ()
    match solveGradShafranov c with
    | .error _ => pure ()
    | .ok _ => throw (IO.userError "accepted-input nonfinite arithmetic succeeded")

/-- Run the exact shared manifest through the public file loader and native solver. -/
def main (args : List String) : IO UInt32 := do
  let manifest := System.FilePath.mk ((args.head?).getD "../validation/polyglot/case_contract/manifest.toml")
  let content ← IO.FS.readFile manifest
  let root ← match ← loadToml (Lean.Parser.mkInputContext content manifest.toString) |>.toBaseIO with
    | .ok table => pure table
    | .error _ => throw (IO.userError "invalid shared corpus manifest")
  check ((← metadataString root `schema) == "scpn.physical-case-corpus.v1") "wrong corpus schema"
  let fixtures ← match root.find? `case with
    | some (.array _ values) => pure values
    | _ => throw (IO.userError "missing shared case array")
  let mut accepted := 0
  let mut solved := 0
  for fixture in fixtures do
    let table ← match fixture with
      | .table _ table => pure table
      | _ => throw (IO.userError "invalid manifest fixture")
    let name ← metadataString table `name
    let path := manifest.parent.getD (System.FilePath.mk ".") / (← metadataString table `path)
    checkFixtureHash path (← metadataString table `sha256)
    let valid ← metadataBool table `valid
    match ← caseFromToml path with
    | .error error => check (!valid) s!"{name}: admitted fixture refused: {error}"
    | .ok c =>
      check valid s!"{name}: invalid fixture admitted"
      let expected ← match table.find? `expected with
        | some (.table _ table) => pure table
        | _ => throw (IO.userError s!"{name}: missing declared physical mapping")
      check (matchesExpected c expected) s!"{name}: physical mapping mismatch"
      accepted := accepted + 1
      if ← metadataBool table `solve then
        checkAffordableSolve c name
        solved := solved + 1
  checkDirectCases
  IO.println s!"PASS: {fixtures.size} shared cases, {accepted} admitted, {solved} affordable solves, 13 direct refusals, 5 arithmetic refusals"
  return 0
