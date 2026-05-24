/-
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Fusion Core — Native Lean Solvers
-/
namespace SCPNFusionSolvers

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

structure GradShafranovResult where
  psi : Array (Array Float)
  residualHistory : Array Float
  deriving Repr

abbrev Matrix := Array (Array Float)

def fmax (a b : Float) : Float :=
  if a >= b then a else b

def fmin (a b : Float) : Float :=
  if a <= b then a else b

def pow10Float : Nat → Float
  | 0 => 1.0
  | n + 1 => 10.0 * pow10Float n

def stripComment (line : String) : String :=
  match line.splitOn "#" with
  | head :: _ => head.trimAscii.toString
  | [] => ""

def parseUnsignedDecimal (raw : String) : Except String Float :=
  match raw.splitOn "." with
  | whole :: [] =>
      match whole.toNat? with
      | some value => pure (Float.ofNat value)
      | none => throw ("invalid decimal value: " ++ raw)
  | whole :: frac :: [] =>
      match whole.toNat?, frac.toNat? with
      | some wholeValue, some fracValue =>
          pure (Float.ofNat wholeValue + Float.ofNat fracValue / pow10Float frac.length)
      | _, _ => throw ("invalid decimal value: " ++ raw)
  | _ => throw ("invalid decimal value: " ++ raw)

def applyScientificExponent (value : Float) (exponent : Int) : Float :=
  if exponent < 0 then
    match (-exponent).toNat? with
    | some n => value / pow10Float n
    | none => value
  else
    match exponent.toNat? with
    | some n => value * pow10Float n
    | none => value

def parseFloatValue (raw : String) : Except String Float := do
  let stripped := raw.trimAscii.toString
  let sign := if stripped.startsWith "-" then -1.0 else 1.0
  let unsigned := if stripped.startsWith "-" || stripped.startsWith "+" then (stripped.drop 1).toString else stripped
  let parts := unsigned.splitOn "e"
  let parts := if parts.length == 1 then unsigned.splitOn "E" else parts
  match parts with
  | mantissa :: [] =>
      return sign * (← parseUnsignedDecimal mantissa)
  | mantissa :: exponentText :: [] =>
      match exponentText.toInt? with
      | some exponent => return sign * applyScientificExponent (← parseUnsignedDecimal mantissa) exponent
      | none => throw ("invalid exponent value: " ++ raw)
  | _ => throw ("invalid float value: " ++ raw)

def parseNatValue (raw : String) : Except String Nat :=
  match raw.trimAscii.toString.toNat? with
  | some value => pure value
  | none => throw ("invalid natural value: " ++ raw)

def requiredCaseFields : List String :=
  [ "R_min", "R_max", "Z_min", "Z_max", "NR", "NZ", "Ip_target", "mu0",
    "n_picard", "n_jacobi", "alpha", "omega_j", "beta_mix" ]

def hasField (fields : List String) (target : String) : Bool :=
  fields.any (fun field => field == target)

def validateRequiredFields (fields : List String) : Except String Unit := do
  for required in requiredCaseFields do
    if !hasField fields required then
      throw ("missing required Grad-Shafranov case field: " ++ required)

def referenceCase : GradShafranovCase :=
  { rMin := 1.0, rMax := 3.0, zMin := -1.2, zMax := 1.2, nr := 17, nz := 17,
    ipTarget := 1000000.0, mu0 := 1.2566370614359173e-6, nPicard := 8,
    nJacobi := 16, alpha := 0.1, omegaJ := 0.6666666666666666, betaMix := 0.5 }

def validateCase (c : GradShafranovCase) : Except String Unit :=
  if !(c.rMax > c.rMin) || !(c.zMax > c.zMin) then
    throw "invalid domain bounds"
  else if c.nr < 3 || c.nz < 3 then
    throw "grid dimensions must be at least 3"
  else if !(c.mu0 > 0.0) || c.nPicard < 1 || c.nJacobi < 1 then
    throw "invalid positive solver scalar"
  else if !(c.alpha > 0.0 && c.alpha <= 1.0) || !(c.omegaJ > 0.0 && c.omegaJ < 2.0) || !(c.betaMix >= 0.0 && c.betaMix <= 1.0) then
    throw "invalid relaxation or profile scalar"
  else
    pure ()

def updateCaseField (c : GradShafranovCase) (key value : String) : Except String GradShafranovCase := do
  match key with
  | "R_min" => pure { c with rMin := ← parseFloatValue value }
  | "R_max" => pure { c with rMax := ← parseFloatValue value }
  | "Z_min" => pure { c with zMin := ← parseFloatValue value }
  | "Z_max" => pure { c with zMax := ← parseFloatValue value }
  | "NR" => pure { c with nr := ← parseNatValue value }
  | "NZ" => pure { c with nz := ← parseNatValue value }
  | "Ip_target" => pure { c with ipTarget := ← parseFloatValue value }
  | "mu0" => pure { c with mu0 := ← parseFloatValue value }
  | "n_picard" => pure { c with nPicard := ← parseNatValue value }
  | "n_jacobi" => pure { c with nJacobi := ← parseNatValue value }
  | "alpha" => pure { c with alpha := ← parseFloatValue value }
  | "omega_j" => pure { c with omegaJ := ← parseFloatValue value }
  | "beta_mix" => pure { c with betaMix := ← parseFloatValue value }
  | _ => pure c

def caseFromToml (path : System.FilePath) : IO (Except String GradShafranovCase) := do
  let content ← IO.FS.readFile path
  let mut inSection := false
  let mut c := referenceCase
  let mut seenFields : List String := []
  for rawLine in content.splitOn "\n" do
    let line := stripComment rawLine
    if line == "" then
      pure ()
    else if line.startsWith "[" && line.endsWith "]" then
      inSection := line == "[grad_shafranov]"
    else if inSection then
      match line.splitOn "=" with
      | key :: value :: [] =>
          let field := key.trimAscii.toString
          match updateCaseField c field value.trimAscii.toString with
          | Except.ok next => c := next
          | Except.error err => return Except.error err
          if hasField requiredCaseFields field && !hasField seenFields field then
            seenFields := field :: seenFields
      | _ => return Except.error ("invalid TOML assignment: " ++ line)
  match validateRequiredFields seenFields with
  | Except.error err => return Except.error err
  | Except.ok _ => pure ()
  match validateCase c with
  | Except.ok _ => return Except.ok c
  | Except.error err => return Except.error err

def zeros (nz nr : Nat) : Matrix :=
  Array.replicate nz (Array.replicate nr 0.0)

def get2D (m : Matrix) (iz ir : Nat) : Float :=
  match m[iz]? with
  | some row => row.getD ir 0.0
  | none => 0.0

def set2D (m : Matrix) (iz ir : Nat) (value : Float) : Matrix :=
  match m[iz]? with
  | some row => m.set! iz (row.set! ir value)
  | none => m

def linspace (start stop : Float) (count : Nat) : Array Float := Id.run do
  let step := (stop - start) / Float.ofNat (count - 1)
  let mut out := #[]
  for i in List.range count do
    out := out.push (start + step * Float.ofNat i)
  return out

def rGrid (c : GradShafranovCase) : Matrix := Id.run do
  let r := linspace c.rMin c.rMax c.nr
  let mut rr := zeros c.nz c.nr
  for iz in List.range c.nz do
    for ir in List.range c.nr do
      rr := set2D rr iz ir (r.getD ir 0.0)
  return rr

def applyZeroBoundary (psi : Matrix) : Matrix := Id.run do
  let nz := psi.size
  let nr := match psi[0]? with | some row => row.size | none => 0
  let mut out := psi
  for ir in List.range nr do
    out := set2D out 0 ir 0.0
    out := set2D out (nz - 1) ir 0.0
  for iz in List.range nz do
    out := set2D out iz 0 0.0
    out := set2D out iz (nr - 1) 0.0
  return out

def initialPsi (c : GradShafranovCase) (rr : Matrix) : Matrix := Id.run do
  let rCenter := 0.5 * (c.rMin + c.rMax)
  let mut psi := zeros c.nz c.nr
  for iz in List.range c.nz do
    for ir in List.range c.nr do
      let delta := get2D rr iz ir - rCenter
      psi := set2D psi iz ir (Float.exp (-(delta * delta) / 0.5) * 0.01)
  return applyZeroBoundary psi

def maxInterior (psi : Matrix) (nz nr : Nat) : Float := Id.run do
  let mut best := get2D psi 1 1
  for iz in List.range nz do
    if iz > 0 && iz + 1 < nz then
      for ir in List.range nr do
        if ir > 0 && ir + 1 < nr then
          best := fmax best (get2D psi iz ir)
  return best

def computeSource (c : GradShafranovCase) (psi rr : Matrix) (dR dZ : Float) : Matrix := Id.run do
  let psiAxis := maxInterior psi c.nz c.nr
  let mut denom := -psiAxis
  if Float.abs denom < 1.0e-9 then
    denom := if denom == 0.0 then 1.0e-9 else if denom < 0.0 then -1.0e-9 else 1.0e-9
  let mut jRaw := zeros c.nz c.nr
  let mut current := 0.0
  for iz in List.range c.nz do
    for ir in List.range c.nr do
      let psiNorm0 := (get2D psi iz ir - psiAxis) / denom
      let psiNorm := fmin 1.0 (fmax 0.0 psiNorm0)
      let profile := if psiNorm >= 0.0 && psiNorm < 1.0 then 1.0 - psiNorm else 0.0
      let rVal := get2D rr iz ir
      let rSafe := fmax rVal 1.0e-10
      let jP := rVal * profile
      let jF := profile / (c.mu0 * rSafe)
      let j := c.betaMix * jP + (1.0 - c.betaMix) * jF
      jRaw := set2D jRaw iz ir j
      current := current + j * dR * dZ
  let scale := c.ipTarget / fmax (Float.abs current) 1.0e-9
  let mut source := zeros c.nz c.nr
  for iz in List.range c.nz do
    for ir in List.range c.nr do
      source := set2D source iz ir (-c.mu0 * get2D rr iz ir * get2D jRaw iz ir * scale)
  return source

def jacobiStep (c : GradShafranovCase) (psi source rr : Matrix) (dR dZ : Float) : Matrix := Id.run do
  let dR2 := dR * dR
  let dZ2 := dZ * dZ
  let aNS := 1.0 / dZ2
  let aC := 2.0 / dR2 + 2.0 / dZ2
  let mut out := psi
  for iz in List.range c.nz do
    if iz > 0 && iz + 1 < c.nz then
      for ir in List.range c.nr do
        if ir > 0 && ir + 1 < c.nr then
          let rSafe := fmax (get2D rr iz ir) 1.0e-10
          let aE := 1.0 / dR2 - 1.0 / (2.0 * rSafe * dR)
          let aW := 1.0 / dR2 + 1.0 / (2.0 * rSafe * dR)
          let update := (aE * get2D psi iz (ir + 1) + aW * get2D psi iz (ir - 1) + aNS * (get2D psi (iz - 1) ir + get2D psi (iz + 1) ir) - get2D source iz ir) / aC
          out := set2D out iz ir ((1.0 - c.omegaJ) * get2D psi iz ir + c.omegaJ * update)
  return out

def maxChange (a b : Matrix) : Float := Id.run do
  let nz := a.size
  let nr := match a[0]? with | some row => row.size | none => 0
  let mut best := 0.0
  for iz in List.range nz do
    for ir in List.range nr do
      best := fmax best (Float.abs (get2D a iz ir - get2D b iz ir))
  return best

def solveGradShafranov (c : GradShafranovCase) : Except String GradShafranovResult := do
  validateCase c
  let rr := rGrid c
  let dR := (c.rMax - c.rMin) / Float.ofNat (c.nr - 1)
  let dZ := (c.zMax - c.zMin) / Float.ofNat (c.nz - 1)
  let mut psi := initialPsi c rr
  let mut residuals := #[]
  for _ in List.range c.nPicard do
    let source := computeSource c psi rr dR dZ
    let mut psiElliptic := psi
    for _ in List.range c.nJacobi do
      psiElliptic := jacobiStep c psiElliptic source rr dR dZ
    let mut psiNext := zeros c.nz c.nr
    for iz in List.range c.nz do
      for ir in List.range c.nr do
        psiNext := set2D psiNext iz ir ((1.0 - c.alpha) * get2D psi iz ir + c.alpha * get2D psiElliptic iz ir)
    residuals := residuals.push (maxChange psiNext psi)
    psi := psiNext
  return { psi := applyZeroBoundary psi, residualHistory := residuals }

end SCPNFusionSolvers
