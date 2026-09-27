/-
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Fusion Core — Native Lean Solvers
-/
import SCPNFusionSolvers.PhysicalCase

namespace SCPNFusionSolvers

structure GradShafranovResult where
  psi : Array (Array Float)
  residualHistory : Array Float
  deriving Repr

abbrev Matrix := Array (Array Float)

abbrev BoolMatrix := Array (Array Bool)

def fmax (a b : Float) : Float :=
  if a >= b then a else b

def fmin (a b : Float) : Float :=
  if a <= b then a else b

def zeros (nz nr : Nat) : Matrix :=
  Array.replicate nz (Array.replicate nr 0.0)

def get2D (m : Matrix) (iz ir : Nat) : Float :=
  match m[iz]? with
  | some row => row.getD ir 0.0
  | none => 0.0

def get2DBool (m : BoolMatrix) (iz ir : Nat) : Bool :=
  match m[iz]? with
  | some row => row.getD ir false
  | none => false

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

/-- Fail typed numerical execution before nonfinite values enter reductions or clipping. -/
def requireFinite (value : Float) (label : String) : Except String Unit :=
  if value.isFinite then pure () else throw s!"nonfinite arithmetic: {label}"

/-- Inspect every cell before a numerical reduction can hide a nonfinite value. -/
def requireFiniteMatrix (matrix : Matrix) (label : String) : Except String Unit := do
  for row in matrix do
    for value in row do
      requireFinite value label

/-- Construct the existing Gaussian seed with checked intermediate arithmetic. -/
def initialPsi (c : GradShafranovCase) (rr : Matrix) : Except String Matrix := do
  let rCenter := 0.5 * (c.rMin + c.rMax)
  requireFinite rCenter "seed radius centre"
  let mut psi := zeros c.nz c.nr
  for iz in List.range c.nz do
    for ir in List.range c.nr do
      let delta := get2D rr iz ir - rCenter
      let square := delta * delta
      requireFinite square "seed radius square"
      let exponent := -square / 0.5
      requireFinite exponent "seed Gaussian exponent"
      let value := Float.exp exponent * 0.01
      requireFinite value "seed flux"
      psi := set2D psi iz ir value
  return applyZeroBoundary psi

def maxInterior (psi : Matrix) (nz nr : Nat) : Float := Id.run do
  let mut best := get2D psi 1 1
  for iz in List.range nz do
    if iz > 0 && iz + 1 < nz then
      for ir in List.range nr do
        if ir > 0 && ir + 1 < nr then
          best := fmax best (get2D psi iz ir)
  return best

/-- Compute the existing profile source while refusing nonfinite intermediate current. -/
def computeSource (c : GradShafranovCase) (psi rr : Matrix) (dR dZ : Float) : Except String Matrix := do
  requireFiniteMatrix psi "source input flux"
  let psiAxis := maxInterior psi c.nz c.nr
  let mut denom := -psiAxis
  if Float.abs denom < 1.0e-9 then
    denom := if denom == 0.0 then 1.0e-9 else if denom < 0.0 then -1.0e-9 else 1.0e-9
  let mut jRaw := zeros c.nz c.nr
  let mut current := 0.0
  for iz in List.range c.nz do
    for ir in List.range c.nr do
      let psiNorm0 := (get2D psi iz ir - psiAxis) / denom
      requireFinite psiNorm0 "normalized flux before clipping"
      let psiNorm := fmin 1.0 (fmax 0.0 psiNorm0)
      let profile := if psiNorm >= 0.0 && psiNorm < 1.0 then 1.0 - psiNorm else 0.0
      let rVal := get2D rr iz ir
      let rSafe := fmax rVal 1.0e-10
      let jP := rVal * profile
      let currentDenom := c.mu0 * rSafe
      requireFinite currentDenom "profile current denominator"
      if currentDenom == 0.0 then throw "zero profile current denominator"
      let jF := profile / currentDenom
      requireFinite jP "pressure current profile"
      requireFinite jF "toroidal current profile"
      let j := c.betaMix * jP + (1.0 - c.betaMix) * jF
      requireFinite j "mixed current profile"
      jRaw := set2D jRaw iz ir j
      current := current + j * dR * dZ
      requireFinite current "integrated current"
  let scale := c.ipTarget / fmax (Float.abs current) 1.0e-9
  requireFinite scale "current scale"
  let mut source := zeros c.nz c.nr
  for iz in List.range c.nz do
    for ir in List.range c.nr do
      let scaledCurrent := get2D jRaw iz ir * scale
      requireFinite scaledCurrent "scaled toroidal current"
      let value := -c.mu0 * get2D rr iz ir * scaledCurrent
      requireFinite value "physical source"
      source := set2D source iz ir value
  return source

/-- Apply one existing Jacobi sweep with checked coefficients, update and flux. -/
def jacobiStep (c : GradShafranovCase) (psi source rr : Matrix) (dR dZ : Float) : Except String Matrix := do
  let dR2 := dR * dR
  let dZ2 := dZ * dZ
  let aNS := 1.0 / dZ2
  let aC := 2.0 / dR2 + 2.0 / dZ2
  requireFinite aNS "vertical coefficient"
  requireFinite aC "centre coefficient"
  let mut out := psi
  for iz in List.range c.nz do
    if iz > 0 && iz + 1 < c.nz then
      for ir in List.range c.nr do
        if ir > 0 && ir + 1 < c.nr then
          let rSafe := fmax (get2D rr iz ir) 1.0e-10
          let radialDenom := 2.0 * rSafe * dR
          requireFinite radialDenom "radial coefficient denominator"
          if radialDenom == 0.0 then throw "zero radial coefficient denominator"
          let aE := 1.0 / dR2 - 1.0 / radialDenom
          let aW := 1.0 / dR2 + 1.0 / radialDenom
          requireFinite aE "east coefficient"
          requireFinite aW "west coefficient"
          let update := (aE * get2D psi iz (ir + 1) + aW * get2D psi iz (ir - 1) + aNS * (get2D psi (iz - 1) ir + get2D psi (iz + 1) ir) - get2D source iz ir) / aC
          requireFinite update "Jacobi update"
          let value := (1.0 - c.omegaJ) * get2D psi iz ir + c.omegaJ * update
          requireFinite value "Jacobi flux"
          out := set2D out iz ir value
  return out

/-- Measure finite flux change before a maximum reduction can hide NaN. -/
def maxChange (a b : Matrix) : Except String Float := do
  let nz := a.size
  let nr := match a[0]? with | some row => row.size | none => 0
  let mut best := 0.0
  for iz in List.range nz do
    for ir in List.range nr do
      let change := Float.abs (get2D a iz ir - get2D b iz ir)
      requireFinite change "Picard flux change"
      best := fmax best change
  return best

def deltaStar (c : GradShafranovCase) (psi : Matrix) : Matrix := Id.run do
  let r := linspace c.rMin c.rMax c.nr
  let dR := (c.rMax - c.rMin) / Float.ofNat (c.nr - 1)
  let dZ := (c.zMax - c.zMin) / Float.ofNat (c.nz - 1)
  let dR2 := dR * dR
  let dZ2 := dZ * dZ
  let mut out := zeros c.nz c.nr
  for iz in List.range c.nz do
    if iz > 0 && iz + 1 < c.nz then
      for ir in List.range c.nr do
        if ir > 0 && ir + 1 < c.nr then
          let radius := r.getD ir 1.0
          let d2DR2 := (get2D psi iz (ir + 1) - 2.0 * get2D psi iz ir + get2D psi iz (ir - 1)) / dR2
          let dDROverR := (get2D psi iz (ir + 1) - get2D psi iz (ir - 1)) / (2.0 * dR * radius)
          let d2DZ2 := (get2D psi (iz + 1) ir - 2.0 * get2D psi iz ir + get2D psi (iz - 1) ir) / dZ2
          out := set2D out iz ir (d2DR2 - dDROverR + d2DZ2)
  return out

def toroidalCurrentDensityFromFlux (c : GradShafranovCase) (psi : Matrix) : Matrix := Id.run do
  let r := linspace c.rMin c.rMax c.nr
  let delta := deltaStar c psi
  let mut out := zeros c.nz c.nr
  for iz in List.range c.nz do
    if iz > 0 && iz + 1 < c.nz then
      for ir in List.range c.nr do
        if ir > 0 && ir + 1 < c.nr then
          out := set2D out iz ir (-(get2D delta iz ir) / (c.mu0 * r.getD ir 1.0))
  return out

def totalToroidalCurrentFromFlux (c : GradShafranovCase) (psi : Matrix) : Float := Id.run do
  let currentDensity := toroidalCurrentDensityFromFlux c psi
  let dR := (c.rMax - c.rMin) / Float.ofNat (c.nr - 1)
  let dZ := (c.zMax - c.zMin) / Float.ofNat (c.nz - 1)
  let mut total := 0.0
  for iz in List.range c.nz do
    if iz > 0 && iz + 1 < c.nz then
      for ir in List.range c.nr do
        if ir > 0 && ir + 1 < c.nr then
          total := total + get2D currentDensity iz ir * dR * dZ
  return total

def totalToroidalCurrentFromFluxTrapezoidal (c : GradShafranovCase) (psi : Matrix) : Float := Id.run do
  let currentDensity := toroidalCurrentDensityFromFlux c psi
  let dR := (c.rMax - c.rMin) / Float.ofNat (c.nr - 1)
  let dZ := (c.zMax - c.zMin) / Float.ofNat (c.nz - 1)
  let mut total := 0.0
  for iz in List.range c.nz do
    let zWeight := if iz == 0 || iz + 1 == c.nz then 0.5 else 1.0
    for ir in List.range c.nr do
      let rWeight := if ir == 0 || ir + 1 == c.nr then 0.5 else 1.0
      total := total + get2D currentDensity iz ir * zWeight * rWeight * dR * dZ
  return total

def totalToroidalCurrentFromFluxMasked
    (c : GradShafranovCase) (psi : Matrix) (domainMask : BoolMatrix) : Float := Id.run do
  let currentDensity := toroidalCurrentDensityFromFlux c psi
  let dR := (c.rMax - c.rMin) / Float.ofNat (c.nr - 1)
  let dZ := (c.zMax - c.zMin) / Float.ofNat (c.nz - 1)
  let mut total := 0.0
  for iz in List.range c.nz do
    if iz > 0 && iz + 1 < c.nz then
      for ir in List.range c.nr do
        if ir > 0 && ir + 1 < c.nr && get2DBool domainMask iz ir then
          total := total + get2D currentDensity iz ir * dR * dZ
  return total

/-- Execute an admitted physical case; arithmetic failures return typed errors. -/
def solveGradShafranov (c : GradShafranovCase) : Except String GradShafranovResult := do
  validateCase c
  let rr := rGrid c
  let dR := (c.rMax - c.rMin) / Float.ofNat (c.nr - 1)
  let dZ := (c.zMax - c.zMin) / Float.ofNat (c.nz - 1)
  requireFinite (dR * dR) "radial spacing square"
  requireFinite (dZ * dZ) "vertical spacing square"
  if dR * dR == 0.0 || dZ * dZ == 0.0 then throw "zero squared grid spacing"
  let mut psi ← initialPsi c rr
  let mut residuals := #[]
  for _ in List.range c.nPicard do
    let source ← computeSource c psi rr dR dZ
    let mut psiElliptic := psi
    for _ in List.range c.nJacobi do
      psiElliptic ← jacobiStep c psiElliptic source rr dR dZ
    let mut psiNext := zeros c.nz c.nr
    for iz in List.range c.nz do
      for ir in List.range c.nr do
        let value := (1.0 - c.alpha) * get2D psi iz ir + c.alpha * get2D psiElliptic iz ir
        requireFinite value "Picard flux"
        psiNext := set2D psiNext iz ir value
    residuals := residuals.push (← maxChange psiNext psi)
    psi := psiNext
  return { psi := applyZeroBoundary psi, residualHistory := residuals }

end SCPNFusionSolvers
