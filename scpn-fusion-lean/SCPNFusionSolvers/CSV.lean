/-
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Fusion Core — Finite binary64 CSV serialization
-/
import Init

namespace SCPNFusionSolvers

/-- Produce seventeen significant decimal digits from the exact binary64 rational.

Integer arithmetic avoids the old scaled-UInt64 saturation and subnormal
underflow. Decimal rounding is nearest with ties to even. The output preserves
negative zero and round trips finite binary64 values through standard parsers.
-/
def formatFloat64 (value : Float) : Except String String := do
  if !value.isFinite then throw "CSV requires finite binary64 values"
  let bits := value.toBits.toNat
  let negative := bits / 2^63 != 0
  let signText := if negative then "-" else ""
  let fraction := bits % 2^52
  let exponentBits := (bits / 2^52) % 2048
  let mantissa := if exponentBits == 0 then fraction else fraction + 2^52
  if mantissa == 0 then return signText ++ "0.0"
  let binaryExponent : Int := if exponentBits == 0 then -1074 else Int.ofNat exponentBits - 1075
  let numerator := if binaryExponent >= 0 then mantissa * 2^binaryExponent.toNat else mantissa
  let denominator := if binaryExponent < 0 then 2^(-binaryExponent).toNat else 1
  let mut exponent := Int.ofNat (toString numerator).length - Int.ofNat (toString denominator).length
  let below := if exponent >= 0 then numerator < denominator * 10^exponent.toNat
    else numerator * 10^(-exponent).toNat < denominator
  if below then exponent := exponent - 1
  let shift := 16 - exponent
  let scaledNumerator := if shift >= 0 then numerator * 10^shift.toNat else numerator
  let scaledDenominator := if shift < 0 then denominator * 10^(-shift).toNat else denominator
  let quotient := scaledNumerator / scaledDenominator
  let remainder := scaledNumerator % scaledDenominator
  let roundUp := 2 * remainder > scaledDenominator ||
    (2 * remainder == scaledDenominator && quotient % 2 == 1)
  let mut rounded := quotient + (if roundUp then 1 else 0)
  if rounded >= 10^17 then
    rounded := rounded / 10
    exponent := exponent + 1
  let digits := toString rounded
  let leading := (digits.take 1).toString
  let rest := (digits.drop 1).toString
  return signText ++ leading ++ "." ++ rest ++ "e" ++ toString exponent

/-- Encode a complete finite row before any CSV output is emitted. -/
def formatRow64 (row : Array Float) : Except String String := do
  let mut cells : Array String := #[]
  for value in row do
    cells := cells.push (← formatFloat64 value)
  return String.intercalate "," cells.toList

end SCPNFusionSolvers
