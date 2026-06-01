/-
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Fusion Core — Lean PID Bounded Output Proof
-/
import SCPNFusionSolvers

namespace SCPNFusionSolvers

/-!
PID actuator safety contract:

The executable controller clamps the requested actuator magnitude before it is
sent to the plant. This proof captures the core bounded-output invariant for
the nonnegative normalized actuator magnitude: after saturation, the command is
never greater than the configured actuator limit.
-/

def saturatePidMagnitude (limit command : Nat) : Nat :=
  if command <= limit then command else limit

theorem saturatePidMagnitude_le_limit (limit command : Nat) :
    saturatePidMagnitude limit command <= limit := by
  unfold saturatePidMagnitude
  by_cases h : command <= limit
  · simp [h]
  · simp [h]

theorem saturatePidMagnitude_idempotent (limit command : Nat) :
    saturatePidMagnitude limit (saturatePidMagnitude limit command)
      = saturatePidMagnitude limit command := by
  unfold saturatePidMagnitude
  by_cases h : command <= limit
  · simp [h]
  · simp [h]

end SCPNFusionSolvers
