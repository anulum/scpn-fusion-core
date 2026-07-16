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
Actuator saturation contract — formal reference model:

`saturatePidMagnitude` formalises the actuator-boundary saturation invariant over
non-negative integer magnitudes: after saturation the command is never greater
than the configured limit and never amplifies the raw command.

Honest scope (what this proof is and is NOT):
- It is a reference contract, NOT an extraction of the shipped controllers. The
  floating-point PIDs (`pid.rs`, `tokamak_flight_sim.pid_step`) emit raw commands;
  the runtime magnitude bound is enforced downstream at the actuator boundary
  (`FirstOrderActuator`, Rust `SafetyEnvelope`, HIL `write_dac`) and is covered by
  unit tests, not by this proof.
- `Nat` cannot represent NaN/inf, so this proof does not model the non-finite
  failure mode; that path is handled by the fail-safe guards in the
  actuator-boundary code.
- Linking these theorems to the shipped controllers would require Lean extraction
  or an FFI binding test (tracked in the Lean verification backlog).
-/

def saturatePidMagnitude (limit command : Nat) : Nat :=
  if command <= limit then command else limit

theorem saturatePidMagnitude_eq_self_when_within
    {limit command : Nat} (h : command <= limit) :
    saturatePidMagnitude limit command = command := by
  unfold saturatePidMagnitude
  simp [h]

theorem saturatePidMagnitude_eq_limit_when_above
    {limit command : Nat} (h : ¬ command <= limit) :
    saturatePidMagnitude limit command = limit := by
  unfold saturatePidMagnitude
  simp [h]

theorem saturatePidMagnitude_le_limit (limit command : Nat) :
    saturatePidMagnitude limit command <= limit := by
  unfold saturatePidMagnitude
  by_cases h : command <= limit
  · simp [h]
  · simp [h]

theorem saturatePidMagnitude_le_command_or_eq_limit (limit command : Nat) :
    saturatePidMagnitude limit command <= command ∨
      saturatePidMagnitude limit command = limit := by
  unfold saturatePidMagnitude
  by_cases h : command <= limit
  · simp [h]
  · simp [h]

theorem saturatePidMagnitude_preserves_zero_limit (command : Nat) :
    saturatePidMagnitude 0 command = 0 := by
  exact Nat.eq_zero_of_le_zero (saturatePidMagnitude_le_limit 0 command)

theorem saturatePidMagnitude_preserves_zero_command (limit : Nat) :
    saturatePidMagnitude limit 0 = 0 := by
  unfold saturatePidMagnitude
  simp

theorem saturatePidMagnitude_at_limit (limit : Nat) :
    saturatePidMagnitude limit limit = limit := by
  unfold saturatePidMagnitude
  simp

theorem saturatePidMagnitude_le_command (limit command : Nat) :
    saturatePidMagnitude limit command <= command := by
  unfold saturatePidMagnitude
  by_cases h : command <= limit
  · simp [h]
  · simp [h]
    exact Nat.le_of_lt (Nat.not_le.mp h)

theorem saturatePidMagnitude_le_limit_and_command (limit command : Nat) :
    saturatePidMagnitude limit command <= limit ∧
      saturatePidMagnitude limit command <= command := by
  constructor
  · exact saturatePidMagnitude_le_limit limit command
  · exact saturatePidMagnitude_le_command limit command

theorem saturatePidMagnitude_eq_command_or_limit (limit command : Nat) :
    saturatePidMagnitude limit command = command ∨
      saturatePidMagnitude limit command = limit := by
  unfold saturatePidMagnitude
  by_cases h : command <= limit
  · simp [h]
  · simp [h]

theorem saturatePidMagnitude_le_common_upper_bound
    {limit command upperBound : Nat}
    (limitWithin : limit <= upperBound) :
    saturatePidMagnitude limit command <= upperBound := by
  exact Nat.le_trans (saturatePidMagnitude_le_limit limit command) limitWithin

theorem saturatePidMagnitude_le_raw_upper_bound
    {limit command upperBound : Nat}
    (commandWithin : command <= upperBound) :
    saturatePidMagnitude limit command <= upperBound := by
  exact Nat.le_trans (saturatePidMagnitude_le_command limit command) commandWithin

theorem saturatePidMagnitude_le_dual_upper_bound
    {limit command upperBound : Nat}
    (limitWithin : limit <= upperBound)
    (commandWithin : command <= upperBound) :
    saturatePidMagnitude limit command <= upperBound := by
  have fromLimit : saturatePidMagnitude limit command <= upperBound :=
    saturatePidMagnitude_le_common_upper_bound
      (limit := limit) (command := command) limitWithin
  have fromCommand : saturatePidMagnitude limit command <= upperBound :=
    saturatePidMagnitude_le_raw_upper_bound
      (limit := limit) (command := command) commandWithin
  exact (fun hLimit _ => hLimit) fromLimit fromCommand

theorem saturatePidMagnitude_monotone_command
    {limit commandA commandB : Nat} (h : commandA <= commandB) :
    saturatePidMagnitude limit commandA <= saturatePidMagnitude limit commandB := by
  by_cases hB : commandB <= limit
  · have hA : commandA <= limit := Nat.le_trans h hB
    rw [saturatePidMagnitude_eq_self_when_within hA]
    rw [saturatePidMagnitude_eq_self_when_within hB]
    exact h
  · rw [saturatePidMagnitude_eq_limit_when_above hB]
    exact saturatePidMagnitude_le_limit limit commandA

theorem saturatePidMagnitude_monotone_limit
    {limitA limitB command : Nat} (h : limitA <= limitB) :
    saturatePidMagnitude limitA command <= saturatePidMagnitude limitB command := by
  by_cases hA : command <= limitA
  · have hB : command <= limitB := Nat.le_trans hA h
    rw [saturatePidMagnitude_eq_self_when_within hA]
    rw [saturatePidMagnitude_eq_self_when_within hB]
    exact Nat.le_refl command
  · rw [saturatePidMagnitude_eq_limit_when_above hA]
    by_cases hB : command <= limitB
    · rw [saturatePidMagnitude_eq_self_when_within hB]
      exact Nat.le_of_lt (Nat.not_le.mp hA)
    · rw [saturatePidMagnitude_eq_limit_when_above hB]
      exact h

theorem saturatePidMagnitude_idempotent (limit command : Nat) :
    saturatePidMagnitude limit (saturatePidMagnitude limit command)
      = saturatePidMagnitude limit command := by
  unfold saturatePidMagnitude
  by_cases h : command <= limit
  · simp [h]
  · simp [h]

theorem saturatePidMagnitude_nested_le_outer_limit
    (outerLimit innerLimit command : Nat) :
    saturatePidMagnitude outerLimit (saturatePidMagnitude innerLimit command)
      <= outerLimit := by
  exact saturatePidMagnitude_le_limit outerLimit (saturatePidMagnitude innerLimit command)

theorem saturatePidMagnitude_nested_le_inner_limit
    (outerLimit innerLimit command : Nat) :
    saturatePidMagnitude outerLimit (saturatePidMagnitude innerLimit command)
      <= innerLimit := by
  exact Nat.le_trans
    (saturatePidMagnitude_le_command outerLimit (saturatePidMagnitude innerLimit command))
    (saturatePidMagnitude_le_limit innerLimit command)

theorem saturatePidMagnitude_nested_le_both_limits
    (outerLimit innerLimit command : Nat) :
    saturatePidMagnitude outerLimit (saturatePidMagnitude innerLimit command)
      <= outerLimit ∧
    saturatePidMagnitude outerLimit (saturatePidMagnitude innerLimit command)
      <= innerLimit := by
  constructor
  · exact saturatePidMagnitude_nested_le_outer_limit outerLimit innerLimit command
  · exact saturatePidMagnitude_nested_le_inner_limit outerLimit innerLimit command

structure PidMagnitudeStep where
  rawCommand : Nat
  limit : Nat
  deriving Repr

def boundedPidCommand (step : PidMagnitudeStep) : Nat :=
  saturatePidMagnitude step.limit step.rawCommand

theorem boundedPidCommand_respects_limit (step : PidMagnitudeStep) :
    boundedPidCommand step <= step.limit := by
  unfold boundedPidCommand
  exact saturatePidMagnitude_le_limit step.limit step.rawCommand

theorem boundedPidCommand_no_amplification_or_saturates (step : PidMagnitudeStep) :
    boundedPidCommand step <= step.rawCommand ∨ boundedPidCommand step = step.limit := by
  unfold boundedPidCommand
  exact saturatePidMagnitude_le_command_or_eq_limit step.limit step.rawCommand

theorem boundedPidCommand_eq_raw_or_limit (step : PidMagnitudeStep) :
    boundedPidCommand step = step.rawCommand ∨ boundedPidCommand step = step.limit := by
  unfold boundedPidCommand
  exact saturatePidMagnitude_eq_command_or_limit step.limit step.rawCommand

theorem boundedPidCommand_le_limit_and_raw_command (step : PidMagnitudeStep) :
    boundedPidCommand step <= step.limit ∧ boundedPidCommand step <= step.rawCommand := by
  unfold boundedPidCommand
  exact saturatePidMagnitude_le_limit_and_command step.limit step.rawCommand

theorem boundedPidCommand_within_limit_is_raw
    {limit rawCommand : Nat} (h : rawCommand <= limit) :
    boundedPidCommand { limit := limit, rawCommand := rawCommand } = rawCommand := by
  unfold boundedPidCommand
  exact saturatePidMagnitude_eq_self_when_within h

theorem boundedPidCommand_above_limit_is_limit
    {limit rawCommand : Nat} (h : ¬ rawCommand <= limit) :
    boundedPidCommand { limit := limit, rawCommand := rawCommand } = limit := by
  unfold boundedPidCommand
  exact saturatePidMagnitude_eq_limit_when_above h

theorem boundedPidCommand_le_common_upper_bound
    {limit rawCommand upperBound : Nat}
    (limitWithin : limit <= upperBound) :
    boundedPidCommand { limit := limit, rawCommand := rawCommand } <= upperBound := by
  unfold boundedPidCommand
  exact saturatePidMagnitude_le_common_upper_bound limitWithin

theorem boundedPidCommand_le_raw_upper_bound
    {limit rawCommand upperBound : Nat}
    (rawWithin : rawCommand <= upperBound) :
    boundedPidCommand { limit := limit, rawCommand := rawCommand } <= upperBound := by
  unfold boundedPidCommand
  exact saturatePidMagnitude_le_raw_upper_bound rawWithin

theorem boundedPidCommand_le_dual_upper_bound
    {limit rawCommand upperBound : Nat}
    (limitWithin : limit <= upperBound)
    (rawWithin : rawCommand <= upperBound) :
    boundedPidCommand { limit := limit, rawCommand := rawCommand } <= upperBound := by
  unfold boundedPidCommand
  exact saturatePidMagnitude_le_dual_upper_bound limitWithin rawWithin

theorem boundedPidCommand_zero_raw_command (limit : Nat) :
    boundedPidCommand { limit := limit, rawCommand := 0 } = 0 := by
  unfold boundedPidCommand
  exact saturatePidMagnitude_preserves_zero_command limit

theorem boundedPidCommand_at_limit (limit : Nat) :
    boundedPidCommand { limit := limit, rawCommand := limit } = limit := by
  unfold boundedPidCommand
  exact saturatePidMagnitude_at_limit limit

theorem boundedPidCommand_zero_limit (rawCommand : Nat) :
    boundedPidCommand { limit := 0, rawCommand := rawCommand } = 0 := by
  unfold boundedPidCommand
  exact saturatePidMagnitude_preserves_zero_limit rawCommand

theorem boundedPidCommand_monotone_raw_command
    {limit rawA rawB : Nat} (h : rawA <= rawB) :
    boundedPidCommand { limit := limit, rawCommand := rawA } <=
      boundedPidCommand { limit := limit, rawCommand := rawB } := by
  unfold boundedPidCommand
  exact saturatePidMagnitude_monotone_command h

theorem boundedPidCommand_monotone_limit
    {limitA limitB rawCommand : Nat} (h : limitA <= limitB) :
    boundedPidCommand { limit := limitA, rawCommand := rawCommand } <=
      boundedPidCommand { limit := limitB, rawCommand := rawCommand } := by
  unfold boundedPidCommand
  exact saturatePidMagnitude_monotone_limit h

theorem boundedPidCommand_idempotent (step : PidMagnitudeStep) :
    boundedPidCommand { step with rawCommand := boundedPidCommand step }
      = boundedPidCommand step := by
  unfold boundedPidCommand
  exact saturatePidMagnitude_idempotent step.limit step.rawCommand

theorem boundedPidCommand_stable_under_repeated_filtering
    (step : PidMagnitudeStep) :
    boundedPidCommand
        { limit := step.limit,
          rawCommand :=
            boundedPidCommand { step with rawCommand := boundedPidCommand step } }
      = boundedPidCommand step := by
  rw [boundedPidCommand_idempotent]
  exact boundedPidCommand_idempotent step

end SCPNFusionSolvers
