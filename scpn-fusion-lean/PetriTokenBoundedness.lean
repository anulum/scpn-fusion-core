/-
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Fusion Core — Lean Petri Token Boundedness Proof
-/
import PIDBoundedOutput

namespace SCPNFusionSolvers

/-!
Finite-capacity Petri marking contract:

The executable Python Petri runtime stores place token densities in bounded
domains. This Lean model captures the safety-filter contract at the natural
number abstraction level: every place token admitted through the capacity
filter is less than or equal to its declared capacity, and repeated filtering
does not change an already filtered marking.
-/

structure PlaceToken where
  capacity : Nat
  tokens : Nat
  deriving Repr

def clampPlaceToken (place : PlaceToken) : Nat :=
  saturatePidMagnitude place.capacity place.tokens

def PlaceToken.Bounded (place : PlaceToken) : Prop :=
  place.tokens <= place.capacity

def filteredPlaceToken (place : PlaceToken) : PlaceToken :=
  { place with tokens := clampPlaceToken place }

theorem clampPlaceToken_respects_capacity (place : PlaceToken) :
    clampPlaceToken place <= place.capacity := by
  unfold clampPlaceToken
  exact saturatePidMagnitude_le_limit place.capacity place.tokens

theorem clampPlaceToken_no_amplification (place : PlaceToken) :
    clampPlaceToken place <= place.tokens := by
  unfold clampPlaceToken
  exact saturatePidMagnitude_le_command place.capacity place.tokens

theorem filteredPlaceToken_is_bounded (place : PlaceToken) :
    (filteredPlaceToken place).Bounded := by
  unfold PlaceToken.Bounded
  unfold filteredPlaceToken
  exact clampPlaceToken_respects_capacity place

theorem filteredPlaceToken_idempotent (place : PlaceToken) :
    filteredPlaceToken (filteredPlaceToken place) = filteredPlaceToken place := by
  unfold filteredPlaceToken
  unfold clampPlaceToken
  rw [saturatePidMagnitude_idempotent]

theorem bounded_place_unchanged_after_filter
    (place : PlaceToken) (h : place.Bounded) :
    filteredPlaceToken place = place := by
  unfold filteredPlaceToken
  unfold clampPlaceToken
  unfold PlaceToken.Bounded at h
  rw [saturatePidMagnitude_eq_self_when_within h]

theorem over_capacity_place_filters_to_capacity
    (place : PlaceToken) (h : ¬ place.tokens <= place.capacity) :
    (filteredPlaceToken place).tokens = place.capacity := by
  unfold filteredPlaceToken
  unfold clampPlaceToken
  exact saturatePidMagnitude_eq_limit_when_above h

theorem filteredPlaceToken_preserves_capacity (place : PlaceToken) :
    (filteredPlaceToken place).capacity = place.capacity := by
  rfl

structure TwoPlaceMarking where
  first : PlaceToken
  second : PlaceToken
  deriving Repr

def TwoPlaceMarking.Bounded (marking : TwoPlaceMarking) : Prop :=
  marking.first.Bounded ∧ marking.second.Bounded

def filteredTwoPlaceMarking (marking : TwoPlaceMarking) : TwoPlaceMarking :=
  { first := filteredPlaceToken marking.first,
    second := filteredPlaceToken marking.second }

theorem filteredTwoPlaceMarking_is_bounded (marking : TwoPlaceMarking) :
    (filteredTwoPlaceMarking marking).Bounded := by
  unfold TwoPlaceMarking.Bounded
  unfold filteredTwoPlaceMarking
  constructor
  · exact filteredPlaceToken_is_bounded marking.first
  · exact filteredPlaceToken_is_bounded marking.second

theorem filteredTwoPlaceMarking_idempotent (marking : TwoPlaceMarking) :
    filteredTwoPlaceMarking (filteredTwoPlaceMarking marking)
      = filteredTwoPlaceMarking marking := by
  unfold filteredTwoPlaceMarking
  rw [filteredPlaceToken_idempotent]
  rw [filteredPlaceToken_idempotent]

theorem bounded_two_place_marking_unchanged_after_filter
    (marking : TwoPlaceMarking) (h : marking.Bounded) :
    filteredTwoPlaceMarking marking = marking := by
  unfold filteredTwoPlaceMarking
  unfold TwoPlaceMarking.Bounded at h
  rw [bounded_place_unchanged_after_filter marking.first h.left]
  rw [bounded_place_unchanged_after_filter marking.second h.right]

abbrev Marking := List PlaceToken

def markingBounded : Marking → Prop
  | [] => True
  | place :: rest => place.Bounded ∧ markingBounded rest

def filteredMarking (marking : Marking) : Marking :=
  marking.map filteredPlaceToken

def markingTokenSum : Marking → Nat
  | [] => 0
  | place :: rest => place.tokens + markingTokenSum rest

def markingCapacitySum : Marking → Nat
  | [] => 0
  | place :: rest => place.capacity + markingCapacitySum rest

theorem filteredMarking_preserves_length (marking : Marking) :
    (filteredMarking marking).length = marking.length := by
  simp [filteredMarking]

theorem filteredMarking_is_bounded (marking : Marking) :
    markingBounded (filteredMarking marking) := by
  induction marking with
  | nil =>
      simp [filteredMarking, markingBounded]
  | cons head tail ih =>
      simp [filteredMarking, markingBounded]
      exact And.intro (filteredPlaceToken_is_bounded head) ih

theorem filteredMarking_idempotent (marking : Marking) :
    filteredMarking (filteredMarking marking) = filteredMarking marking := by
  induction marking with
  | nil =>
      simp [filteredMarking]
  | cons head tail ih =>
      simp [filteredMarking, filteredPlaceToken_idempotent]

theorem bounded_marking_unchanged_after_filter
    (marking : Marking) (h : markingBounded marking) :
    filteredMarking marking = marking := by
  induction marking with
  | nil =>
      simp [filteredMarking]
  | cons head tail ih =>
      unfold markingBounded at h
      simp [filteredMarking, bounded_place_unchanged_after_filter head h.left]
      exact ih h.right

theorem filteredMarking_capacity_sum_preserved (marking : Marking) :
    markingCapacitySum (filteredMarking marking) = markingCapacitySum marking := by
  induction marking with
  | nil =>
      simp [filteredMarking, markingCapacitySum]
  | cons head tail ih =>
      simp [filteredMarking, markingCapacitySum, filteredPlaceToken_preserves_capacity]
      exact ih

theorem filteredMarking_token_sum_le_original (marking : Marking) :
    markingTokenSum (filteredMarking marking) <= markingTokenSum marking := by
  induction marking with
  | nil =>
      simp [filteredMarking, markingTokenSum]
  | cons head tail ih =>
      simp [filteredMarking, markingTokenSum]
      exact Nat.add_le_add (clampPlaceToken_no_amplification head) ih

theorem filteredMarking_token_sum_le_capacity_sum (marking : Marking) :
    markingTokenSum (filteredMarking marking) <= markingCapacitySum marking := by
  induction marking with
  | nil =>
      simp [filteredMarking, markingTokenSum, markingCapacitySum]
  | cons head tail ih =>
      simp [filteredMarking, markingTokenSum, markingCapacitySum]
      exact Nat.add_le_add (clampPlaceToken_respects_capacity head) ih

theorem filteredMarking_token_sum_le_filtered_capacity_sum (marking : Marking) :
    markingTokenSum (filteredMarking marking) <=
      markingCapacitySum (filteredMarking marking) := by
  rw [filteredMarking_capacity_sum_preserved marking]
  exact filteredMarking_token_sum_le_capacity_sum marking

theorem filteredMarking_token_sum_safe_against_original_and_capacity
    (marking : Marking) :
    markingTokenSum (filteredMarking marking) <= markingTokenSum marking ∧
      markingTokenSum (filteredMarking marking) <= markingCapacitySum marking := by
  constructor
  · exact filteredMarking_token_sum_le_original marking
  · exact filteredMarking_token_sum_le_capacity_sum marking

theorem bounded_marking_token_sum_le_capacity_sum
    (marking : Marking) (h : markingBounded marking) :
    markingTokenSum marking <= markingCapacitySum marking := by
  induction marking with
  | nil =>
      simp [markingTokenSum, markingCapacitySum]
  | cons head tail ih =>
      unfold markingBounded at h
      simp [markingTokenSum, markingCapacitySum]
      exact Nat.add_le_add h.left (ih h.right)

theorem repeated_filteredMarking_capacity_sum_stable (marking : Marking) :
    markingCapacitySum (filteredMarking (filteredMarking marking)) =
      markingCapacitySum (filteredMarking marking) := by
  exact filteredMarking_capacity_sum_preserved (filteredMarking marking)

theorem repeated_filteredMarking_token_sum_stable (marking : Marking) :
    markingTokenSum (filteredMarking (filteredMarking marking)) =
      markingTokenSum (filteredMarking marking) := by
  rw [filteredMarking_idempotent marking]

theorem repeated_filteredMarking_is_bounded (marking : Marking) :
    markingBounded (filteredMarking (filteredMarking marking)) := by
  exact filteredMarking_is_bounded (filteredMarking marking)

theorem repeated_filteredMarking_token_sum_le_capacity_sum (marking : Marking) :
    markingTokenSum (filteredMarking (filteredMarking marking)) <=
      markingCapacitySum (filteredMarking (filteredMarking marking)) := by
  exact filteredMarking_token_sum_le_filtered_capacity_sum (filteredMarking marking)

end SCPNFusionSolvers
