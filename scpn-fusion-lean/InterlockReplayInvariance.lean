/-
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Fusion Core — Lean Petri-to-SNN Interlock & Replay Invariance Proof
-/
import SNNReachabilityPreservation

namespace SCPNFusionSolvers

/-!
Petri-to-SNN interlock and replay contract (M-1 compiler+runtime contracts):

`SNNReachabilityPreservation` proves the *static* graph contract — every Petri
adjacency edge survives compilation, so reachability is preserved and reflected.
This module adds the *dynamic* contract over token markings:

* **Interlock semantics.** A transition fires only when each input place holds a
  token *and* every interlock guard place is clear (empty). A raised interlock
  (a marked guard place) disables the transition; firing a disabled transition is
  a no-op. The compiled SNN step is defined to reuse the identical enabledness and
  update rule, so `compile_preserves_enabled` / `compile_step_commutes` machine-
  check that compilation cannot weaken an interlock.

* **Replay invariance.** `replay` folds `fire` over a firing sequence. It is a
  pure, total function of the initial marking and the sequence — no floats, no IO,
  no randomness — so a replayed episode is deterministic and machine-independent.
  `compile_replay_commutes` proves compiling then replaying equals replaying then
  compiling; `replay_append` gives prefix determinism (the checkpoint law behind a
  replay certificate); `replay_keeps_guard_clear` is a safety invariant: an
  interlock guard that starts clear and is never produced stays clear across the
  whole episode, in both the Petri net and its compiled SNN.

The SNN semantics is pinned to the Petri semantics by definition; these theorems
fail to compile if either side later drifts, which is the compiler contract.
-/

/-- A marking assigns a token count to each place (place identifiers are `Nat`). -/
abbrev Marking := Nat → Nat

/-- Point update of a marking at a single place. -/
def upd (m : Marking) (p v : Nat) : Marking := fun q => if q = p then v else m q

/-- An interlock transition: input places consumed, output places produced, and
interlock guard places that must be clear for the transition to be enabled. -/
structure InterlockTransition where
  inputs : List Nat
  outputs : List Nat
  guards : List Nat
  deriving Repr

/-- Consume one token from a single place (saturating at zero via `Nat` subtraction). -/
def consume1 (m : Marking) (p : Nat) : Marking := upd m p (m p - 1)

/-- Produce one token at a single place. -/
def produce1 (m : Marking) (p : Nat) : Marking := upd m p (m p + 1)

/-- Consume one token from each place in `ps`. -/
def consume (m : Marking) (ps : List Nat) : Marking := ps.foldl consume1 m

/-- Produce one token at each place in `ps`. -/
def produce (m : Marking) (ps : List Nat) : Marking := ps.foldl produce1 m

/-- Every input place holds at least one token. -/
def inputsAvailable (m : Marking) (t : InterlockTransition) : Bool :=
  t.inputs.all (fun p => decide (0 < m p))

/-- Every interlock guard place is clear (no raised interlock). -/
def guardsClear (m : Marking) (t : InterlockTransition) : Bool :=
  t.guards.all (fun p => decide (m p = 0))

/-- A transition is enabled when its inputs are available and no interlock is raised. -/
def enabled (m : Marking) (t : InterlockTransition) : Bool :=
  inputsAvailable m t && guardsClear m t

/-- Fire a transition: if enabled, consume inputs then produce outputs; otherwise
the marking is unchanged (a disabled/interlocked transition is a safe no-op). -/
def fire (m : Marking) (t : InterlockTransition) : Marking :=
  if enabled m t then produce (consume m t.inputs) t.outputs else m

/-- Compilation carries the place-marking onto the place-neurons unchanged. -/
def compileMarking (m : Marking) : Marking := m

/-- Compilation carries the transition incidence unchanged (cf. `compilePetriToSnnGraph`). -/
def compileTransition (t : InterlockTransition) : InterlockTransition := t

/-- The SNN enabledness predicate is pinned to the Petri one. -/
def snnEnabled (m : Marking) (t : InterlockTransition) : Bool := enabled m t

/-- The SNN step is pinned to the Petri firing rule. -/
def snnFire (m : Marking) (t : InterlockTransition) : Marking := fire m t

/-- Replay a firing sequence over the Petri semantics. -/
def replay (m : Marking) (ts : List InterlockTransition) : Marking := ts.foldl fire m

/-- Replay a firing sequence over the SNN semantics. -/
def snnReplay (m : Marking) (ts : List InterlockTransition) : Marking := ts.foldl snnFire m

/-! ### Point-update lemmas -/

theorem upd_same (m : Marking) (p v : Nat) : upd m p v p = v := by
  simp [upd]

theorem upd_other (m : Marking) (p v q : Nat) (h : q ≠ p) : upd m p v q = m q := by
  simp [upd, h]

theorem consume1_other (m : Marking) (p q : Nat) (h : q ≠ p) : consume1 m p q = m q := by
  simp [consume1, upd_other, h]

theorem produce1_other (m : Marking) (p q : Nat) (h : q ≠ p) : produce1 m p q = m q := by
  simp [produce1, upd_other, h]

/-- Folding single-place consumes leaves a place outside the list untouched. -/
theorem consume_untouched (q : Nat) :
    ∀ (ps : List Nat) (m : Marking), q ∉ ps → consume m ps q = m q := by
  intro ps
  induction ps with
  | nil => intro m _; rfl
  | cons p ps ih =>
      intro m hq
      have hne : q ≠ p := fun h => hq (by simp [h])
      have hnotin : q ∉ ps := fun h => hq (by simp [h])
      have : consume (consume1 m p) ps q = (consume1 m p) q := ih (consume1 m p) hnotin
      simpa [consume, List.foldl, consume1_other m p q hne] using this

/-- Folding single-place produces leaves a place outside the list untouched. -/
theorem produce_untouched (q : Nat) :
    ∀ (ps : List Nat) (m : Marking), q ∉ ps → produce m ps q = m q := by
  intro ps
  induction ps with
  | nil => intro m _; rfl
  | cons p ps ih =>
      intro m hq
      have hne : q ≠ p := fun h => hq (by simp [h])
      have hnotin : q ∉ ps := fun h => hq (by simp [h])
      have : produce (produce1 m p) ps q = (produce1 m p) q := ih (produce1 m p) hnotin
      simpa [produce, List.foldl, produce1_other m p q hne] using this

/-! ### Interlock semantics -/

/-- A raised interlock (a marked guard place) makes the guard check fail. -/
theorem guard_marked_blocks (m : Marking) (t : InterlockTransition)
    (p : Nat) (hp : p ∈ t.guards) (hmarked : m p ≠ 0) :
    guardsClear m t = false := by
  cases hg : guardsClear m t with
  | false => rfl
  | true =>
      exfalso
      rw [guardsClear, List.all_eq_true] at hg
      have hp0 := hg p hp
      rw [decide_eq_true_eq] at hp0
      exact hmarked hp0

/-- A blocked guard disables the transition. -/
theorem guard_marked_disables (m : Marking) (t : InterlockTransition)
    (p : Nat) (hp : p ∈ t.guards) (hmarked : m p ≠ 0) :
    enabled m t = false := by
  simp [enabled, guard_marked_blocks m t p hp hmarked]

/-- Firing a disabled transition changes nothing. -/
theorem fire_disabled_noop (m : Marking) (t : InterlockTransition)
    (h : enabled m t = false) : fire m t = m := by
  simp [fire, h]

/-- A raised interlock makes firing a no-op (interlock semantics). -/
theorem interlock_raised_noop (m : Marking) (t : InterlockTransition)
    (p : Nat) (hp : p ∈ t.guards) (hmarked : m p ≠ 0) :
    fire m t = m :=
  fire_disabled_noop m t (guard_marked_disables m t p hp hmarked)

/-! ### Compilation preserves the dynamic contract -/

theorem compile_preserves_enabled (m : Marking) (t : InterlockTransition) :
    snnEnabled (compileMarking m) (compileTransition t) = enabled m t := rfl

theorem compile_step_commutes (m : Marking) (t : InterlockTransition) :
    compileMarking (fire m t) = snnFire (compileMarking m) (compileTransition t) := rfl

/-- Compilation cannot weaken a raised interlock: the compiled step is also inert. -/
theorem compile_preserves_interlock_block (m : Marking) (t : InterlockTransition)
    (p : Nat) (hp : p ∈ t.guards) (hmarked : m p ≠ 0) :
    snnFire (compileMarking m) (compileTransition t) = compileMarking m := by
  show fire m t = m
  exact interlock_raised_noop m t p hp hmarked

/-! ### Replay invariance -/

theorem replay_nil (m : Marking) : replay m [] = m := rfl

theorem replay_cons (m : Marking) (t : InterlockTransition) (ts : List InterlockTransition) :
    replay m (t :: ts) = replay (fire m t) ts := rfl

/-- Prefix determinism: the state after a firing sequence splits at any prefix.
This is the checkpoint law a deterministic replay certificate relies on. -/
theorem replay_append (m : Marking) :
    ∀ (pre suf : List InterlockTransition),
      replay m (pre ++ suf) = replay (replay m pre) suf := by
  intro pre
  induction pre generalizing m with
  | nil => intro suf; rfl
  | cons t pre ih =>
      intro suf
      exact ih (fire m t) suf

/-- Replaying commutes with compilation: compile-then-replay equals
replay-then-compile. Replay invariance survives the Petri→SNN compiler. -/
theorem compile_replay_commutes (m : Marking) :
    ∀ (ts : List InterlockTransition),
      compileMarking (replay m ts) = snnReplay (compileMarking m) (ts.map compileTransition) := by
  intro ts
  induction ts generalizing m with
  | nil => rfl
  | cons t ts ih =>
      simpa [replay, snnReplay, List.foldl, List.map, compileMarking, snnFire, compileTransition]
        using ih (fire m t)

/-! ### Safety invariant across a replayed episode -/

/-- Consuming tokens keeps a clear place clear (`Nat` subtraction saturates at 0). -/
theorem consume_keeps_zero (q : Nat) :
    ∀ (ps : List Nat) (m : Marking), m q = 0 → consume m ps q = 0 := by
  intro ps
  induction ps with
  | nil => intro m hzero; simpa [consume] using hzero
  | cons p ps ih =>
      intro m hzero
      have hstep : consume1 m p q = 0 := by
        by_cases hpq : q = p
        · subst hpq; simp [consume1, upd_same, hzero]
        · rw [consume1_other m p q hpq]; exact hzero
      have := ih (consume1 m p) hstep
      simpa [consume, List.foldl] using this

/-- Firing keeps a place clear if it starts clear and is not produced. Inputs may
list the place (a consume of zero stays zero under `Nat` subtraction). -/
theorem fire_keeps_zero (m : Marking) (t : InterlockTransition) (q : Nat)
    (hzero : m q = 0) (hnotout : q ∉ t.outputs) : fire m t q = 0 := by
  unfold fire
  by_cases henabled : enabled m t
  · simp only [henabled, if_true]
    rw [produce_untouched q t.outputs (consume m t.inputs) hnotout]
    exact consume_keeps_zero q t.inputs m hzero
  · simp only [henabled]; exact hzero

/-- Safety invariant: an interlock guard place that starts clear and is produced by
no transition in the episode stays clear throughout the whole replay — so it never
spuriously raises the interlock. By `compile_replay_commutes` the compiled SNN
exhibits the identical marking. -/
theorem replay_keeps_guard_clear (q : Nat) :
    ∀ (ts : List InterlockTransition) (m : Marking),
      m q = 0 → (∀ t ∈ ts, q ∉ t.outputs) → replay m ts q = 0 := by
  intro ts
  induction ts with
  | nil => intro m hzero _; simpa [replay] using hzero
  | cons t ts ih =>
      intro m hzero hnotout
      have hthis : q ∉ t.outputs := hnotout t (by simp)
      have hrest : ∀ t' ∈ ts, q ∉ t'.outputs := fun t' ht' => hnotout t' (by simp [ht'])
      have hfire : fire m t q = 0 := fire_keeps_zero m t q hzero hthis
      simpa [replay, List.foldl] using ih (fire m t) hfire hrest

end SCPNFusionSolvers
