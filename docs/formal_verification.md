# Formal Verification of SCPN Controller Properties

This document records the formal verification status of the stochastic
Petri net (SCPN) controller compilation pipeline.

## 1. Boundedness

**Property**: All place markings remain in [0, 1] at every simulation step.

**Proof method**: Constructive (software-enforced clamp).

Every `SCPNController.step()` call enforces:
```python
marking = np.clip(marking, 0.0, 1.0)
```
after the stochastic firing step. This guarantees boundedness by
construction: the marking vector is bounded in the unit hypercube.

**Verification**: `structure.py:verify_boundedness()` performs a
coverability analysis on the compiled net, confirming that no reachable
marking can exceed the [0, 1] bounds given the clamp.

**Status**: VERIFIED (constructive proof).

## 2. Liveness

**Property**: Every enabled transition can eventually fire from any
reachable marking.

**Proof method**: Monte Carlo liveness campaign.

The function `_run_flux_liveness_campaign()` in
`validation/task3_error_handling_audit.py` runs 1000 random initial
markings through 200 simulation steps each, tracking which transitions
fire at least once. A transition is considered "live" if it fires in
>= 99% of campaigns.

**Results**: >= 99% liveness across all transitions in the compiled
SPARC and ITER controller nets.

**Status**: EMPIRICALLY VERIFIED (99% coverage, not formally proved).

**Limitation**: This is a statistical argument, not a formal proof.
The continuous marking space means exhaustive state enumeration is not
applicable (unlike discrete Petri nets where coverability trees can
prove liveness). A formal proof would require showing that:
1. The stochastic firing rates are strictly positive for all enabled transitions.
2. The marking dynamics are ergodic (irreducible Markov chain on the continuous state space).

## 3. Reachability

**Property**: The set of reachable markings forms a connected subset
of the unit hypercube.

**Argument**: Because:
- The initial marking is in [0, 1]^n (guaranteed by construction).
- Each firing step applies continuous (not discrete) changes.
- The clamp to [0, 1] preserves connectivity.
- Stochastic transitions have positive probability for any non-zero marking.

The reachable set is path-connected in [0, 1]^n.

**Status**: ARGUED (not machine-checked).

## 4. Safety (Disruption Avoidance)

**Property**: The SNN-compiled controller does not command actuator
values that violate physical constraints (coil current limits, rate
limits, plasma current ramp rates).

**Verification**: The `disruption_contracts.py:run_disruption_episode()`
function enforces actuator clamps at every step. The
`stress_test_campaign.py` runs 100+ episodes verifying constraint
satisfaction.

**Status**: VERIFIED (by simulation).

## 5. Stability (Lyapunov)

**Property**: The closed-loop system with the H-infinity controller
is asymptotically stable.

**Verification**: The H-infinity controller synthesis (Riccati equations)
guarantees that all closed-loop eigenvalues have negative real parts.
This is verified by `HInfinityController.is_stable` property.

**Status**: VERIFIED (eigenvalue check).

## 6. Gaps and Future Work

### Not Machine-Checked
- Boundedness proof relies on software invariant (np.clip). A formal
  tool (Coq, Isabelle, TLA+) could verify the implementation matches
  the specification.
- Liveness is statistical, not formal. A measure-theoretic proof of
  ergodicity for the continuous-marking Markov chain is future work.

### Coq Formalization (Future)
- Encode the Petri net structure as an inductive type.
- Prove boundedness via induction on step count.
- Prove liveness via constructive witness (firing sequence).
- Target: Coq proof artifact for paper submission.

### Machine-Checked Bounds
- Consider interval arithmetic (MPFI) for guaranteed floating-point
  bounds on marking evolution.
- Relevant for FPGA deployment where bit-exact properties matter.
