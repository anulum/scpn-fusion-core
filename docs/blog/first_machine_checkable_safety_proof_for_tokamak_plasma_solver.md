# First Machine-Checkable Safety Proof for a Tokamak Plasma Solver

SCPN Fusion Core now carries a committed Lean 4 proof surface for a safety
boundary in the Grad-Shafranov solver path. The first theorem is intentionally
small and strict: if physical-case validation rejects an input, the solver must
return that exact validation error before numerical work can begin.

The proof lives in `scpn-fusion-lean/SafetyProof.lean`:

```lean
theorem solveGradShafranov_rejects_validation_error
    (c : GradShafranovCase) (err : String)
    (h : validateCase c = Except.error err) :
    solveGradShafranov c = Except.error err := by
  simp [solveGradShafranov, h]
```

## Why this matters

Fusion control software has to reject invalid physical states predictably. A
simulation that silently accepts malformed geometry, invalid grid dimensions, or
nonphysical inputs can produce convincing-looking numbers that should never have
entered the solver core.

This proof makes the first boundary machine-checkable: validation failure is not
just tested by examples; Lean verifies that the solver propagates validation
errors exactly.

## Evidence boundary

This is not a claim that the full tokamak plant model is formally verified. It
is a first committed proof in a growing proof surface.

Current machine-checked scope:

- Lean 4 project: `scpn-fusion-lean/`
- Proof file: `scpn-fusion-lean/SafetyProof.lean`
- Property: invalid Grad-Shafranov case descriptions fail closed before solver
  execution
- CI lane: Lean safety-proof build in `.github/workflows/ci.yml`

Current non-proof scope:

- PID bounded-output proof is planned, not yet proven.
- Petri-net to SNN reachability preservation is planned, not yet proven.
- Full nonlinear plasma physics correctness is not claimed.
- Regulatory certification is not claimed.

## Next proof targets

The next useful proofs are control-facing because they connect directly to
safety certification:

1. PID bounded-output property: prove that configured actuator saturation bounds
   are never exceeded.
2. SCPN compiler reachability preservation: prove that the SNN artifact
   preserves the relevant Petri-net reachability contract.
3. Solver input-domain contracts: expand fail-closed proofs from validation
   rejection to individual geometry and grid invariants.

The engineering strategy is incremental: prove one small, executable safety
contract at a time; keep each proof wired into CI; avoid marketing claims that
outrun the machine-checked artifact.
