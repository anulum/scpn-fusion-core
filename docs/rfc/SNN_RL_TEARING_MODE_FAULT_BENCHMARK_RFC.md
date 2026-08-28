<!--
SCPN Fusion Core — SNN/RL Tearing-Mode Fault Benchmark RFC
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# SNN/RL Tearing-Mode Fault Benchmark

Status: Implemented synthetic-validation contract

## Scope

This benchmark compares two deterministic in-repository controller paths over
the same synthetic tearing-mode feature stream: a compiled stochastic Petri
net controller and a lightweight RL-style policy head. It also measures
recovery after a bounded bit flip in the SNN risk sequence.

The campaign is a reproducible software-validation lane. It is not a TORAX,
DIII-D, experimental, plant-control, or external-RL parity result. The
RL-style policy is a fixed analytic baseline, not a trained or independently
qualified external agent.

## Implementation surfaces

- `validation/snn_rl_tearing_mode_fault_benchmark.py`
- `tests/test_snn_rl_tearing_mode_fault_benchmark.py`
- `src/scpn_fusion/control/disruption_predictor.py`
- `src/scpn_fusion/scpn/compiler.py`
- `src/scpn_fusion/scpn/controller.py`
- `validation/reports/snn_rl_tearing_mode_fault_benchmark.json`
- `validation/reports/snn_rl_tearing_mode_fault_benchmark.md`

The benchmark remains offline and uses only NumPy plus repository-owned
simulation, compiler, artifact, and controller surfaces.

## Campaign contract

For each seeded episode, the runner:

1. generates a deterministic synthetic tearing-mode trace;
2. extracts shared disruption and toroidal-mode features;
3. evaluates the fixed RL-style risk head and compiled SNN controller;
4. records decision agreement, risk delta, and stochastic-versus-float
   controller diagnostics;
5. injects one bounded bit flip into the SNN sequence; and
6. measures P95 recovery time against the nominal sequence.

The public CLI exposes every acceptance threshold. `--strict` returns status 2
when any threshold fails, while the non-strict lane still writes the complete
scorecard for diagnosis.

## Default acceptance gates

- SNN/RL decision agreement: at least `0.95`.
- Mean absolute risk delta: at most `0.08`.
- Stochastic-versus-float equivalence error: at most `0.05`.
- Oracle-versus-SC mean marking delta: at most `0.05`.
- Oracle-versus-SC mean firing delta: at most `0.05`.
- P95 bit-flip recovery: at most `1.0 ms`.

Agreement describes the two in-repository decision labels only. No metric is
reported as external-solver parity.

## Serialized evidence

Reports use schema version 2 and report kind
`snn_rl_tearing_mode_fault_benchmark`. The descriptive report-kind key owns
the benchmark payload. Unversioned coded payloads are stale and rejected; no
compatibility alias is retained.

JSON and Markdown reports include the seed, campaign dimensions, threshold
values, per-gate outcomes, aggregate controller metrics, generation timestamp,
and measured runtime. Fixed-seed scorecards are deterministic except for the
timestamp and runtime metadata.

## Safety and claim boundary

- Risk values and injected fault values remain bounded to `[0, 1]`.
- Public input validation rejects invalid or non-finite campaign and threshold
  values before controller construction.
- The benchmark does not alter production controller defaults.
- Passing this synthetic lane does not authorize experimental, deployment,
  superiority, publication, or release claims.
