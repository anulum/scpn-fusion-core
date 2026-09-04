<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Controller stress campaign

`validation/stress_test_campaign.py` evaluates the complete controller registry
under one immutable, unit-explicit contract. That contract fixes the plant,
targets, observations, full-coil action vector, actuator dynamics, disturbance
trace, timing boundary, integration rule, and disruption convention. A lane is
comparable only after every requested episode completes with the same scenario,
evaluation-contract, and per-episode trace digests. Unavailable and failed lanes
remain visible with null aggregates; they never become attractive zero-valued
measurements.

## Physical scenario

The scenario has three scientific inputs:

- `measurement_noise_std_m`: standard deviation, in metres, of independent
  zero-mean Gaussian errors on radial and vertical magnetic-axis measurements;
- `actuator_delay_s`: pure transport delay applied to controller commands before
  the existing actuator lag and rate/saturation limits;
- `master_seed`: unsigned 64-bit root seed.

Episode seeds are derived from the master seed, episode index, and scenario
digest with SHA-256. Before a controller runs, the common Python plant
materializes one two-channel NumPy/PCG64 noise trace. Every lane receives those
exact samples for that episode, proven by the stored trace digest. Execution
order, interruption, and resume therefore cannot alter the disturbance.

Every policy is called exactly once per plant step and returns the complete
ordered coil-current offset vector in MA. Pure command delay, first-order lag,
slew limit, and saturation are applied independently to every coil. Applied
current is always the immutable initial current plus the current offset, never
a repeatedly accumulated offset. The Rust lane uses the Rust PID through PyO3
as a policy on this same plant; it does not substitute a different reduced-order
plant or random-number generator. Policy latency encloses only that single
policy call. Simulation wall time is recorded separately.

Disruption is checked at `t=0` and after every actuation/state transition.
Radial and vertical absolute error use trapezoidal integration including the
initial and final samples. Magnetic effort is explicitly the time integral of
the sum of absolute applied coil-current offsets in MA s; it is not an energy
or efficiency measurement.

The neural surrogate currently has negligible response to these small
full-coil offsets. A surrogate campaign is therefore marked `wiring_only` and
can validate orchestration, replay, and policy plumbing, but can never serve as
controller-ranking or promotion evidence. Promotion-grade comparison requires
the real kernel and a physically admissible initial condition/criterion; a run
that disrupts at `t=0` remains a valid execution record but does not demonstrate
control performance.

## Controller registry

The registry is never environment-filtered:

- `PID`: Python two-axis `IsoFluxController` policy;
- `H-infinity`: two signed offset-setpoint state-space policies, protected by
  `SCPN_ENABLE_HINF_RESEARCH=1`;
- `LQR`: two signed offset-setpoint state-space policies;
- `MPC`: a full-coil linear policy calibrated against a separate equilibrium
  kernel instance;
- `NMPC-JAX`: explicitly `unavailable` because its current random, untrained
  dynamics MLP has neither a calibrated artifact nor a held-out gate;
- `LIF-NEF-SNN`: two-channel LIF/NEF policy advanced in exact 1 ms substeps;
- `Rust-PID`: native Rust two-axis PID called through PyO3.

The `policy_implementation` value in every lane binds checkpoints and results
to the specific implementation class above.

## Run and resume

```bash
python validation/stress_test_campaign.py \
  --episodes 1000 \
  --shot-duration 30 \
  --measurement-noise-std-m 0.2 \
  --actuator-delay-ms 50 \
  --seed 1942 \
  --checkpoint-dir /durable/path/stress-campaign \
  --output /durable/path/stress-campaign.json
```

Resume the exact campaign with the same command, same explicit `--seed`, same
checkpoint directory, and `--resume`. Recovery binds the checkpoint to all
scenario values, ordered controller set and implementation identities,
requested episode count, shot duration, H-infinity gate, surrogate selection,
configuration bytes, relevant Python and Rust source bytes, surrogate weights,
Python/platform identity, CPU count, and dependency versions. The immutable
`campaign.identity.json` and every lane checkpoint are content-validated; any
mismatch or corruption fails closed. A missing resume directory never creates a
new run silently.

Each attempted episode atomically replaces its lane checkpoint.
`progress.json` reports writer UUID/PID, UTC update timestamp, active lane,
episode index and seed, success/failure/unavailable counts, remaining work, and
current-process ETA. A background heartbeat refreshes the file during a long
episode. ETA is null while an episode is active because its duration is not yet
known.

If `--seed` is omitted, the campaign generates a 64-bit seed and records it. If
`--checkpoint-dir` is omitted, an identity-scoped directory below
`.cache/stress_campaign/` is used. `--quick` changes only the requested episode
count to ten; it is a development check and is not promotion evidence.

## Result interpretation

Lane statuses are:

- `complete`: all requested episodes succeeded under the declared scenario;
- `partial_failure`: at least one episode succeeded and at least one did not;
- `failed`: no episode succeeded and failures were recorded;
- `unavailable`: the lane was deliberately unavailable, for example because a
  research gate was disabled.

`campaign_complete` means that every requested lane and episode completed under
identical identities and traces. It does not by itself mean that the evidence is
scientifically promotion-grade. `promotion_eligible` additionally requires the
real-kernel `controller_comparison` scope. A completed surrogate report uses
`campaign_status: complete_wiring_only` and remains promotion-ineligible.

An episode may complete computationally and still be physically disrupted; the
disruption flag, time, rate, and DEF must therefore be evaluated separately.
The CLI exits with status 2 when any requested lane is unavailable, failed,
partial, or incomparable. H-infinity promotion additionally requires the real
kernel, explicit research gate, exact PID/H-infinity disturbance traces, at
least 100 complete episodes, and the declared reward, latency, and disruption
thresholds.

The schema-v3 JSON stores individual episode seeds and trace digests, realized
disturbance RMS, every policy-latency sample, simulation wall time, unit-explicit
R/Z and actuator metrics, bounded exception traceback, lane implementation,
scenario/evaluation/campaign identities, host/software provenance, execution
completion, and separate promotion eligibility. A historical result produced
before this contract cannot be treated as comparable.
