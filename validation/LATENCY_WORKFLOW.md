# Replicable controller-latency measurement workflow

This is the reproducible workflow behind the per-update controller latencies reported
in Paper B (Table 5) and `RESULTS.md`. It measures the wall-clock per-control-step
latency of each available controller and records a **host-provenance block** so every
number is attributable to the machine that produced it — never a hard-coded or
fabricated value.

## Run it

```bash
# Full campaign (writes a provenance-stamped artifact)
python validation/stress_test_campaign.py \
    --episodes 200 \
    --seed 42 \
    --output validation/reports/stress_test_campaign.json

# Quick smoke (CI): 10 episodes
python validation/stress_test_campaign.py --quick --output /tmp/stress_quick.json
```

Controllers run when their optional dependency is importable (`MPC` / `NMPC-JAX` need
JAX; `Nengo-SNN` needs Nengo; `Rust-PID` needs the `scpn_fusion_rs` wheel; `H-infinity`
is an opt-in research lane behind `--enable-hinf-research`). Absent controllers are
skipped, so the artifact records exactly what the host could measure.

## What the artifact contains

```jsonc
{
  "provenance": {
    "schema": "scpn-fusion-core.stress_test_campaign_provenance.v1",
    "timestamp_utc": "...",
    "git_sha": "...",
    "host":     { "cpu_model": "...", "machine": "...", "platform": "...", "logical_cpus": N },
    "software": { "python": "...", "numpy": "...", "jax": "...", "nengo": "...", "scpn_fusion_rs": "present|absent" },
    "methodology": { "n_episodes": N, "shot_duration_s": N, "seed": N|null,
                     "latency_metric": "per-control-step wall time (perf_counter_ns), p50/p95/p99 over episodes", ... }
  },
  "controllers": { "<name>": { "p50_latency_us": ..., "p95_latency_us": ..., "p99_latency_us": ..., ... } },
  "hinf_graduation": { ... }
}
```

`host.cpu_model` is read from the running machine (`/proc/cpuinfo` / `platform`) at run
time. **The CPU model is never hard-coded**, so a figure or table can cite the exact box
that produced its numbers.

## Independent verification (second host)

Run the identical command on a second, independent host (for example a cloud instance):

```bash
python validation/stress_test_campaign.py --episodes 200 --seed 42 \
    --output stress_test_campaign.<host-tag>.json
```

Compare the two `provenance.host` blocks and the per-controller `p50_latency_us`.
Latency is host-dependent, so absolute numbers differ between machines — but the
relative ordering (e.g. Rust-PID ≪ Python-PID ≪ float SNN ≪ NMPC-JAX) is
hardware-independent and is the claim that should be reproduced.

## Provenance discipline

- Latencies are **measured**, not projected. Any projected figure (e.g. dedicated-silicon
  FPGA/Loihi estimates) must be labelled *projected* and kept out of the measured bars.
- Do not attribute measured numbers to a CPU the artifact did not record. Cite
  `provenance.host.cpu_model` verbatim.
- Regenerate `RESULTS.md` / paper figures from the committed artifact rather than editing
  numbers by hand.
