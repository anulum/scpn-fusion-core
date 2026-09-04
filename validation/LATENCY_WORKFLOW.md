<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Replicable control-policy latency workflow

This workflow records the latency of exactly one policy call per plant step on
the common stress-campaign plant. It does not measure a different plant for any
controller and does not treat simulation wall time as policy latency. Every
number is bound to the host, software, scenario, evaluation contract, policy
implementation, episode seed, and exact disturbance trace that produced it.

## Run it

```bash
# Full campaign (writes a provenance-stamped artifact)
python validation/stress_test_campaign.py \
    --episodes 200 \
    --seed 42 \
    --output validation/reports/stress_test_campaign.json

# Ten-episode development run; not promotion evidence
python validation/stress_test_campaign.py --quick --output /tmp/stress_quick.json
```

The complete seven-lane registry is always present. A missing dependency,
disabled H-infinity research gate, or uncalibrated NMPC lane is serialized as
`unavailable` with null timing aggregates, and makes the command exit with
status 2. It is never silently skipped. See
[`docs/STRESS_CAMPAIGN.md`](../docs/STRESS_CAMPAIGN.md) for the controller and
physical contracts.

## What the artifact contains

```jsonc
{
  "provenance": {
    "schema": "scpn-fusion-core.stress-test-campaign-provenance.v3",
    "timestamp_utc": "...",
    "git_sha": "...",
    "host":     { "cpu_model": "...", "machine": "...", "platform": "...", "logical_cpus": N },
    "software": { "python": "...", "numpy": "...", "jax": "...", "nengo": "...", "scpn_fusion_rs": "present|absent" },
    "methodology": { "n_episodes": N, "shot_duration_s": N, "seed": N,
                     "latency_metric": "exactly one policy step", ... }
  },
  "campaign_status": "complete|complete_wiring_only|incomplete",
  "promotion_eligible": false,
  "controllers": { "<name>": { "p50_control_policy_latency_us": ..., ... } },
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

Compare the two `provenance.host` blocks and policy-latency distributions only
after verifying identical scenario, evaluation-contract, implementation, and
trace digests. Absolute values and relative ordering are both host- and
implementation-dependent; neither may be generalized without multi-host
evidence.

## Provenance discipline

- Latencies are **measured**, not projected. Any projected figure (e.g. dedicated-silicon
  FPGA/Loihi estimates) must be labelled *projected* and kept out of the measured bars.
- Do not attribute measured numbers to a CPU the artifact did not record. Cite
  `provenance.host.cpu_model` verbatim.
- A surrogate run is wiring-only even when computationally complete. It must not
  update controller ranking, `RESULTS.md`, or publication figures.
- Regenerate downstream tables from a promotion-eligible committed artifact;
  never edit measurements by hand.
