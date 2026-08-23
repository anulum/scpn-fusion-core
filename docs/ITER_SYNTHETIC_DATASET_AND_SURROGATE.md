<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# ITER-like synthetic dataset and surrogate baseline

This page documents the `iter-synthetic-equilibrium-v1-10000` dataset and the
`neural-equilibrium-iter-synthetic-v2` model. They form a reproducible baseline
for one SCPN Fusion Core synthetic equilibrium distribution. They do not contain
ITER Organisation data or experimental shots.

## Getting started

Download the raw arrays and verify them:

```bash
gh release download iter-synthetic-dataset-v1 \
  --repo anulum/scpn-fusion-core \
  --pattern 'iter_X.npy' \
  --pattern 'iter_Y.npy' \
  --dir data
python tools/verify_iter_synthetic_dataset.py \
  --data data \
  --manifest data/iter_synthetic_v1_manifest.json \
  --full-field-scan
```

The manifest pins each file by byte size, shape, dtype, and SHA-256. The raw
field array is 1,310,720,128 bytes, so the project publishes it as a GitHub
release asset instead of a Git blob.

## Source provenance

The project generated the dataset with the repository snapshot
`13af94b0b968fc25bc1527dd19c0ae7fbbcf31e6`:

- generator: `tools/parallel_gen_iter.py`, SHA-256
  `0a5d16a972ebeeedf62f6b09b237b0f84b80ec62e142afeb32176bae280706b6`;
- solver: `src/scpn_fusion/core/fusion_kernel.py`, SHA-256
  `51ce4ece660d2a139aaf7936809c527e68a123e2cedd56285977b469c9619a17`;
- config: `validation/iter_config.json`, SHA-256
  `b608764c65051aa37a45289359300f59fb1640e5d7f56b0e24501e730e56a645`;
- 10,000 requested and valid samples, 12 worker chunks, seeds 42 through 53;
- PF/CS current multipliers sampled uniformly from 0.85 to 1.15;
- plasma-current multipliers sampled uniformly from 0.8 to 1.2.

The preserved NPY files no longer rely on an undocumented provenance claim.
All 10,000 encoded plasma-current values match the historical worker RNG stream
exactly. A detached copy of the pinned source snapshot reproduces the first
field of every worker chunk bit for bit: 12 of 12 checked fields, maximum
absolute difference 0.0. The current `FusionKernel` does not reproduce these
fields bitwise because its solver has evolved; reproduction therefore must use
the pinned historical snapshot.

The historical generator wrote an NPZ container. The retained arrays were
exported losslessly as `iter_X.npy` and `iter_Y.npy`; the exact conversion
command was not retained. File hashes bind the published arrays directly.

## Dataset schema

`iter_X.npy` has shape `(10000, 12)`. `iter_Y.npy` has shape
`(10000, 16384)` and each row reshapes to a `128 x 128` flux field.

| Index | Feature | Cohort role | Variation |
|---:|---|---|---|
| 0 | encoded plasma current | sampled input | 10,000 unique values |
| 1 | `B_T` | fixed descriptor | constant 5.3 T |
| 2 | `R_axis` | post-solve value | 36 grid values |
| 3 | `Z_axis` | post-solve value | 21 grid values |
| 4 | `pprime_scale` | fixed descriptor | constant 1.0 |
| 5 | `FFprime_scale` | fixed descriptor | constant 1.0 |
| 6 | `psi_axis` | post-solve value | 10,000 unique values |
| 7 | `psi_x` | post-solve value | 10,000 unique values |
| 8 | `kappa` | fixed descriptor | constant 1.7 |
| 9 | `delta_up` | fixed descriptor | constant 0.33 |
| 10 | `delta_low` | fixed descriptor | constant 0.33 |
| 11 | `q95` | fixed descriptor | constant 3.0 |

The generator perturbed seven coil currents, but it did not store those currents
in `X`. This omission makes the baseline under-conditioned: different coil
settings can map to inputs that expose only their post-solve axis and flux
summaries. Those post-solve features also leak solved-state information into a
nominal forward-prediction interface.

## Machine-conditioned v2 generator

The successor generator removes that leakage and uses the compiled predictive
Grad–Shafranov solver in SI units. Its 17 per-sample inputs are target plasma
current, six signed PF effective currents, five `pprime` knots, and five
`FFprime` knots. Grid, coil centres and turns, analytic wall and limiter,
COCOS 3, flux gauge, profile boundary knots, and a static analytic D-shaped
plasma support are machine metadata rather than repeated feature columns.

The fixed support is an explicit scientific boundary: this dataset models
equilibria within one machine-defined plasma region. It does not claim to learn
free-boundary shape evolution. Every coil filament is outside the numerical
rectangle, so the stored vacuum field is exactly reconstructed from the six
unit-current Green maps without an in-domain filament singularity.

Generate and verify a cohort with:

```bash
python tools/generate_iter_machine_conditioned_dataset.py \
  --spec validation/iter_machine_conditioned_v2_spec.json \
  --samples 50000 \
  --seed 20260822 \
  --grid-resolution 129 129 \
  --output-dir /path/to/iter-machine-conditioned-v2
python tools/verify_iter_machine_conditioned_dataset.py \
  --dataset-dir /path/to/iter-machine-conditioned-v2 \
  --full-field-scan
```

Generation is fail-closed. Accepted rows must stop below the iteration cap,
close the canonical Ip-normalized GS residual and target current, have a
non-zero smooth axis/X flux span, and contain a non-zero plasma self-field.
The tracked three-sample 33×33 reference cohort is at
`validation/reference/iter_machine_conditioned_v2_n3_seed20260822_33x33`;
its manifest binds all source and array hashes and records the exact replay
commands. Production datasets and checkpoints belong on owner-controlled
FTP/storage with public HTTPS retrieval where available; Git retains small
references, manifests, checksums, provenance, licensing, and reproduction
commands.

## Training and selection

The recoverable training path uses a deterministic seed-42 80/20 split: 8,000
training samples and 2,000 held-out samples. It fits a 20-component PCA on the
training subset, normalizes inputs and latent targets, and trains a
`12 -> 256 -> 128 -> 64 -> 20` JAX MLP with Adam.

```bash
python tools/train_iter_surrogate.py \
  --data data \
  --epochs 10000 \
  --out candidate_epoch10000.npz \
  --report training_report_epoch10000.json \
  --checkpoint-dir checkpoints \
  --checkpoint-every 1000 \
  --validation-fraction 0.2 \
  --seed 42
```

The selected epoch-10,000 model reports:

| Held-out metric | Value |
|---|---:|
| Mean relative L2 | 3.0502% |
| P95 relative L2 | 6.8996% |
| Maximum relative L2 | 16.1436% |
| Field RMSE | 52.5319 |
| Runtime/training path maximum absolute difference | `9.095e-13` |

Epoch 20,000 reduced mean relative L2 by only 0.87% relative to epoch 10,000,
while p95 worsened from 6.8996% to 7.0070% and the maximum worsened from
16.1436% to 17.2674%. The selection therefore favors the stronger error tail.
The machine-readable record is
[`validation/reports/iter_surrogate_v2_selection.json`](../validation/reports/iter_surrogate_v2_selection.json).

## Usage

The repository publishes the selected weights as
`weights/neural_equilibrium_iter_synthetic_v2.npz`. This artifact is a named
baseline; it does not replace the default v1 runtime weight.

```python
import numpy as np

from scpn_fusion.core.neural_equilibrium import NeuralEquilibriumAccelerator

model = NeuralEquilibriumAccelerator()
model.load_weights("weights/neural_equilibrium_iter_synthetic_v2.npz")
with np.load("weights/neural_equilibrium_iter_synthetic_v2.npz", allow_pickle=False) as weights:
    input_mean = np.asarray(weights["input_mean"], dtype=np.float64)
psi = model.predict(input_mean)
assert psi.shape == (128, 128)
assert np.all(np.isfinite(psi))
```

## Claim boundary and next model

The 3.05% held-out mean error measures interpolation inside one synthetic
generator distribution. It does not measure generalization to another machine,
another solver, or an experimental shot cohort. The baseline is not facility
validated, control-grade, safety certified, or an IDA/EFIT replacement.

A machine-conditioned successor must make causes, not solved consequences, its
inputs:

- versioned wall, limiter, LCFS, grid, and coordinate conventions;
- coil geometry, turns, polarity, measured currents, commanded currents, and
  Green/mqdsk mapping;
- normalized `pprime`, `FFprime`, pressure, `fpol`, current, and boundary
  profiles with explicit COCOS and gauge conventions;
- actuator history, diagnostics, reconstruction provenance, uncertainty, and
  quality flags;
- time and previous-state context for shot prediction;
- machine-held-out and shot-disjoint validation, including MAST and frozen
  same-case OMFIT/IDA references when available.

The physics solver should generate or constrain state targets. The learned model
should accelerate the solver, provide a warm start, or learn a bounded residual;
it should not hide omitted actuator variables behind post-solve features.
