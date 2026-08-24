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
  --checkpoint-every 100 \
  --output-dir /path/to/iter-machine-conditioned-v2
python tools/verify_iter_machine_conditioned_dataset.py \
  --dataset-dir /path/to/iter-machine-conditioned-v2 \
  --full-field-scan
```

The generator writes fixed-shape float64 NPY memmaps directly to a hidden
`.partial` directory and advances a sibling recovery JSON only after flushing
the arrays. An interrupted run resumes with the identical immutable run
contract and candidate order:

```bash
python tools/generate_iter_machine_conditioned_dataset.py \
  --spec validation/iter_machine_conditioned_v2_spec.json \
  --samples 50000 \
  --seed 20260822 \
  --grid-resolution 129 129 \
  --checkpoint-every 100 \
  --resume \
  --output-dir /path/to/iter-machine-conditioned-v2
```

The visible output directory appears only after full-field verification.
`--pause-after-accepted N` provides an intentional operator checkpoint for
testing or maintenance; it is not needed during an uninterrupted run.

Generation is fail-closed. Accepted rows must stop below the iteration cap,
close the canonical Ip-normalized GS residual and target current, have a
non-zero smooth axis/X flux span, and contain a non-zero plasma self-field.
The tracked three-sample 33×33 reference cohort is at
`validation/reference/iter_machine_conditioned_v2_n3_seed20260822_33x33`;
its manifest binds all source and array hashes and records the exact replay
commands. Production datasets and checkpoints remain in local or owner-approved
large-artifact storage until a publication endpoint is explicitly selected.
Git retains small references, manifests, checksums, provenance, licensing, and
reproduction commands rather than the large arrays.

## Historical v1 training and selection

The historical v1 training path uses a deterministic seed-42 80/20 split: 8,000
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

This result remains a reproducible historical baseline. It must not be used as
evidence for the 17-input machine-conditioned v2 cohort because its data,
feature contract and model-selection problem are different.

## Machine-conditioned v2 successor training

The successor trainer consumes only a fully authenticated v2 dataset. Its v2
training contract uses a deterministic seed-42 80/10/5/5 split before fitting
any transform. Input scaling, field mean, randomized PCA basis, and latent
scaling are learned exclusively from the 40,000 training rows. The 5,000
validation rows select the epoch, 2,500 calibration rows construct post-hoc
uncertainty evidence only after selection, and 2,500 final-test rows remain
untouched until both model and epoch are frozen. Index hashes for all four roles
are bound into recovery state, weights, and the report.

The default model is a `17 -> 512 -> 256 -> 128 -> 64` float64 JAX MLP. The
64-dimensional target is produced by recovery-safe streaming randomized PCA
with 16 oversampling vectors and one power iteration. The implementation never
forms the infeasible sample-by-sample Gram matrix and never materialises all
50,000 fields as one in-memory matrix.

The default loss is 90% field-aligned and 10% standardised-latent MSE. Because
the PCA basis is orthonormal, raw latent squared error equals squared error in
the reconstructible field subspace. Per-component weights undo latent
standardisation and per-row weights divide by the field norm; both are scaled
only by training-set constants. This corrects the old objective's tendency to
give equal importance to low-energy PCA tail modes. Validation uses the same
declared objective. A finite-sample 95% split-conformal relative-L2 bound is
then fitted on calibration residuals and its empirical coverage is reported on
the untouched test partition.

```bash
python tools/train_machine_conditioned_equilibrium_surrogate.py \
  --dataset-dir data/iter_machine_conditioned_v2_n50000_seed20260822_129x129 \
  --out artifacts/iter_machine_conditioned_field_aware_20260824_seed42/candidate_local.npz \
  --report artifacts/iter_machine_conditioned_field_aware_20260824_seed42/report_local.json \
  --checkpoint-dir artifacts/iter_machine_conditioned_field_aware_20260824_seed42/checkpoints \
  --epochs 20000 \
  --seed 42 \
  --validation-fraction 0.10 \
  --calibration-fraction 0.05 \
  --test-fraction 0.05 \
  --field-loss-weight 0.9 \
  --pca-components 64 \
  --pca-oversampling 16 \
  --pca-power-iterations 1 \
  --pca-chunk-rows 256 \
  --evaluation-every 100 \
  --checkpoint-every 500 \
  --early-stopping-patience 50
```

The completed local seed-42 run selected epoch 20,000 from validation and then
opened calibration and final test exactly once:

| Untouched-test measure | Result |
| --- | ---: |
| Field RMSE | `0.020297107883367773 Wb/rad` |
| Mean relative L2 | `0.0015291318885462526` |
| P95 relative L2 | `0.0028262867732278905` |
| Maximum relative L2 | `0.004641217924994206` |
| 95% conformal relative-L2 bound | `0.002787281010345701` |
| Test coverage of that bound | `0.9464` |
| Runtime/training maximum absolute difference | `7.105427357601002e-15 Wb/rad` |

Training took `6692.994974019006 s` on the local CUDA device. The candidate
SHA-256 is
`a6684a82ec0b683384b512b1723be5ac16218852f516b46fb2134c99a7216d41`;
the report SHA-256 is
`4b8185b24d58d1b3e59acef5381170c894c54557a2bcba60d7d0838dece9c396`.
The final optimiser stage and complete PCA stage authenticate as
`8b21b2e630f6a8e11de8f3909b8e5d4f8c4db3f57a4015d83d16affabb0ce7a4`
and
`bda830de8c22e15a6e9bb32a442e4fbddd67143243e17c12aab8282e7c822698`.
All remain local and unpromoted. Against the earlier 80/20 objective baseline,
the new final-test values are lower by 56.42% for field RMSE, 52.94% for mean,
59.81% for p95, and 76.79% for maximum relative L2. That comparison is
descriptive rather than a strict paired benchmark because the old run used a
different holdout and model-selection contract.

Resume repeats dataset authentication, verifies the SHA-bound PCA stage and
optimiser state, and then continues the exact Adam trajectory:

```bash
python tools/train_machine_conditioned_equilibrium_surrogate.py \
  --dataset-dir data/iter_machine_conditioned_v2_n50000_seed20260822_129x129 \
  --out artifacts/iter_machine_conditioned_field_aware_20260824_seed42/candidate_local.npz \
  --report artifacts/iter_machine_conditioned_field_aware_20260824_seed42/report_local.json \
  --checkpoint-dir artifacts/iter_machine_conditioned_field_aware_20260824_seed42/checkpoints \
  --epochs 20000 \
  --resume
```

Recovery identities bind the dataset and array hashes, all four split-index
hashes, loss contract, PCA configuration and result, optimiser hyperparameters,
and the exact trainer, PCA, and production-runtime source files. The final NPZ
records the feature order, grid shape, transforms, network weights, selected
epoch, split hashes, dataset manifest hash, and source hashes. Before
completion, the trainer loads that NPZ through
`NeuralEquilibriumAccelerator`, performs a finite prediction, and checks its
numerical parity with the training path.

The report deliberately labels every output
`completed_local_candidate_not_promoted`. Selection requires the validation
objective; reporting separately exposes full-field RMSE and relative-L2 on
validation, calibration, and final test, the PCA-only test floor, conformal
coverage, and runtime parity. A good synthetic final-test result does
not make the artifact facility-validated, free-boundary capable, or an IDA/EFIT
replacement. Dataset, checkpoints, candidate weights, and reports remain local
until an owner-approved large-artifact publication endpoint is chosen; Git
receives only bounded references, provenance, checksums and reproduction
instructions.

## DeepONet operator candidate

`tools/train_machine_conditioned_deeponet.py` implements a genuine branch-trunk
operator following the branch-trunk construction of Lu et al. (2021),
DOI [`10.1038/s42256-021-00302-5`](https://doi.org/10.1038/s42256-021-00302-5).
The branch consumes the same 17 causal pre-solve controls; the trunk consumes
normalised physical `(R, Z)` coordinates; their 64-dimensional inner
product predicts the field residual around a training-only spatial mean. It
does not use PCA targets. Coordinate/shot minibatches optimise a field-norm
weighted physical objective with float32 AdamW, while final artifacts and all
reported field metrics use float64.

```bash
python tools/train_machine_conditioned_deeponet.py \
  --dataset-dir data/iter_machine_conditioned_v2_n50000_seed20260822_129x129 \
  --out artifacts/iter_machine_conditioned_deeponet_20260824_seed42_74c05fda/candidate_local.npz \
  --report artifacts/iter_machine_conditioned_deeponet_20260824_seed42_74c05fda/report_local.json \
  --checkpoint-dir artifacts/iter_machine_conditioned_deeponet_20260824_seed42_74c05fda/checkpoints \
  --steps 20000 \
  --seed 42 \
  --validation-fraction 0.10 \
  --calibration-fraction 0.05 \
  --test-fraction 0.05 \
  --basis-width 64 \
  --shot-batch-size 256 \
  --coordinate-batch-size 512 \
  --validation-probe-shots 1024 \
  --validation-probe-coordinates 2048 \
  --evaluation-every 250 \
  --checkpoint-every 500 \
  --early-stopping-patience 40
```

The completed seed-42 run from published source SHA `74c05fda` evaluated all
20,000 requested optimiser steps and selected step 19,500 from validation
before opening calibration and the untouched final test:

| Untouched-test measure | Result |
| --- | ---: |
| Field RMSE | `0.023554936122933533 Wb/rad` |
| Mean relative L2 | `0.0018768039361196063` |
| P95 relative L2 | `0.0027345910550329064` |
| Maximum relative L2 | `0.0042368007914158155` |
| 95% conformal relative-L2 bound | `0.0027451468653940715` |
| Test coverage of that bound | `0.9516` |
| JAX/runtime maximum absolute difference | `1.587064968333607e-7 Wb/rad` |
| Rust/NumPy untouched-test maximum absolute difference | `8.881784197001252e-15 Wb/rad` |

Training took `432.47219220298575 s` on the local CUDA device, including the
authenticated full-cohort scan and final evaluation. The candidate SHA-256 is
`68a432399bc647308ee081eb6ef53603ace53c323a9f2d4c9b41cd8817b67fb4`;
the report SHA-256 is
`2def1624c7009d2490a3433a333405b32d17461332d678e3f6566d610f72f769`.
The final optimiser and statistics stages authenticate as
`2cf84617b68f06c6c12afec66935d5f65c3a1a8c752eeb5df30366e04bbd091a`
and
`c2849f7283eb7fba1e9fed323ab196e6c82f9a6b369fac63537027e2aac1f1de`.
The Rust/NumPy sweep covered all 2,500 untouched-test shots and remained
inside the declared `1e-14` relative/absolute tolerance. All 14 branch/trunk
network arrays are bit-identical to the pre-publication run; only the recorded
source-hash member changed after the logging-contract fix. Candidate, report,
statistics and recovery checkpoints remain local and unpromoted.

The operator is not uniformly better than the field-aware PCA-MLP on this
one fixed-machine cohort: the PCA-MLP has lower field RMSE and mean relative
L2, while DeepONet has lower p95 and maximum relative L2 and closer-to-nominal
95% conformal coverage. Those complementary outcomes justify retaining both
as independently reproducible candidates rather than promoting either from a
single synthetic-machine experiment.

Every minibatch is a pure function of the seed and absolute optimiser step.
Statistics and optimiser checkpoints are atomically replaced and SHA-256
checked; recovery rejects symlinked stage files. Their identities are bound to
the dataset, four split hashes, validation probe, hyperparameters, and exact
trainer/runtime sources. A resumed run is tested against an
uninterrupted trajectory, array for array. The final artifact is loaded through
`DeepONetEquilibriumAccelerator` and checked against the JAX training path.
The source identity includes Python orchestration, mathematics, persistence,
safe loading and backend dispatch, plus the Rust/PyO3 implementation, crate
manifests, and workspace `Cargo.lock` dependency resolution.
When the current `scpn_fusion_rs` extension is installed, inference dispatches
first to the native `fusion-ml` branch-trunk kernel through PyO3; otherwise it
uses the NumPy reference path. The parity test loads one authenticated artifact
through both backends and requires agreement within `1e-14` relative and
absolute tolerance. The fixed-coefficient runtime fixture currently agrees bit
for bit. The authenticated tiny-training fixture differs by at most
`1.4210854715202004e-14 Wb/rad` (`256` ULP; normalised tolerance ratio
`0.15228015611094084`) while remaining inside the declared bound. Portable bit
identity is not claimed across different BLAS implementations, CPU instruction
sets, or compiler floating-point policies. Finalisation also
streams both backends over every untouched-test row and records maximum
absolute difference, normalised tolerance ratio, and IEEE-754 ULP distance.
An installed native backend fails closed when the normalised ratio exceeds
one. Run the 129x129 native regression benchmark with:

```bash
cargo bench --manifest-path scpn-fusion-rs/Cargo.toml \
  -p fusion-ml --bench deeponet_equilibrium_bench
```

Two 2026-08-24 local non-isolated regression runs measured
`1.0636–1.3730 ms` per one-shot 129x129 prediction; the latest range was
`1.2301–1.3730 ms`. These are workstation regression ranges, not an
isolated-core production latency claim.

Implementation responsibilities remain explicit: `deeponet_training.py` owns
only branch-trunk mathematics and AdamW; `deeponet_training_data.py` owns
coordinate, minibatch, and metric preparation;
`deeponet_training_recovery.py` owns identity-bound persistence;
`deeponet_training_report.py` owns evidence composition; and
`machine_conditioned_deeponet_cli.py` owns argument adaptation. The top-level
tool only orchestrates those contracts. Runtime tests exercise malformed and
non-finite artifacts, shape rejection, and public inference. Training tests
exercise exact interrupted-versus-uninterrupted recovery, tamper rejection,
four-way evidence, runtime parity, and a leakage test in which changing only
calibration/test values cannot change selected network weights. The training
suite derives its fixture from the tracked authenticated v2 reference cohort,
rewrites every changed array contract, and exercises the real verifier, loader,
CLI subprocess, recovery files, candidate artifact, and production runtime.

This v1 operator is deliberately *manifest-bound*. The cohort contains one
fixed ITER-like analytic geometry, so constant machine descriptors cannot teach
cross-machine dependence. The artifact stores that machine manifest digest and
the complete coordinate grid; it is evidence for shot-disjoint interpolation
and learned spatial evaluation on that one machine only. A cross-machine v2
must train on multiple independently varied machine geometries and coil maps,
hold out whole machines, and then move versioned geometry/coil descriptors into
the branch contract. No cross-machine, facility, free-boundary, IDA, EFIT, PCS,
or safety claim is made here.

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
