# RustBCA particle-surface diagnostics

The optional reference uses the public sputtering-yield and reflection APIs
from [lcpp-org/RustBCA](https://github.com/lcpp-org/RustBCA) at commit
`18920a26263c21b9e337db1bab1fd17f23f0fec5`. Each energy, angle and seed runs in
an isolated Python process. The report preserves seed batches, reflected and
sputtered counts, coefficient means, sampling standard errors and file digests.

## Build the optional reference

Use Linux CPython, an available Rust toolchain compatible with the pinned
upstream, and `wheel==0.47.0` in the builder's Python environment. The observed
build used CPython 3.12 and Rust 1.98.1. Other compiler/runtime combinations need
their own integration evidence. No cross-platform or manylinux compatibility
is inferred from a local wheel.

Acquire the tar.gz for that exact upstream commit separately. The builder
requires archive SHA256
`3aded40a7a3282ac41e4a1c90198511f26bf8d418ec2994914b6bf930424f4b7`;
it checks the separately pinned 140-file inventory and Cargo lock before
compilation. By default Cargo dependencies must already be available in the selected
cache. For a fresh environment, pass `--fetch-dependencies` to run bounded
`cargo fetch --locked` before frozen compilation. Both phases share the deadline,
and the lockfile must remain unchanged. Downloads are never enabled implicitly.

From the SCPN Fusion Core source checkout:

```bash
python -m tools.build_rustbca_reference /path/to/upstream.tar.gz \
  /path/to/new-build --python /path/to/runtime/bin/python \
  --cargo /path/to/toolchain/bin/cargo --rustc /path/to/toolchain/bin/rustc
```

The output directory must be new. The builder retains the source snapshot,
lockfile, recipe, command/environment, compiler identities and logs. A fresh
Cargo target directory and two compilation jobs avoid mixing prior build output
with this observation. `--timeout-seconds` sets the execution budget (default
600, maximum 1800). Failed work retains a failed `result.json`; compiler timeout
or interruption kills and reaps the owned compiler process group. `process.json`
identifies that group. Do not treat a partially written output directory as a
successful build.

A successful build writes a local-platform `rustbca-3.0.0-*.whl` and records its
SHA256 and the compiled extension SHA256. The wheel includes the upstream
licence and a wheel RECORD. Install that exact wheel into the intended runtime:

```bash
/path/to/runtime/bin/python -m pip install --no-index --no-deps /path/to/built.whl
```

Upstream licensing remains with the optional RustBCA package. This integration
does not relicense the upstream solver. Retain the source archive and build
records alongside local scientific evidence; redistribution needs its own
licensing and publication checks.

## Run and retain a request

The source checkout includes `validation/reference_data/rustbca_request.json`,
a D/W example derived from the pinned upstream example materials. Those material
values are example inputs, not independently validated material data.

```bash
python -m validation.rustbca_reference \
  validation/reference_data/rustbca_request.json /path/to/new-run \
  --python /path/to/runtime/bin/python \
  --binary-sha256 RECORDED_BINARY_SHA256 \
  --build-receipt /path/to/new-build/result.json \
  --build-receipt-sha256 RECORDED_RECEIPT_SHA256
```

Record the receipt's SHA256 after a successful build and check the extension
identity against that build. The adapter checks receipt, source inventory,
source bytes, lockfile, build log, recipe and binary before dispatch and again
after the seed batches. It also retains the receipt bytes in the run directory.
The source tree must match the entire inventory: extra files are refused.

The two build-receipt arguments are optional as a pair. Without them the report
has no build-custody result and verifies only the caller-selected binary hash.
With them, `retained_build_inputs_verified` means that the local observation and
retained files agree. It does not establish who created the record, prove its
asserted compilation history or authenticate a remote producer. Independent
attestation and binary-reproducibility flags remain false.

Request validation rejects invalid material fields, nonfinite inputs, duplicate
energies/angles/seeds and excessive work. Units are eV, degrees, atomic mass
units and target number density in m^-3. Seeds are explicit unsigned 64-bit
integers; threads and samples per seed are bounded. The API docstrings define
the exact contract. Existing result directories are preserved rather than reused.

The worker and the parent both validate diagnostic values before aggregation.
Coefficients must be finite numeric scalars, sputtering yield must be nonnegative,
and number/energy reflection must lie in [0, 1]. Multiplying sputtering yield and number
reflection by the sample count must produce finite integer event counts within
the numerical tolerance. Retained integer counts must agree with those values.
Invalid retained results are refused before dispatching another seed or writing
a successful report. These checks do not authenticate a producer or establish
physical validity of otherwise consistent values.

## Interpretation and limitations

Sputtering and reflection use different upstream recoil settings. Their results
must not be combined into a purported single-cascade energy ledger. Seed-batch
standard errors quantify observed sampling dispersion only. Zero events or zero
observed dispersion produce an unresolved uncertainty status and no confidence
interval; they do not establish zero uncertainty.

The local erosion surrogate does not expose incident-ion species, so comparing
its values to this explicit D/W case does not demonstrate species-matched model
parity. Do not tune it solely to these reference outputs or present the example
as experimental material validation.

`provenance_verified`, `material_evidence_verified`, `actionable`, `federated`
and `evidence_claimed` remain false. Neither a successful build nor a numerical
report grants control, safety, licensing or manufacturing admission.

## Integration checks

For the full integration lane, build with `--fetch-dependencies` and an initially
empty `CARGO_HOME`, install the wheel, and supply the actual artifacts:

```bash
export RUSTBCA_BUILD_RECEIPT=/path/to/new-build/result.json
export RUSTBCA_SOURCE_ARCHIVE=/path/to/upstream.tar.gz
export RUSTBCA_PYTHON=/path/to/runtime/bin/python
python -m pytest integration_tests/rustbca --no-cov
```

The tests include real wheel installation, real runtime/CLI execution, retained
source mutation and failed-build cases. They require an actual build whose
recipe digest matches the current builder. Tests and file consistency checks do
not replace independent review or measured scientific validation.

The dedicated RustBCA workflow uses a fresh Cargo cache, hash-locked Python
tooling and the explicit dependency-fetch mode before building and installing
the actual wheel. Hosted execution is separate from local verification. The
workflow collects branch coverage from the builder and runtime subprocesses.
It retains the raw parallel datasets and combines with `--append --keep`; reports
use `--keep-combined` so later batches cannot silently discard earlier build
observations. Instrumentation is scoped to the disposable integration environments.
The integration gate requires 100% combined statement and branch coverage;
reporting below that threshold fails the workflow while retaining observations.
Independent acceptance review is also required. Collecting a report alone does
not establish complete coverage.
