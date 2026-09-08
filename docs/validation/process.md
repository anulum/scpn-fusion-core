# PROCESS power diagnostics

`validation.process_power_report.read_process_power_report` reads an actual
PROCESS MFILE through the optional upstream parser. Supply the artifact SHA256,
the exact active constraint IDs from the producing case and an explicit equality
residual tolerance. Every scan is retained; missing fields, changed files and
inconsistent scan counts refuse parsing. Scan labels must be consecutive from one,
and every point must contain its own diagnostics. A truncated final point or a
missing earlier field is rejected even when the upstream parser could return a
shorter array. All repeated required values, including equality and inequality constraints,
must be finite and numerically identical within a point. Equal repeats are
retained; a contradictory first or later occurrence rejects the artifact.

The report separates solver convergence, upstream error status, equality norm,
inequality feasibility, recorded power imbalances and electrical accounting.
Passing a recorded check does not establish upstream source provenance or
physical admission. Authority/evidence flags remain false. The 0.1 MW power
threshold follows the inspected PROCESS power diagnostic threshold; it is not
an experimentally established uncertainty bound.

The reader requires the optional `ukaea/PROCESS` package. A caller-supplied
hash establishes file identity only; it does not prove which solver generated
that file. Retain the producing run's input, source revision and full logs
separately. Never derive manufacturing or control permission from this report.


## Source-bound execution

`validation.process_reference_run.run_process_reference` verifies all source and
physical-data files in PROCESS commit
`620d1e9a38f1b3c6d2597956c8556e9ab6c17037`, verifies the input case, and runs
`SingleRun` in a child process with a 240 second default timeout. Set
`timeout_seconds` (CLI `--timeout-seconds`) to a finite positive execution limit. It requires a new output
directory and retains input, execution log and upstream output even if execution
fails. Successful parsing writes `power-report.json` with local custody hashes.
This local provenance check is not a signed external attestation or a validation
of transitive dependencies. Build-generated version metadata may report `0.0.0`;
the source commit and file manifest define the inspected implementation.

Install `requirements/process.txt` in an isolated Python environment, then run:

```bash
PYTHONPATH=. /path/to/process-env/bin/python -m validation.process_reference_run \
  /path/to/pinned-PROCESS/examples/data/large_tokamak_eval_IN.DAT \
  /path/to/new-result-directory --equality-tolerance 1e-8
```

The dedicated PROCESS workflow installs the upstream, checks out the exact case
revision and generates real output during its tests. No pre-existing workstation
MFILE or mocked solver is used. PROCESS is distributed by UKAEA under MIT;
upstream notices remain with the separately installed package. The source
manifest records the inspected license digest.


## Integration verification

The optional workflow measures both adapter modules with branch coverage,
including the real CLI and subprocess failure paths. `.coveragerc.process`
requires 100% statement and branch coverage and contains no coverage exclusions.
The tests generate single-point and two-point upstream output and exercise
constraint conflicts, incomplete runtimes, execution timeout and changes to
input/artifact custody. Coverage measures exercised code paths; it does not
establish physical model accuracy.

To repeat the workflow in the isolated runtime with `pytest==9.1.1` and
`coverage==7.15.2` installed:

```bash
export PYTHONPATH=.
export PROCESS_TEST_CASE=/path/to/pinned-PROCESS/examples/data/large_tokamak_eval_IN.DAT
export COVERAGE_RCFILE="$PWD/.coveragerc.process"
export COVERAGE_FILE=/path/to/new-coverage-directory/process.coverage
/path/to/process-env/bin/python -m coverage run -m pytest integration_tests/process -q
/path/to/process-env/bin/python -m coverage combine
/path/to/process-env/bin/python -m coverage json --fail-under=0 -o /path/to/coverage.json
/path/to/process-env/bin/python -m coverage report
```

Create the coverage directory before running. The JSON command retains diagnostics
without enforcing the threshold; the final report command enforces the 100% gate.


## Power-accounting boundaries

The local `PowerPlantModel` is a reduced balance-of-plant model. Matching a
PROCESS variable name to one of its outputs does not establish equal physical
boundaries. At the inspected source revision:

| Quantity | PROCESS | Local `PowerPlantModel` |
| --- | --- | --- |
| Heat available for conversion | First-wall/blanket deposited heat plus shield and divertor terms selected by the conversion model | `1.15 * 0.8 * fusion + 0.2 * fusion + absorbed_auxiliary` |
| Gross electricity | Primary heat times turbine efficiency, or separate liquid-breeder conversion in the dual-coolant branch | Total model heat times `eta_thermal` |
| Heating electricity | Complete heating/current-drive electrical load | Absorbed auxiliary power divided by `eta_heating` |
| Other recirculating loads | Cryogenics, facility loads, applicable centrepost pumping, magnet supplies, tritium and vacuum systems | Fixed cryogenic and miscellaneous loads |
| Coolant pumping | Selected upstream multi-circuit pumping model | Constant-property equal-flow parallel paths with explicitly supplied geometry |
| Net electricity | Gross minus recirculating electricity | Gross minus recirculating electricity |

The final subtraction is a shared accounting identity; its agreement cannot
validate either physical plant model. A numerical component comparison requires
matching coolant circuits, heat allocation, efficiencies and instantaneous versus
pulse-averaged quantities. The historical local single-pipe default is not a
reactor cooling layout. Do not choose channel counts merely to reproduce an
upstream net-power figure. The reference diagnostic report does not claim such
component parity.

The upstream definitions are in the pinned
[PROCESS power model](https://github.com/ukaea/PROCESS/blob/620d1e9a38f1b3c6d2597956c8556e9ab6c17037/process/models/power.py),
particularly `plant_electric_production` and the primary/secondary heat allocation.
