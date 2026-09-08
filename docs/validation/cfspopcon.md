# Source-pinned cfspopcon operating maps

This optional integration evaluates a bounded uniform deuterium–tritium
operating map through cfspopcon 8.0.0 public formula APIs. It compares reaction
rates with SCPN's existing `bosch_hale_reactivity` in a separate Python process.
Both implementations use the same published fit: agreement is code parity,
not independent experimental validation.

The prescribed case uses equal D/T populations, uniform profiles, Ti=Te,
elliptical torus volume and ITER98y2 confinement. Density and temperature vary;
geometry, magnetic field and plasma current remain fixed. Impurity radiation,
L–H accessibility, self-consistent equilibrium and net electric power are not
calculated. The example geometry is not a reproduction of the SPARC PRD case.

## Environments and execution

Use Python 3.12 or later for cfspopcon in a separate environment. Install the
optional requirements without upgrading an existing SCPN environment:

```bash
python3.12 -m venv /path/to/cfspopcon-env
/path/to/cfspopcon-env/bin/python -m pip install -r requirements/cfspopcon.txt
PYTHONPATH=src:. /path/to/cfspopcon-env/bin/python -m validation.cfspopcon_operating_map \
  validation/reference_data/cfspopcon_uniform_dt.json /path/to/map.json \
  --reference-python /path/to/scpn-env/bin/python
```

The reference interpreter must support the normal public SCPN imports from this
checkout. Source validation compares all upstream Python/YAML package files
against `validation/reference_data/cfspopcon_source.json` before calculation.
The upstream commit is `b9ed8c3fd973bd2ad3d8226acbf77aa0e4caf7d1`.
The result includes the local reaction module's hash and interpreter/NumPy
versions. Runtime process failure propagates; there is no replacement model.
NumPy 2.4.4 and 2.5.3 both exposed an existing SCPN import incompatibility during
integration; process separation permits each project to retain its own runtime.

## Public API

`validation.cfspopcon_operating_map.run_operating_map(request,
reference_python=Path(...), reference_root=Path(...))` returns a JSON-compatible dictionary.
The optional reference root selects a specific SCPN checkout for candidate comparison.
`validate_request(request)` checks the exact input schema, and
`verify_upstream(package_root)` checks package source custody.

Input units are explicit in field names: metres, tesla, megaamperes, megawatts,
number density per cubic metre and kiloelectronvolts. Axes are nonempty, unique,
positive finite lists with at most 256 product points. Temperature is restricted
to the common 0.2–100 keV fit interval. Coupling is in (0,1]; minor radius is below
major radius. Geometry and limits are positive; ohmic power may be zero.

For each point the upstream stored energy and confinement scaling determine
loss power. Alpha heating is one fifth of upstream DT fusion power. Required
launched auxiliary power is `(loss - alpha - ohmic) / coupling`. Reported gain
is `fusion / (ohmic + launched auxiliary)`; it is not absorbed-power gain.

Negative auxiliary demand, requested auxiliary/Greenwald limit violations,
nonfinite outputs, reaction-code discrepancies above 1e-12 relative, and gain
denominators at or below the upstream 1 W floor retain their coordinates with
reasons and null gain. Arithmetic/value failures retain a failed point.
Nonfinite numeric outputs are null, never JSON NaN. A point within the declared
limits is not a claim that all physical operating limits are satisfied.

Every result preserves the request and its hash. `actionable`,
`evidence_claimed` and `federated` remain false. This artifact does not allocate
a producer role or constitute admission to a Studio/SPO runtime contract.

## Upstream attribution

cfspopcon is Copyright Commonwealth Fusion Systems, distributed under the MIT
license: https://github.com/cfs-energy/cfspopcon/blob/main/LICENSE.
This integration calls its public APIs rather than vendoring its implementation.
The source manifest records the license hash. Model descriptions and source
references are available at https://github.com/cfs-energy/cfspopcon.

## Focused verification

Install pytest and pytest-cov into the optional environment, then run this
integration's dedicated test module with the existing SCPN interpreter selected:

```bash
SCPN_REFERENCE_PYTHON=/path/to/scpn-env/bin/python PYTHONPATH=src:. \
  /path/to/cfspopcon-env/bin/python -m pytest -o addopts= \
  integration_tests/cfspopcon/test_operating_map.py
```

These tests require the real upstream runtime. They exercise the complete grid,
unit conversions, power accounting, fit boundaries, invalid requests, source
corruption and subprocess failure, plus command-line artifact generation.

The default pytest collection is `tests/`. This external integration lives in
`integration_tests/cfspopcon/` and runs explicitly in the cfspopcon workflow on
push and pull requests. That workflow creates both interpreters, installs the
reference CI lock separately and executes these tests plus the public CLI;
missing upstream dependencies fail the dedicated job.

For a prospective commit, `python tools/capability_manifest.py --check --index`
checks source and generated bytes from the Git index in an isolated snapshot.
It ignores unstaged changes; every staged path belongs to the candidate. The
ordinary generator continues to inventory HEAD in a checkout. Generate updated
projections from an exact candidate source archive and stage them with the
source change before running the index guard.
