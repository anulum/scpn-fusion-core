# Complete FAIR-MAST magnetic archive contract

SCPN-FUSION-CORE captures a FAIR-MAST `magnetics` group as one complete,
canonical, review-only evidence envelope. It does not select a probe, resample a
clock, infer a positional join, fill missing values, or promote raw diagnostics
to control authority.

The public surface is:

- `acquire_mast_complete_magnetic_archive`: downloads every magnetic object
  declared by the tracked parent provenance manifest, verifies its byte count
  and SHA-256, decodes the complete group, and returns the envelope;
- `build_mast_complete_magnetic_archive_envelope`: performs the same complete
  verification and decoding against an already materialised shot archive;
- `verify_mast_complete_magnetic_archive_source`: re-verifies every declared
  source object and rejects missing, extra, symlinked, or modified objects;
- `decode_mast_complete_magnetic_archive_envelope`: validates canonical bytes,
  payload identity, all structural invariants, and permanent non-actuating
  authority.

The schema is generic across FAIR-MAST shots. The tracked shot 27707 envelope is
a complete regression witness, not a fixed schema template. Array names,
shapes, native clocks, chunk-object paths, attributes, and cardinalities are
discovered from the source group and bound into the envelope. No implementation
constant limits a valid shot to the witness's 72 arrays, 253 objects, or four
clocks.

## Preserved evidence

For each group, the contract retains:

- the exact Zarr v3 group metadata and its source-object digest;
- every array's complete Zarr metadata, attributes, data type, native shape,
  dimension names, data-object paths, decoded value count, non-finite count,
  and decoded-content digest;
- every self-coordinate measured in seconds as a distinct native clock, with
  sample count, bounds, interval statistics, and source array binding;
- the full parent and magnetic-subgroup object inventories, byte counts,
  manifest digests, source URLs, retrieval timestamp, licence, limitations,
  upstream FAIR-MAST ingestion revision, and producer-module digests;
- explicit qualification and authority states.

Zarr scalar arrays retain the source metadata's `dimension_names: null` while
the normalized decoded dimension list is empty. This distinction is validated;
it is not silently rewritten in the source metadata.

The current upstream witness identifies its FAIR-MAST ingestion tree as
`dirty`. That state is retained as evidence and cannot be presented as a clean
upstream release, even though every served object is independently hash-bound.

## Diagnostic qualification companion

The immutable archive envelope remains the raw-source contract. A separate
`mast-magnetic-diagnostic-qualification.v1` document now reproduces every fact
that the exact FAIR-MAST ingestion mapping and complete shot can support:

- all 72 arrays are classified by role and all 11 measurements are covered;
- every configured signal channel has per-channel and aggregate finite, NaN,
  infinite, zero, unique-level, and minimum-positive-level-spacing statistics;
- the applied scale, target-unit conversion intent, background sample range,
  selected source, source channel list, shot validity range, and IMAS quantity
  path are recorded for every measurement;
- all four coordinate arrays are verified against the upstream Level-2
  interpolation definitions;
- all available signal-to-geometry identifiers are compared explicitly.

These facts are derived from the exact upstream mapping at revision
`ab435c799d892956fb042d55391f7d1be0c950e6`, whose tracked tree state is
`dirty`. The four coordinates are therefore described as Level-2 archive grids,
not raw instrument clocks. The saddle background correction `[0, 10]` is
recorded as the sample-index range actually used by the ingestion code, not as
a time interval.

The source does not provide calibration lineage, transfer functions, provider
quality flags, uncertainty, raw-instrument-clock relations, or a facility event
identifier. Geometry associations are identifier correspondences only because
the ingestion implementation does not establish a physical join. FAIR-MAST
issue 211 reports numerical-resolution and saddle-NaN behaviour for shot 29980;
the qualification records that limitation but does not assume it applies to
the tracked shot 27707. Unsupported fields remain explicitly unresolved.

The archive and qualification documents remain permanently review-only. They do
not infer a plasma phase, classify a regime, enter semantic control ingress,
execute a command, or actuate hardware. Full fidelity here means complete
source/evidence accounting and explicit unknowns; it never means inventing
missing authority.

## Upstream evidence

- [UKAEA FAIR-MAST repository](https://github.com/ukaea/fair-mast/)
- [Pinned FAIR-MAST ingestion mapping](https://raw.githubusercontent.com/ukaea/fair-mast-ingestion/ab435c799d892956fb042d55391f7d1be0c950e6/mappings/level2/mast.yml)
- [UKAEA MAST magnetic diagnostics publication record](https://scientific-publications.ukaea.uk/papers/mast-magnetic-diagnostics/)
- [FAIR-MAST issue 211](https://github.com/ukaea/fair-mast/issues/211), retained with applicability to shot 27707 explicitly unassumed

## Reproduction

Install the isolated Python 3.12 profile and run the exact complete-group gate:

```bash
python -m pip install --require-hashes -r requirements/mast.txt
python -m pip install --no-deps -e .
python validation/verify_mast_magnetic_diagnostic_qualification.py \
  --report artifacts/mast-magnetic-diagnostic-qualification.json
```

The dedicated `Complete FAIR-MAST Magnetic Archive` workflow performs that
single authentic acquisition, complete archive verification, pinned mapping
download, and byte-identical qualification reproduction on every push and pull
request. Raw Zarr objects are not committed; the repository retains the source
manifest, canonical archive and qualification documents, schemas, hashes,
provenance, and reproduction command.
