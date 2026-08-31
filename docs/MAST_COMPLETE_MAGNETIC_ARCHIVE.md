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

## Deliberate fail-closed boundary

Some diagnostic signal and geometry coordinate cardinalities differ in the
source archive. The contract preserves those arrays independently and does not
invent positional correspondence. Calibration, event-clock mapping, validity,
quality, uncertainty, and observation operators remain unresolved. Therefore
the envelope is not eligible for phase inference, classification, semantic
ingress, execution, or actuation.

This is full acquisition fidelity, not full diagnostic qualification. Later
qualification requires authoritative channel mapping, calibration, uncertainty,
quality, and event-identity evidence without weakening this raw-source contract.

## Reproduction

Install the isolated Python 3.12 profile and run the exact complete-group gate:

```bash
python -m pip install --require-hashes -r requirements/mast.txt
python -m pip install --no-deps -e .
python validation/verify_mast_complete_magnetic_archive.py \
  --report artifacts/mast-complete-magnetic-archive.json
```

The dedicated `Complete FAIR-MAST Magnetic Archive` workflow performs that
download-and-decode proof on every push and pull request. Raw Zarr objects are
not committed; the repository retains the complete source manifest, canonical
envelope, schema, hashes, provenance, and reproduction command.
