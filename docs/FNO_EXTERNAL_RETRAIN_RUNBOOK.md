# FNO External Retrain Runbook

This runbook defines the production route for FNO retraining in v4.0:
retraining is executed on the external service, then imported back into
`scpn-fusion-core` with checksum and provenance checks.

## 1) Export retrain request

```bash
python tools/export_fno_external_retrain_request.py \
  --validation-report validation/reports/full_validation_pipeline.json \
  --output-json artifacts/fno_external_retrain_request.json
```

Send `artifacts/fno_external_retrain_request.json` to the external retrain
service.

## 2) Receive service outputs

Expected files returned by the service:

- `artifacts/external_fno_retrain_manifest.json`
- `artifacts/external_fno_retrain_weights.npz`

Required manifest fields:

- `schema_version`
- `service`
- `weights_sha256`
- `trained_datasets` (must include GENE and/or CGYRO provenance)
- `data_license`

## 3) Import and validate

```bash
python tools/import_external_fno_weights.py \
  --manifest artifacts/external_fno_retrain_manifest.json \
  --weights artifacts/external_fno_retrain_weights.npz \
  --output weights/fno_turbulence_retrained_from_empirical.npz \
  --summary-json artifacts/external_fno_retrain_import_summary.json
```

## 4) Re-run validation

```bash
python validation/full_validation_pipeline.py \
  --experimental \
  --experimental-ack I_UNDERSTAND_EXPERIMENTAL \
  --strict \
  --auto-retrain-fno \
  --fno-retrain-mode external-service \
  --fno-external-manifest artifacts/external_fno_retrain_manifest.json \
  --fno-external-weights artifacts/external_fno_retrain_weights.npz
```

The run report includes `fno_retrain_plan.summary` with import status.
