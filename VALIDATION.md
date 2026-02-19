# Silver Base Validation Notes

This document covers data provenance and validation notes for:

- `examples/neuro_symbolic_control_demo_silver_base.ipynb`

## DIII-D Shot Provenance

- The notebook uses:
  - `validation/reference_data/diiid/disruption_shots/shot_166000_beta_limit.npz`
- In this repository, files in `validation/reference_data/diiid/disruption_shots/` are synthetic DIII-D-like profiles generated from physics-informed templates, not raw downloaded EFIT/MDSplus dumps.
- Generation source:
  - `tools/generate_disruption_profiles.py`
  - The script defines a deterministic manifest of disruption/safe scenarios and seeds each shot with `np.random.default_rng(shot_number)` for reproducibility.

## If The Shot File Is Missing

Generate or regenerate the bundled disruption profiles from repo root:

```powershell
python tools/generate_disruption_profiles.py
```

Optional explicit output directory:

```powershell
python tools/generate_disruption_profiles.py --output-dir validation/reference_data/diiid/disruption_shots
```

Verification-only pass:

```powershell
python tools/generate_disruption_profiles.py --verify-only
```

## Data License And Copyright Context

- Bundled `disruption_shots/*.npz` in this repo are generated synthetic reference profiles and are distributed with this codebase workflow.
- Real DIII-D experimental data are not bundled in this specific `disruption_shots` folder.
- DIII-D itself is a U.S. DOE Office of Science user facility operated by General Atomics; real-shot data access and reuse follow facility/data-access terms and publication rules.
- If replacing synthetic files with true DIII-D data, document:
  - acquisition source,
  - shot list,
  - processing steps,
  - applicable data rights/attribution terms.

## Copyright Clarity

- Concepts: Copyright 1996-2026
- Code: Copyright 2024-2026
