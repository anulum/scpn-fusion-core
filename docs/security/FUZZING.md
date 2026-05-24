# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Fuzzing Guide

# Fuzzing guide

The repository ships Atheris-compatible fuzz targets for malformed user-controlled file inputs:

- `fuzz/fuzz_geqdsk.py`: G-EQDSK parser input, including absurd grid dimensions and malformed Fortran tokens.
- `fuzz/fuzz_fusion_config.py`: `FusionKernel` JSON configuration loading and schema validation.
- `fuzz/fuzz_disruption_npz.py`: disruption-shot NumPy archive loading with pickle disabled.

Install the optional fuzzing dependencies before running targets:

```bash
python -m pip install '.[fuzz]'
```

Run one target at a time with a dedicated corpus and findings directory:

```bash
python fuzz/fuzz_geqdsk.py corpus/geqdsk findings/geqdsk
python fuzz/fuzz_fusion_config.py corpus/fusion_config findings/fusion_config
python fuzz/fuzz_disruption_npz.py corpus/disruption_npz findings/disruption_npz
```

All targets truncate single generated inputs to bounded sizes before writing temporary files. Production loaders enforce 10 MiB gates for JSON, GEQDSK, and NumPy archive paths before parsing.
