<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN-FUSION-CORE — conformal-coverage paper build instructions. -->

# Build

Run from this directory:

```bash
export SOURCE_DATE_EPOCH=1787756047
export FORCE_SOURCE_DATE=1
export TZ=UTC
PYTHONPATH=../../../src ../../../.venv/bin/python generate_evidence_manifest.py
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
```

The expected final artefact is `manuscript.pdf`. LaTeX auxiliary files are
ignored. The source, the bibliography, the evidence, the custody manifest and
the final PDF are intentional records.

The evidence generator packages and checks tracked files; it runs no model. To
reproduce the reports themselves, run from the repository root, with the
processed QLKNN-10D splits that the certificate names under
`data/qlknn10d_processed/`:

1. `python validation/qlknn_conformal_regime_coverage.py`;
2. `python validation/qlknn_conformal_variants.py`;
3. `python validation/qlknn_conformal_stability_sets.py`;
4. `python validation/qlknn_conformal_profile_envelope.py`.

Each command takes `--help`. The certificate itself comes from
`tools/generate_surrogate_conformal_certificates.py`.
