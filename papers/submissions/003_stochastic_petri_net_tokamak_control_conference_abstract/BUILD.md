<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN-FUSION-CORE — extended-abstract build instructions. -->

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

Expected final artefact: `manuscript.pdf`.  LaTeX auxiliary files are ignored;
the source, bibliography, exact evidence, custody manifest and final PDF are
intentional records.  The evidence generator validates and packages existing
records; it does not execute a new physics or controller campaign.
