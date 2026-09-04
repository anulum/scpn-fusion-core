<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN-FUSION-CORE — equilibrium-paper build instructions. -->

# Build

Run from this directory with the repository virtual environment available:

```bash
export SOURCE_DATE_EPOCH=1787756047
export FORCE_SOURCE_DATE=1
export TZ=UTC
PYTHONPATH=../../../src ../../../.venv/bin/python figures/generate_figures.py
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
```

Expected final artefact: `manuscript.pdf`.  LaTeX auxiliary files are ignored;
the source, evidence, generated figures and final PDF are intentional records.
The figure command also validates both frozen scientific payloads and
regenerates `evidence/evidence_manifest.json`.  From the repository root, the
focused disposable-package check is:

```bash
papers/verify_submissions.sh 001_hybrid_rust_python_grad_shafranov_equilibrium_solver
```
