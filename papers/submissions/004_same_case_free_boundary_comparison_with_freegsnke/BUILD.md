<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN-FUSION-CORE — same-case FreeGSNKE comparison build instructions. -->

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

The evidence generator packages and checks two tracked reports. It does not run
FreeGSNKE or the SCPN solver.

To reproduce the reports themselves, run from the repository root:

1. Install the pinned FreeGSNKE environment as
   `.github/workflows/freegsnke-inverse.yml` does: `requirements/ci-freegsnke.txt`,
   then FreeGSNKE at commit `00b79a08fbb6e642ac5ed2e440383cc5962e0358`, checked
   out under `data/external/full_fidelity_public_sources/repos/freegsnke` or
   passed with `--source`.
2. Run `python validation/benchmark_freegsnke_multimachine.py --strict` for the
   forward report.
3. Run `python validation/benchmark_freegsnke_inverse_fit.py --strict` for the
   inverse report. It accepts `--case` and `--start`, so the six case-and-start
   runs can go on separate processes and be joined with `--merge`.

The inverse runs are long; the run times are recorded in the report.
