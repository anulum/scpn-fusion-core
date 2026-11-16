<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN-FUSION-CORE — TORAX model-intersection paper build instructions. -->

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

The evidence generator packages and checks tracked files; it runs neither code.
To reproduce the reports themselves, run from the repository root:

1. `python validation/torax_coupled_reference_runner.py` with TORAX 1.4.3
   installed, to remake the TORAX reference run;
2. `python validation/benchmark_torax_coupled_transport_parity.py`;
3. `JAX_ENABLE_X64=true python validation/benchmark_torax_differentiable_transport.py`.

Each command takes `--help`.
