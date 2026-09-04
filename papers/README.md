<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN-FUSION-CORE — authoritative manuscript collection. -->

# Papers

This directory is the canonical public home for manuscript artefacts in
SCPN-Fusion-Core.  Review drafts live under [`submissions/`](submissions/) and
use stable three-digit ordering prefixes.

Each submission directory is self-contained: manuscript source, bibliography,
figures and their generators, evidence, citation metadata, build instructions,
and the generated review PDF are kept together.  A directory is a review
workspace, not evidence that the manuscript has been submitted or accepted.

From the repository root, rebuild and validate every numbered package in a
disposable tree with:

```bash
./papers/verify_submissions.sh
```

The verifier does not modify the committed review PDFs or use them as build
inputs.

The superseded top-level manuscript and shared-figure layout was retired after
the three numbered packages became self-contained.  Exact historical custody is
recorded in [`legacy_layout_manifest.json`](legacy_layout_manifest.json): every
retired path names its source commit, Git blob, SHA-256 digest, byte size,
retrieval command and current disposition.  Successor packages are corrected
review artifacts; the manifest does not claim byte equivalence with the legacy
content.

Repository code retains its existing software licence.  The licence for
manuscript content will be stated separately when selected by the author.
