==========
Onboarding
==========

This page gives first-time readers one controlled path through setup, a useful
demo, and evidence review.  It mirrors ``docs/ONBOARDING.md`` but keeps the
Sphinx navigation self-contained.

First-Run Contract
------------------

Use the onboarding path when you need a reproducible first look, not a parity
claim:

1. create a local virtual environment,
2. install from the hash-pinned lock file when reviewing release state,
3. run the hero demo,
4. refresh the checksummed evidence bundle,
5. read the blocked and accepted rows before quoting any result.

Pinned Environment
------------------

For a current source checkout on Python 3.12, use the minimal hash-pinned lock
file before installing the editable package::

    python -m venv .venv
    . .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install --require-hashes -r requirements/minimal.txt
    python -m pip install --no-deps -e .

For documentation work, use ``requirements/docs.txt``.  For CI-equivalent local
review on Python 3.12, use ``requirements/ci-py312.txt``.  Regenerate lock files
only through ``tools/regenerate-requirements.sh`` so the hash-pinned surfaces
stay consistent.

Hero Demo
---------

The shortest useful demo is the minimal equilibrium run followed by the
full-reproduction evidence wrapper::

    python examples/minimal.py --grid 17 --equilibrium-iters 4
    scpn-fusion repro --full

The first command exercises an inspectable local Grad-Shafranov path.  The
second command refreshes ``validation/reports/full_reproduction_evidence.json``
and ``validation/reports/full_reproduction_evidence.md`` with SHA-256 checksums
for the full-fidelity campaign source reports.  The expected top-level status is
``not_full_fidelity`` until external same-case parity artifacts close the
blocked rows.

Reader Tracks
-------------

.. list-table::
   :header-rows: 1
   :widths: 24 44 32

   * - Reader
     - Start with
     - Evidence boundary
   * - New user
     - ``README.md`` and this page
     - Run the hero demo before reading benchmark values.
   * - Contributor
     - ``docs/API_OVERVIEW.md`` and the relevant tests
     - Update code, tests, docs, and generated reports together.
   * - Fusion-domain reviewer
     - ``docs/BENCHMARKS.md`` and ``validation/reports/``
     - Treat blocked rows as missing parity evidence, not hidden failures.
   * - Release reviewer
     - ``docs/RELEASE_READINESS.md`` and ``requirements/*.txt``
     - Confirm hash-pinned dependency state and report freshness.

Next Reading
------------

- :doc:`quickstart`
- :doc:`installation`
- :doc:`userguide/validation`
- `Public benchmark taxonomy <../BENCHMARKS.md>`_
- `Generated full-reproduction report <../../validation/reports/full_reproduction_evidence.md>`_
