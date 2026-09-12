# Maintaining the CI workflows

The CI coordinator calls reusable workflows grouped by their build or validation
responsibility. `tools/ci_workflow_ownership.json` declares each group, its jobs,
dependencies, condition, permissions, exact secret forwarding and reusable
workflow interface. Changes to a workflow must update this
contract in the same change.

Run `python tools/check_ci_workflow_ownership.py` before committing workflow
changes. The guard checks the declared graph, action pins, permissions, file
limits and final aggregate. It rejects duplicate YAML keys, unknown or duplicate
dependencies, cycles and unowned jobs. Pre-commit, release preflight and the
Python CI lanes run this check. The guard reports source consistency; hosted
execution remains necessary to verify runner behaviour and check identities.

Secret forwarding must match the explicit per-group mapping. The guard rejects
`secrets: inherit`, extra names, substituted expressions and forwarding on groups
whose contract is empty. Reusable secret declarations, including whether a token
is required, must match the reviewed interface. Changing the policy itself still
requires review; the guard does not establish authority for a policy change.

`workflow_sources(root)` provides validated source text for the complete graph.
Tests and tools that previously read only `.github/workflows/ci.yml` should use
this interface when checking commands distributed across reusable workflows.

The coordinator's final check retains the name **Python 3.12**. It requires every
category to succeed. The stress category may be skipped outside a main-branch
push because its original scheduling condition runs only on main-branch pushes.
Failures, cancellations, other skipped categories and missing or extra category
results fail the aggregate. The separate **pre-commit** required check remains.

Python test environments include pinned PyYAML and its type annotations. The
workflow ownership tests execute the actual aggregate shell across successful,
failed, cancelled, skipped and incomplete-result cases. They also mutate copies
of the maintained workflow files to verify rejection. Coverage is supplementary
evidence and does not replace these behavioural checks.
