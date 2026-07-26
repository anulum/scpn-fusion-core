# Tutorial Notebook HTML Exports


## Purpose and audience

This notebook index is a practical starting point for reproducing documented workflows. It points to the material that maps code-level implementation to benchmark and validation evidence.

This directory contains six pre-built HTML exports from the numbered Jupyter
notebooks in `examples/`. CI regenerates that published subset on every push to
`main` (see `.github/workflows/docs.yml`, job `build-notebook-html`). The source
directory also contains advanced and hero notebooks that are intentionally not
all duplicated as tracked HTML.

## Evidence boundary

Notebook pages are tutorial and onboarding material. They are not production
parity evidence unless the notebook links to a tracked report under
`validation/reports/` and that report marks the relevant gate as accepted.

For current production-parity status, use:

- `validation/reports/full_fidelity_end_to_end_campaign.md`
- `validation/reports/full_fidelity_acceptance_benchmark.md`
- `validation/reports/production_decomposition_contract.md`
- `validation/reports/gk_external_nonlinear_parity.md`
- `validation/reports/gk_electromagnetic_fidelity.md`

The full-fidelity campaign remains fail-closed: native local contracts can pass
while external GENE/CGYRO/GS2, DREAM, Aurora/STRAHL, FreeGS, and distributed
MPI/multi-GPU evidence remains blocked.


## Suggested reading order

1. Start with the neuro-symbolic compiler notebook to understand Petri-net to SNN control.
2. Run the Grad-Shafranov equilibrium notebook to see the core solver contract.
3. Review inverse and transport benchmarks to connect solver outputs to validation reports.
4. Treat HTML exports as tutorials; use tracked validation reports for public benchmark claims.

## Source notebook catalogue

| Source notebook | Best for | Evidence boundary |
|---|---|---|
| `01_compact_reactor_search.ipynb` | Design-space exploration | Educational reduced-order search |
| `02_neuro_symbolic_compiler.ipynb` | Petri-net to SNN compilation | Compiler tutorial |
| `02_neuro_symbolic_compiler_secondary.ipynb` | Alternate compiler walkthrough | Secondary tutorial, not a separate acceptance gate |
| `03_grad_shafranov_equilibrium.ipynb` | Equilibrium basics | Local solver demonstration |
| `04_divertor_and_neutronics.ipynb` | Engineering/nuclear coupling | Educational model coupling |
| `05_validation_against_experiments.ipynb` | Reading validation metrics | Use linked reports for claim state |
| `06_inverse_and_transport_benchmarks.ipynb` | Inverse/transport workflow | Benchmark support, not parity by itself |
| `07_multi_ion_transport.ipynb` | Multi-ion conservation workflow | Local contract plus linked benchmark |
| `08_mhd_stability.ipynb` | MHD stability criteria | Educational/local analysis |
| `09_coil_optimization.ipynb` | Coil design workflow | Research optimization |
| `10_uncertainty_quantification.ipynb` | UQ and uncertainty propagation | Method tutorial |
| `Q10_closed_loop_demo.ipynb` | Closed-loop scenario demonstration | Reduced-order scenario |
| `neuro_symbolic_control_demo.ipynb` | Original end-to-end control demo | Historical tutorial |
| `neuro_symbolic_control_demo_v2.ipynb` | Recommended hero control demo | Current tutorial entry point |
| `platinum_standard_demo_v1.ipynb` | Broad integrated demonstration | Showcase; follow every claim to its report |

The `metadata.language_info.version` field records the Python interpreter used
when a notebook was saved or executed. It is not the SCPN Fusion Core package
version and should not be rewritten during release-documentation updates unless
the notebook is actually re-executed in a different interpreter.

## How notebooks fit the product story

The notebooks are the fastest way to see the project’s potential without
reading the whole codebase. They show how a control idea, a physics model, and a
reportable artifact connect. They are also deliberately bounded: a notebook can
teach the workflow, but it does not by itself certify solver parity, plant
safety, or market readiness.

Use notebooks for:

- learning the control-first architecture,
- demonstrating a reproducible local workflow,
- preparing a benchmark or validation report,
- onboarding collaborators before they run heavier campaigns.

Use `validation/reports/` for accepted, blocked, or diagnostic claim state.

## Notebook roles

| Notebook family | Role | Claim status |
|---|---|---|
| Neuro-symbolic control | Shows the controller and compiler workflow | Tutorial unless linked to replay reports |
| Grad-Shafranov equilibrium | Demonstrates equilibrium inputs, solves, and diagnostics | Local contract unless linked to accepted GEQDSK/FreeGS reports |
| Inverse and transport benchmarks | Connects model outputs to validation workflows | Diagnostic or benchmark support depending on the linked report |
| Compact reactor and Q10 demos | Explains scenario exploration | Educational; not plant-design certification |

When exporting or updating notebooks, keep setup cells deterministic, avoid
private paths, and link any public performance or physics claim to a tracked
report under `validation/reports/`.
