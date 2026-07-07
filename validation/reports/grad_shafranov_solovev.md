<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Grad-Shafranov Solov'ev Validation

- Schema: `scpn-fusion-core.grad-shafranov-solovev-validation.v1`
- Generated (UTC): 2026-07-07T16:55:44Z
- Target: `local-grad-shafranov-solovev`
- Status: **pass**

## Discrete operator (`FusionKernel._apply_gs_operator`)

- Order of accuracy: 2.000 (gate ≥ 1.8)
- Finest-grid max truncation error: 6.104e-05 (gate < 5.0e-04)
- Passed: True

| resolution | h | max |Δ* error| |
| --- | --- | --- |
| 33 | 4.6875e-02 | 5.4932e-04 |
| 49 | 3.1250e-02 | 2.4414e-04 |
| 65 | 2.3438e-02 | 1.3733e-04 |
| 97 | 1.5625e-02 | 6.1035e-05 |

## SOR reconstruction (`FusionKernel._sor_step`)

- Order of accuracy: 2.019 (gate ≥ 1.8)
- Finest-grid NRMSE: 1.194e-06 (gate < 1.0e-04)
- Passed: True

| resolution | h | NRMSE | sweeps | converged |
| --- | --- | --- | --- | --- |
| 33 | 4.6875e-02 | 1.0969e-05 | 651 | True |
| 49 | 3.1250e-02 | 4.8253e-06 | 1451 | True |
| 65 | 2.3438e-02 | 2.7003e-06 | 2551 | True |
| 97 | 1.5625e-02 | 1.1939e-06 | 5701 | True |

## Dispatched multigrid full solve (`multigrid_solve` kernel)

| tier | resolution | NRMSE | residual | cycles | converged | meets tolerance |
| --- | --- | --- | --- | --- | --- | --- |
| numpy | 97 | 1.1939e-06 | 2.3357e-10 | 9 | True | True |
| rust | 97 | 1.1939e-06 | 2.3516e-10 | 9 | True | True |

- Multigrid gate passed: True
