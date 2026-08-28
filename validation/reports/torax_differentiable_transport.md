# Differentiable Coupled Transport

Status: `differentiable_model_intersection_evaluated`
Overall pass: `True`
Performance superiority claimed: `False`
General transport differentiability claimed: `False`

The evidence covers only the frozen circular prescribed-coefficient coupled model intersection and a synthetic production-perturbation recovery target. The TORAX row is its pinned nominal cold baseline, not a TORAX optimiser comparison. Loaded-host timings establish cost completeness for this case only, not portable performance superiority or differentiability of other production transport models.

## Gradient and optimisation evidence

- Maximum AD/central-FD relative error: `2.18396459473e-08`
- Initial objective: `6.14973983171e-06`
- Final objective: `3.53357311148e-08`
- TORAX nominal baseline objective: `0.000380625152985`

## Gates

- `central_finite_difference`: `True`
- `deterministic_optimisation`: `True`
- `finite_full_chain_gradients`: `True`
- `optimisation_quality`: `True`
- `perturbation_replay`: `True`
- `production_forward_replay`: `True`
- `same_case_cost`: `True`
