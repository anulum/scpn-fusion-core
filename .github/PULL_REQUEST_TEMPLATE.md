## Description

<!-- What does this PR do? Why is it needed? Link related issues with "Fixes #123" or "Refs #123". -->

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Performance improvement (non-breaking change that improves speed/memory)
- [ ] Refactor (no behavior change)
- [ ] Documentation update
- [ ] Test improvement
- [ ] CI/CD change

## Testing

- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] All Rust tests pass (`cargo test --all-features`)
- [ ] New tests added for this change
- [ ] Validation scripts run successfully (if physics/solver changes)

## Code Quality

- [ ] Python formatted with `black` and linted with `ruff`
- [ ] Rust formatted with `cargo fmt` and linted with `cargo clippy -- -D warnings`
- [ ] No new `unwrap()`/`expect()` in Rust library code
- [ ] Public API has docstrings (Python) or `///` doc comments (Rust)

## Documentation

- [ ] Docstrings updated for changed APIs
- [ ] CHANGELOG.md updated (if user-facing change)
- [ ] Sphinx docs updated (if new public API)

## Physics Validation (if applicable)

- [ ] Numerical invariants preserved (conservation laws, symmetry properties)
- [ ] Results compared against known analytical solutions or published data
- [ ] No regression in validation RMSE metrics
