# Contributing to SCPN Fusion Core

Thank you for your interest in contributing to SCPN Fusion Core.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/<you>/scpn-fusion-core.git`
3. Create a feature branch: `git checkout -b feature/your-feature`
4. Install in development mode: `pip install -e ".[dev]"`

## Development Setup

### Python

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Rust

```bash
cd scpn-fusion-rs
cargo test
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

## Code Style

### Python

- **Formatter**: [black](https://github.com/psf/black) with default settings (line length 88)
- **Linter**: [ruff](https://github.com/astral-sh/ruff) recommended
- **Type hints**: Use where practical, especially on public APIs
- **Docstrings**: Google or NumPy style (Sphinx will parse both)

```bash
# Format
black src/ tests/

# Lint
ruff check src/ tests/
```

### Rust

- **Formatter**: `rustfmt` with default settings — run `cargo fmt` before committing
- **Linter**: `clippy` with `-D warnings` — zero warnings policy
- **Doc comments**: All `pub` items must have `///` doc comments

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
```

### Copyright Headers

**Every new source file** must include a copyright header. See any existing file for the template. Key fields:

```
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
License: GNU AGPL v3 | Commercial licensing available
```

## Pull Request Process

1. **Branch**: Create from `main` with a descriptive name (`feature/`, `fix/`, `docs/`)
2. **Tests**: All tests must pass — `pytest tests/ -v` and `cargo test`
3. **Style**: Code must be formatted (`black`, `cargo fmt`) and lint-clean
4. **Coverage**: Add tests for new functionality; aim for the same or higher coverage
5. **Docs**: Update Sphinx/rustdoc if you change public APIs
6. **Commits**: One logical change per commit; describe *why*, not just *what*
7. **PR description**: Summarize changes, link related issues, note any breaking changes
8. **Review**: At least one maintainer approval required before merge

### What Makes a Good PR

- Focused scope (one feature or fix)
- Tests that fail without the change and pass with it
- No unrelated formatting changes
- Clear commit messages

## Reporting Issues

Use our [issue templates](https://github.com/anulum/scpn-fusion-core/issues/new/choose):

- **Bug Report**: Include reproduction steps, environment, and error output
- **Feature Request**: Describe the problem and proposed solution
- **Question**: Check docs first, then ask with context

## Architecture Notes

- The Python package lives in `src/scpn_fusion/` (src layout)
- The Rust workspace in `scpn-fusion-rs/` mirrors the Python structure
- PyO3 bindings are in `scpn-fusion-rs/crates/fusion-python/`
- The neuro-symbolic compiler (`scpn/`) optionally depends on `sc-neurocore`
- Simulation modes are dispatched via `run_fusion_suite.py`

## Priority Areas for Contribution

We especially welcome contributions in:

- **Benchmarks**: criterion.rs benchmarks, Python `%timeit` comparisons
- **Validation**: Cross-checking against published tokamak data (JET, DIII-D, ITER)
- **Solver improvements**: SOR/multigrid upgrades, adaptive grid refinement
- **Documentation**: Tutorials, Jupyter notebooks, docstring improvements
- **Testing**: Increasing coverage, property-based testing with hypothesis/proptest
- **GPU acceleration**: CUDA/OpenCL offload for SOR, FFT, transport solvers
- **Data integration**: IMAS IDS import/export, MDSplus readers, CHEASE format

### Good First Issues

Look for issues tagged `good first issue` on our [issue tracker](https://github.com/anulum/scpn-fusion-core/issues). Some ideas:

- Add `%timeit` benchmarks to existing Jupyter notebooks
- Expand docstrings on public API functions
- Add new GEQDSK test cases from publicly available tokamak data
- Implement additional elliptic integral identities in proptest
- Add CI badges for code coverage

### Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| v1.0 | Core solvers, PyO3 bindings, PyPI release | Done |
| v1.1 | Adaptive mesh refinement, IMAS compatibility | Planned |
| v1.2 | GPU offload (CUDA via cupy/wgpu), FNO turbulence | Planned |
| v1.3 | Real-time digital twin integration, WebSocket API | Planned |
| v2.0 | Full ITER scenario modelling with validated transport | Planned |

## License

By contributing, you agree that your contributions will be licensed under the GNU AGPL v3.0.
