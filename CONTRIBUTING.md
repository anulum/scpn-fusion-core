# Contributing to SCPN Fusion Core

Thank you for your interest in contributing to SCPN Fusion Core -- a neuro-symbolic
tokamak simulation and control framework. Contributions from the fusion plasma
physics, scientific computing, and open-source communities are welcome and valued.

Please read this guide carefully before submitting your first pull request.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Code Style](#code-style)
- [Testing Requirements](#testing-requirements)
- [Branch Naming Conventions](#branch-naming-conventions)
- [Commit Message Format](#commit-message-format)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Architecture Overview](#architecture-overview)
- [Priority Areas for Contribution](#priority-areas-for-contribution)
- [License Implications](#license-implications)

---

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md). By
participating, you agree to uphold a welcoming, inclusive, and harassment-free
environment for everyone. Violations can be reported to **protoscience@anulum.li**.

---

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork:
   ```bash
   git clone https://github.com/<your-username>/scpn-fusion-core.git
   cd scpn-fusion-core
   ```
3. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Install** in development mode:
   ```bash
   pip install -e ".[dev]"
   ```
5. **Verify** your setup by running the test suite:
   ```bash
   pytest tests/ -v
   ```

---

## Development Environment

### Python

**Required:** Python 3.9 or later.

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
.venv\Scripts\activate       # Windows

# Install with development dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import scpn_fusion; print(scpn_fusion.__version__)"
pytest tests/ -v
```

### Rust (Optional)

The Rust workspace provides high-performance kernels. All functionality works
without Rust via NumPy/SciPy fallbacks. If you want to contribute to the Rust
codebase:

**Required:** Rust stable toolchain (1.75+), installed via [rustup](https://rustup.rs/).

```bash
cd scpn-fusion-rs

# Build
cargo build --release

# Test
cargo test --all-features

# Format check
cargo fmt --all -- --check

# Lint (zero warnings policy)
cargo clippy --all-targets --all-features -- -D warnings

# Build Python bindings (requires maturin)
pip install maturin
cd crates/fusion-python
maturin develop --release
```

### Docker

For a reproducible environment without local toolchain setup:

```bash
docker build --build-arg INSTALL_DEV=1 -t scpn-fusion-core:dev .
docker run scpn-fusion-core:dev pytest tests/ -v
```

---

## Code Style

### Python

| Tool | Purpose | Configuration |
|------|---------|---------------|
| [black](https://github.com/psf/black) | Formatter | Default settings (line length 88) |
| [ruff](https://github.com/astral-sh/ruff) | Linter | Default rule set |

```bash
# Format
black src/ tests/

# Lint
ruff check src/ tests/
```

**Additional conventions:**
- **Type hints:** Use on all public API functions and class methods.
- **Docstrings:** NumPy or Google style. All public functions, classes, and modules
  must have docstrings.
- **Imports:** Standard library, then third-party, then local -- separated by blank
  lines. Use absolute imports within the package.

### Rust

| Tool | Purpose | Policy |
|------|---------|--------|
| `rustfmt` | Formatter | Default settings; run `cargo fmt` before every commit |
| `clippy` | Linter | Zero warnings (`-D warnings`); no `#[allow]` without justification |

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
```

**Additional conventions:**
- All `pub` items must have `///` doc comments.
- Avoid `unwrap()` and `expect()` in library code. Use `FusionResult<T>` for
  fallible operations.
- Use `thiserror` for error type definitions.

### Copyright Headers

**Every new source file** must include a copyright header at the top. Use the
following template:

**Python:**
```python
# SCPN Fusion Core
# Copyright (c) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available
```

**Rust:**
```rust
// SCPN Fusion Core
// Copyright (c) 1998-2026 Miroslav Sotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// License: GNU AGPL v3 | Commercial licensing available
```

---

## Testing Requirements

**All tests must pass before a pull request will be reviewed.**

### Python Tests

```bash
# Run the full test suite
pytest tests/ -v

# Run strict typing with CI-parity settings
python tools/run_mypy_strict.py

# Run a specific test file
pytest tests/test_physics.py -v

# Run with coverage reporting
pytest tests/ -v --cov=scpn_fusion --cov-report=term-missing
```

### Rust Tests

```bash
cd scpn-fusion-rs
cargo test --all-features

# Run benchmarks (optional, for performance-related changes)
cargo bench
```

### What to Test

- **New features:** Add unit tests that cover the happy path, edge cases, and
  expected error conditions.
- **Bug fixes:** Add a regression test that fails without the fix and passes
  with it.
- **Physics modules:** Include at least one numerical invariant test (e.g.,
  conservation law, symmetry property, known analytical solution).
- **Property-based tests:** Use [Hypothesis](https://hypothesis.readthedocs.io/)
  (Python) or [proptest](https://crates.io/crates/proptest) (Rust) for numerical
  routines where applicable.

### Continuous Integration

The CI pipeline runs on every push and pull request. It checks:
- `black --check` and `ruff check` (Python formatting/linting)
- `python tools/run_mypy_strict.py` (strict Python typing)
- `cargo fmt --check` and `cargo clippy` (Rust formatting/linting)
- `pytest tests/` (Python test suite)
- `cargo test` (Rust test suite)
- `cargo audit` (Rust dependency vulnerability scan)

Your PR will not be merged until all CI checks pass.

---

## Local Operational Artifacts

Operational handover notes under `.handovers/` are strictly local-only working
artifacts and must not be committed or pushed to public remotes.

If you generate local handover notes, keep them under `.handovers/` and verify
they are excluded from `git status` before committing.

---

## Branch Naming Conventions

Use descriptive branch names with the following prefixes:

| Prefix | Use Case | Example |
|--------|----------|---------|
| `feature/` | New functionality | `feature/multigrid-v-cycle` |
| `fix/` | Bug fixes | `fix/sor-convergence-check` |
| `docs/` | Documentation only | `docs/solver-tuning-examples` |
| `refactor/` | Code restructuring (no behavior change) | `refactor/transport-module-split` |
| `test/` | Test additions or improvements | `test/property-based-inverse` |
| `bench/` | Benchmarking additions | `bench/criterion-fft-kernel` |
| `ci/` | CI/CD pipeline changes | `ci/add-coverage-reporting` |

Always branch from `main`. Keep branches focused on a single concern.

---

## Commit Message Format

Use clear, descriptive commit messages following this pattern:

```
<type>(<scope>): <short summary>

<optional body explaining WHY, not WHAT>

<optional footer with references>
```

**Types:** `feat`, `fix`, `docs`, `test`, `refactor`, `bench`, `ci`, `chore`

**Scope:** The module or area affected (e.g., `kernel`, `transport`, `scpn`,
`fusion-core`, `ci`).

**Examples:**

```
feat(transport): add neoclassical bootstrap current model

Implements the Sauter et al. (1999) bootstrap current formula
for use in the integrated transport solver. Validated against
Table 2 of the original paper.

Refs: Sauter et al., Phys. Plasmas 6, 2834 (1999)
```

```
fix(inverse): prevent singular Jacobian in Levenberg-Marquardt

The LM solver could encounter a singular Jacobian when coil
currents were near zero. Added a Tikhonov regularization floor
of 1e-8 to prevent division by zero.

Fixes #42
```

```
test(stability): add proptest for Mercier criterion symmetry
```

---

## Pull Request Process

### Before Submitting

1. **Rebase** on latest `main` to avoid merge conflicts.
2. **Run the full test suite** locally (`pytest tests/ -v` and `cargo test`).
3. **Format and lint** your code (`black`, `ruff`, `cargo fmt`, `cargo clippy`).
4. **Update documentation** if you changed any public API.
5. **Review your own diff** -- remove debugging artifacts, unrelated changes,
   and commented-out code.

### PR Requirements

- **Focused scope:** One feature, one fix, or one refactor per PR. Do not mix
  unrelated changes.
- **Tests:** Include tests that fail without your change and pass with it.
- **Documentation:** Update docstrings, Sphinx docs, or rustdoc for any public
  API changes.
- **No regressions:** All existing tests must continue to pass.
- **Descriptive title and body:** Summarize the change, explain *why* it is
  needed, and link any related issues.

### Review Process

1. At least **one maintainer approval** is required before merge.
2. Reviewers may request changes. Please address all comments or explain why
   you disagree.
3. Once approved, a maintainer will merge using squash-merge to keep the
   commit history clean.
4. The CI pipeline must be fully green before merge.

### What Makes a Good PR

- Focused scope (one logical change).
- Tests that demonstrate the change works.
- No unrelated formatting or whitespace changes.
- Clear commit messages that explain *why*, not just *what*.
- Links to relevant issues, papers, or external references.

---

## Reporting Issues

Use our [issue templates](https://github.com/anulum/scpn-fusion-core/issues/new/choose):

- **Bug Report:** Include reproduction steps, environment details, and full
  error output.
- **Feature Request:** Describe the problem being solved and the proposed
  approach, including any relevant physics or engineering context.
- **Security Vulnerability:** Do **not** open a public issue. See
  [SECURITY.md](SECURITY.md) for responsible disclosure instructions.

---

## Architecture Overview

Understanding the project structure will help you contribute effectively:

- **`src/scpn_fusion/`** -- Python package (src layout), the primary interface.
- **`scpn-fusion-rs/`** -- Rust workspace mirroring the Python structure for
  performance-critical paths.
- **`scpn-fusion-rs/crates/fusion-python/`** -- PyO3 bindings connecting Rust
  to Python.
- **`scpn/`** -- Neuro-symbolic compiler (Petri nets to stochastic neurons).
- **`validation/`** -- Reference data from real tokamaks (SPARC, ITPA, ITER).
- **`tests/`** -- Python test suite.
- **`docs/`** -- Technical documentation, solver guides, benchmark reports.
- **`examples/`** -- Jupyter notebooks and standalone scripts.

For a detailed architecture diagram, see the [README](README.md#architecture).

---

## Priority Areas for Contribution

We especially welcome contributions in these areas:

| Area | Examples | Skill Level |
|------|----------|-------------|
| **Benchmarks** | Criterion.rs benchmarks, Python timing comparisons | Intermediate |
| **Validation** | Cross-checking against published tokamak data (JET, DIII-D, ITER) | Advanced |
| **Solver improvements** | SOR/multigrid upgrades, adaptive mesh refinement | Advanced |
| **Documentation** | Tutorials, Jupyter notebooks, docstring improvements | Beginner |
| **Testing** | Increasing coverage, property-based testing (Hypothesis/proptest) | Beginner-Intermediate |
| **GPU acceleration** | CUDA/OpenCL offload for SOR, FFT, transport solvers | Advanced |
| **Data integration** | IMAS IDS import/export, MDSplus readers, CHEASE format | Intermediate |

### Good First Issues

Look for issues tagged [`good first issue`](https://github.com/anulum/scpn-fusion-core/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
on our issue tracker. Some ideas:

- Expand docstrings on public API functions.
- Add new GEQDSK test cases from publicly available tokamak data.
- Implement additional elliptic integral identities in proptest.
- Add `%timeit` benchmarks to existing Jupyter notebooks.
- Improve error messages in input validation guards.

---

## License Implications

SCPN Fusion Core is licensed under the **GNU Affero General Public License v3.0**
(AGPL-3.0-or-later). By submitting a pull request, you agree to the following:

1. **Your contributions will be licensed under the AGPL-3.0-or-later**, the same
   license as the rest of the project. You retain copyright on your contributions.
2. **You certify** that you have the right to submit the contribution under this
   license (i.e., it is your original work, or you have permission from the
   copyright holder).
3. **Derivative works** that incorporate your contribution must also be released
   under the AGPL-3.0-or-later when distributed or made available over a network.
4. **Commercial licensing** is available separately from the project maintainer
   for organizations that cannot comply with AGPL requirements. Contact
   protoscience@anulum.li for details.

If you have questions about licensing, please ask before submitting your contribution.

---

## Questions?

- Check the [documentation](docs/) and [README](README.md) first.
- Open a [Question issue](https://github.com/anulum/scpn-fusion-core/issues/new/choose)
  on GitHub.
- Email the maintainer at protoscience@anulum.li.

Thank you for helping improve fusion plasma simulation and control.
