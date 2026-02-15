============
Contributing
============

SCPN-Fusion-Core is developed under the GNU Affero General Public
License v3.0.  Contributions are welcome and should follow the
guidelines below.

Getting Started
-----------------

1. Fork the repository on GitHub.
2. Clone your fork::

       git clone https://github.com/<your-username>/scpn-fusion-core.git
       cd scpn-fusion-core

3. Install in development mode::

       pip install -e ".[dev]"

4. Create a feature branch::

       git checkout -b feature/my-new-feature

Development Workflow
---------------------

**Python code:**

- Follow PEP 8 style (enforced by CI linter)
- Add type annotations for all public functions
- Write Google-style or NumPy-style docstrings
- Add unit tests for new functionality
- Run the test suite before submitting::

      pytest tests/ -v

**Rust code:**

- Follow ``rustfmt`` conventions (enforced by ``cargo fmt --check``)
- Address all ``clippy`` warnings
- Use ``FusionResult<T>`` for fallible operations (no ``unwrap()`` in
  production paths)
- Add property-based tests with ``proptest`` where appropriate
- Run the Rust test suite::

      cd scpn-fusion-rs
      cargo test --all-features

**Documentation:**

- Update docstrings when modifying public APIs
- Add user guide sections for new subsystems
- Build and review documentation locally::

      cd docs/sphinx
      make html
      # Open _build/html/index.html

Pull Request Guidelines
-------------------------

- Keep PRs focused: one feature or fix per PR
- Include tests that demonstrate the change
- Update CHANGELOG.md with a summary of the change
- Ensure CI passes (lint, test, fmt checks)
- Reference related issues in the PR description

Code Health Standards
-----------------------

The codebase has undergone 248 hardening tasks across 8 waves.  New
code should maintain these standards:

- No silent clamping or coercion -- raise explicit errors
- No ``unwrap()`` in Rust production paths -- use ``?`` operator with
  ``FusionResult<T>``
- Scoped RNG isolation for deterministic replay
- Input validation guards on all public API boundaries
- Property-based tests for numerical invariants

Contact
---------

For questions about contributing, contact: protoscience@anulum.li

For commercial licensing inquiries: www.anulum.li
