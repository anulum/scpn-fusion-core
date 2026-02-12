# Contributing to SCPN Fusion Core

Thank you for your interest in contributing to SCPN Fusion Core.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/<you>/scpn-fusion-core.git`
3. Create a feature branch: `git checkout -b feature/your-feature`
4. Install in development mode: `pip install -e .`

## Development Setup

### Python

```bash
pip install -e .
pip install pytest
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

- **Python**: PEP 8. Use type hints where practical.
- **Rust**: `rustfmt` defaults. All public items must have doc comments.

## Pull Request Process

1. Ensure all tests pass (`pytest tests/ -v` and `cargo test`)
2. Add tests for new functionality
3. Update documentation if you change public APIs
4. Keep commits focused — one logical change per commit
5. Write clear commit messages describing *why*, not just *what*

## Reporting Issues

Open a GitHub issue with:
- A clear title and description
- Steps to reproduce (if a bug)
- Expected vs actual behavior
- Python/Rust version and OS

## Architecture Notes

- The Python package lives in `src/scpn_fusion/` (src layout)
- The Rust workspace in `scpn-fusion-rs/` mirrors the Python structure
- PyO3 bindings are in `scpn-fusion-rs/crates/fusion-python/`
- The neuro-symbolic compiler (`scpn/`) optionally depends on `sc-neurocore`

## License

By contributing, you agree that your contributions will be licensed under the GNU AGPL v3.0.
