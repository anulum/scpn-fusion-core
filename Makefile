# SPDX-License-Identifier: AGPL-3.0-or-later
.PHONY: test lint fmt docs bench clean build preflight bandit sast install-hooks

test:
	pytest tests/ -v --cov=scpn_fusion --cov-report=term

test-rust:
	cd scpn-fusion-rs && cargo test

test-all: test test-rust

lint:
	ruff check src/ tests/

fmt:
	ruff format src/ tests/
	ruff check --fix src/ tests/
	cd scpn-fusion-rs && cargo fmt

bandit:
	bandit -r src/scpn_fusion/ -c pyproject.toml -q

sast: bandit

preflight:
	python tools/run_python_preflight.py

preflight-fast:
	python tools/run_python_preflight.py --no-tests

docs:
	mkdocs serve

docs-build:
	mkdocs build --strict

bench:
	python validation/full_validation_pipeline.py

build:
	python -m build

install-hooks:
	git config core.hooksPath .githooks
	@echo "Git hooks installed (.githooks/pre-push)"

clean:
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	cd scpn-fusion-rs && cargo clean
