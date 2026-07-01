# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
.PHONY: test lint fmt docs docs-build docs-serve bench clean build preflight bandit sast install-hooks

PYTHON ?= python
SPHINXBUILD ?= sphinx-build
DOCS_SOURCE ?= docs/sphinx
DOCS_BUILD ?= docs/sphinx/_build/html

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
	$(MAKE) docs-build

docs-build:
	PYTHONPATH=src $(SPHINXBUILD) -W -b html $(DOCS_SOURCE) $(DOCS_BUILD)

docs-serve: docs-build
	$(PYTHON) -m http.server --directory $(DOCS_BUILD) 8000

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
