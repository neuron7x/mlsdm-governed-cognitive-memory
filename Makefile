.PHONY: help install install-dev test lint type cov format security build clean ci-local

# ============================================
# MLSDM Governed Cognitive Memory - Development Commands
# ============================================

help:
	@echo "MLSDM Governed Cognitive Memory - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install all development dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint         - Run ruff linter on src and tests"
	@echo "  make format       - Run ruff formatter on src and tests"
	@echo "  make type         - Run mypy type checker on src/mlsdm"
	@echo "  make security     - Run security checks (pip-audit, bandit)"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run all tests (excludes load tests)"
	@echo "  make test-unit    - Run unit tests only"
	@echo "  make test-int     - Run integration tests only"
	@echo "  make test-e2e     - Run end-to-end tests only"
	@echo "  make test-prop    - Run property-based tests only"
	@echo "  make test-sec     - Run security tests only"
	@echo "  make cov          - Run tests with coverage report"
	@echo ""
	@echo "Build:"
	@echo "  make build        - Build wheel and sdist packages"
	@echo "  make docker       - Build Docker image locally"
	@echo "  make clean        - Remove build artifacts"
	@echo ""
	@echo "CI/CD:"
	@echo "  make ci-local     - Run full CI pipeline locally"
	@echo "  make ci-quick     - Run quick CI checks (lint, type, unit tests)"
	@echo ""
	@echo "Note: These commands match what CI runs. Run them before pushing."

# ============================================
# Setup
# ============================================

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev,test]"

# ============================================
# Code Quality
# ============================================

lint:
	ruff check src tests

format:
	ruff format src tests
	ruff check --fix src tests

type:
	mypy src/mlsdm

security:
	@echo "Running pip-audit for dependency vulnerabilities..."
	pip-audit --strict --progress-spinner=off || true
	@echo ""
	@echo "Running bandit security linter..."
	bandit -r src -ll -ii -x "*/tests/*" || true

# ============================================
# Testing
# ============================================

test:
	pytest --ignore=tests/load -q

test-unit:
	pytest tests/unit -v --tb=short

test-int:
	pytest tests/integration -v --tb=short

test-e2e:
	LLM_BACKEND=local_stub DISABLE_RATE_LIMIT=1 pytest tests/e2e -v -m "not slow" --tb=short

test-prop:
	pytest tests/property -v --tb=short --maxfail=5

test-sec:
	pytest tests/security -v --tb=short

cov:
	pytest --ignore=tests/load --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=90

# ============================================
# Build
# ============================================

build:
	pip install build
	python -m build

docker:
	docker build -f Dockerfile.neuro-engine-service -t mlsdm-neuro-engine:local .

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ============================================
# CI/CD
# ============================================

ci-local: lint type test-unit test-int test-e2e test-sec cov
	@echo ""
	@echo "============================================"
	@echo "CI local checks completed successfully!"
	@echo "============================================"

ci-quick: lint type test-unit
	@echo ""
	@echo "============================================"
	@echo "Quick CI checks completed successfully!"
	@echo "============================================"
