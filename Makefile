.PHONY: test lint type cov help run-dev run-cloud-local run-agent health-check eval-moral_filter test-memory-obs

help:
	@echo "MLSDM Governed Cognitive Memory - Development Commands"
	@echo ""
	@echo "Testing & Linting:"
	@echo "  make test     - Run all tests (uses pytest.ini config)"
	@echo "  make lint     - Run ruff linter on src and tests"
	@echo "  make type     - Run mypy type checker on src/mlsdm"
	@echo "  make cov      - Run tests with coverage report"
	@echo ""
	@echo "Observability Tests:"
	@echo "  make test-memory-obs - Run memory observability tests"
	@echo ""
	@echo "Evaluations:"
	@echo "  make eval-moral_filter - Run moral filter evaluation suite"
	@echo ""
	@echo "Runtime Modes:"
	@echo "  make run-dev        - Start development server (hot reload, debug logging)"
	@echo "  make run-cloud-local - Start local production server (multiple workers)"
	@echo "  make run-agent      - Start agent/API server (for LLM integration)"
	@echo "  make health-check   - Run health check"
	@echo ""
	@echo "Note: These commands match what CI runs. Run them before pushing."

# Testing & Linting
test:
	pytest --ignore=tests/load

lint:
	ruff check src tests

type:
	mypy src/mlsdm

cov:
	pytest --ignore=tests/load --cov=src --cov-report=html --cov-report=term-missing

# Observability Tests
test-memory-obs:
	pytest tests/observability/test_memory_observability.py -v

# Runtime Modes
run-dev:
	python -m mlsdm.entrypoints.dev

run-cloud-local:
	python -m mlsdm.entrypoints.cloud

run-agent:
	python -m mlsdm.entrypoints.agent

health-check:
	python -m mlsdm.entrypoints.health

# Evaluation Suites
eval-moral_filter:
	python -m evals.moral_filter_runner
