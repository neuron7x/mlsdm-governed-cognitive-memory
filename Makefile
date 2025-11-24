.PHONY: test lint type cov help

help:
	@echo "MLSDM Governed Cognitive Memory - Development Commands"
	@echo ""
	@echo "make test     - Run all tests (uses pytest.ini config)"
	@echo "make lint     - Run ruff linter on src and tests"
	@echo "make type     - Run mypy type checker on src/mlsdm"
	@echo "make cov      - Run tests with coverage report"
	@echo ""
	@echo "Note: These commands match what CI runs. Run them before pushing."

test:
	pytest --ignore=tests/load

lint:
	ruff check src tests

type:
	mypy src/mlsdm

cov:
	pytest --ignore=tests/load --cov=src --cov-report=html --cov-report=term-missing
