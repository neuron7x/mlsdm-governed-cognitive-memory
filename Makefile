.PHONY: test test-fast coverage-gate lint type cov bench bench-drift help run-dev run-cloud-local run-agent health-check eval-moral_filter test-memory-obs \
        build-package test-package docker-build-neuro-engine docker-run-neuro-engine docker-smoke-neuro-engine \
        docker-compose-up docker-compose-down

export PYTHONPATH := $(PYTHONPATH):$(CURDIR)/src

help:
	@echo "MLSDM Governed Cognitive Memory - Development Commands"
	@echo ""
	@echo "Testing & Linting:"
	@echo "  make test          - Run all tests (uses pytest.ini config)"
	@echo "  make test-fast     - Run fast unit tests (excludes slow/comprehensive)"
	@echo "  make coverage-gate - Run coverage gate with threshold check"
	@echo "  make lint          - Run ruff linter on src and tests"
	@echo "  make type          - Run mypy type checker on src/mlsdm"
	@echo "  make cov           - Run tests with coverage report"
	@echo "  make bench         - Run performance benchmarks (matches CI)"
	@echo "  make bench-drift   - Check benchmark results against baseline"
	@echo ""
	@echo "Package Building:"
	@echo "  make build-package  - Build wheel and sdist distributions"
	@echo "  make test-package   - Test package installation in fresh venv"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build-neuro-engine  - Build neuro-engine Docker image"
	@echo "  make docker-run-neuro-engine    - Run neuro-engine container locally"
	@echo "  make docker-smoke-neuro-engine  - Run smoke test on container"
	@echo "  make docker-compose-up          - Start local stack with docker-compose"
	@echo "  make docker-compose-down        - Stop local stack"
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

test-fast:
	@echo "Running fast unit tests (excluding slow/comprehensive)..."
	pytest tests/unit tests/state -m "not slow and not comprehensive" -q --tb=short

coverage-gate:
	@echo "Running coverage gate..."
	./coverage_gate.sh
	@if [ ! -f coverage.xml ]; then \
		echo "ERROR: coverage.xml was not generated"; \
		exit 1; \
	fi
	@echo "✓ coverage.xml generated successfully"

lint:
	ruff check src tests

type:
	mypy src/mlsdm

cov:
	pytest --ignore=tests/load --cov=src --cov-report=html --cov-report=term-missing

bench:
	@echo "Running performance benchmarks..."
	pytest benchmarks/test_neuro_engine_performance.py -v -s --tb=short

bench-drift:
	@echo "Checking benchmark drift against baseline..."
	@if [ -f benchmark-metrics.json ]; then \
		python scripts/check_benchmark_drift.py benchmark-metrics.json; \
	else \
		echo "Error: benchmark-metrics.json not found. Run 'make bench' first."; \
		exit 1; \
	fi

# Package Building
build-package:
	python -m build

test-package:
	@echo "Testing package installation in fresh venv..."
	rm -rf /tmp/mlsdm-test-venv
	python -m venv /tmp/mlsdm-test-venv
	/tmp/mlsdm-test-venv/bin/pip install --upgrade pip -q
	/tmp/mlsdm-test-venv/bin/pip install dist/*.whl -q
	/tmp/mlsdm-test-venv/bin/python scripts/test_package_install.py
	rm -rf /tmp/mlsdm-test-venv
	@echo "✓ Package test passed"

# Docker
DOCKER_IMAGE_NAME ?= ghcr.io/neuron7x/mlsdm-neuro-engine
DOCKER_IMAGE_TAG ?= latest

docker-build-neuro-engine:
	docker build -f Dockerfile.neuro-engine-service -t $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG) .

docker-run-neuro-engine:
	docker run --rm -p 8000:8000 \
		-e LLM_BACKEND=local_stub \
		-e ENABLE_METRICS=true \
		--name mlsdm-neuro-engine \
		$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)

docker-smoke-neuro-engine:
	@echo "Starting container for smoke test..."
	docker run -d --rm -p 8000:8000 \
		-e LLM_BACKEND=local_stub \
		-e DISABLE_RATE_LIMIT=1 \
		--name mlsdm-smoke-test \
		$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)
	@echo "Waiting for container to start..."
	@sleep 10
	@echo "Running smoke tests..."
	curl -s http://localhost:8000/health | grep -q "healthy" && echo "✓ Health check passed" || (echo "✗ Health check failed" && docker stop mlsdm-smoke-test && exit 1)
	curl -s http://localhost:8000/health/ready | grep -q "ready" && echo "✓ Readiness check passed" || (echo "✗ Readiness check failed" && docker stop mlsdm-smoke-test && exit 1)
	curl -s -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{"prompt": "Hello"}' | grep -q "response" && echo "✓ Generate endpoint passed" || (echo "✗ Generate test failed" && docker stop mlsdm-smoke-test && exit 1)
	@echo "Stopping container..."
	docker stop mlsdm-smoke-test
	@echo "✓ All smoke tests passed"

docker-compose-up:
	docker compose -f docker/docker-compose.yaml up -d

docker-compose-down:
	docker compose -f docker/docker-compose.yaml down

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
