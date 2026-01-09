.PHONY: test test-fast coverage-gate verify-metrics verify-evidence lint type cov bench bench-drift help run-dev run-cloud-local run-agent health-check eval-moral_filter test-memory-obs \
        determinism-check security-audit test-hygiene log-hygiene docs-lint policy-drift-check ci-summary \
        readiness-preview readiness-apply \
        build-package test-package docker-build-neuro-engine docker-run-neuro-engine docker-smoke-neuro-engine \
        docker-compose-up docker-compose-down lock sync evidence iteration-metrics

export PYTHONPATH := $(PYTHONPATH):$(CURDIR)/src
ITERATION_METRICS_PATH ?= artifacts/tmp/iteration-metrics.jsonl
EVIDENCE_INPUTS_PATH ?= artifacts/tmp/evidence-inputs.json

help:
	@echo "MLSDM Governed Cognitive Memory - Development Commands"
	@echo ""
	@echo "Testing & Linting:"
	@echo "  make test          - Run all tests (uses pytest.ini config)"
	@echo "  make test-fast     - Run fast unit tests (excludes slow/comprehensive)"
	@echo "  make coverage-gate - Run coverage gate with threshold check"
	@echo "  make verify-metrics - Validate latest evidence snapshot integrity"
	@echo "  make determinism-check - Validate lockfile + requirements determinism"
	@echo "  make security-audit - Run dependency vulnerability audit"
	@echo "  make test-hygiene - Enforce pytest skip/xfail hygiene"
	@echo "  make log-hygiene - Enforce structured logging rules"
	@echo "  make docs-lint - Lint documentation for duplicate headings"
	@echo "  make policy-drift-check - Detect policy drift requiring approval token"
	@echo "  make verify-evidence - Validate latest evidence snapshot completeness"
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
	@echo "Dependency Management:"
	@echo "  make sync           - Install dependencies from uv.lock"
	@echo "  make lock           - Update uv.lock with latest compatible versions"
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
	@echo "Readiness:"
	@echo "  make readiness-preview TITLE=\"Message\" [BASE_REF=origin/main] - Preview readiness change log update"
	@echo "  make readiness-apply   TITLE=\"Message\" [BASE_REF=origin/main] - Apply readiness change log update"
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

verify-metrics:
	@LATEST_SNAPSHOT=$$(ls -1d artifacts/evidence/*/*/* 2>/dev/null | LC_ALL=C sort | tail -n 1); \
	if [ -z "$$LATEST_SNAPSHOT" ]; then \
		echo "No evidence snapshot found under artifacts/evidence"; \
		exit 1; \
	fi; \
	echo "Validating evidence snapshot: $$LATEST_SNAPSHOT"; \
	python scripts/evidence/verify_evidence_snapshot.py --evidence-dir "$$LATEST_SNAPSHOT"

verify-evidence: verify-metrics

determinism-check:
	python scripts/ci/determinism_check.py

security-audit:
	python scripts/security/run_pip_audit.py --output artifacts/security/pip-audit.json

test-hygiene:
	python scripts/ci/test_hygiene.py

log-hygiene:
	python scripts/ci/logging_hygiene.py

docs-lint:
	python scripts/ci/docs_lint.py

policy-drift-check:
	python scripts/policy/check_policy_drift.py --output artifacts/tmp/policy-drift.json

ci-summary:
	python scripts/evidence/generate_ci_summary.py artifacts/tmp/ci-summary.json

lint:
	@echo "Running ruff linter (src + tests)..."
	@ruff check src tests --show-fixes || (echo "❌ Lint failed.  Run 'ruff check src tests --fix' to auto-fix" && exit 1)

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

TITLE ?=
BASE_REF ?= origin/main

readiness-preview:
	python scripts/readiness/changelog_generator.py --title "$(TITLE)" --base-ref "$(BASE_REF)" --mode preview

readiness-apply:
	python scripts/readiness/changelog_generator.py --title "$(TITLE)" --base-ref "$(BASE_REF)" --mode apply

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

# Dependency Management
sync:
	@echo "Installing dependencies from uv.lock..."
	uv sync

lock:
	@echo "Updating uv.lock with latest compatible versions..."
	uv lock --upgrade
	@echo "✓ uv.lock updated"

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
	mlsdm serve --mode dev --reload --log-level debug --disable-rate-limit

run-cloud-local:
	mlsdm serve --mode cloud-prod

run-agent:
	mlsdm serve --mode agent-api

health-check:
	python -m mlsdm.entrypoints.health

# Evaluation Suites
eval-moral_filter:
	python -m evals.moral_filter_runner

# Evidence Snapshot
iteration-metrics:
	@echo "Generating deterministic iteration metrics..."
	@mkdir -p $(dir $(ITERATION_METRICS_PATH))
	python scripts/eval/generate_iteration_metrics.py --out $(ITERATION_METRICS_PATH)

evidence: iteration-metrics
	@$(MAKE) security-audit
	@$(MAKE) ci-summary
	@mkdir -p $(dir $(EVIDENCE_INPUTS_PATH))
	@echo '{"iteration_metrics": "$(ITERATION_METRICS_PATH)", "pip_audit_json": "artifacts/security/pip-audit.json", "ci_summary": "artifacts/tmp/ci-summary.json"}' > $(EVIDENCE_INPUTS_PATH)
	@echo "Capturing evidence snapshot..."
	DISABLE_UV_RUN=1 python scripts/evidence/capture_evidence.py --mode build --inputs $(EVIDENCE_INPUTS_PATH)
	@echo "Verifying captured evidence snapshot..."
	$(MAKE) verify-evidence
