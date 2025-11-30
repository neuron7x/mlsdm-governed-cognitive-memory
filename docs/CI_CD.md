# CI/CD Pipeline Documentation

This document describes the Continuous Integration and Continuous Deployment (CI/CD) pipeline for the MLSDM (Governed Cognitive Memory) project.

## Overview

The CI/CD pipeline consists of three main workflows:

1. **CI** (`ci.yml`) - Runs on every pull request and push to main
2. **Build** (`build.yml`) - Builds artifacts on push to main
3. **Release** (`release.yml`) - Full release pipeline triggered by version tags

## Workflow Triggers

| Workflow | Trigger | Description |
|----------|---------|-------------|
| CI | `pull_request`, `push` to main | Code quality and test gates |
| Build | `push` to main | Build Python wheel and Docker image |
| Release | `push` tags `v*.*.*` | Full release with all gates |

## CI Workflow (`ci.yml`)

The CI workflow runs on every pull request and push to main branch. It ensures code quality and correctness before merge.

### Jobs

| Job | Description | Required |
|-----|-------------|----------|
| `lint` | Ruff linter and format check | ✅ |
| `type-check` | Mypy type checking | ✅ |
| `security-scan` | pip-audit and bandit | ⚠️ (continues on error) |
| `unit-tests` | Unit tests (Python 3.10, 3.11, 3.12) | ✅ |
| `integration-tests` | Integration tests | ✅ |
| `e2e-tests` | End-to-end tests | ✅ |
| `property-tests` | Property-based tests (Hypothesis) | ✅ |
| `security-tests` | Security-specific tests | ✅ |
| `coverage` | Coverage report (≥90%) | ✅ |
| `effectiveness-validation` | SLO validation suite | ✅ |
| `ci-passed` | Summary gate | ✅ |

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     Pull Request / Push                      │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
    ┌─────────┐         ┌──────────┐        ┌───────────┐
    │  lint   │         │type-check│        │security   │
    │         │         │          │        │scan       │
    └────┬────┘         └────┬─────┘        └─────┬─────┘
         │                   │                    │
         ▼                   ▼                    │
    ┌─────────────────────────────────────┐      │
    │          Test Jobs (parallel)        │      │
    │  unit-tests | integration-tests     │      │
    │  e2e-tests  | property-tests        │      │
    │  security-tests                      │      │
    └────────────────┬────────────────────┘      │
                     │                           │
                     ▼                           │
    ┌────────────────────────────────────┐      │
    │           coverage                  │      │
    │    effectiveness-validation         │      │
    └────────────────┬───────────────────┘      │
                     │                           │
                     ▼                           │
    ┌────────────────────────────────────┐      │
    │           ci-passed                 │◄─────┘
    │       (all checks gate)             │
    └────────────────────────────────────┘
```

## Build Workflow (`build.yml`)

The build workflow runs on push to main and builds artifacts.

### Jobs

| Job | Description | Artifacts |
|-----|-------------|-----------|
| `build-wheel` | Build Python wheel and sdist | `dist/*.whl`, `dist/*.tar.gz` |
| `build-docker` | Build and push Docker image | `ghcr.io/<owner>/mlsdm-neuro-engine:latest` |
| `validate-artifacts` | Verify built artifacts | - |

### Docker Image Tags

- `ghcr.io/<owner>/mlsdm-neuro-engine:latest` - Latest main branch
- `ghcr.io/<owner>/mlsdm-neuro-engine:<sha>` - Specific commit
- `ghcr.io/<owner>/mlsdm-neuro-engine:main` - Main branch

## Release Workflow (`release.yml`)

The release workflow runs when a version tag is pushed (e.g., `v1.2.0`).

### Release Gates

All gates must pass before release artifacts are published:

1. **Unit Tests** - Python 3.10, 3.11, 3.12
2. **Integration Tests**
3. **Property-Based Tests**
4. **Validation Tests**
5. **Security Tests**
6. **Observability Tests**
7. **Performance Benchmarks**
8. **Code Quality** - Lint + Type check
9. **Coverage Check** - ≥90% coverage

### Release Artifacts

| Artifact | Location |
|----------|----------|
| Docker Image | `ghcr.io/<owner>/mlsdm-neuro-engine:<version>` |
| Python Wheel | GitHub Release assets |
| Source Dist | GitHub Release assets |
| Release Notes | GitHub Release (from CHANGELOG.md) |

### Release Process

1. Update version in `src/mlsdm/__init__.py`
2. Update `CHANGELOG.md` with release notes
3. Create and push tag:
   ```bash
   git tag -a v1.2.0 -m "Release v1.2.0"
   git push origin v1.2.0
   ```
4. GitHub Actions will automatically:
   - Run all test gates
   - Build Docker image and push to GHCR
   - Build Python wheel and sdist
   - Create GitHub Release with notes from CHANGELOG
   - Upload wheel/sdist to GitHub Release
   - Optionally publish to TestPyPI

## Specialized Workflows

### Property Tests (`property-tests.yml`)

Runs property-based tests with Hypothesis on changes to core code.

- Triggers on changes to `src/mlsdm/**`, `tests/property/**`
- Generates counterexamples report
- Checks invariant coverage

### Aphasia/NeuroLang CI (`aphasia-ci.yml`)

Specialized CI for NeuroLang extension changes.

- Triggers on changes to `src/mlsdm/extensions/**`
- Installs `neurolang` extra dependencies
- Runs aphasia-specific test suite

## Local Development

### Quick CI Check

Run minimal CI checks locally before pushing:

```bash
make ci-quick
```

This runs:
- `lint` - Ruff linter
- `type` - Mypy type checker
- `test-unit` - Unit tests

### Full CI Check

Run the complete CI pipeline locally:

```bash
make ci-local
```

This runs:
- `lint` - Ruff linter
- `type` - Mypy type checker
- `test-unit` - Unit tests
- `test-int` - Integration tests
- `test-e2e` - End-to-end tests
- `test-sec` - Security tests
- `cov` - Coverage report

### Individual Commands

```bash
# Code quality
make lint          # Run linter
make format        # Auto-format code
make type          # Type check
make security      # Security scans

# Tests
make test          # All tests
make test-unit     # Unit tests only
make test-int      # Integration tests
make test-e2e      # E2E tests
make test-prop     # Property tests
make test-sec      # Security tests
make cov           # Coverage report

# Build
make build         # Build wheel/sdist
make docker        # Build Docker image
make clean         # Clean build artifacts
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PYTHON_VERSION` | Python version for CI | `3.11` |
| `LLM_BACKEND` | LLM backend for E2E tests | `local_stub` |
| `DISABLE_RATE_LIMIT` | Disable rate limiting in tests | `1` |

### Secrets

| Secret | Description | Required For |
|--------|-------------|--------------|
| `GITHUB_TOKEN` | Auto-provided by GitHub | Docker push, Releases |
| `TEST_PYPI_API_TOKEN` | TestPyPI API token | TestPyPI publishing |

## Troubleshooting

### CI Failures

1. **Lint failures**: Run `make format` to auto-fix
2. **Type errors**: Check `make type` output
3. **Test failures**: Run `make test` locally to reproduce
4. **Coverage failure**: Ensure ≥90% coverage with `make cov`

### Build Failures

1. **Docker build**: Check `Dockerfile.neuro-engine-service`
2. **Wheel build**: Run `make build` locally

### Release Failures

1. Check all test gates pass
2. Verify CHANGELOG.md has entry for version
3. Ensure tag format is `v*.*.*` (e.g., `v1.2.0`)

## See Also

- [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md) - Deployment instructions
- [TESTING_GUIDE.md](../TESTING_GUIDE.md) - Testing documentation
- [CHANGELOG.md](../CHANGELOG.md) - Release history
