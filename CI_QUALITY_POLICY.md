# CI Quality Policy

## Overview

This document defines the CI/CD quality standards and Zero-Trust security practices for the MLSDM Governed Cognitive Memory project.

## CI Pipeline

### Workflow: CI Tests & Quality Checks

The `ci-tests.yml` workflow enforces quality standards on all pull requests and pushes to main and feature branches.

### Jobs

#### 1. Tests
**Purpose:** Ensure code functionality and correctness

**Execution:**
- Runs on Python 3.10, 3.11, and 3.12
- Executes full test suite (excluding load tests and benchmarks)
- Fails on any test failure

**Command:**
```bash
pytest -q --ignore=tests/load --ignore=benchmarks --tb=short
```

#### 2. Lint (Ruff)
**Purpose:** Enforce code style and catch common errors

**Execution:**
- Checks `src/` and `tests/` directories
- Enforces PEP 8, import sorting, and code simplifications
- Zero tolerance for linting errors

**Command:**
```bash
ruff check src tests
```

#### 3. Type Check (Mypy)
**Purpose:** Ensure type safety and prevent type-related bugs

**Execution:**
- Runs strict type checking on `src/mlsdm/`
- Installs required type stubs automatically
- Enforces type annotations and consistency

**Command:**
```bash
mypy src/mlsdm --install-types --non-interactive
```

**Configuration:** See `pyproject.toml` [tool.mypy] section

#### 4. All CI Checks Passed (Aggregator)
**Purpose:** Provide honest aggregate status

**Execution:**
- Depends on: tests, lint-ruff, type-check-mypy
- Runs always (even if dependencies fail)
- Explicitly checks each job's status
- Fails if ANY check fails
- **NO** `continue-on-error` or `|| true` workarounds

**Logic:**
```bash
if any job != success:
  print detailed status
  exit 1
else:
  print success message
```

## Local Development

### Before Committing

Run these commands locally to catch issues early:

```bash
# Run tests
pytest -q --ignore=tests/load --ignore=benchmarks

# Check linting
ruff check src tests

# Run type checking
mypy src/mlsdm --install-types --non-interactive
```

### Auto-fix Style Issues

```bash
# Auto-fix safe issues
ruff check src tests --fix

# Auto-fix including unsafe fixes (review changes!)
ruff check src tests --fix --unsafe-fixes
```

## Zero-Trust Principles

### 1. No Direct Push to Main
- All changes must go through pull requests
- CI checks must pass before merging
- No exceptions for "quick fixes"

### 2. Honest Failure Reporting
- CI never hides failures
- Aggregator job accurately reflects status
- No masking with `continue-on-error`

### 3. Type Safety
- Strict mypy checking enabled
- Minimal use of `type: ignore` comments
- All `type: ignore` must have justification

### 4. Code Quality
- Ruff enforces consistent style
- No unused imports or variables
- Modern Python idioms preferred

## Type Ignore Guidelines

When `type: ignore` is necessary:

1. **Document the reason:**
   ```python
   # type: ignore[arg-type]  # numpy stubs don't reflect runtime behavior
   ```

2. **Be specific:**
   ```python
   # Good: type: ignore[arg-type]
   # Bad:  type: ignore
   ```

3. **Provide context:**
   ```python
   # Runtime-safe despite type checker limitations
   # Tests in test_data_serializer.py verify correctness
   np.savez(path, **arrays)  # type: ignore[arg-type]
   ```

4. **Create typed wrappers when possible:**
   ```python
   def save_arrays(path: str, arrays: Mapping[str, NDArray]) -> None:
       """Typed wrapper with full documentation."""
       np.savez(path, **dict(arrays))  # type: ignore[arg-type]
   ```

## Branch Protection Requirements

### For `main` branch:
- ✅ Require pull request reviews before merging
- ✅ Require status checks to pass: `all-ci-checks-passed`
- ✅ Require branches to be up to date before merging
- ✅ Include administrators in restrictions
- ❌ No force pushes
- ❌ No deletions

## Continuous Improvement

This policy is a living document. Improvements should:
1. Maintain or raise quality standards
2. Preserve Zero-Trust principles
3. Be documented with rationale
4. Be tested before merging

## Troubleshooting

### CI Fails But Passes Locally

**Possible causes:**
1. Different Python version (test on 3.10, 3.11, 3.12)
2. Missing dependencies in requirements.txt
3. Platform-specific behavior (CI runs on Ubuntu)
4. Cache issues (rare)

**Solutions:**
- Match Python version: `python --version`
- Check dependencies: `pip list`
- Run in clean virtualenv
- Clear pip cache if needed

### Mypy Type Errors

**Approach:**
1. Understand the error (don't suppress immediately)
2. Fix the actual type issue if possible
3. Add proper type annotations
4. Use typed wrappers for library limitations
5. Document `type: ignore` as last resort

### Ruff Linting Errors

**Auto-fixable:**
```bash
ruff check src tests --fix
```

**Requires manual fix:**
- Unused variables (delete or rename with `_`)
- Complex logic simplifications (review carefully)
- Import organization (follow suggestions)

## Contact

For questions or improvements to this policy, open an issue or pull request.

---

**Last Updated:** 2025-11-24
**Maintainers:** neuron7x, GitHub Copilot Workspace
