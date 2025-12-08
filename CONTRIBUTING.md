# Contributing Guide

**Document Version:** 1.0.0  
**Project Version:** 1.0.0  
**Last Updated:** November 2025  
**Minimum Coverage:** 90%

Thank you for your interest in contributing to MLSDM Governed Cognitive Memory! This document provides comprehensive guidelines and instructions for contributors.

> **See also:** [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for a comprehensive guide to the project layout, development workflow, patterns, and debugging.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation Requirements](#documentation-requirements)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to professional engineering standards. We expect:

- **Respectful communication** in all interactions
- **Technical excellence** in contributions
- **Constructive feedback** during code reviews
- **Focus on project goals** and user needs

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Git for version control
- Understanding of cognitive architectures and LLM systems (recommended)

### First Steps

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/mlsdm.git
   cd mlsdm
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/neuron7x/mlsdm.git
   ```

## Development Setup

### Local Dev / Tests

#### 1. Create Virtual Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 2. Install Development Dependencies

```bash
# Install all dev/test dependencies
pip install -r requirements-dev.txt

# This includes:
# - Core dependencies (from requirements.txt)
# - Testing tools (pytest, pytest-cov, hypothesis)
# - Linting/type checking (ruff, mypy)
# - Load testing (locust)
# - OpenTelemetry (for full observability testing)
```

#### 3. Verify Installation

```bash
# Run quick smoke tests
./scripts/dev_smoke_tests.sh

# Or use the canonical test command directly
PYTHONPATH=src pytest -q --ignore=tests/load
```

### Development Tools

We use the following tools (all included in requirements-dev.txt):

- **pytest**: Testing framework
- **pytest-cov**: Code coverage
- **pytest-asyncio**: Async test support
- **hypothesis**: Property-based testing
- **ruff**: Linting and formatting
- **mypy**: Type checking
- **httpx**: HTTP client for testing
- **locust**: Load testing
- **OpenTelemetry**: Observability (optional in prod, included in dev)

### Canonical Development Commands

These commands match what CI runs. **Always run these before pushing:**

```bash
# CANONICAL TEST COMMAND (matches CI exactly)
PYTHONPATH=src pytest -q --ignore=tests/load

# Quick smoke tests (same as above but wrapped in a script)
./scripts/dev_smoke_tests.sh

# Run linter (ruff)
make lint
# Or: ruff check src tests

# Run type checker (mypy)
make type
# Or: mypy src/mlsdm

# Run tests with coverage
make cov
# Or: pytest --ignore=tests/load --cov=src --cov-report=html --cov-report=term-missing

# Show all available commands
make help
```

## Development Workflow

### 1. Create a Feature Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following our [coding standards](#coding-standards)
- Add tests for new functionality
- Update documentation as needed
- Commit frequently with clear messages

### 3. Test Your Changes

```bash
# Run all tests
pytest tests/ src/tests/ -v

# Run with coverage
pytest --cov=src --cov-report=html tests/ src/tests/

# Run linting
ruff check src/ tests/

# Run type checking
mypy src/
```

### 4. Commit Guidelines

Write clear, descriptive commit messages:

```
<type>: <short summary>

<detailed description if needed>

<issue reference if applicable>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Example:**
```
feat: Add adaptive batch processing to memory consolidation

Implements dynamic batch sizing based on memory pressure.
Reduces consolidation time by 30% under high load.

Closes #42
```

## Coding Standards

### Python Style

We follow **PEP 8** with these additions:

- **Maximum line length**: 100 characters
- **Type hints**: Required for all public functions
- **Docstrings**: Required for all public classes and functions (Google style)
- **Import order**: stdlib, third-party, local (use isort)

### Code Quality Standards

1. **Type Hints**
   ```python
   from typing import Optional, List
   import numpy as np

   def process_vector(
       vector: np.ndarray,
       threshold: float = 0.5,
       options: Optional[List[str]] = None
   ) -> dict:
       """Process vector with given threshold.
       
       Args:
           vector: Input vector of shape (dim,)
           threshold: Processing threshold (0.0-1.0)
           options: Optional processing options
           
       Returns:
           Dictionary containing processing results
           
       Raises:
           ValueError: If vector dimension is invalid
       """
       pass
   ```

2. **Error Handling**
   ```python
   def safe_operation(data: np.ndarray) -> np.ndarray:
       """Perform operation with proper error handling."""
       if data.size == 0:
           raise ValueError("Input data cannot be empty")
       
       try:
           result = complex_operation(data)
       except RuntimeError as e:
           logger.error(f"Operation failed: {e}")
           raise
       
       return result
   ```

3. **Docstrings** (Google Style)
   ```python
   class MoralFilter:
       """Adaptive moral filtering with homeostatic threshold.
       
       The filter maintains a dynamic threshold that adapts based on
       acceptance rates to achieve approximately 50% acceptance.
       
       Attributes:
           threshold: Current moral threshold value (0.30-0.90)
           ema: Exponential moving average of acceptance rate
           
       Example:
           >>> filter = MoralFilter(initial_threshold=0.5)
           >>> accepted = filter.evaluate(moral_value=0.8)
           >>> filter.adapt(accepted)
       """
       pass
   ```

### Architecture Principles

1. **Immutability**: Prefer immutable data structures where possible
2. **Single Responsibility**: Each class/function should have one clear purpose
3. **Dependency Injection**: Pass dependencies rather than creating them
4. **Interface Segregation**: Keep interfaces focused and minimal
5. **Fail Fast**: Validate inputs early and raise clear errors

## Testing Requirements

### Test Coverage

- **Minimum coverage**: 90% for all new code
- **Critical paths**: 100% coverage required for:
  - Moral filtering logic
  - Memory management
  - Phase transitions
  - Thread-safe operations

### Test Types

1. **Unit Tests** (`src/tests/unit/`)
   - Test individual components in isolation
   - Use mocks for dependencies
   - Fast execution (< 1ms per test)

2. **Integration Tests** (`tests/integration/`)
   - Test component interactions
   - Test end-to-end workflows
   - Moderate execution time (< 1s per test)

3. **Property-Based Tests** (`src/tests/unit/`)
   - Use Hypothesis for invariant testing
   - Test mathematical properties
   - Generate edge cases automatically

4. **Validation Tests** (`tests/validation/`)
   - Test effectiveness claims
   - Measure performance characteristics
   - Validate system behavior under load

### Writing Tests

```python
import pytest
import numpy as np
from hypothesis import given, strategies as st

class TestMoralFilter:
    """Test suite for MoralFilter component."""
    
    def test_basic_evaluation(self):
        """Test basic moral evaluation."""
        filter = MoralFilter(0.5)
        assert filter.evaluate(0.8) is True
        assert filter.evaluate(0.2) is False
    
    @given(moral_value=st.floats(0.0, 1.0))
    def test_threshold_bounds_property(self, moral_value):
        """Threshold always stays in valid range."""
        filter = MoralFilter(0.5)
        filter.evaluate(moral_value)
        filter.adapt(True)
        assert 0.30 <= filter.threshold <= 0.90
    
    def test_thread_safety(self):
        """Test concurrent access is safe."""
        filter = MoralFilter(0.5)
        # Thread safety test implementation
        pass
```

### Running Tests

```bash
# All tests
pytest tests/ src/tests/ -v

# Specific test file
pytest tests/integration/test_end_to_end.py -v

# Specific test function
pytest tests/integration/test_end_to_end.py::test_basic_flow -v

# With coverage
pytest --cov=src --cov-report=html tests/ src/tests/

# Property-based tests only
pytest -k property -v

# Integration tests only
pytest tests/integration/ -v
```

## Documentation Requirements

### Code Documentation

1. **Module Docstrings**: Every module needs a docstring describing its purpose
2. **Class Docstrings**: All classes need comprehensive documentation
3. **Function Docstrings**: All public functions need complete docstrings
4. **Inline Comments**: Use for complex logic only, prefer self-documenting code

### Documentation Files

When adding new features, update:

- **README.md**: If feature affects user-facing API
- **USAGE_GUIDE.md**: Add usage examples
- **ARCHITECTURE_SPEC.md**: Document architectural changes
- **IMPLEMENTATION_SUMMARY.md**: Update implementation status

### Example Documentation

See existing files for style guidelines:
- [README.md](README.md) - Feature overview
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Detailed usage
- [examples/](examples/) - Working code examples

## Pull Request Process

### Before Submitting

- [ ] All tests pass locally
- [ ] Code coverage ≥ 90%
- [ ] Linting passes (ruff)
- [ ] Type checking passes (mypy)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- Describe testing performed
- List any new tests added

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] Changes are backwards compatible (or documented)
```

### Review Process

1. **Automated Checks**: CI runs tests, linting, type checking
2. **Code Review**: Maintainer reviews code quality and design
3. **Testing**: Maintainer may test changes locally
4. **Merge**: Once approved, changes are merged to main

### Branch Protection (CICD-002)

The `main` branch has branch protection rules that require the following status checks to pass before merging:

**Required Status Checks:**

| Check Name | Workflow | Description |
|------------|----------|-------------|
| `Lint and Type Check` | `ci-neuro-cognitive-engine.yml` | Ruff linting and mypy type checking |
| `Security Vulnerability Scan` | `ci-neuro-cognitive-engine.yml` | pip-audit dependency scanning |
| `test (3.10)` | `ci-neuro-cognitive-engine.yml` | Unit tests on Python 3.10 |
| `test (3.11)` | `ci-neuro-cognitive-engine.yml` | Unit tests on Python 3.11 |
| `End-to-End Tests` | `ci-neuro-cognitive-engine.yml` | E2E integration tests |
| `Effectiveness Validation` | `ci-neuro-cognitive-engine.yml` | SLO and effectiveness validation |
| `All CI Checks Passed` | `ci-neuro-cognitive-engine.yml` | Gate job requiring all checks |

**Additional Branch Protection Settings:**

- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Require at least 1 approval (recommended)
- ✅ Dismiss stale pull request approvals when new commits are pushed
- ❌ Do not allow bypassing the above settings

**Configure via GitHub CLI:**

Repository administrators can configure branch protection using the GitHub CLI:

```bash
# Enable branch protection with required status checks
gh api repos/{owner}/{repo}/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["Lint and Type Check","Security Vulnerability Scan","test (3.10)","test (3.11)","End-to-End Tests","Effectiveness Validation","All CI Checks Passed"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":1,"dismiss_stale_reviews":true}' \
  --field restrictions=null

# Verify branch protection is configured
gh api repos/{owner}/{repo}/branches/main/protection
```

### Aphasia / NeuroLang CI Gate

Each PR that modifies NeuroLang/Aphasia-Broca components triggers a dedicated CI job (`aphasia-neurolang`) that:

- **Runs all Aphasia/NeuroLang tests**, including:
  - `tests/validation/test_aphasia_detection.py` - Core aphasia detection tests
  - `tests/eval/test_aphasia_eval_suite.py` - Evaluation suite tests
  - `tests/observability/test_aphasia_logging.py` - Logging tests
  - `tests/security/test_aphasia_*` - Security-related tests
  - `tests/packaging/test_neurolang_optional_dependency.py` - Dependency tests
  - `tests/scripts/test_*neurolang*` - NeuroLang script tests

- **Runs AphasiaEvalSuite quality gate** (`scripts/run_aphasia_eval.py --fail-on-low-metrics`) which verifies:
  - True Positive Rate (TPR) ≥ 0.8
  - True Negative Rate (TNR) ≥ 0.8
  - Mean severity for telegraphic samples ≥ 0.3

- **Blocks merge** if:
  - Any aphasia/neurolang test fails
  - Metrics fall below required thresholds

**Local Reproduction:**

To run the same checks locally before pushing:

```bash
# Install with neurolang extras
pip install '.[neurolang]'

# Run aphasia/neurolang tests
pytest tests/validation/test_aphasia_detection.py
pytest tests/eval/test_aphasia_eval_suite.py
pytest tests/observability/test_aphasia_logging.py
pytest tests/security/test_aphasia_*
pytest tests/packaging/test_neurolang_optional_dependency.py
pytest tests/scripts/test_*neurolang*

# Run quality gate
python scripts/run_aphasia_eval.py \
  --corpus tests/eval/aphasia_corpus.json \
  --fail-on-low-metrics
```

**Trigger Conditions:**

The Aphasia/NeuroLang CI job is triggered when PRs modify:
- `src/mlsdm/extensions/**` - NeuroLang extension code
- `src/mlsdm/observability/**` - Observability components
- `tests/eval/**` - Evaluation tests and data
- `tests/validation/test_aphasia_*` - Aphasia validation tests
- `tests/observability/test_aphasia_*` - Aphasia observability tests
- `tests/security/test_aphasia_*` - Aphasia security tests
- `tests/packaging/test_neurolang_optional_dependency.py` - Dependency tests
- `scripts/*neurolang*` - NeuroLang-related scripts
- `scripts/run_aphasia_eval.py` - Evaluation script

### Review Criteria

Reviewers will check:

- **Correctness**: Does code work as intended?
- **Design**: Is architecture sound?
- **Testing**: Are tests comprehensive?
- **Documentation**: Is documentation clear and complete?
- **Performance**: Are there performance implications?
- **Security**: Are there security considerations?

## Release Process

### Version Numbering

We follow **Semantic Versioning** (semver.org):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

### Release Checklist

1. Update version in all relevant files
2. Update CHANGELOG.md
3. Update documentation
4. Run full test suite
5. Create git tag
6. Build and test package
7. Publish release notes

## Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: See README.md and USAGE_GUIDE.md
- **Email**: Contact maintainer for sensitive issues

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

## Acknowledgments

Thank you for contributing to advancing neurobiologically-grounded AI systems with built-in safety and governance!

---

**Note**: This is a professional, production-ready project. We maintain high standards for code quality, testing, and documentation. Please take time to understand these guidelines before contributing.
