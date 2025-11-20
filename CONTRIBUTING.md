# Contributing to MLSDM Governed Cognitive Memory

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Code Quality Standards](#code-quality-standards)

## Code of Conduct

This project adheres to professional engineering standards. We expect:
- Respectful communication
- Focus on technical merit
- Constructive feedback
- Collaborative problem-solving

## Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/neuron7x/mlsdm-governed-cognitive-memory.git
   cd mlsdm-governed-cognitive-memory
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set up development environment**
   ```bash
   pip install -r requirements.txt
   pip install pre-commit
   pre-commit install
   ```

## Development Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git
- pip

### Install Development Dependencies
```bash
pip install -r requirements.txt
pip install pytest-asyncio pre-commit
```

### Verify Installation
```bash
# Run unit tests
pytest src/tests/unit/ -v

# Run integration tests
python tests/integration/test_end_to_end.py

# Check linting
ruff check .

# Check types
mypy . --strict --ignore-missing-imports
```

## Testing Guidelines

### Test Categories

#### 1. Unit Tests (`src/tests/unit/`)
**Required for**: All new components and functions
- Test individual functions and classes
- Use mocking for external dependencies
- Aim for >90% code coverage
- Follow existing test patterns

Example:
```python
import pytest
from src.cognition.moral_filter_v2 import MoralFilterV2

def test_moral_filter_initialization():
    """Test moral filter initializes with correct defaults"""
    filter = MoralFilterV2(initial_threshold=0.50)
    assert 0.3 <= filter.threshold <= 0.9
    assert filter.ema_accept_rate == 0.5
```

#### 2. Property-Based Tests (`src/tests/unit/test_property_based.py`)
**Required for**: Core invariants and mathematical properties
- Use Hypothesis for generating test cases
- Verify invariants hold across wide input ranges
- Document expected properties

Example:
```python
from hypothesis import given, strategies as st
from src.cognition.moral_filter_v2 import MoralFilterV2

@given(threshold=st.floats(min_value=0.0, max_value=1.0))
def test_threshold_always_bounded(threshold):
    """Threshold should always be clamped to [0.3, 0.9]"""
    filter = MoralFilterV2(initial_threshold=threshold)
    assert 0.3 <= filter.threshold <= 0.9
```

#### 3. Integration Tests (`tests/integration/`)
**Required for**: Multi-component interactions
- Test realistic usage scenarios
- Verify component integration
- Test end-to-end flows

#### 4. Chaos Engineering Tests (`tests/chaos/`)
**Recommended for**: Resilience-critical changes
- Test system under failure conditions
- Verify graceful degradation
- Test concurrent access patterns

#### 5. Adversarial Tests (`tests/adversarial/`)
**Required for**: Security and moral filter changes
- Test resistance to attacks
- Verify safety boundaries
- Test manipulation attempts

#### 6. Performance Tests (`tests/performance/`)
**Required for**: Performance-critical changes
- Benchmark latency and throughput
- Verify memory bounds
- Check for regressions

### Running Tests

```bash
# All tests
pytest src/tests/ tests/ -v --cov=src

# Specific category
pytest src/tests/unit/ -v
python tests/chaos/test_fault_injection.py
python tests/adversarial/test_jailbreak_resistance.py
python tests/performance/test_benchmarks.py

# With coverage report
pytest src/tests/unit/ --cov=src --cov-report=html

# Property-based tests with statistics
pytest src/tests/unit/test_property_based.py --hypothesis-show-statistics
```

### Test Quality Requirements

1. **Coverage**: Maintain ≥90% code coverage
2. **Assertions**: Every test must have clear assertions
3. **Documentation**: Docstrings explaining what is tested
4. **Independence**: Tests must not depend on execution order
5. **Speed**: Unit tests should run quickly (<1s each)
6. **Determinism**: Tests must be reproducible

### Adding New Tests

When adding new functionality:

1. **Write tests first** (TDD approach recommended)
2. **Cover edge cases**: Zero, negative, extreme values
3. **Test error handling**: Invalid inputs, exceptions
4. **Add property tests**: For mathematical invariants
5. **Consider chaos tests**: For concurrent/resilient code

## Pull Request Process

### Before Submitting

1. **Run all tests locally**
   ```bash
   pytest src/tests/ tests/ -v --cov=src --cov-fail-under=90
   ```

2. **Check code quality**
   ```bash
   ruff check . --fix
   mypy . --strict --ignore-missing-imports
   ```

3. **Run security checks**
   ```bash
   bandit -r src/ -f json
   ```

4. **Update documentation** if needed

### PR Checklist

- [ ] Tests added for new functionality
- [ ] All tests passing locally
- [ ] Code coverage ≥90%
- [ ] Linting and type checking pass
- [ ] Security scan passes
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)
- [ ] Commit messages are clear and descriptive

### PR Title Format

Use conventional commit format:
- `feat:` New feature
- `fix:` Bug fix
- `test:` Adding or updating tests
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

Example: `feat: add phase-based memory retrieval optimization`

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Test improvement

## Testing
- Unit tests: X tests added/updated
- Integration tests: Y scenarios covered
- Performance: Z% improvement / No regression

## Checklist
- [ ] Tests pass locally
- [ ] Coverage ≥90%
- [ ] Documentation updated
- [ ] No security vulnerabilities
```

### Review Process

1. **Automated checks** run via GitHub Actions
2. **Maintainer review** for code quality and design
3. **Address feedback** and update PR
4. **Final approval** and merge

## Code Quality Standards

### Style Guide

Follow Python PEP 8 with project-specific conventions:
- Line length: 100 characters
- Use type hints for function signatures
- Docstrings for all public functions/classes
- Ruff for linting and formatting

### Type Checking

Use type hints and pass MyPy strict checking:
```python
def process_event(vector: np.ndarray, moral_value: float) -> dict[str, Any]:
    """Process cognitive event with type safety"""
    ...
```

### Documentation

- **Docstrings**: Use Google style for all public APIs
- **Comments**: Explain "why", not "what"
- **Type hints**: Required for function signatures
- **Module docs**: Brief description at top of file

Example:
```python
"""
Moral Filter v2

Adaptive moral threshold with EMA-based convergence.
Ensures bounded threshold in [0.3, 0.9] range.
"""

def evaluate(self, moral_value: float) -> bool:
    """
    Evaluate if content passes moral threshold.
    
    Args:
        moral_value: Moral score in range [0, 1]
        
    Returns:
        True if content passes threshold, False otherwise
    """
    return bool(moral_value >= self.threshold)
```

### Security

- No hardcoded secrets or credentials
- Validate all external inputs
- Use safe dependencies (check with pip-audit)
- Follow secure coding practices
- Pass Bandit security linting

### Performance

- Maintain P95 latency < 120ms
- Keep memory footprint ≤1.4GB
- Ensure thread safety for concurrent operations
- No memory leaks (verify with profiling tests)

## Project Structure

```
mlsdm-governed-cognitive-memory/
├── src/                      # Source code
│   ├── cognition/           # Moral filter, ontology
│   ├── core/                # Controller, memory manager
│   ├── memory/              # QILM, multi-level memory
│   ├── rhythm/              # Cognitive rhythm
│   ├── tests/               # Unit tests (co-located)
│   │   └── unit/
│   └── utils/               # Utilities
├── tests/                    # Integration & specialized tests
│   ├── integration/         # End-to-end tests
│   ├── validation/          # Effectiveness validation
│   ├── chaos/               # Fault injection tests
│   ├── adversarial/         # Jailbreak resistance
│   └── performance/         # Benchmarks
├── .github/workflows/       # CI/CD workflows
├── config/                  # Configuration files
└── scripts/                 # Utility scripts
```

## Getting Help

- **Issues**: Search existing issues or create new one
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: See README.md and TESTING_STRATEGY.md
- **Architecture**: See ARCHITECTURE_SPEC.md

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to production-grade neurobiological AI systems!**
