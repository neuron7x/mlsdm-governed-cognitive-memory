# Contributing to MLSDM Governed Cognitive Memory

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributors.

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
   git clone https://github.com/YOUR_USERNAME/mlsdm-governed-cognitive-memory.git
   cd mlsdm-governed-cognitive-memory
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/neuron7x/mlsdm-governed-cognitive-memory.git
   ```

## Development Setup

### Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Run tests to ensure everything works
pytest tests/ src/tests/ -v

# Run integration tests
python tests/integration/test_end_to_end.py
```

### Development Tools

We use the following tools (all included in requirements.txt):

- **pytest**: Testing framework
- **pytest-cov**: Code coverage
- **hypothesis**: Property-based testing
- **ruff**: Linting and formatting
- **mypy**: Type checking

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
