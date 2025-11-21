# GitHub Actions Workflows

This document describes the GitHub Actions workflows configured for this project.

## Overview

The project uses GitHub Actions for continuous integration, security scanning, and automated dependency management. All workflows are located in `.github/workflows/`.

## Workflows

### 1. CI Workflow (`ci.yml`)

**Trigger:** Push and Pull Requests to `main` and `develop` branches, manual dispatch

**Purpose:** Continuous integration testing and validation

**Jobs:**

- **Lint:** Checks code style with `ruff`
  - Runs on: Ubuntu Latest
  - Python: 3.12
  
- **Type Check:** Validates type hints with `mypy`
  - Runs on: Ubuntu Latest
  - Python: 3.12
  - Uses strict mode
  
- **Test:** Runs unit tests with coverage requirements
  - Runs on: Ubuntu Latest
  - Python: 3.12
  - Coverage threshold: 90%
  - Uploads coverage to Codecov
  
- **Integration Test:** Runs integration test suite
  - Runs on: Ubuntu Latest
  - Python: 3.12
  - Depends on: lint, typecheck, test
  
- **Validation Test:** Runs validation tests
  - Runs on: Ubuntu Latest
  - Python: 3.12
  - Depends on: lint, typecheck, test
  - Allowed to fail (continue-on-error: true)

**Status Badge:**
```markdown
[![CI](https://github.com/neuron7x/mlsdm-governed-cognitive-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/neuron7x/mlsdm-governed-cognitive-memory/actions/workflows/ci.yml)
```

### 2. CodeQL Security Scan (`codeql.yml`)

**Trigger:** Push and Pull Requests to `main` and `develop` branches, weekly schedule (Monday midnight UTC), manual dispatch

**Purpose:** Automated security vulnerability scanning

**Jobs:**

- **Analyze:** Performs CodeQL analysis on Python code
  - Runs on: Ubuntu Latest
  - Language: Python
  - Queries: security-extended, security-and-quality
  - Schedule: Weekly on Mondays

**Permissions:**
- actions: read
- contents: read
- security-events: write

**Status Badge:**
```markdown
[![CodeQL](https://github.com/neuron7x/mlsdm-governed-cognitive-memory/actions/workflows/codeql.yml/badge.svg)](https://github.com/neuron7x/mlsdm-governed-cognitive-memory/actions/workflows/codeql.yml)
```

### 3. Dependency Review (`dependency-review.yml`)

**Trigger:** Pull Requests to `main` and `develop` branches

**Purpose:** Reviews dependency changes for security vulnerabilities

**Jobs:**

- **Dependency Review:** Checks for vulnerable dependencies
  - Runs on: Ubuntu Latest
  - Fail on: Moderate severity or higher
  - Posts summary comment in PR

**Permissions:**
- contents: read
- pull-requests: write

### 4. Documentation Check (`docs.yml`)

**Trigger:** Push and Pull Requests to `main` (only for documentation changes), manual dispatch

**Purpose:** Validates documentation quality and completeness

**Jobs:**

- **Check Docs:** Validates documentation files
  - Runs on: Ubuntu Latest
  - Python: 3.12
  - Checks markdown links
  - Verifies required documentation files exist

**Required Files:**
- README.md
- CONTRIBUTING.md
- LICENSE
- SECURITY_POLICY.md
- API_REFERENCE.md
- USAGE_GUIDE.md

**Status Badge:**
```markdown
[![Documentation](https://github.com/neuron7x/mlsdm-governed-cognitive-memory/actions/workflows/docs.yml/badge.svg)](https://github.com/neuron7x/mlsdm-governed-cognitive-memory/actions/workflows/docs.yml)
```

### 5. Publish to PyPI (`publish.yml`)

**Trigger:** Release published, manual dispatch

**Purpose:** Publishes package to PyPI

**Jobs:**

- **Build and Publish:** Builds and publishes Python package
  - Runs on: Ubuntu Latest
  - Python: 3.12
  - Uses trusted publishing (OIDC)
  - Skips existing versions

**Permissions:**
- contents: read
- id-token: write (for trusted publishing)

**Setup Required:**
To enable this workflow, configure PyPI trusted publishing:
1. Go to PyPI project settings
2. Enable trusted publishing
3. Add GitHub repository as a trusted publisher

## Dependabot Configuration

**File:** `.github/dependabot.yml`

**Purpose:** Automated dependency updates

**Configuration:**

- **Python Dependencies:**
  - Schedule: Weekly (Monday at 09:00 UTC)
  - Max open PRs: 5
  - Labels: dependencies, python
  
- **GitHub Actions:**
  - Schedule: Weekly (Monday at 09:00 UTC)
  - Max open PRs: 3
  - Labels: dependencies, github-actions

## Issue and PR Templates

### Issue Templates

Located in `.github/ISSUE_TEMPLATE/`:

1. **Bug Report** (`bug_report.yml`)
   - Structured template for bug reports
   - Includes: description, reproduction steps, environment info
   
2. **Feature Request** (`feature_request.yml`)
   - Structured template for feature requests
   - Includes: problem statement, proposed solution, priority

3. **Config** (`config.yml`)
   - Disables blank issues
   - Links to documentation and discussions

### Pull Request Template

**File:** `.github/PULL_REQUEST_TEMPLATE.md`

**Includes:**
- Description
- Type of change
- Related issues
- Testing checklist
- Documentation checklist
- General checklist

## Workflow Best Practices

### For Contributors

1. **Before Pushing:**
   ```bash
   # Run linting
   ruff check .
   
   # Run type checking
   mypy . --strict
   
   # Run tests
   pytest src/tests/unit/ -v --cov=src --cov-fail-under=90
   ```

2. **Monitoring Workflows:**
   - Check the Actions tab for workflow runs
   - Address any failures promptly
   - Review Dependabot PRs regularly

3. **Security:**
   - Review CodeQL findings
   - Keep dependencies up to date
   - Follow security best practices

### For Maintainers

1. **Releases:**
   - Create a new release on GitHub
   - The publish workflow will automatically trigger
   - Verify package on PyPI

2. **Workflow Maintenance:**
   - Keep actions up to date (Dependabot helps)
   - Review and adjust coverage thresholds
   - Monitor workflow execution times

3. **Security:**
   - Review security advisories weekly
   - Enable secret scanning
   - Use branch protection rules

## Troubleshooting

### Common Issues

**1. CI Fails on Coverage:**
- Ensure tests cover at least 90% of code
- Add tests for new functionality
- Check coverage report: `pytest --cov=src --cov-report=html`

**2. CodeQL False Positives:**
- Review alert in Security tab
- Add suppression comment if necessary
- Document reason for suppression

**3. Dependabot PRs:**
- Review changes carefully
- Run tests locally before merging
- Check for breaking changes

**4. Type Check Failures:**
- Add type hints to new code
- Use `# type: ignore` sparingly
- Consider updating mypy if needed

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [CodeQL Documentation](https://codeql.github.com/docs/)
- [Dependabot Documentation](https://docs.github.com/en/code-security/dependabot)
- [Python Packaging Guide](https://packaging.python.org/)

## Contact

For questions about workflows, open an issue or discussion in the repository.
