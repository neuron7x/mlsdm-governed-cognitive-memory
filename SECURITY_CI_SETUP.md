# Security & CI Configuration Guide for MLSDM

This document describes the security and CI/CD hardening implemented for the MLSDM Governed Cognitive Memory repository.

## Table of Contents

1. [Overview](#overview)
2. [CI/CD Workflows](#cicd-workflows)
3. [Security Scanning](#security-scanning)
4. [Branch Protection](#branch-protection)
5. [Automated PR Management](#automated-pr-management)
6. [Dependency Management](#dependency-management)
7. [Secret Management](#secret-management)
8. [Configuration Guide](#configuration-guide)

---

## Overview

The MLSDM repository implements a comprehensive security and CI/CD strategy aligned with the neuro-cognitive architecture's high security and reliability requirements. This setup ensures:

- **Zero direct pushes to `main`** - all changes go through reviewed PRs
- **Automated quality checks** - lint, type-check, tests on every PR
- **Security scanning** - CodeQL analysis, secret detection, dependency vulnerabilities
- **Automated maintenance** - Dependabot for dependency updates
- **Clear ownership** - CODEOWNERS for critical code areas
- **Standardized PRs** - template ensuring security and testing considerations

---

## CI/CD Workflows

### 1. CI Tests & Quality Checks (`.github/workflows/ci-tests.yml`)

**Triggers:**
- On every PR to `main` or `feature/*` branches
- On push to `main`

**Jobs:**

#### Lint (ci/lint)
- Runs `ruff` linter on source code and tests
- Enforces code style and catches common issues
- Status check: **REQUIRED** for merge

#### Type Check (ci/types)
- Runs `mypy` for static type checking
- Validates type hints across the codebase
- Status check: **REQUIRED** for merge

#### Tests (ci/pytest)
- Runs on Python 3.10, 3.11, and 3.12
- Executes:
  - Unit tests (`tests/unit/`)
  - Integration tests (`tests/integration/`)
  - Security tests (`tests/security/`)
- Generates coverage reports (uploaded as artifacts for Python 3.11)
- Status check: **REQUIRED** for merge

#### Property-Based Tests
- Runs Hypothesis-based property tests (`tests/property/`)
- Validates formal invariants and safety properties
- 15-minute timeout
- Status check: **REQUIRED** for merge

**All checks must pass before a PR can be merged.**

---

### 2. CodeQL Security Analysis (`.github/workflows/codeql-analysis.yml`)

**Triggers:**
- On every PR to `main`
- On push to `main`
- Weekly on Sundays at 00:00 UTC (schedule)

**Configuration:**
- Language: Python
- Query suites: `security-extended` and `security-and-quality`
- Excludes: `tests/`, `benchmarks/`, `examples/`, `docs/`

**Scanning:**
- Detects security vulnerabilities (SQL injection, XSS, etc.)
- Identifies code quality issues
- Results uploaded to GitHub Security tab
- **Fails on error** - blocks PR merge if critical/high severity issues found

**Status check: REQUIRED for merge**

---

### 3. Auto Labeler (`.github/workflows/labeler.yml`)

**Triggers:**
- When PR is opened, synchronized, or reopened

**Function:**
- Automatically applies labels based on changed files
- Labels include: `core`, `security`, `api`, `documentation`, `tests`, etc.
- Helps with PR triage and review assignment

**Configuration:** See `.github/labeler.yml`

---

## Security Scanning

### CodeQL Code Scanning

**Status:** ✅ Enabled via workflow

CodeQL performs deep semantic analysis of Python code to detect:
- Security vulnerabilities
- Code quality issues
- Common bug patterns

**Access Results:**
- Navigate to **Security → Code scanning** tab in GitHub
- Review alerts and take action

### Secret Scanning

**Status:** ⚠️ Requires repository settings configuration

**To Enable:**

1. Go to **Settings → Code security and analysis**
2. Enable **Secret scanning**
3. Enable **Push protection** (recommended)

**What it does:**
- Scans for accidentally committed secrets (API keys, tokens, passwords)
- Blocks pushes containing secrets when push protection is enabled
- Alerts repository administrators of detected secrets

**Best Practices:**
- Never commit secrets to the repository
- Use GitHub Actions Secrets for CI/CD credentials
- Use environment variables for runtime secrets
- Reference: `env.example` for required environment variables

### Dependabot Security Updates

**Status:** ✅ Enabled via `.github/dependabot.yml`

Dependabot automatically:
- Scans for vulnerable dependencies
- Creates PRs for security updates (immediately, not grouped)
- Creates PRs for regular updates (grouped and scheduled)

See [Dependency Management](#dependency-management) section for details.

---

## Branch Protection

### Configuration Required

**⚠️ Important:** Branch protection rules must be configured in repository settings.

**To Configure:**

1. Go to **Settings → Branches**
2. Add rule for branch `main`
3. Apply the following settings:

#### Required Settings

**Protect matching branches:**
- ✅ **Require a pull request before merging**
  - ✅ Require approvals: **1** (or **2** for security-sensitive changes)
  - ✅ Dismiss stale pull request approvals when new commits are pushed
  - ✅ Require review from Code Owners
  
- ✅ **Require status checks to pass before merging**
  - ✅ Require branches to be up to date before merging
  - **Required status checks:**
    - `Lint (Ruff)` (from ci-tests.yml)
    - `Type Check (Mypy)` (from ci-tests.yml)
    - `Tests (Python 3.10)` (from ci-tests.yml)
    - `Tests (Python 3.11)` (from ci-tests.yml)
    - `Tests (Python 3.12)` (from ci-tests.yml)
    - `Property-Based Tests` (from ci-tests.yml)
    - `Analyze Python Code` (from codeql-analysis.yml)
    - `All CI Checks Passed` (from ci-tests.yml)

- ✅ **Require conversation resolution before merging**
  - Ensures all review comments are addressed

- ✅ **Do not allow bypassing the above settings**
  - Applies to administrators (recommended for high-security repos)

#### Optional Settings

- ✅ **Require linear history** - prevents merge commits, enforces rebase or squash
- ✅ **Require signed commits** - enforces GPG signature verification

### For Feature Branches

Consider applying similar (but potentially less strict) rules to `feature/*` branches if needed.

---

## Automated PR Management

### PR Template (`.github/pull_request_template.md`)

Every PR uses a standardized template that requires:

**Required Sections:**
- **Description** - what the PR does
- **Type of Change** - bug fix, feature, breaking change, etc.
- **Scope of Changes** - affected components
- **Security Impact Assessment** - security risk level and considerations
- **Testing** - tests run and results
- **Regression Risk** - risk level and justification

**Benefits:**
- Ensures thorough consideration of security implications
- Standardizes information for reviewers
- Improves PR quality and review efficiency

### CODEOWNERS (`.github/CODEOWNERS`)

Automatically requests reviews from designated owners when PRs touch specific code areas:

**Protected Areas:**
- Core architecture: `src/mlsdm/core/`, `src/mlsdm/engine/`
- Security: `src/mlsdm/security/`, security-related utils
- Configuration: `src/mlsdm/utils/config_*`
- API: `src/mlsdm/api/`
- Documentation: Security, SLO, architecture docs

**Owner:** `@neuron7x` (adjust as team grows)

### Auto-Labeling

PRs are automatically labeled based on changed files:
- `core` - core engine changes
- `security` - security-related changes
- `api` - API changes
- `documentation` - doc changes
- `tests` - test changes
- `ci` - CI/CD changes
- etc.

---

## Dependency Management

### Dependabot Configuration (`.github/dependabot.yml`)

**Python Dependencies:**
- **Schedule:** Weekly on Mondays at 09:00 UTC
- **Limit:** 10 concurrent PRs
- **Grouping:**
  - Development dependencies (pytest, mypy, ruff, etc.) - minor/patch updates
  - Production dependencies (fastapi, numpy, etc.) - patch updates only
  - Security updates - always created individually
- **Ignored:** Major version updates (for stability)

**GitHub Actions:**
- **Schedule:** Weekly on Mondays at 09:00 UTC
- **Limit:** 5 concurrent PRs
- Keeps workflow actions up to date

**Workflow:**
1. Dependabot creates PR with dependency update
2. CI runs automatically
3. Review and merge if checks pass
4. For security updates, prioritize review and merge

---

## Secret Management

### Best Practices

1. **Never commit secrets** to the repository
2. **Use GitHub Actions Secrets** for CI/CD:
   - Go to **Settings → Secrets and variables → Actions**
   - Add secrets (they won't be visible after creation)
   - Reference in workflows as `${{ secrets.SECRET_NAME }}`

3. **Use environment variables** at runtime:
   - Reference `env.example` for required variables
   - Set in deployment environment (Docker, Kubernetes, etc.)

4. **Rotate secrets regularly**
5. **Audit secret usage** - check who has access

### GitHub Actions Secrets

Currently used secrets (if any):
- `GITHUB_TOKEN` - automatically provided, used for PR comments, labels, etc.

To add more:
```yaml
# In workflow file
env:
  API_KEY: ${{ secrets.API_KEY }}
```

---

## Configuration Guide

### Initial Setup Checklist

- [x] 1. CI workflows created (ci-tests.yml, codeql-analysis.yml)
- [x] 2. Dependabot configuration added
- [x] 3. PR template created
- [x] 4. CODEOWNERS file created
- [x] 5. Auto-labeler configured
- [ ] 6. **Enable Secret Scanning** (in Settings)
- [ ] 7. **Enable Push Protection** (in Settings)
- [ ] 8. **Configure Branch Protection** (in Settings)

### Post-Merge Setup (Repository Settings)

After this PR is merged, complete the following in repository settings:

#### 1. Enable Secret Scanning

**Settings → Code security and analysis:**
- Click **Enable** for "Secret scanning"
- Click **Enable** for "Push protection" (recommended)

#### 2. Configure Branch Protection

**Settings → Branches → Add rule:**

**Branch name pattern:** `main`

**Protection rules:**
1. ✅ Require a pull request before merging
   - Required approvals: 1
   - Dismiss stale reviews: Yes
   - Require review from Code Owners: Yes

2. ✅ Require status checks to pass before merging
   - Require branches to be up to date: Yes
   - Add required checks (listed in [Branch Protection](#branch-protection))

3. ✅ Require conversation resolution before merging

4. ✅ Do not allow bypassing settings (optional but recommended)

5. Save changes

#### 3. Verify Configuration

After setup:
1. Create a test PR
2. Verify all CI checks run
3. Verify labels are applied automatically
4. Verify CODEOWNERS review requests
5. Verify branch protection prevents merge until checks pass

---

## Monitoring & Maintenance

### Weekly Tasks
- Review Dependabot PRs and merge approved updates
- Check CodeQL alerts in Security tab
- Review any secret scanning alerts

### Monthly Tasks
- Audit CODEOWNERS assignments
- Review and update branch protection rules if needed
- Check CI workflow performance and optimization opportunities

### Quarterly Tasks
- Review security posture
- Update documentation
- Evaluate new GitHub security features

---

## Support & Questions

For questions about this setup:
- Review this document
- Check GitHub documentation:
  - [Branch protection rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
  - [CodeQL](https://docs.github.com/en/code-security/code-scanning/introduction-to-code-scanning/about-code-scanning-with-codeql)
  - [Dependabot](https://docs.github.com/en/code-security/dependabot)
  - [Secret scanning](https://docs.github.com/en/code-security/secret-scanning/about-secret-scanning)
- Open an issue in the repository

---

## Summary

This security and CI/CD setup provides:
- ✅ Automated quality checks on every PR
- ✅ Comprehensive security scanning (code, dependencies, secrets)
- ✅ Branch protection preventing unsafe merges
- ✅ Automated dependency updates
- ✅ Clear code ownership
- ✅ Standardized PR process

**Result:** High confidence in code quality, security, and stability of the MLSDM neuro-cognitive architecture.
